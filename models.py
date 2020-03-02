import sys
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_valid_padding(kernel_size: int, dilation: int) -> int:
  kernel_size = kernel_size + (kernel_size-1)*(dilation-1)
  padding = (kernel_size-1) // 2
  return padding

def activation(act_type='relu', inplace=True, n_parameters=1, slope=0.2) -> nn.Module:
  if not act_type:
    return None
  elif act_type == 'relu':
    act = nn.ReLU(True)
  elif act_type == 'lrelu':
    act = nn.LeakyReLU(0.2, True)
  elif act_type == 'prelu':
    act = nn.PReLU(1, 0.2)
  else:
    raise NotImplementedError(act_type)
  return act

def batch_norm(n_features: int, norm_type='bn') -> nn.Module:
  if not norm_type:
    return None
  elif norm_type == 'bn':
    norm = nn.BatchNorm2d(n_features)
  else:
    raise NotImplementedError(norm_type)
  return norm

def ConvBlock(in_channels: int, 
              out_channels: int, 
              kernel_size: int, 
              stride=1, 
              padding=0, 
              valid_padding=True,
              dilation=1, 
              bias=True, 
              act_type='relu', 
              norm_type='bn') -> nn.Sequential:
  if valid_padding:
    padding = get_valid_padding(kernel_size, dilation)
  conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)
  act = activation(act_type)
  norm = batch_norm(out_channels, norm_type)
  
  m = []
  m.append(conv)
  if act: m.append(act)
  if norm: m.append(norm)
  return nn.Sequential(*m)

def DeconvBlock(in_channels: int, 
                out_channels: int, 
                kernel_size: int, 
                stride=1, 
                padding=0, 
                dilation=1, 
                bias=True, 
                act_type='relu', 
                norm_type='bn') -> nn.Sequential:
  # TODO: Why DeconvBlock doesn't have valid_padding?
  deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)
  act = activation(act_type)
  norm = batch_norm(out_channels, norm_type)
  
  m = []
  m.append(deconv)
  if act: m.append(act)
  if norm: m.append(norm)
  return nn.Sequential(*m)

class MeanShift(nn.Conv2d):
  def __init__(self, 
               color_mean: List[float], 
               color_std: List[float], 
               sign=-1):
    c = min(len(color_mean), len(color_std))
    super(MeanShift, self).__init__(c, c, 1)
    
    mean = torch.Tensor(color_mean[:c])
    std = torch.Tensor(color_std[:c])
    
    self.weight.data = torch.eye(c).view(c, c, 1, 1)
    self.weight.data.div_(std.view(c, 1, 1, 1))
    self.bias.data = sign * 255. * mean
    self.bias.data.div_(std)
    
    for p in self.parameters():
      p.requires_grad = False
    
class FeedbackBlock(nn.Module):
  def __init__(self,
               device: torch.device,
               n_features: int, 
               n_groups: int, 
               upscale_factor: int, 
               act_type='relu', 
               norm_type=None):
    super(FeedbackBlock, self).__init__()
    
    if upscale_factor == 2:
      stride = 2
      padding = 2
      kernel_size = 6
    elif upscale_factor == 3:
      stride = 3
      padding = 2
      kernel_size = 7
    elif upscale_factor == 4:
      stride = 4
      padding = 2
      kernel_size = 8
    elif upscale_factor == 8:
      stride = 8
      padding = 2
      kernel_size = 12
    
    self.device = device
    self.n_groups = n_groups
    
    self.compress_in = ConvBlock(2*n_features, n_features, 1, act_type=act_type, norm_type=norm_type)
    
    up_blocks = []
    down_blocks = []
    uptran_blocks = []
    downtran_blocks = []
    for idx in range(self.n_groups):
      up_blocks.append(DeconvBlock(n_features, n_features, kernel_size, stride, padding, act_type=act_type, norm_type=norm_type))
      down_blocks.append(ConvBlock(n_features, n_features, kernel_size, stride, padding, act_type=act_type, norm_type=norm_type))
      if idx > 0:
        uptran_blocks.append(ConvBlock(n_features*(idx+1), n_features, 1, 1, act_type=act_type, norm_type=norm_type))
        downtran_blocks.append(ConvBlock(n_features*(idx+1), n_features, 1, 1, act_type=act_type, norm_type=norm_type))
    self.up_blocks = nn.Sequential(*up_blocks)
    self.down_blocks = nn.Sequential(*down_blocks)
    self.uptran_blocks = nn.Sequential(*uptran_blocks)
    self.downtran_blocks = nn.Sequential(*downtran_blocks)

    self.compress_out = ConvBlock(n_groups*n_features, n_features, 1, act_type=act_type, norm_type=norm_type)
    
    self.should_reset = True
    self.last_hidden = None

  def forward(self, x):
    if self.should_reset:
      self.last_hidden = torch.zeros(x.size()).to(self.device)
      self.last_hidden.copy_(x)
      self.should_reset = False
    
    x = torch.cat((x, self.last_hidden), dim=1)
    x = self.compress_in(x)
    
    lr_features = []
    hr_features = []
    lr_features.append(x)
    
    for idx in range(self.n_groups):
      LD_L = torch.cat(tuple(lr_features), 1)
      if idx > 0:
        LD_L = self.uptran_blocks[idx-1](LD_L)
      LD_H = self.up_blocks[idx](LD_L)
      
      hr_features.append(LD_H)
      
      LD_H = torch.cat(tuple(hr_features), 1)
      if idx > 0:
        LD_H = self.downtran_blocks[idx-1](LD_H)
      LD_L = self.down_blocks[idx](LD_H)
      
      lr_features.append(LD_L)
    
    del hr_features
    output = torch.cat(tuple(lr_features[1:]), 1)
    output = self.compress_out(output)
    self.last_hidden = output
    return output
  
  def reset_state(self):
    self.should_reset = True

class SRFBN(nn.Module):
  def __init__(self, 
               device: torch.device,
               in_channels: int, 
               out_channels: int, 
               n_features: int, 
               n_steps: int, 
               n_groups: int, 
               upscale_factor: int, 
               color_mean: List[float], 
               color_std: List[float],
               act_type='relu', 
               norm_type=None):
    super(SRFBN, self).__init__()
    
    if upscale_factor == 2:
      stride = 2
      padding = 2
      kernel_size = 6
    elif upscale_factor == 3:
      stride = 3
      padding = 2
      kernel_size = 7
    elif upscale_factor == 4:
      stride = 4
      padding = 2
      kernel_size = 8
    elif upscale_factor == 8:
      stride = 8
      padding = 2
      kernel_size = 12
    
    in_color_mean = color_mean[:in_channels]
    in_color_std = color_std[:in_channels]
    out_color_mean = color_mean[:out_channels]
    out_color_std = color_std[:out_channels]
    
    self.n_steps = n_steps
    self.n_features = n_features
    self.upscale_factor = upscale_factor
    
    self.sub_mean = MeanShift(in_color_mean, in_color_std, sign=-1)
    
    # LR feature extraction block
    self.conv_in = ConvBlock(in_channels, 4*n_features, 3, act_type=act_type, norm_type=norm_type)
    self.feat_in = ConvBlock(4*n_features, n_features, 1, act_type=act_type, norm_type=norm_type)
    
    # basic block
    self.block = FeedbackBlock(device, n_features, n_groups, upscale_factor, act_type=act_type, norm_type=norm_type)
    
    self.out = DeconvBlock(n_features, n_features, kernel_size, stride, padding, act_type='prelu', norm_type=norm_type)
    self.conv_out = ConvBlock(n_features, out_channels, 3, act_type=act_type, norm_type=norm_type)
    
    self.add_mean = MeanShift(out_color_mean, out_color_std, sign=1)
    
  def forward(self, x):
    self._reset_state()
    
    x = self.sub_mean(x)
    
    inter_res = F.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
    
    # LRFB - LR Feature Block
    x = self.conv_in(x)
    x = self.feat_in(x)
    
    # FB - Feedback Block
    outs = []
    for _ in range(self.n_steps):
      h = self.block(x)
      
      # RB - Reconstruction Block
      h = self.out(h)
      h = self.conv_out(h)
      
      # add upsample(bilinear interpolation) and RB output
      h = torch.add(inter_res, h)
      outs.append(h)
    
    return outs

  def _reset_state(self):
    self.block.reset_state()
