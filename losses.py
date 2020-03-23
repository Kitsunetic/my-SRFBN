from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg as vgg

LOSS_TYPES = ['cosine', 'l1', 'l2']


class VGG19(nn.Module):
  def __init__(self, requires_grad=False):
    super(VGG19, self).__init__()
    vgg_pretrained_features  = vgg.vgg19(pretrained=True).features
    self.slice1 = nn.Sequential()
    self.slice2 = nn.Sequential()
    self.slice3 = nn.Sequential()
    self.slice4 = nn.Sequential()
    self.slice5 = nn.Sequential()
    for x in range(4):
      self.slice1.add_module(str(x), vgg_pretrained_features[x])
    for x in range(4, 9):
      self.slice2.add_module(str(x), vgg_pretrained_features[x])
    for x in range(9, 18):
      self.slice3.add_module(str(x), vgg_pretrained_features[x])
    for x in range(18, 27):
      self.slice4.add_module(str(x), vgg_pretrained_features[x])
    for x in range(27, 36):
      self.slice5.add_module(str(x), vgg_pretrained_features[x])
    if not requires_grad:
      for param in self.parameters():
        param.requires_grad = False

  def forward(self, x):
    h = self.slice1(x)
    h_relu1_2 = h
    h = self.slice2(h)
    h_relu2_2 = h
    h = self.slice3(h)
    h_relu3_4 = h
    h = self.slice4(h)
    h_relu4_4 = h
    h = self.slice5(h)
    h_relu5_4 = h
    
    vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4'])
    out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_4, h_relu4_4, h_relu5_4)
    return out

def compute_cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  # mean shifting by channel-wise mean of `y`
  y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
  x_centered = x - y_mu
  y_centered = y - y_mu
  
  # L2 normalization
  x_normalized = F.normalize(x_centered, p=2, dim=1)
  y_normalized = F.normalize(y_centered, p=2, dim=1)
  
  # channel-wise vectorization(flattening)
  N, C, *_ = x.size()
  x_normalized = x_normalized.reshape(N, C, -1) # (N, C, H*W)
  y_normalized = y_normalized.reshape(N, C, -1) # (N, C, H*W)
  
  # cosine similarity
  cosine_sim = torch.bmm(x_normalized.transpose(1, 2), 
                         y_normalized) # (N, H*W, H*W)
  
  # convert to distance
  dist = 1 - cosine_sim
  
  return dist

def compute_l1_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  assert x.size() == y.size(), \
    f'expected input tensors have same size but {x.size()} and {y.size()}'
  
  N, C, H, W = x.size()
  x_vec = x.view(N, C, -1) # (N, C, H*W)
  y_vec = y.view(N, C, -1) # (N, C, H*W)
  
  dist = x_vec.unsqueeze(2) - y_vec.unsqueeze(3) # (N, C, H*W, H*W)
  dist = dist.sum(dim=1).abs() # (N, H*W, H*W)
  dist = dist.transpose(1, 2).reshape(N, H*W, H*W) # (N, H*W, H*W)
  dist = dist.clamp(min=0.) # remove negative
  
  return dist

def compute_l2_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  assert x.size() == y.size(), \
    f'expected input tensors have same size but {x.size()} and {y.size()}'
  
  N, C, H, W = x.size()
  x_vec = x.view(N, C, -1)
  y_vec = y.view(N, C, -1)
  x_s = torch.sum(x_vec ** 2, dim=1)
  y_s = torch.sum(y_vec ** 2, dim=1)
  
  A = y_vec.transpose(1, 2) @ x_vec
  dist = y_s - 2*A + x_s.transpose(0, 1)
  dist = dist.transpose(1, 2).reshape(N, H*W, H*W)
  dist = dist.clamp(min=0.)
  
  return dist

def compute_relative_distance(dist_raw: torch.Tensor, eps=1e-5):
  dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
  dist_tilde = dist_raw / (dist_min + eps)
  return dist_tilde

def compute_cx(dist_tilde: torch.Tensor, bandwidth: float):
  w = torch.exp((1 - dist_tilde) / bandwidth) # Eq(3)
  cx = w / torch.sum(w, dim=2, keepdim=True) # Eq(4)
  return cx

def compute_meshgrid(shape):
  N, C, H, W = shape
  rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
  cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)
  
  feature_grid = torch.meshgrid(rows, cols)
  feature_grid = torch.stack(feature_grid).unsqueeze(0)
  feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)
  
  return feature_grid

def calculate_contextual_loss(x: torch.Tensor, y: torch.Tensor,
                              bandwidth: float, loss_type: str, eps=1e-5) -> torch.Tensor:
  """
  Computes contextual loss between x and y.
  The most of this code is copied from
      https://gist.github.com/yunjey/3105146c736f9c1055463c33b4c989da.

  Parameters
  ---
  x : torch.Tensor
      features of shape (N, C, H, W).
  y : torch.Tensor
      features of shape (N, C, H, W).
  band_width : float, optional
      a band-width parameter used to convert distance to similarity.
      in the paper, this is described as :math:`h`.
  loss_type : str, optional
      a loss type to measure the distance between features.
      Note: `l1` and `l2` frequently raises OOM.

  Returns
  ---
  cx_loss : torch.Tensor
      contextual loss between x and y (Eq (1) in the paper)
  """
  
  assert x.size() == y.size(), 'input tensors must have the same size.'
  assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'
  
  N, C, H, W = x.size()
  
  if loss_type == 'cosine':
    dist_raw = compute_cosine_distance(x, y)
  elif loss_type == 'l1':
    dist_raw = compute_l1_distance(x, y)
  elif loss_type == 'l2':
    dist_raw = compute_l2_distance(x, y)
  
  dist_tilde = compute_relative_distance(dist_raw, eps=eps)
  cx = compute_cx(dist_tilde, bandwidth)
  cx = torch.mean(torch.max(cx, dim=1)[0], dim=1)
  cx_loss = torch.mean(-torch.log(cx + eps))
  
  return cx_loss

def calculate_contextual_bilateral_loss(x: torch.Tensor,
                                        y: torch.Tensor,
                                        weight_sp: float=0.1,
                                        bandwidth: float=1.,
                                        loss_type: str='cosine',
                                        eps: float=1e-5) -> torch.Tensor:
  """
  Computes Contextual Bilateral (CoBi) Loss between x and y,
      proposed in https://arxiv.org/pdf/1905.05169.pdf.

  Parameters
  ---
  x : torch.Tensor
      features of shape (N, C, H, W).
  y : torch.Tensor
      features of shape (N, C, H, W).
  band_width : float, optional
      a band-width parameter used to convert distance to similarity.
      in the paper, this is described as :math:`h`.
  loss_type : str, optional
      a loss type to measure the distance between features.
      Note: `l1` and `l2` frequently raises OOM.

  Returns
  ---
  cx_loss : torch.Tensor
      contextual loss between x and y (Eq (1) in the paper).
  k_arg_max_NC : torch.Tensor
      indices to maximize similarity over channels.
  """
  
  assert x.size() == y.size(), 'input tensors must have the same size.'
  assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'
  
  # spatial loss
  grid = compute_meshgrid(x.shape).to(x.device)
  dist_raw = compute_l2_distance(grid, grid)
  dist_tilde = compute_relative_distance(dist_raw)
  cx_sp = compute_cx(dist_tilde, bandwidth)
  
  # feature loss
  if loss_type == 'cosine':
    dist_raw = compute_cosine_distance(x, y)
  elif loss_type == 'l1':
    dist_raw = compute_l1_distance(x, y)
  elif loss_type == 'l2':
    dist_raw = compute_l2_distance(x, y)
  dist_tilde = compute_relative_distance(dist_raw, eps=eps)
  cx_feat = compute_cx(dist_tilde, bandwidth)
  
  # combined loss
  cx_combine = (1. - weight_sp)*cx_feat + weight_sp*cx_sp
  
  k_max_NC, _ = torch.max(cx_combine, dim=2, keepdim=True)
  
  cx = k_max_NC.mean(dim=1)
  cx_loss = torch.mean(-torch.log(cx + eps))
  
  return cx_loss

class ContextualLoss(nn.Module):
  """
  Creates a criterion that measures the contextual loss.

  Parameters
  ---
  band_width : int, optional
    a band_width parameter described as :math:`h` in the paper.
  loss_type : str, optional
    a loss type to measure the distance between features.
    Note: `l1` and `l2` frequently raises OOM.
  use_vgg : bool, optional
    if you want not to use VGG feature, set this `False`.
  vgg_layer : str, optional
    intermidiate layer name for VGG feature.
    Now we support layer names:
      `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
  """
  def __init__(self, 
               bnadwidth=0.5, 
               loss_type='cosine', 
               use_vgg=True, 
               vgg_layer='relu3_4',
               eps=1e-5):
    super(ContextualLoss, self).__init__()
    
    assert bnadwidth > 0, 'bandwidth must be positive.'
    assert loss_type in LOSS_TYPES, f'select loss type from {LOSS_TYPES}.'
    
    self.bnadwidth = bnadwidth
    self.loss_type = loss_type
    self.eps = eps
    
    if use_vgg:
      self.vgg_model = VGG19()
      self.vgg_layer = vgg_layer
      self.register_buffer(
        'vgg_mean', 
        torch.tensor([[[0.485]], [[0.456]], [[0.406]]], requires_grad=False)
      )
      self.register_buffer(
        'vgg_std', 
        torch.tensor([[[0.229]], [[0.224]], [[0.225]]], requires_grad=False)
      )

  def forward(self, x: torch.Tensor, y: torch.Tensor):
    if hasattr(self, 'vgg_model'):
      assert x.shape[1] == 3 and y.shape[1] == 3, \
        f'Expected 3 channel input but inputs have {x.shape[1]}, {y.shape[1]}.'
      
      # normalization
      x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
      y = y.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
      
      # picking up vgg feature maps
      x = getattr(self.vgg_model(x), self.vgg_layer)
      y = getattr(self.vgg_model(y), self.vgg_layer)
    
    return calculate_contextual_loss(x, y, 
                                     bandwidth=self.bnadwidth, 
                                     loss_type=self.loss_type, 
                                     eps=self.eps)

class ContextualBilateralLoss(nn.Module):
  """
  Creates a criterion that measures the contextual bilateral loss.

  Parameters
  ---
  weight_sp : float, optional
      a balancing weight between spatial and feature loss.
  band_width : int, optional
      a band_width parameter described as :math:`h` in the paper.
  use_vgg : bool, optional
      if you want not to use VGG feature, set this `False`.
  vgg_layer : str, optional
      intermidiate layer name for VGG feature.
      Now we support layer names:
          `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
  """
  
  def __init__(self,
               weight_sp=0.1,
               bandwidth=0.5,
               loss_type='cosine',
               use_vgg=True,
               vgg_layer='relu3_4',
               eps=1e-5):
    super(ContextualBilateralLoss, self).__init__()
    
    assert bandwidth > 0, 'bandwidth must be positive.'
    assert loss_type in LOSS_TYPES, \
      f'select a loss type from {LOSS_TYPES}.'
    
    self.weight_sp = weight_sp
    self.bandwidth = bandwidth
    self.loss_type = loss_type
    self.eps = eps
    
    if use_vgg:
      self.vgg_model = VGG19()
      self.vgg_layer = vgg_layer
      self.register_buffer(
        name='vgg_mean',
        tensor=torch.tensor([[[0.485]], [[0.456]], [[0.406]]], requires_grad=False)
      )
      self.register_buffer(
        name='vgg_std',
        tensor=torch.tensor([[[0.229]], [[0.224]], [[0.225]]], requires_grad=False)
      )
    
  def forward(self, x, y):
    if hasattr(self, 'vgg_model'):
      assert x.shape[1] == 3 and y.shape[1] == 3, \
        f'Expected 3 channel input but inputs have {x.shape[1]}, {y.shape[1]}.'
      
      # normalization
      x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
      y = y.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
      
      # picking up vgg feature map
      x = getattr(self.vgg_model(x), self.vgg_layer)
      y = getattr(self.vgg_model(y), self.vgg_layer)
      
    return calculate_contextual_bilateral_loss(x, y, 
                                               weight_sp=self.weight_sp,
                                               bandwidth=self.bandwidth, 
                                               loss_type=self.loss_type,
                                               eps=self.eps)
