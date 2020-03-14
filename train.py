import json
import os
import sys
from datetime import datetime
from typing import Iterable, List, Tuple

import rawpy
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

import utils


class Trainer:
  def __init__(self,
               device: torch.device,
               model: nn.Module,
               criterion: nn.Module,
               optimizer: torch.optim.Optimizer,
               train_loader: torch.utils.data.DataLoader,
               val_loader: torch.utils.data.DataLoader,
               n_epochs: int,
               patch_size: int,
               result_path: str,
               test_path: str=None,
               pretrained_path: str=None,
               checkpoint_path: str=None):
    self.device = device
    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer
    self.train_loader = train_loader
    self.test_path = test_path
    self.val_loader = val_loader
    self.n_epochs = n_epochs
    self.patch_size = patch_size
    self.result_path = result_path
    
    self.model = self.model.to(device)
    self.criterion = self.criterion.to(device)
    
    if checkpoint_path:
      checkpoint = torch.load(checkpoint_path)
      model.load_state_dict(checkpoint['model_state_dict'])
      self.model.eval()
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      epoch = checkpoint['epoch']
      loss = checkpoint['loss']
      # TODO: 
    elif pretrained_path:
      pretrained_model_state_dict = torch.load(pretrained_path)
      self.model.load_state_dict(pretrained_model_state_dict)
      self.model.eval()
  
  def train(self):
    history = {'loss': [], 'mean_loss': []}
    for epoch in range(1, self.n_epochs+1):
      with tqdm(desc='[%04d/%04d]'%(epoch, self.n_epochs),
                total=len(self.train_loader),
                unit='batch', ncols=128, miniters=1) as t:
        loss_list = []
        for batch_idx, (hr, lr) in enumerate(self.train_loader):
          hr = hr.to(self.device)
          lr = lr.to(self.device)
          
          self.optimizer.zero_grad()
          sr_list = self.model(lr)
          sr = sr_list[-1]
          
          # calculate loss
          loss = self.criterion(sr, hr)
          loss.backward()
          self.optimizer.step()
          
          # update tqdm
          loss_list.append(loss.item())
          mean_loss = sum(loss_list) / len(loss_list)
          t.set_postfix_str('loss %.4f'%mean_loss)
          t.update()
          
          # update history
          history['loss'].append(loss.item())
      history['mean_loss'].append(mean_loss)
      
      # save hr, lr, sr as example image
      self.save_example(epoch, hr[0], lr[0], sr[0])
      
      # TODO: validation
      self.validation()
    
    # save test images
    self.test()
    
    # save result graph
    print('Save history[loss] graph into \'loss.png\' ...')
    utils.save_graph(history['loss'], os.path.join(self.result_path, 'loss.png'))
    
    # save model
    self.save_model()
  
  def save_example(self, epoch, hr, lr, sr):
    hr = hr.cpu().detach()
    lr = lr.cpu().detach()
    sr = sr.cpu().detach()
    
    hrpath = os.path.join(self.result_path, '%04d-hr.png'%epoch)
    lrpath = os.path.join(self.result_path, '%04d-lr.png'%epoch)
    srpath = os.path.join(self.result_path, '%04d-sr.png'%epoch)
    
    lr = utils.demosaic_tensor(lr) # 4ch -> 3ch
    hr = utils.to_numpy_image(hr)
    sr = utils.to_numpy_image(sr)
    
    hr = utils.norm(hr, hr.max(), hr.min())
    lr = utils.norm(lr, lr.max(), lr.min())
    sr = utils.norm(sr, sr.max(), sr.min())
    
    hr = (255. * hr).astype(np.uint8)
    lr = (255. * lr).astype(np.uint8)
    sr = (255. * sr).astype(np.uint8)
    
    imageio.imwrite(hrpath, hr)
    imageio.imwrite(lrpath, lr)
    imageio.imwrite(srpath, sr)
    
  def test(self):
    torch.set_grad_enabled(False)
    if not os.path.exists(self.test_path):
      return
    
    files = os.listdir(self.test_path)
    files = filter(lambda x: x.endswith('.ARW'), files)
    files = list(files)
    for i, fname in tqdm(enumerate(files), desc='saving test images', unit='file'):
      name, ext = os.path.splitext(fname)
      lrpath = os.path.join(self.test_path, fname)
      hrpath = os.path.join(self.test_path, name + '.JPG')
      result = self.test_large_image(lrpath, hrpath)
      imageio.imwrite(os.path.join(self.result_path, 'test-%05d.png'%i), result)
    
    torch.set_grad_enabled(True)
  
  def test_large_image(self, lrpath, hrpath):
    # load image
    with rawpy.imread(lrpath) as raw:
      bayer = raw.raw_image_visible
      rggb = utils.demosaic_bayer(bayer)
      rggb = utils.percentile(rggb, 95)
      rggb = utils.norm(rggb)
      rgb = utils.demosaic(rggb)
      
      hr = imageio.imread(hrpath)
      hr = hr.astype(np.float32) / 255.
      
      # adjust wb
      wb = utils.calculate_wb(hr, rgb)
      rggb[:, :, 0] *= wb[0]
      rggb[:, :, 1] *= wb[1]
      rggb[:, :, 2] *= wb[1]
      rggb[:, :, 3] *= wb[2]
      rggb = utils.percentile(rggb, 95)
      rggb = utils.norm(rggb, rggb.max(), rggb.min())
      
      tr = transforms.ToTensor()

      result = np.zeros((*bayer.shape, 3), dtype=np.uint8)      
      p = self.patch_size
      lr_patches = utils.image_patches(rggb, p)
      for l, dh, dw in lr_patches:
        # make tensor
        l = tr(l)
        l = l.view((1, *l.shape))
        l = l.to(self.device)
        
        sr_list = self.model(l)
        sr = sr_list[-1]
        sr = utils.to_numpy_image(sr[0])
        sr = utils.limit(sr)
        sr = (255. * sr).astype(np.uint8)
        result[dh*2:(dh+p)*2, dw*2:(dw+p)*2, :] = sr
      return result
  
  def validation(self):
    #torch.set_grad_enabled(False)
    #torch.set_grad_enabled(True)
    pass
  
  def load_model(self):
    
    pass
  
  def save_model(self):
    
    pass
  
  def save_checkpoint(self):
    
    pass
  
  # TODO: make losses list to make loss plot
