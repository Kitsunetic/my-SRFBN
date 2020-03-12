import json
import os
import sys
from datetime import datetime
from typing import Iterable, List, Tuple

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
               test_loader: torch.utils.data.DataLoader,
               val_loader: torch.utils.data.DataLoader,
               n_epochs: int,
               result_path: str,
               pretrained_path: str=None,
               checkpoint_path: str=None):
    self.device = device
    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.val_loader = val_loader
    self.n_epochs = n_epochs
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
    history = {'loss': []}
    for epoch in range(1, self.n_epochs+1):
      with tqdm(desc='[%04d/%04d]'%(epoch, self.n_epochs),
                total=len(self.train_loader),
                unit='batch', ncols=128, miniters=1) as t:
        loss_list = []
        for batch_idx, (hr, lr, inter_res) in enumerate(self.train_loader):
          hr = hr.to(self.device)
          lr = lr.to(self.device)
          inter_res = inter_res.to(self.device)
          
          self.optimizer.zero_grad()
          sr_list = self.model(lr, inter_res)
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
          
      # save hr, lr, sr as example image
      self.save_example(epoch, hr[0], lr[0], sr[0], inter_res[0])
      
      # test
      self.validation()
      
    # save result graph
    print('Save history[loss] graph into \'loss.png\' ...')
    utils.save_graph(history['loss'], os.path.join(self.result_path, 'loss.png'))
    
    # save model
    self.save_model()
  
  def save_example(self, epoch, hr, lr, sr, inter_res):
    hr = hr.cpu().detach()
    lr = lr.cpu().detach()
    sr = sr.cpu().detach()
    inter_res = inter_res.cpu().detach()
    
    hr = utils.to_numpy_image(hr)
    sr = utils.to_numpy_image(sr)
    inter_res = utils.to_numpy_image(inter_res)
    
    hrpath = os.path.join(self.result_path, '%04d-hr.png'%epoch)
    lrpath = os.path.join(self.result_path, '%04d-lr.png'%epoch)
    srpath = os.path.join(self.result_path, '%04d-sr.png'%epoch)
    pppath = os.path.join(self.result_path, '%04d-inter-res.png'%epoch)
    
    # decrease channel. 4ch -> 3ch
    lr = utils.demosaic_tensor(lr)
    
    hr = utils.norm(hr, hr.max(), hr.min())
    lr = utils.norm(lr, lr.max(), lr.min())
    sr = utils.norm(sr, sr.max(), sr.min())
    inter_res = utils.norm(inter_res, sr.max(), sr.min())
    
    hr = (255. * hr).astype(np.uint8)
    lr = (255. * lr).astype(np.uint8)
    sr = (255. * sr).astype(np.uint8)
    inter_res = (255. * inter_res).astype(np.uint8)
    
    imageio.imwrite(hrpath, hr)
    imageio.imwrite(lrpath, lr)
    imageio.imwrite(srpath, sr)
    imageio.imwrite(pppath, inter_res)
    
  def test(self):
    #torch.set_grad_enabled(False)
    #torch.set_grad_enabled(True)
    pass

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
