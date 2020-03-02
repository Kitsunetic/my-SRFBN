import os
import json
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

import options
import models
import train
import datasets


def main():
  args = options.parser.parse_args(sys.argv[1:])
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # load dataset
  trainset = datasets.ImageSet(args.dataset_path, args.patch_size)
  trainloader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=args.batch_size,
                                            shuffle=args.shuffle,
                                            num_workers=args.num_workers)
  
  # load color_mean
  color_path = os.path.join(args.dataset_path, 'color.json')
  with open(color_path, 'r') as f:
    data = json.load(f)
    color_mean = data['color_mean']
    color_std = data['color_std']
  
  # make model
  model = models.SRFBN(device=device,
                       in_channels=args.in_channels,
                       out_channels=args.out_channels,
                       n_features=args.n_features,
                       n_steps=args.n_steps,
                       n_groups=args.n_groups,
                       upscale_factor=args.scale,
                       color_mean=color_mean,
                       color_std=color_std,
                       act_type='relu',
                       norm_type=None)
  model = model.to(device)
  
  # make loss
  criterion = nn.L1Loss()
  optimizer = torch.optim.Adam(model.parameters())
  
  # train
  for epoch in range(1, args.n_epochs+1):
    with tqdm(desc='[%04d/%04d]'%(epoch, args.n_epochs), total=len(trainloader), unit='batch', miniters=1) as t:
      for batch_idx, (hr, lr) in enumerate(trainloader):
        #print(hr, hr.shape)
        #print(lr, lr.shape)
        #exit()
        
        hr, lr = hr.to(device), lr.to(device)
        sr = model(lr)
        loss = criterion(sr, hr)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        t.set_postfix_str('loss%.4f'%loss.item())
        t.update()
    
    # save result
    tr = transforms.ToPILImage()
    imhr = tr(hr[0].cpu())
    imlr = tr(lr[0].cpu())
    imsr = tr(sr[0].cpu())
    imhr.save(os.path.join(args.result_path, '%05d-hr.png'%epoch))
    imlr.save(os.path.join(args.result_path, '%05d-lr.png'%epoch))
    imsr.save(os.path.join(args.result_path, '%05d-sr.png'%epoch))

if __name__ == "__main__":
  main()
