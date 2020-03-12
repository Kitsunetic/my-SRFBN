import os
import pickle
import re
import random

import rawpy
import imageio
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import cv2

import utils


class SRRAW(torch.utils.data.Dataset):
  def __init__(self, dataset_path: str, patch_size: int, white_lv=16383, black_lv=512):
    super(SRRAW, self).__init__()
    self.patch_size = patch_size
    self.white_lv = white_lv
    self.black_lv = black_lv
    
    self.image_files = []
    for fname in os.listdir(dataset_path):
      name, ext = os.path.splitext(fname)
      if ext.lower() != '.arw':
        continue
      
      path = os.path.join(dataset_path, fname)
      self.image_files.append(path)

  def __getitem__(self, index):
    path = self.image_files[index]
    
    # load images
    with rawpy.imread(path) as raw:
      pp = raw.postprocess(gamma=(1, 1),
                           no_auto_bright=True,
                           use_camera_wb=True,
                           output_bps=16)
      pp = utils.percentile(pp, 95)
      pp = utils.norm(pp, pp.max(), pp.min())
      
      bayer = raw.raw_image_visible.copy()
    
    # calculate wb
    rggb = utils.demosaic_bayer(bayer)
    rggb = utils.percentile(rggb, 95)
    rggb = utils.norm(rggb)
    rgb = utils.demosaic(rggb)
    wb = utils.calculate_wb(pp, rgb)
    
    # make patch
    p = self.patch_size*2
    h, w = rggb.shape[:2]
    dh, dw = random.randint(0, h-p), random.randint(0, w-p)
    dh, dw = dh - dh%2, dw - dw%2
    hr = rggb[dh:dh+p, dw:dw+p]
    
    # norm
    hr[:, :, 0] *= wb[0]
    hr[:, :, 1] *= wb[1]
    hr[:, :, 2] *= wb[1]
    hr[:, :, 3] *= wb[2]
    hr = utils.percentile(hr, 95)
    hr = utils.norm(hr, hr.max(), hr.min())
    
    # make lr
    #lr = cv2.resize(hr, (p//2, p//2), interpolation=cv2.INTER_CUBIC)
    #lr = utils.percentile(lr, 95)
    #lr = utils.norm(lr, lr.max(), lr.min())
    lr = hr[::2, ::2, :]
    
    # transform
    tr = transforms.ToTensor()
    hr = tr(hr)
    lr = tr(lr)
    pp = tr(pp)
    
    return hr, lr, wb, pp

  def __len__(self):
    return len(self.image_files)

class ImageSet(torch.utils.data.Dataset):
  def __init__(self, dataset_path: str, patch_size: int):
    super(ImageSet, self).__init__()
    
    self.patch_size = patch_size
    
    self.image_files = []
    for imfile in sorted(os.listdir(dataset_path)):
      ext = os.path.splitext(imfile)[1].lower()
      if ext not in ['.jpg', '.png']:
        continue
      
      impath = os.path.join(dataset_path, imfile)
      self.image_files.append(impath)
    
    #print(self.image_files)
    #exit()

  def __getitem__(self, idx: int):
    hr = imageio.imread(self.image_files[idx])
    
    # make patch
    p = self.patch_size
    h, w = hr.shape[:2]
    dh, dw = random.randint(0, h-p), random.randint(0, w-p)
    hr = hr[dh:dh+p, dw:dw+p, :]
    
    lr = hr.copy()
    #lr = lr[0::2, 0::2, :]
    lr = Image.fromarray(lr)
    trans_hr = transforms.Compose([
      transforms.ToTensor()
    ])
    trans_lr = transforms.Compose([
      transforms.Resize((hr.shape[0]//2, hr.shape[1]//2)),
      transforms.ToTensor()
    ])
    hr = trans_hr(hr)
    lr = trans_lr(lr)
    
    return hr, lr

  def __len__(self):
    return len(self.image_files)
