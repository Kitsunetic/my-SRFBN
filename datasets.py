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
      hrpath = os.path.join(dataset_path, name + '.JPG')
      lrpath = os.path.join(dataset_path, fname)
      self.image_files.append((hrpath, lrpath))

  def __getitem__(self, index):
    hrpath, lrpath = self.image_files[index]
    
    # load images and norm
    with rawpy.imread(lrpath) as raw:
      bayer = raw.raw_image_visible.copy()
    hr = imageio.imread(hrpath)
    hr = hr.astype(np.float32) / 255.
    
    # crop
    bayer = bayer[8:-8, 8:-8]
    if bayer.shape[0] > hr.shape[0]:
      bayer = bayer[224:-224, :]
    
    # norm
    rggb = utils.demosaic_bayer(bayer)
    rggb = utils.percentile(rggb, 95)
    rggb = utils.norm(rggb)
    rgb = utils.demosaic(rggb)
    
    # adjust wb
    wb = utils.calculate_wb(hr, rgb)
    rggb[:, :, 0] *= wb[0]
    rggb[:, :, 1] *= wb[1]
    rggb[:, :, 2] *= wb[1]
    rggb[:, :, 3] *= wb[2]
    rggb = utils.percentile(rggb, 95)
    rggb = utils.norm(rggb, rggb.max(), rggb.min())
    
    # make patch
    p = self.patch_size
    h, w = rggb.shape[:2]
    dh, dw = random.randint(0, h-p), random.randint(0, w-p)
    dh, dw = dh - dh%2, dw - dw%2
    lr = rggb[dh:dh+p, dw:dw+p, :]
    hr = hr[dh*2:(dh+p)*2, dw*2:(dw+p)*2, :]
    rgb_patch = rgb[dh:dh+p, dw:dw+p, :]
    
    # transform
    tr = transforms.ToTensor()
    hr = tr(hr)
    lr = tr(lr)
    
    return hr, lr

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
