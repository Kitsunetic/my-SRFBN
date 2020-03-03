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


class SRRAW(torch.utils.data.Dataset):
  def __init__(self, dataset_path: str, patch_size: int, black_lv=512, white_lv=16384):
    super(SRRAW, self).__init__()
    self.patch_size = patch_size
    self.black_lv = black_lv
    self.white_lv = white_lv
    
    self.image_files = []
    for fname in os.listdir(dataset_path):
      name, ext = os.path.splitext(fname)
      if ext.lower() != '.arw':
        continue
      
      lrpath = os.path.join(dataset_path, fname)
      hrpath = os.path.join(dataset_path, name + '.JPG')
      self.image_files.append((hrpath, lrpath))

  def __getitem__(self, index):
    hrpath, lrpath = self.image_files[index]
    
    # load images
    with rawpy.imread(lrpath) as raw:
      lr = raw.raw_image_visible.copy()[8:-8, 8:-8]
    hr = imageio.imread(hrpath)
    
    # crop
    if hr.shape[0] < lr.shape[0]:
      lr = lr[224:-224, :]
    
    # make patch
    p = self.patch_size
    h, w = lr.shape
    dh = random.randint(0, h-p)
    dw = random.randint(0, w-p)
    hr = hr[dh:dh+p, dw:dw+p, :]
    lr = lr[dh:dh+p, dw:dw+p]
    
    # norm
    hr = hr.astype(np.float32) / 255.
    lr = (lr.astype(np.float32)-self.black_lv) / (self.white_lv-self.black_lv)
    
    # make raw to 4ch
    lr_ = np.zeros((p//2, p//2, 4), dtype=np.float32)
    lr_[:, :, 0] = lr[0::2, 0::2]
    lr_[:, :, 1] = lr[1::2, 0::2]
    lr_[:, :, 2] = lr[1::2, 1::2]
    lr_[:, :, 3] = lr[0::2, 1::2]
    lr = lr_
    
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
