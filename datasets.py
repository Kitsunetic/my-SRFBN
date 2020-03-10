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
  def __init__(self, dataset_path: str, patch_size: int, white_lv=16384, black_lv=512):
    super(SRRAW, self).__init__()
    self.patch_size = patch_size
    self.white_lv = white_lv
    self.black_lv = black_lv
    
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
    dh, dw = random.randint(0, h-p), random.randint(0, w-p)
    dh, dw = dh - dh%2, dw - dw%2
    hr = hr[dh:dh+p, dw:dw+p, :]
    lr = lr[dh:dh+p, dw:dw+p]

    # norm
    hr = hr.astype(np.float32) / 255.
    lr = utils.norm(lr)
    
    # get wb
    lr = utils.demosaic_bayer(lr)
    pp = utils.demosaic(lr)
    wb = utils.calculate_wb(pp, hr)
    
    # adjust wb
    lr[..., 0] *= wb[0]
    lr[..., 1] *= wb[1]
    lr[..., 2] *= wb[1]
    lr[..., 3] *= wb[2]
    lr = utils.percentile(lr, 99)
    lr = utils.norm(lr, lr.max(), lr.min())
    
    # transform
    tr = transforms.ToTensor()
    hr = tr(hr)
    lr = tr(lr)
    pp = tr(pp)
    
    return hr, lr, pp

  def __len__(self):
    return len(self.image_files)

class RAW2RAW(torch.utils.data.Dataset):
  def __init__(self, dataset_path: str, patch_size: int, black_lv=512, white_lv=16384):
    super(RAW2RAW, self).__init__()
    self.patch_size = patch_size
    self.black_lv = black_lv
    self.white_lv = white_lv
    
    self.image_files = []
    for fname in os.listdir(dataset_path):
      name, ext = os.path.splitext(fname)
      if ext.lower() == '.arw':
        fpath = os.path.join(dataset_path, fname)
        self.image_files.append(fpath)

  def __getitem__(self, index):
    # load images
    impath = self.image_files[index]
    with rawpy.imread(impath) as raw:
      img = raw.raw_image_visible.copy()

    # make patch
    p = self.patch_size
    p2 = 2 * p # make 4ch == resize 0.5
    h, w = img.shape
    dh = random.randint(0, h-p2)
    dw = random.randint(0, w-p2)
    img = img[dh:dh+p2, dw:dw+p2]
    
    # norm
    img = img.astype(np.float32)
    img = (img-self.black_lv) / (self.white_lv-self.black_lv)
    
    # make raw to 4ch
    img_ = np.zeros((p, p, 4), dtype=np.float32)
    img_[:, :, 0] = img[0::2, 0::2]
    img_[:, :, 1] = img[1::2, 0::2]
    img_[:, :, 2] = img[1::2, 1::2]
    img_[:, :, 3] = img[0::2, 1::2]
    img = img_
    
    # make lr image
    lr = cv2.resize(img, (p//2, p//2), interpolation=cv2.INTER_CUBIC)
    
    # transform
    tr = transforms.ToTensor()
    hr = tr(img)
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

class SRRAW64(torch.utils.data.Dataset):
  def __init__(self, dataset_path: str, patch_size: int):
    super(SRRAW64, self).__init__()
    self.patch_size = patch_size
    
    self.file_list = list(map(lambda x: os.path.join(dataset_path, '%05d.pkl'%x), range(32)))

  def __getitem__(self, index):
    with open(self.file_list[index], 'rb') as f:
      data = pickle.load(f)
    hr = data['hr']
    lr = data['lr']
    wb2 = data['wb2']
    wb3 = data['wb3']
    
    # make patch
    p = self.patch_size
    h, w = lr.shape[:2]
    dh, dw = random.randint(0, h-p), random.randint(0, w-p)
    lr = lr[dh:dh+p, dw:dw+p, :]
    hr = hr[dh*2:(dh+p)*2, dw*2:(dw+p)*2]
    
    # tensor transform
    tr = transforms.ToTensor()
    lr = tr(lr)
    hr = tr(hr)
    
    return lr, hr, wb2, wb3

  def __len__(self):
    return len(self.file_list)
