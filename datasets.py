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
    
    # load images
    with rawpy.imread(lrpath) as raw:
      pp = raw.postprocess(gamma=(1, 1),
                           no_auto_bright=True,
                           use_camera_wb=True,
                           output_bps=16)
      pp = utils.percentile(pp, 95)
      pp = utils.norm(pp, pp.max(), pp.min())
      
      bayer = raw.raw_image_visible.copy()
    
    hr = imageio.imread(hrpath)
    hr = hr.astype(np.float32) / 255.
    
    # crop
    bayer = bayer[8:-8, 8:-8]
    pp = pp[8:-8, 8:-8, :]
    if bayer.shape[0] > hr.shape[0]:
      bayer = bayer[224:-224, :]
      pp = pp[224:-224, :, :]
    
    # calculate wb
    rggb = utils.demosaic_bayer(bayer)
    rggb = utils.percentile(rggb, 95)
    rggb = utils.norm(rggb)
    rgb = utils.demosaic(rggb)
    wb1 = utils.calculate_wb(hr, rgb)
    wb2 = utils.calculate_wb(hr, pp)
    
    # make patch
    p = self.patch_size
    h, w = rggb.shape[:2]
    dh, dw = random.randint(0, h-p), random.randint(0, w-p)
    dh, dw = dh - dh%2, dw - dw%2
    lr = rggb[dh:dh+p, dw:dw+p, :]
    hr = hr[dh*2:(dh+p)*2, dw*2:(dw+p)*2, :]
    inter_res = pp[dh*2:(dh+p)*2, dw*2:(dw+p)*2, :]
    
    # lr wb
    lr[:, :, 0] *= wb1[0]
    lr[:, :, 1] *= wb1[1]
    lr[:, :, 2] *= wb1[1]
    lr[:, :, 3] *= wb1[2]
    lr = utils.percentile(lr, 95)
    lr = utils.norm(lr, lr.max(), lr.min())
    
    # hr norm
    hr = utils.norm(hr, hr.max(), hr.min())
    
    # inter_res wb
    inter_res[:, :, 0] *= wb2[0]
    inter_res[:, :, 1] *= wb2[1]
    inter_res[:, :, 2] *= wb2[2]
    inter_res = utils.percentile(inter_res, 95)
    inter_res = utils.norm(inter_res, inter_res.max(), inter_res.min())
    
    # transform
    tr = transforms.ToTensor()
    hr = tr(hr)
    lr = tr(lr)
    inter_res = tr(inter_res)
    
    return hr, lr, inter_res

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
