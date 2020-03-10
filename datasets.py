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
      
      path = os.path.join(dataset_path, fname)
      self.image_files.append(path)

  def __getitem__(self, index):
    path = self.image_files[index]
    
    # load images
    with rawpy.imread(path) as raw:
      bayer = raw.raw_image_visible.copy()
      pp = raw.postprocess(gamma=(1, 1),
                           no_auto_bright=True,
                           use_camera_wb=True,
                           output_bps=16)
    
    # calculate wb
    rgbg = utils.demosaic_bayer(bayer)
    wb = utils.calculate_wb(pp, rgbg)
    
    # make patch
    p = self.patch_size
    h, w = bayer.shape
    dh, dw = random.randint(0, h-p), random.randint(0, w-p)
    dh, dw = dh - dh%2, dw - dw%2
    hr = rgbg

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
