import os
import random
from typing import Iterator, List, Optional, Tuple, Union

import cv2
import imageio
import numpy as np
import rawpy
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


def bayer2rgbg(bayer):
  H, W = bayer.shape
  rgbg = np.zeros((H//2, W//2, 4), dtype=np.float32)
  rgbg[..., 0] = bayer[0::2, 0::2]
  rgbg[..., 1] = bayer[1::2, 0::2]
  rgbg[..., 2] = bayer[1::2, 1::2]
  rgbg[..., 3] = bayer[0::2, 1::2]
  return rgbg

def rgbg2rgb(rgbg, combine_green_channel=True):
  rgb = np.zeros((*rgbg.shape[:2], 3), dtype=np.float32)
  rgb[..., 0] = rgbg[..., 0]
  rgb[..., 2] = rgbg[..., 2]
  if combine_green_channel:
    rgb[..., 1] = rgbg[..., 1]/2 + rgbg[..., 3]/2
  else:
    rgb[..., 1] = rgbg[..., 1]
  return rgb

def patches(img: np.ndarray, patch_size: int) -> Iterator[Tuple[np.ndarray, int, int]]:
  p = patch_size
  h, w = img.shape[:2]
  
  # get image patches
  dh, dw = 0, 0
  lh, lw = False, False
  while True:
    while True:
      patch = img[dh:dh+p, dw:dw+p, :]
      yield patch, dh, dw
      
      if lw: break
      
      dw += p
      if dw >= w-p:
        dw = w-p
        lw = True
        
    if lh: break
    
    dw = 0
    lw = False
    dh += p
    if dh >= h-p:
      dh = h-p
      lh = True

def adjust_wb(img, wb):
  img_copy = img.copy()
  for i in range(len(wb)):
    img_copy[..., i] *= wb[i]
  return img_copy

def calculate_wb(a, b):
  wb = [a[..., i].mean() / b[..., i].mean() for i in range(a.shape[2])]
  return wb

def cramp(img, percentile):
  u = np.percentile(img, percentile)
  d = np.percentile(img, 100-percentile)
  img[img > u] = u
  u, d = img.max(), img.min()
  img = (img - d) / (u - d) * 255.
  img = img.astype(np.uint8)
  return img

def norm(img, up, down):
  if up == down:
    return img
  
  out = (img.astype(np.float32) - down) / (up - down)
  return out
