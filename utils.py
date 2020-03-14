import os
import random
from typing import Iterator, List, Optional, Tuple, Union

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import rawpy
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

_to_pil_image = transforms.ToPILImage()


def save_tensor_image(tensor: torch.Tensor, path: str):
  img = _to_pil_image(tensor.cpu())
  img.save(path)
  img.close()

def image_patches(img: np.ndarray, patch_size: int) -> Iterator[Tuple[np.ndarray, int, int]]:
  p = patch_size
  h, w = img.shape[:2]
  
  # get image patches
  dh, dw = 0, 0
  lh, lw = False, False
  while True:
    while True:
      patch = img[dh:dh+p, dw:dw+p, :]
      yield patch, dh, dw
      
      if lw:
        break
      
      dw += p
      if dw >= w-p:
        dw = w-p
        lw = True
        
    if lh:
      break
    
    dw = 0
    lw = False
    dh += p
    if dh >= h-p:
      dh = h-p
      lh = True

def model_large_image(model: nn.Module, device: torch.device, impath: str, 
                      result_path: str, patch_size: int) -> torch.Tensor:
  with rawpy.imread(impath) as raw:
    img = raw.raw_image_visible.astype(np.float32)
    img = (img-512) / (16383-512)
    img_ = np.zeros((img.shape[0]//2, img.shape[1]//2, 4), dtype=np.float32)
    img_[:, :, 0] = img[0::2, 0::2]
    img_[:, :, 1] = img[1::2, 0::2]
    img_[:, :, 2] = img[1::2, 1::2]
    img_[:, :, 3] = img[0::2, 1::2]
    img = img_
    dst = np.zeros((img.shape[0]*2, img.shape[1]*2, 3), dtype=np.uint8)
    
    p = patch_size * 2
    for patch, dh, dw in image_patches(img, patch_size):
      patch = transforms.ToTensor()(patch)
      patch = patch.view((1, *patch.shape))
      patch = patch.to(device)
      res = model(patch)
      res = transforms.ToPILImage()(res[-1][0].cpu())
      dst[dh*2:dh*2+p, dw*2:dw*2+p, :] = np.array(res)
      res.close()
    imageio.imwrite(result_path, dst)

def save_4ch_image(img: np.ndarray, path: str, white_lv=16383, black_lv=512):
  # convert float32 to 16bit
  img_ = (img*(white_lv-black_lv) + black_lv).astype(np.uint16)
  
  # convert 4ch image into bayer
  h, w = img.shape[:2]
  bayer = np.zeros((h*2, w*2), dtype=np.uint16)
  bayer[0::2, 0::2] = img_[:, :, 0]
  bayer[1::2, 0::2] = img_[:, :, 1]
  bayer[1::2, 1::2] = img_[:, :, 2]
  bayer[0::2, 1::2] = img_[:, :, 3]
  
  # convert bayer to RGB
  rgb16 = cv2.cvtColor(bayer, cv2.COLOR_BAYER_BG2RGB)
  rgb8 = (255 * (rgb16.astype(np.float32) - black_lv) / (white_lv - black_lv)).astype(np.uint8)
  rgb8_pil = Image.fromarray(rgb8)
  rgb8_pil.save(path)
  rgb8_pil.close()

def demosaic(img4ch):
  h, w = img4ch.shape[:2]
  out = np.zeros((h, w, 3), dtype=np.float32)
  out[:, :, 0] = img4ch[:, :, 0]
  out[:, :, 1] = (img4ch[:, :, 1] + img4ch[:, :, 2]) / 2
  out[:, :, 2] = img4ch[:, :, 3]
  return out

def demosaic_tensor(img4ch):
  h, w = img4ch.shape[1:3]
  out = np.zeros((h, w, 3), dtype=np.float32)
  out[:, :, 0] = img4ch[0, :, :]
  out[:, :, 1] = (img4ch[1, :, :] + img4ch[2, :, :]) / 2
  out[:, :, 2] = img4ch[3, :, :]
  return out

def demosaic_bayer(bayer):
  """RGGB"""
  h, w = bayer.shape
  out = np.zeros((h//2, w//2, 4), dtype=bayer.dtype)
  out[:, :, 0] = bayer[0::2, 0::2]
  out[:, :, 1] = bayer[1::2, 0::2]
  out[:, :, 2] = bayer[0::2, 1::2]
  out[:, :, 3] = bayer[1::2, 1::2]
  return out

def adjust_wb(img, wb):
  img[:, :, 0] *= wb[0].item()
  img[:, :, 1] *= wb[1].item()
  img[:, :, 2] *= wb[2].item()
  return img

def calculate_wb(rgb1, rgb2):
  wb = [rgb1[:, :, 0].mean() / (rgb2[:, :, 0].mean() + 1e-8),
        rgb1[:, :, 1].mean() / (rgb2[:, :, 1].mean() + 1e-8),
        rgb1[:, :, 2].mean() / (rgb2[:, :, 2].mean() + 1e-8)]
  wb = np.array(wb, dtype=np.float32)
  return wb

def normalization(img, percentile):
  u = np.percentile(img, percentile)
  d = np.percentile(img, 100-percentile)
  img[img > u] = u
  u, d = img.max(), img.min()
  img = (img - d) / (u - d) * 255.
  img = img.astype(np.uint8)
  return img

def norm(img, white=16383, black=512):
  if white == black:
    return img
  
  out = (img.astype(np.float32) - black) / (white - black)
  return out

def percentile(img, percentile=99):
  if len(img.shape) == 3:
    h, w, c = img.shape
    for i in range(c):
      ch = img[..., i]
      u = np.percentile(ch, percentile)
      ch[ch > u] = u
      img[..., i] = ch
  else:
    up = np.percentile(img, percentile)
    img[img > up] = up
  return img

def limit(a: np.ndarray, maximum=1, minimum=0):
  a[a > maximum] = maximum
  a[a < minimum] = minimum
  return a

def patch(img, patch_size):
  p = patch_size
  if len(img.shape) == 2:
    h, w = img.shape
    c = 1
  else:
    h, w, c = img.shape
    
  dh, dw = random.randint(0, h-p), random.randint(0, w-p)
  
  out = np.zeros((p, p, c), dtype=img.dtype)
  out = img[dh:dh+p, dw:dw+p]
  return out

def to_numpy_image(img: torch.Tensor) -> Union[np.ndarray, List[np.ndarray]]:
  img = img.cpu().detach()
  
  def _to_numpy_image(img: torch.Tensor):
    channels, h, w = img.shape
    out = np.zeros((h, w, channels), dtype=np.float32)
    for c in range(channels):
      out[:, :, c] = img[c, :, :]
    return out
  
  if len(img.shape) == 3:
    return _to_numpy_image(img)
  elif len(img.shape) == 4:
    out = []
    for i in range(img.shape[0]):
      out.append(_to_numpy_image(img[i]))
    return out
  else:
    raise NotImplementedError('img\'s shape must have 3 or 4 length')

def save_graph(data: List[float], path: str):
  plt.plot(data)
  plt.savefig(path)
  plt.close()
