import os
import random
from typing import Iterator

import rawpy
import imageio
import numpy as np
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

def image_patches(img: np.ndarray, patch_size: int) -> Iterator[np.ndarray]:
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

if __name__ == "__main__":
  img = imageio.imread('test.png')
  for i, patch in tqdm(enumerate(image_patches(img, 96))):
    imageio.imsave('results/%04d.png'%i, patch)
