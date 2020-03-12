import json
import os
import random
import shutil
import sys
from datetime import datetime

import imageio
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import datasets
import losses
import models
import options
import train
import utils


def main():
  args = options.parser.parse_args(sys.argv[1:])
  os.makedirs(args.result_path, exist_ok=True)
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # load dataset
  trainset = datasets.SRRAW(args.dataset_path, args.patch_size)
  trainloader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=args.batch_size,
                                            shuffle=args.shuffle,
                                            num_workers=args.num_workers)
  
  # make model
  model = models.SRFBN_RAW(device=device,
                           n_features=args.n_features,
                           n_steps=args.n_steps,
                           n_groups=args.n_groups,
                           act_type='relu',
                           norm_type=None)
  model = model.to(device)
  
  # make loss
  criterion = nn.L1Loss()
  #criterion = losses.Contextual_Loss({"conv_1_1": 1.0, "conv_3_2": 1.0}, max_1d_size=64)
  criterion = criterion.to(device)
  optimizer = torch.optim.Adam(model.parameters())
  
  # train
  trainer = train.Trainer(device=device,
                          model=model,
                          criterion=criterion,
                          optimizer=optimizer,
                          train_loader=trainloader,
                          test_loader=None,
                          val_loader=None,
                          n_epochs=args.n_epochs,
                          result_path=args.result_path,
                          pretrained_path=args.pretrained_path)
  trainer.train()
  
if __name__ == "__main__":
  main()
