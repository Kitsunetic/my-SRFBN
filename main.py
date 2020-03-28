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
import losses2
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
                           act_type='prelu',
                           norm_type=None)
  
  # make loss
  criterion = losses.ContextualBilateralLoss()
  optimizer = torch.optim.Adam(model.parameters())
  
  # train
  trainer = train.Trainer(device=device,
                          model=model,
                          criterion=criterion,
                          optimizer=optimizer,
                          train_loader=trainloader,
                          val_loader=None,
                          n_epochs=args.n_epochs,
                          patch_size=args.patch_size,
                          result_path=args.result_path,
                          test_path=args.test_path,
                          pretrained_path=args.pretrained_path)
  trainer.train()
  
if __name__ == "__main__":
  main()
