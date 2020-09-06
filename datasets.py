import os
import pickle
import random
import re

import imageio
import torch
from PIL import Image
from torchvision import transforms


class RAW2RGB(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str, black_lv=512, white_lv=16384):
        self.black_lv = black_lv
        self.white_lv = white_lv

        self.file_list = []
        for file in os.listdir(dataset_path):
            if re.match('\\d{5}\\.pkl', file):
                self.file_list.append(os.path.join(dataset_path, file))

        self.transform = transforms.ToTensor()

    def __getitem__(self, idx: int):
        with open(self.file_list[idx], 'rb') as f:
            data = pickle.load(f)

        lr = self.transform(data['lr'])
        hr = self.transform(data['hr'])

        return lr, hr

    def __len__(self):
        return len(self.file_list)


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

        # print(self.image_files)
        # exit()

    def __getitem__(self, idx: int):
        hr = imageio.imread(self.image_files[idx])

        # make patch
        p = self.patch_size
        h, w = hr.shape[:2]
        dh, dw = random.randint(0, h - p), random.randint(0, w - p)
        hr = hr[dh:dh + p, dw:dw + p, :]

        lr = hr.copy()
        # lr = lr[0::2, 0::2, :]
        lr = Image.fromarray(lr)
        trans_hr = transforms.Compose([
            transforms.ToTensor()
        ])
        trans_lr = transforms.Compose([
            transforms.Resize((hr.shape[0] // 2, hr.shape[1] // 2)),
            transforms.ToTensor()
        ])
        hr = trans_hr(hr)
        lr = trans_lr(lr)

        return hr, lr

    def __len__(self):
        return len(self.image_files)
