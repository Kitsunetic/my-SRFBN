import os
import random
import shutil
import sys
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm

import datasets
import models
import options
import utils


def main():
    args = options.parser.parse_args(sys.argv[1:])
    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load dataset
    trainset = datasets.ImageSet(args.dataset_path, args.patch_size)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=args.shuffle,
                                              num_workers=args.num_workers)

    # load color_mean
    # color_path = os.path.join(args.dataset_path, 'color.json')
    # with open(color_path, 'r') as f:
    #  data = json.load(f)
    #  color_mean = data['color_mean']
    #  color_std = data['color_std']
    color_mean = [0., 0., 0.]
    color_std = [0., 0., 0.]

    # make model
    model = models.SRFBN(device=device,
                         in_channels=args.in_channels,
                         out_channels=args.out_channels,
                         n_features=args.n_features,
                         n_steps=args.n_steps,
                         n_groups=args.n_groups,
                         upscale_factor=args.scale,
                         color_mean=color_mean,
                         color_std=color_std,
                         act_type='relu',
                         norm_type=None)
    model = model.to(device)

    # make loss
    criterion = nn.L1Loss()
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # train
    for epoch in range(1, args.n_epochs + 1):
        with tqdm(desc='[%04d/%04d]' % (epoch, args.n_epochs), total=len(trainloader),
                  unit='batch', ncols=96, miniters=1) as t:
            losses = []
            for batch_idx, (hr, lr) in enumerate(trainloader):
                hr, lr = hr.to(device), lr.to(device)
                sr_list = model(lr)
                sr = sr_list[-1][0]
                loss = criterion(sr, hr)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                mean_loss = sum(losses) / len(losses)
                t.set_postfix_str('loss %.4f' % mean_loss)
                t.update()

        # save result
        if epoch % args.save_result_interval == 0:
            utils.save_tensor_image(hr[0], os.path.join(args.result_path, '%05d-hr.png' % epoch))
            utils.save_tensor_image(lr[0], os.path.join(args.result_path, '%05d-lr.png' % epoch))
            for i, sr in enumerate(sr_list):
                utils.save_tensor_image(sr[0], os.path.join(args.result_path, '%05d-sr%d.png' % (epoch, i)))

    # save model
    model_path = os.path.join(args.model_path, '{}-epoch{}-loss{:.04f}.pth'.format(
        datetime.now().strftime('%y%m%d-%H%M%S'),
        epoch,
        sum(losses) / len(losses)))
    with open(model_path, 'wb') as f:
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, f)

    torch.set_grad_enabled(False)

    # save example image
    result_images = []
    for file in os.listdir(args.dataset_path):
        if os.path.splitext(file)[1].lower() in ['.jpg', '.png']:
            image_file = os.path.join(args.dataset_path, file)
            result_images.append(image_file)
    result_images = random.sample(result_images, args.save_examples)

    for i, imfile in tqdm(enumerate(result_images), total=len(result_images),
                          desc='saving results', ncols=96, miniters=1, unit='file'):
        shutil.copy(imfile, os.path.join(args.result_path, 'result-%05d-org.png' % i))
        res_path = os.path.join(args.result_path, 'result-%05d.png' % i)
        utils.model_large_image(model, device, imfile, res_path, args.patch_size)


if __name__ == "__main__":
    main()
