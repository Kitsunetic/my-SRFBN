import argparse

parser = argparse.ArgumentParser(description='SRFBN')

# raw image normalization
parser.add_argument('--black_lv', type=int, default=512)
parser.add_argument('--white_lv', type=int, default=16383)

# image preprocessing
parser.add_argument('--patch_size', type=int, default=256)

# train
parser.add_argument('--n_epochs', type=int, default=50)

# dataset
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--validation_split', type=float, default=0.05)

# file system
parser.add_argument('--dataset_path', type=str, default='../../datasets/SR_testing_datasets/Urban100')
parser.add_argument('--result_path', type=str, default='./results')

# model
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--out_channels', type=int, default=3)
parser.add_argument('--n_features', type=int, default=64)
parser.add_argument('--n_steps', type=int, default=4)
parser.add_argument('--n_groups', type=int, default=6)
parser.add_argument('--scale', type=int, default=2)
