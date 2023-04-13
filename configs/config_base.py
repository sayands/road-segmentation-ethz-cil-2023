import os 
import os.path as osp 
import argparse
from easydict import EasyDict as edict

from utils import define
from utils.common import ensure_dir

_C = edict()

# common 
_C.seed = 42
_C.num_workers = 4

# path params
_C.data_dir = define.DATA_DIR
_C.file_dir = define.FILES_DIR

ensure_dir(_C.file_dir)

# Data Params
_C.data = edict()
_C.data.label_ext = '.png'
_C.data.image_ext = '.png'

# Training params
_C.train = edict()
_C.train.batch_size = 4
_C.train.use_augmentation = True

# Augmentation
_C.augment = edict()
_C.augment.h_flip = 0.5
_C.augment.v_flip = 0.5
_C.augment.rot_degree = 20
_C.augment.brightness = 0.2
_C.augment.crop_size = (224, 224)

# Validation params
_C.val = edict()
_C.val.batch_size = 4

# model param
_C.model = edict()

# optim
_C.optim = edict()
_C.optim.lr = 1e-3
_C.optim.lr_decay = 0.95
_C.optim.lr_decay_steps = 1
_C.optim.weight_decay = 1e-6
_C.optim.max_epoch = 50
_C.optim.grad_acc_steps = 1

# loss
_C.loss = edict()

# inference
_C.metrics = edict()

def make_cfg():
    return _C

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_output', dest='link_output', action='store_true', help='link output dir')
    args = parser.parse_args()
    return args

def main():
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, 'output')