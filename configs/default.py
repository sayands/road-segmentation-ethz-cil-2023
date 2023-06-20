from yacs.config import CfgNode as CN
import os.path as osp

from utils import common

_C = CN()

# common 
_C.seed = 42
_C.num_workers = 4
_C.log_dir = ''
_C.log_level = 'INFO'

# path params
_C.data = CN()
_C.data.root_dir = ''
_C.data.label_ext = '.png'
_C.data.image_ext = '.png'

# Training params
_C.train = CN()
_C.train.batch_size = 4
_C.train.use_augmentation = True
_C.train.epochs = 50
_C.train.log_every = 500
_C.train.save_every = 2

# Augmentation
_C.augment = CN()
_C.augment.h_flip = 0.5
_C.augment.v_flip = 0.5
_C.augment.rot_degree = 20
_C.augment.brightness = 0.2
_C.augment.crop_size = (256, 256)

# Validation params
_C.validation = CN()
_C.validation.batch_size = 4
_C.validation.valid_every = 1

# model param
_C.model = CN()

# optim
_C.optim = CN()
_C.optim.lr = 1e-3
_C.optim.lr_decay = 0.95
_C.optim.lr_decay_steps = 1
_C.optim.weight_decay = 1e-6
_C.optim.max_epoch = 50
_C.optim.grad_acc_steps = 1

# loss
_C.loss = CN()

# inference
_C.metrics = CN()

# wandb
_C.wandb = CN()
_C.wandb.wandb = False
_C.wandb.entity = ''
_C.wandb.project = ''
_C.wandb.name = ''
_C.wandb.group = None
_C.wandb.id = None

def update_config(cfg, filename):
    cfg.defrost()
    cfg.merge_from_file(filename)
    
    if cfg.log_dir == '':
        working_dir = osp.dirname(osp.abspath(__file__))
        root_dir = osp.dirname(working_dir)
        cfg.log_dir = osp.join(root_dir, 'logs')
        common.ensure_dir(cfg.log_dir)
        
    cfg.freeze()
    
    return cfg