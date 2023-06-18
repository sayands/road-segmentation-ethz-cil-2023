from yacs.config import CfgNode as CN
import os.path as osp

_C = CN()

# common 
_C.seed = 42
_C.num_workers = 4

# path params
_C.data = CN()
_C.data.root_dir = '/Users/sayands/Documents/Work/Courses/Computational-Intelligence-Lab/Project/data/seg-data'
_C.data.label_ext = '.png'
_C.data.image_ext = '.png'

# Training params
_C.train = CN()
_C.train.batch_size = 4
_C.train.use_augmentation = True

# Augmentation
_C.augment = CN()
_C.augment.h_flip = 0.5
_C.augment.v_flip = 0.5
_C.augment.rot_degree = 20
_C.augment.brightness = 0.2
_C.augment.crop_size = (256, 256)

# Validation params
_C.val = CN()
_C.val.batch_size = 4

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


def update_config(cfg, filename):
    cfg.defrost()
    cfg.merge_from_file(filename)
    cfg.freeze()
    
    return cfg