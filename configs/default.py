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
_C.augment.rot_degree = 90
_C.augment.brightness = 0.5
_C.augment.crop_size = (256, 256)

# Validation params
_C.validation = CN()
_C.validation.batch_size = 4
_C.validation.valid_every = 1

# model param
_C.model = CN()
_C.model.type = 'DeepLabV3Plus' #Unet, PSPNet
_C.model.encoder = 'efficientnet-b3' #resnet34, densenet169, xception, mobilenet_v2, vgg16

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
_C.loss.loss_type = ["ce"]
_C.loss.epochs = [100]
_C.loss.wlambda = [0.5]
_C.loss.alpha = [0.7]
_C.loss.gamma = [1.5]

# testing params
_C.test = CN()
_C.test.test_path = ''
_C.test.mask_results_path = ''
_C.test.submission_path = ''
_C.test.model_path = ''
_C.test.device = 'cpu'
_C.test.stride = 1
_C.test.auto_padding = False
_C.test.padding = 0
_C.test.model_ensemble = []

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