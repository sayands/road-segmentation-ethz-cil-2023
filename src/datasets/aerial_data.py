import os
import os.path as osp
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import cv2

import sys
sys.path.append('..')
from configs import config, update_config

class AerialSeg(Dataset):
    def __init__(self, cfg, split=''):
        self.data_root_dir = cfg.data.root_dir
        self.split = split
        self.image_names = np.genfromtxt(osp.join(self.data_root_dir, '{}.txt'.format(self.split)) , dtype=str) 

        self.image_ext = cfg.data.image_ext
        self.label_ext = cfg.data.label_ext
        
        # Define data augmentation transforms using Albumentations library
        self.train_transform = A.Compose([
                               A.RandomCrop(width = cfg.augment.crop_size[0], height=cfg.augment.crop_size[1]),
                               A.HorizontalFlip(p=cfg.augment.h_flip),
                               A.VerticalFlip(p=cfg.augment.v_flip),
                               A.Rotate(limit=cfg.augment.rot_degree, p=0.5),
                               A.RandomBrightnessContrast(p=cfg.augment.brightness),
                               ToTensorV2(),
                            ])
        print('[INFO] Initialised Dataset.')
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_path = os.path.join(osp.join(self.data_root_dir, 'images'), self.image_names[idx] + self.image_ext)
        mask_path = os.path.join(osp.join(self.data_root_dir, 'groundtruth'), self.image_names[idx] + self.label_ext)
        
        image = cv2.imread(image_path)
        image = image[:, :, [2, 1, 0]] # Convert to RGB
        image = image.astype(np.float32)
        image /= 255.0
        
        mask = cv2.imread(mask_path, 0)
        mask[mask > 0] = 1
        mask = mask.astype(np.uint8)
        
        if self.train_transform is not None:
            augmented = self.train_transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']            
        return image, mask

if __name__ == '__main__':
    config_file_name = '../../configs/base.yaml'
    cfg = update_config(config, config_file_name)
    
    dataset = AerialSeg(cfg)
    image, mask = dataset[0]
    print(image.size(), mask.size())