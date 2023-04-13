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
from configs import config_base

class AerialSegCustom(Dataset):
    def __init__(self, cfg, split=''):
        self.data_dir = cfg.data_dir
        self.files_dir = cfg.file_dir
        self.image_fileset_name = osp.join(self.files_dir, 'imageset.txt') # TODO : Add split here
        self.image_fileset = np.genfromtxt(self.image_fileset_name, dtype=str) 

        self.image_ext = cfg.data.image_ext
        self.label_ext = cfg.data.label_ext

        self.visualise = True

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
        return len(self.image_fileset)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_fileset[idx] + self.image_ext)
        mask_path = os.path.join(self.data_dir, self.image_fileset[idx] + '_label' + self.label_ext)
        
        image = cv2.imread(image_path)
        image = image[:, :, [2, 1, 0]] # Convert to RGB
        image = image.astype(np.float32)
        image /= 255.0
        
        mask = cv2.imread(mask_path, 0)
        mask[mask > 0] = 1
        mask = mask.astype(np.uint8)

        if self.visualise:
            # Create a subplot with 1 row, 2 columns
            fig, axs = plt.subplots(1, 2)

            # Plot the first image on the left subplot
            axs[0].imshow(image)
            axs[0].set_title('Aerial Image')

            # Plot the second image on the right subplot
            axs[1].imshow(mask)
            axs[1].set_title('Road Mask')

            # Remove ticks from both subplots
            axs[0].axis('off')
            axs[1].axis('off')

            # Display the subplot
            plt.show()
        
        if self.train_transform is not None:
            augmented = self.train_transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']            
        return image, mask

if __name__ == '__main__':
    cfg = config_base.make_cfg()
    dataset = AerialSegCustom(cfg)
    image, mask = dataset[0]
    print(image.size(), mask.size())