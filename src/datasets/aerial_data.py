import os
import os.path as osp
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import sys

sys.path.append('..')
from configs import config, update_config


class AerialSeg(Dataset):
    """
    Class which will perform data augmentation for better diversity of the dataset.
    """

    def __init__(self, cfg, split=''):
        assert split in ['training', 'validation']
        self.data_root_dir = cfg.data.root_dir
        self.split = split
        self.is_training = True if self.split == 'training' else False
        self.image_names = np.genfromtxt(osp.join(self.data_root_dir, '{}.txt'.format(self.split)), dtype=str)

        self.image_ext = cfg.data.image_ext
        self.label_ext = cfg.data.label_ext
        self.img_shape = cfg.augment.crop_size

        # Define data augmentation transforms using Albumentations library
        self.train_transform = A.Compose([
            A.RandomCrop(width=cfg.augment.crop_size[0], height=cfg.augment.crop_size[1]),
            A.HorizontalFlip(p=cfg.augment.h_flip),
            A.VerticalFlip(p=cfg.augment.v_flip),
            A.Rotate(limit=cfg.augment.rot_degree, p=0.5),
            A.RandomBrightnessContrast(p=cfg.augment.brightness),
            ToTensorV2(transpose_mask=True),
        ])

        self.val_transform = A.Compose([
            A.CenterCrop(width=cfg.augment.crop_size[0], height=cfg.augment.crop_size[1]),
            ToTensorV2(transpose_mask=True)])

        print('[INFO] Initialised {} Dataset.'.format(self.split))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = os.path.join(osp.join(self.data_root_dir, 'images'), self.image_names[idx] + self.image_ext)
        mask_path = os.path.join(osp.join(self.data_root_dir, 'groundtruth'), self.image_names[idx] + self.label_ext)

        image = cv2.imread(image_path)
        image = image[:, :, [2, 1, 0]]  # Convert to RGB
        image = image.astype(np.float32)
        image /= 255.0

        binary_mask = cv2.imread(mask_path, 0)
        binary_mask[binary_mask > 0] = 1
        binary_mask = binary_mask.astype(np.uint8)

        mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 2), dtype=np.uint8)
        mask[..., 0] = (1 - binary_mask)  # First channel corresponds to 0s
        mask[..., 1] = binary_mask
        mask = mask.astype(np.float32)

        print

        if self.is_training:
            augmented = self.train_transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        else:
            augmented = self.val_transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        out = {}
        out['images'] = image
        out['groundtruth'] = mask
        return out

    def collate_fn(self, batch):
        for data in batch:
            print(data)


if __name__ == '__main__':
    config_file_name = '../configs/base.yaml'
    cfg = update_config(config, config_file_name)
    print(cfg)

    dataset = AerialSeg(cfg, 'training')
    dataset[0]

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True)
