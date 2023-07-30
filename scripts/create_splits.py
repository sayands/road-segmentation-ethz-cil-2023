"""
This script splits the dataset into validation and training subsets which are used in the training pipeline
"""

import numpy as np
import os
import os.path as osp
import random

data_root_dir = '/home/ivan/PycharmProjects/ETH/road-segmentation-ethz-cil-2023/data'
out_data_dir = osp.join(data_root_dir, 'seg-data')
out_data_groundtruth_dir = osp.join(out_data_dir, 'groundtruth')
out_data_images_dir = osp.join(out_data_dir, 'images')

groundtruth_images = os.listdir(out_data_groundtruth_dir)
groundtruth_images = [image[:-4] for image in groundtruth_images]
train_ratio = 0.8
val_ratio = 1.0 - train_ratio

train_size = int(len(groundtruth_images) * train_ratio)
train_images = random.sample(groundtruth_images, train_size)
val_images = [image for image in groundtruth_images if image not in train_images]

np.savetxt(osp.join(out_data_dir, 'all_images.txt'), np.array(groundtruth_images, dtype=str), fmt='%s')
np.savetxt(osp.join(out_data_dir, 'training.txt'), np.array(train_images, dtype=str), fmt='%s')
np.savetxt(osp.join(out_data_dir, 'validation.txt'), np.array(val_images, dtype=str), fmt='%s')






