"""
The purpose of this script is to reorganize and concatenate the different sources of data
into one dataset with which  structure it would be  easier to operate
"""


import os
import os.path as osp
import shutil

# root directory for all data collected
data_root_dir = '/home/ivan/PycharmProjects/ETH/road-segmentation-ethz-cil-2023/data'
# location of the data provided by kaggle
kaggle_data_dir = osp.join(data_root_dir, 'kaggle-data')
# location of the collected data by craw_aerial_seg.py
scraped_data_dir = osp.join(data_root_dir, 'collected')
# destination directory for the whole dataset
out_data_dir = osp.join(data_root_dir, 'seg-data')

# groundtruth samples destination location
out_data_groundtruth_dir = osp.join(out_data_dir, 'groundtruth')
# satellite images destination location
out_data_images_dir = osp.join(out_data_dir, 'images')

# Create any if the directories if it is non-existent
if not osp.exists(out_data_dir): os.makedirs(out_data_dir)
if not osp.exists(out_data_groundtruth_dir): os.makedirs(out_data_groundtruth_dir)
if not osp.exists(out_data_images_dir): os.makedirs(out_data_images_dir)

# Only move kaggle training
split = 'training'
kaggle_split_dir = osp.join(kaggle_data_dir, split)

# count number of images
kaggle_images = os.listdir(osp.join(kaggle_split_dir, 'images'))
start_count = len(kaggle_images)

# convert scraped images to same format
scraped_folders = os.listdir(scraped_data_dir)

for scraped_folder in scraped_folders:
    if not osp.isdir(osp.join(scraped_data_dir, scraped_folder)): continue
    
    images = os.listdir(osp.join(scraped_data_dir, scraped_folder))
    groundtruth_names = [image for image in images if '_label' in image]

    # Change file names to be consistent and save them according the the output(destination) directories
    for groundtruth_name in groundtruth_names:
        groundtruth_path = osp.join(scraped_data_dir, scraped_folder, groundtruth_name)
        image_path = groundtruth_path.replace('_label', '')
        
        assert osp.exists(groundtruth_path)
        assert osp.exists(image_path)
        
        groundtruth_path_dest = osp.join(out_data_groundtruth_dir, 'satimage_{}.png'.format(start_count))
        image_path_dest = osp.join(out_data_images_dir, 'satimage_{}.png'.format(start_count))
        
        shutil.copy(groundtruth_path, groundtruth_path_dest)
        shutil.copy(image_path, image_path_dest)
        
        start_count += 1
