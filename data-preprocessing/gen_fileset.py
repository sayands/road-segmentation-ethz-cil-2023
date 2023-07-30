import os 
import os.path as osp
import numpy as np
from glob import glob 

import sys
sys.path.append('.')
from configs import config_base


def gen_fileset(data_dir, file_name):
    """
    Generates file for indexing filestes.
    @param data_dir: directory in which the data is stored
    @param file_name: the name of the file where the file ids will be saved
    """
    files = glob(osp.join(data_dir, '*/*.png'), recursive=True)
    files = [file for file in files if 'label' not in file]
    files = ['/'.join(file.split('/')[-2:]) for file in files]
    image_names = [file[:-4] for file in files]

    np.savetxt(file_name, image_names, fmt='%s')


if __name__ == '__main__':
    cfg = config_base.make_cfg()
    data_dir = cfg.data_dir
    file_dir = cfg.file_dir
    file_name = osp.join(file_dir, 'imageset.txt')
    gen_fileset(data_dir, file_name)
