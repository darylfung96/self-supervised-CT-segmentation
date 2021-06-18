import numpy as np
import os
from shutil import copyfile
from skimage import io, feature


STICHNET_DIR = 'StichNet_Dataset'

OUTPUT_DIR = 'StichNet_processed_Dataset'

types = ['train', 'test', 'val']
for current_type in types:
    current_dir = os.path.join(STICHNET_DIR, current_type)
    output_dir = os.path.join(OUTPUT_DIR, current_type)

    img_folder_dir = os.path.join(current_dir, 'img')
    msk_folder_dir = os.path.join(current_dir, 'msk')

    img_output_dir = os.path.join(output_dir, 'Imgs')
    prior_output_dir = os.path.join(output_dir, 'Prior')
    msk_output_dir = os.path.join(output_dir, 'GT')
    edge_output_dir = os.path.join(output_dir, 'Edge')
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(prior_output_dir, exist_ok=True)
    os.makedirs(msk_output_dir, exist_ok=True)
    os.makedirs(edge_output_dir, exist_ok=True)

    img_folders = os.listdir(img_folder_dir)
    img_folders.sort()
    msk_folders = os.listdir(msk_folder_dir)
    msk_folders.sort()
    for img_folder in img_folders:

        # img and msk should have the same name
        img_dirname = os.path.join(img_folder_dir, img_folder)
        img_filename = os.listdir(img_dirname)[0]
        new_filename = f'{img_folder}_{img_filename}'

        copyfile(os.path.join(img_dirname, img_filename), os.path.join(img_output_dir, new_filename))

    for msk_folder in msk_folders:
        # msk
        msk_dirname = os.path.join(msk_folder_dir, msk_folder)
        msk_filename = os.listdir(msk_dirname)[0]
        new_filename = f'{msk_folder}_{msk_filename}'
        copyfile(os.path.join(msk_dirname, msk_filename), os.path.join(msk_output_dir, new_filename))

        # prior
        msk = os.path.join(msk_dirname, msk_filename)
        binary_msk = io.imread(msk).astype(np.uint8)
        binary_msk[binary_msk > 1] = 1
        io.imsave(os.path.join(prior_output_dir, new_filename), binary_msk)

        # edge
        edge_msk = io.imread(msk)
        edge_msk = feature.canny(edge_msk).astype(np.uint8)
        edge_msk[edge_msk >= 1] = 255
        io.imsave(os.path.join(edge_output_dir, new_filename), edge_msk)
