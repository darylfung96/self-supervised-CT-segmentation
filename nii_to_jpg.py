import nibabel as nib
import os
import cv2
from argparse import ArgumentParser


if __name__ == '__main__':
    arg_parse = ArgumentParser()
    arg_parse.add_argument('--nii_file', type=str, required=True)
    arg_parse.add_argument('--output_folder', type=str, required=True)
    arg_parse.add_argument('--filename_prefix', type=str, required=True)
    arg_parse.add_argument('--save_type', type=str, required=True, help='jpg or png')
    arg_parse.add_argument('--is_binary', type=bool, default=False)  # mask out all number of different classes to 0 or 1
                                                                     # this is for the prior mask, that masks everything
                                                                     # else that has a class

    arg = arg_parse.parse_args()

    img = nib.load(arg.nii_file)
    data = img.get_fdata()

    # make output folder if not exist yet
    os.makedirs(arg.output_folder, exist_ok=True)

    prefix = arg.filename_prefix
    num_images = data.shape[2]
    for i in range(num_images):
        current_img = data[:, :, i]

        if arg.is_binary:
            current_img[current_img > 1] = 1
        output_filename = os.path.join(arg.output_folder, f'{prefix}_{i}.{arg.save_type}')
        cv2.imwrite(output_filename, current_img)

