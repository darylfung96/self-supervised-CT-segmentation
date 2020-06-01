import nibabel as nib
import os
import imageio
from argparse import ArgumentParser


if __name__ == '__main__':
    arg_parse = ArgumentParser()
    arg_parse.add_argument('--nii_file', type=str, required=True)
    arg_parse.add_argument('--output_folder', type=str, required=True)
    arg_parse.add_argument('--save_type', type=str, required=True, help='jpg or png')

    arg = arg_parse.parse_args()

    img = nib.load(arg.nii_file)
    data = img.get_fdata()

    # make output folder if not exist yet
    os.makedirs(arg.output_folder, exist_ok=True)

    prefix = os.path.basename(arg.nii_file).split('.')[0]
    num_images = data.shape[2]
    for i in range(num_images):
        current_img = data[:, :, i]

        output_filename = os.path.join(arg.output_folder, f'{prefix}_{i}.{arg.save_type}')
        imageio.imwrite(output_filename, current_img)

