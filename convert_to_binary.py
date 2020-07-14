import os
import argparse
import numpy as np
from PIL import Image


def convert_to_binary(input_folder_dir, output_folder_dir):
    images = os.listdir(input_folder_dir)

    for image in images:
        image_filename = os.path.join(input_folder_dir, image)
        img = Image.open(image_filename).convert('L')
        np_img = np.array(img)
        np_img[np_img > 1] = 255

        output_image_filename = os.path.join(output_folder_dir, image)
        img = Image.fromarray(np_img)
        img.save(output_image_filename)


if __name__ == '__main__':
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--input_folder_dir', required=True, type=str)
    arg_parse.add_argument('--output_folder_dir', required=True, type=str)
    args = arg_parse.parse_args()

    os.makedirs(args.output_folder_dir, exist_ok=True)
    convert_to_binary(args.input_folder_dir, args.output_folder_dir)
