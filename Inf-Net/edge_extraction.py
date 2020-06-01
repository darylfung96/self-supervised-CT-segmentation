from argparse import ArgumentParser
import os
import numpy as np
from skimage import feature, io

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--gt_folder', type=str, required=True)
    arg_parser.add_argument('--output_folder', type=str, required=True)

    args = arg_parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    for img_str in os.listdir(args.gt_folder):
        img_filename = os.path.join(args.gt_folder, img_str)
        img = io.imread(img_filename)
        edge = feature.canny(img).astype(np.uint8)

        new_img_str = img_str.replace('parenchyma', 'edge')
        output_filename = os.path.join(args.output_folder, new_img_str)
        io.imsave(output_filename, edge)


