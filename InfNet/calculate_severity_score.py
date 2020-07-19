import os
import cv2
import argparse

def calculate_severity(input_dir):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)

    args = parser.parse_args()
    calculate_severity(args.input_dir)
