import os
import argparse
from shutil import copyfile


def process_images(input_folder, output_folder):
    all_patients_folder = os.listdir(input_folder)
    for patient_folder in all_patients_folder:
        if 'Patient' not in patient_folder:
            continue

        patient_dir = os.path.join(input_folder, patient_folder)
        patient_images = os.listdir(patient_dir)
        for patient_image in patient_images:
            patient_image_filename = os.path.join(patient_dir, patient_image)

            output_image_filename = os.path.join(output_folder, f'{patient_folder}_{patient_image}')
            copyfile(patient_image_filename, output_image_filename)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input_folder', type=str, required=True)
    arg_parser.add_argument('--output_folder', type=str, required=True)
    args = arg_parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    process_images(args.input_folder, args.output_folder)
