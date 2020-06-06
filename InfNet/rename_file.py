import os
from argparse import ArgumentParser


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--directory', type=str, required=True)
    args = arg_parser.parse_args()

    filenames = os.listdir(args.directory)
    for filename in filenames:
        new_filename = filename.replace('parenchyma', 'edge')
        old_file_directory = os.path.join(args.directory, filename)
        new_file_directory = os.path.join(args.directory, new_filename)
        os.rename(old_file_directory, new_file_directory)
