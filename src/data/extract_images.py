import os
import shutil
from py7zr import unpack_7zarchive
import shutil
from sklearn.model_selection import train_test_split
import argparse

shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)

def main(zip_file_path):
    print(f"Extracting images from {zip_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip_file_path',
                         type=str, help='Path to the .7z file containing images', 
                         default="ZooScan77.7z")
    args = parser.parse_args()
    main(args.zip_file_path)
