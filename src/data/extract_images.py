import os
import shutil
from py7zr import unpack_7zarchive
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import py7zr

shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)


def extract_images(zip_path, extract_to):
   # shutil.unpack_archive(zip_path, "tmp")
    src = os.path.join(os.path.join("tmp", 'ZooScan77'), "Images")
    dst = os.path.join(extract_to, 'ZooScan77')
    i=0
    for item in os.listdir(src):
        print(f"{i}: {item}")
        i += 1
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        shutil.move(s, d)
    shutil.rmtree(src)

def split_data(base_directory):
    data = []
    images_dir = os.path.join(base_directory, 'ZooScan77')
    for subdir in os.listdir(images_dir):
        subdir_path = os.path.join(images_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith('.jpg'):  
                    data.append((os.path.join('ZooScan77', subdir, file), subdir))
    filenames, labels = zip(*data)
    train_filenames, temp_filenames, train_labels, temp_labels = train_test_split(
        filenames, labels, test_size=0.3, random_state=42, stratify=labels)
    val_filenames, test_filenames, val_labels, test_labels = train_test_split(
        temp_filenames, temp_labels, test_size=1./3, random_state=42, stratify=temp_labels)
    return train_filenames, train_labels, val_filenames, val_labels, test_filenames, test_labels

def move_files(filenames, labels, dirname, base_dir, target_base_dir):
    for filename, label in zip(filenames, labels):
        os.makedirs(os.path.join(target_base_dir, dirname, label), exist_ok=True)
        shutil.move(os.path.join(base_dir, filename), os.path.join(target_base_dir, dirname, label))
        


def main(zip_file_path, target_base_dir):
    print(f"Extracting images from {zip_file_path} to {target_base_dir}")
    extract_images(zip_file_path, target_base_dir)
    print("Done Extracting images")
    train_filenames, train_labels, val_filenames, val_labels, test_filenames, test_labels = split_data(target_base_dir)
    # Copy files to respective directories
    move_files(train_filenames, train_labels, 'train', target_base_dir, os.path.join(target_base_dir, "ZooScan77"))
    move_files(val_filenames, val_labels, 'val', target_base_dir, os.path.join(target_base_dir, "ZooScan77"))
    move_files(test_filenames, test_labels, 'test', target_base_dir, os.path.join(target_base_dir, "ZooScan77"))

    for subdir in os.listdir(os.path.join(target_base_dir, "ZooScan77")):
            if subdir not in ['train', 'val', 'test']:
                shutil.rmtree(os.path.join(target_base_dir, os.path.join("ZooScan77", subdir)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip_file_path',
                         type=str, 
                         help='Path to the .7z file containing images', 
                         default="ZooScan77.7z")
    parser.add_argument('--target_base_dir', 
                        type=str, 
                        help='Base directory to store the images',
                        default="datasets")
    args = parser.parse_args()
    main(args.zip_file_path, args.target_base_dir)
