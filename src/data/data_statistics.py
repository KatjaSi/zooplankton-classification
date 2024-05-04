import os
import argparse
from pathlib import Path


def count_files(directory, class_name=None, category_name=None):
    """
      one of class_name, category_name should be None for correct usage
    """
    total_files = 0    
    for root, dir, files in os.walk(directory):
        if os.path.basename(root) == class_name or (category_name is not None and os.path.basename(root).startswith(category_name)):
            total_files += len(files)
    return total_files

def main(class_name, category_name):
    if (bool(class_name) == bool(category_name)):
        raise ValueError("Exactly one of class_name and category name should be specified")
    total = 961167
    if class_name is not None:
        class_num = count_files("datasets/ZooScan77", class_name=class_name)
        print(f"{class_name} is {class_num*100/total:.2f} % of dataset" )
    if category_name is not None:
        category_num = count_files("datasets/ZooScan77", category_name=category_name)
        print(f"{category_name} is {category_num*100/total:.2f} % of dataset" )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_name', type=str)
    parser.add_argument('--category_name', type=str)
    args = parser.parse_args()
    main(args.class_name, args.category_name)