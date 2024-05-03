import os
import argparse
from pathlib import Path


def count_files(directory, class_name=None):
    total_files = 0
    for root, dirs, files in os.walk(directory):
        if class_name is None or os.path.basename(root) == class_name:
            total_files += len(files)
    return total_files

def main(class_name):
    
    #path = os.path.join("datasets","ZooScan77")
    total = 961167
    class_num = count_files("datasets/ZooScan77", class_name)
    print(f"{class_name} is {class_num*100/total:.2f} % of dataset" )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('class_name', type=str)
    args = parser.parse_args()
    main(args.class_name)