import os
import shutil
import json
import random
import ipdb

source_directory = 'datasets/ZooScan77/train'
target_directory = 'datasets/ZooScan77_010_final/train'

with open('file_counts.json', 'r') as json_file:
    file_counts = json.load(json_file)


for class_name, data in file_counts.items():
    sample_diff = data['sample_diff']
    if sample_diff == 0:
        continue

    source_class_path = os.path.join(source_directory, class_name)
    target_class_path = os.path.join(target_directory, class_name)

    existing_files = set(os.listdir(target_class_path))

    if sample_diff < 0:
        files_to_remove = random.sample(list(existing_files), abs(sample_diff))
        for file in files_to_remove:
            os.remove(os.path.join(target_class_path, file))
        print(f"Removed {len(files_to_remove)} files from {target_class_path}")

    else:
        all_source_files = set(os.listdir(source_class_path))
        available_files = list(all_source_files - existing_files)

        files_to_add = random.sample(available_files, sample_diff)
        for file in files_to_add:
            shutil.copy(os.path.join(source_class_path, file), os.path.join(target_class_path, file))
        print(f"Added {len(files_to_add)} files to {target_class_path}")
   

print("Sampling based on sample_diff completed!")
