import os
import shutil
from PIL import Image
import numpy as np

def combine_images(base_dir):
    cup_dir = os.path.join(base_dir, 'masks', 'cup')
    disk_dir = os.path.join(base_dir, 'masks', 'disk')
    both_dir = os.path.join(base_dir, 'masks', 'both')

    if not os.path.exists(both_dir):
        os.makedirs(both_dir)

    for filename in os.listdir(cup_dir):
        if filename.endswith('.png'):
            cup_path = os.path.join(cup_dir, filename)
            disk_path = os.path.join(disk_dir, filename)
            both_path = os.path.join(both_dir, filename.replace('.png', '.bmp'))
            try:
                if os.path.exists(disk_path):
                    cup = np.array(Image.open(cup_path))
                    disk = np.array(Image.open(disk_path))

                    combined = np.zeros_like(cup) + 255
                    combined[disk == 255] = 128  # disk
                    combined[cup == 255] = 0  # cup

                    Image.fromarray(combined).save(both_path)
                else:
                    print(f"Disk file not found for {filename} in {disk_dir}")
            except Exception as e:
                print(f"Error processing file {filename} in {base_dir}: {e}")

def handle_g1020(base_dir):
    if base_dir.endswith('G1020'):
        image_dir = os.path.join(base_dir, 'images')
        both_dir = os.path.join(base_dir, 'masks', 'both')
        labelled_dir = os.path.join(base_dir, 'labelled')
        unlabelled_dir = os.path.join(base_dir, 'unlabelled')

        if not os.path.exists(labelled_dir):
            os.makedirs(labelled_dir)
        if not os.path.exists(unlabelled_dir):
            os.makedirs(unlabelled_dir)

        for filename in os.listdir(image_dir):
            if filename.endswith('.png'):  # or other image formats
                both_path = os.path.join(both_dir, filename.replace('.png', '.bmp'))
                image_path = os.path.join(image_dir, filename)
                target_dir = labelled_dir if os.path.exists(both_path) else unlabelled_dir
                shutil.copy(image_path, target_dir)

# Example usage
parent_directory = 'C:\\Users\\Josh\\Desktop\\4YP\\Datasets\\RAW\\Kaggle\\Labelled'  # Replace with the path to your parent directory
for base_dir_name in os.listdir(parent_directory):
    base_dir_path = os.path.join(parent_directory, base_dir_name)
    if os.path.isdir(base_dir_path):
        combine_images(base_dir_path)
        handle_g1020(base_dir_path)