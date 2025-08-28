import os
from glob import glob  # You forgot this import
path_to_datasets = "/Users/katjahellgren/YOLOv8-pt-fedn/split_datasets"

# TRAIN
image_dir = os.path.join(path_to_datasets, "train/images/")
output_txt = os.path.join(path_to_datasets, "train.txt")

image_paths = sorted(
    glob(os.path.join(image_dir, "*.jpg")) +
    glob(os.path.join(image_dir, "*.jpeg"))
)

with open(output_txt, "w") as f:
    for path in image_paths:
        f.write(os.path.abspath(path) + "\n")

print(f"Saved {len(image_paths)} image paths to {output_txt}")

# VALID
image_dir = os.path.join(path_to_datasets, "valid/images/")
output_txt = os.path.join(path_to_datasets, "valid.txt")

image_paths = sorted(
    glob(os.path.join(image_dir, "*.jpg")) +
    glob(os.path.join(image_dir, "*.jpeg"))
)

with open(output_txt, "w") as f:
    for path in image_paths:
        f.write(os.path.abspath(path) + "\n")

print(f"Saved {len(image_paths)} image paths to {output_txt}")