import os
import random
import shutil
from glob import glob  # You forgot this import

path_to_datasets = "/Users/katjahellgren/YOLOv8-pt-fedn/split_datasets"
LOCAL_PATH = "/Users/katjahellgren/YOLOv8-pt-fedn/datasets"
# Output folders
OUTPUT_PATH = "/Users/katjahellgren/YOLOv8-pt-fedn/split_datasets"
TRAIN_PATH = os.path.join(OUTPUT_PATH, "train")
TEST_PATH = os.path.join(OUTPUT_PATH, "valid")

# File extensions
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
TEXT_EXT = ".txt"

TRAIN_RATIO = 0.8


def make_dirs():
    for path in [TRAIN_PATH, TEST_PATH]:
        os.makedirs(path, exist_ok=True)


def get_pairs():
    """Return a list of (image_path, text_path) pairs."""
    pairs = []
    for fname in os.listdir(LOCAL_PATH):
        if fname.lower().endswith(IMAGE_EXTS):
            base = os.path.splitext(fname)[0]
            img_path = os.path.join(LOCAL_PATH, fname)
            txt_path = os.path.join(LOCAL_PATH, base + TEXT_EXT)
            if os.path.exists(txt_path):
                pairs.append((img_path, txt_path))
            else:
                print(f"⚠️  Warning: text file missing for {img_path}")
    return pairs

def split_pairs():
    pairs = get_pairs()
    random.shuffle(pairs)

    split_idx = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]

    # ✅ Ensure train/test subdirectories exist
    for subset in ["train", "valid"]:
        os.makedirs(os.path.join(OUTPUT_PATH, subset, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_PATH, subset, "labels"), exist_ok=True)

    # Copy train pairs
    for img, txt in train_pairs:
        shutil.copy2(img, os.path.join(TRAIN_PATH, "images"))
        shutil.copy2(txt, os.path.join(TRAIN_PATH, "labels"))

    # Copy test pairs
    for img, txt in test_pairs:
        shutil.copy2(img, os.path.join(TEST_PATH, "images"))
        shutil.copy2(txt, os.path.join(TEST_PATH, "labels"))

    print(f"✅ Split done: {len(train_pairs)} train pairs, {len(test_pairs)} test pairs")

split_pairs()
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