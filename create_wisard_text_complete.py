import os
import random
import shutil
from glob import glob  # You forgot this import
import allure
from paramiko.proxy import ProxyCommand
import paramiko
import stat

root = os.getcwd()
path_to_datasets = os.path.join(root, "split_datasets")
LOCAL_PATH = os.path.join(root, "datasets")
# Output folders
TRAIN_PATH = os.path.join(path_to_datasets, "train")
TEST_PATH = os.path.join(path_to_datasets, "valid")

# File extensions
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
TEXT_EXT = ".txt"

TRAIN_RATIO = 0.8

ACROSSER = {
    "host": "100.124.13.41",  # Acrosser IP
    "user": "nviduser",
    "password": "nVidia64GB"
}

JETSONS = [
    {"host": "192.168.1.2", "user": "nviduser", "password": "nVidia64GB", "path": "/home/nviduser/pt1"},
    {"host": "192.168.1.3", "user": "nviduser", "password": "nVidia64GB", "path": "/home/nviduser/pt2"},
    {"host": "192.168.1.4", "user": "nviduser", "password": "nVidia64GB", "path": "/home/nviduser/pt3"},
]

def make_dirs():
    for path in [TRAIN_PATH, TEST_PATH]:
        os.makedirs(path, exist_ok=True)

def get_pairs():
    """Collect all (image, text) file pairs from dataset subfolders."""
    pairs = []
    for root, _, files in os.walk(LOCAL_PATH):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                base = os.path.splitext(file)[0]
                img_path = os.path.join(root, file)
                txt_path = os.path.join(root, base + ".txt")
                if os.path.exists(txt_path):
                    pairs.append((img_path, txt_path))
    return pairs

def split_pairs():
    pairs = get_pairs()

    # Make sure train/test subfolders exist
    for folder in [TRAIN_PATH, TEST_PATH]:
        os.makedirs(os.path.join(folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(folder, "labels"), exist_ok=True)

    # Rule-based split
    train_pairs, test_pairs = [], []
    for img, txt in pairs:
        if "SuddenValley" in os.path.basename(img):  # check filename
            test_pairs.append((img, txt))
        else:
            train_pairs.append((img, txt))

    # Copy files
    for img, txt in train_pairs:
        shutil.copy2(img, os.path.join(TRAIN_PATH, "images"))
        shutil.copy2(txt, os.path.join(TRAIN_PATH, "labels"))

    for img, txt in test_pairs:
        shutil.copy2(img, os.path.join(TEST_PATH, "images"))
        shutil.copy2(txt, os.path.join(TEST_PATH, "labels"))

    print(f"Train: {len(train_pairs)} pairs, Test: {len(test_pairs)} pairs")

def sftp_get_dir(sftp, remote_dir, local_dir, jetson_prefix):
    """Recursively fetch a directory via SFTP and aggregate into one folder."""
    os.makedirs(local_dir, exist_ok=True)

    for entry in sftp.listdir_attr(remote_dir):
        remote_path = f"{remote_dir}/{entry.filename}"

        if stat.S_ISDIR(entry.st_mode):
            # Recurse into subdir, keep structure
            new_local_dir = os.path.join(local_dir, entry.filename)
            sftp_get_dir(sftp, remote_path, new_local_dir, jetson_prefix)
        else:
            # Prefix filename with Jetson IP to avoid overwrite
            base, ext = os.path.splitext(entry.filename)
            new_filename = f"{jetson_prefix}_{base}{ext}"
            local_path = os.path.join(local_dir, new_filename)

            print(f"Fetching {remote_path} -> {local_path}")
            sftp.get(remote_path, local_path)

@allure.step("Fetch Data Partitions")
def fetch_and_aggregate():
    for jetson in JETSONS:
        print(f"Connecting to Jetson {jetson['host']}...")

        proxy_cmd = (
            f"ssh -o StrictHostKeyChecking=no "
            f"-W {jetson['host']}:22 "
            f"{ACROSSER['user']}@{ACROSSER['host']}"
        )
        sock = ProxyCommand(proxy_cmd)

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        ssh.connect(
            hostname=jetson["host"],
            username=jetson["user"],
            sock=sock,
            password=jetson.get("password")  # if needed
        )

        sftp = ssh.open_sftp()

        jetson_prefix = jetson["host"].replace('.', '_')
        print(f"Downloading directory {jetson['path']} (Jetson {jetson_prefix}) into {LOCAL_PATH}")
        sftp_get_dir(sftp, jetson["path"], LOCAL_PATH, jetson_prefix)

        sftp.close()
        ssh.close()
        print(f"✅ Done with {jetson['host']}\n")
    run()

def run():
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
