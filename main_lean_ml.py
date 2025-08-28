from dataloader import get_concatenated_dataloader
from trainer import Trainer
import os
import yaml
import argparse
import csv
from trainer import PersistentDataLoader
import paramiko
import allure
import time
from paramiko.proxy import ProxyCommand
import stat

exp_name = 'lean_ml_fhl_airfield_lr_0.001-500_exp2'
LOCAL_PATH = "/Users/katjahellgren/YOLOv8-pt-fedn/datasets"
ACROSSER = {
    "host": "100.124.13.41",  # Acrosser IP
    "user": "nviduser",
    "password": "nVidia64GB"
}
JETSONS = [
    {"host": "192.168.1.2", "user": "nviduser", "password": "nVidia64GB", "path": "/home/nviduser/pt1/200614_SuddenValley_Phantom_VIS_0005"},
    #{"host": "192.168.1.3", "user": "nviduser", "password": "nVidia64GB", "path": "/home/nviduser/pt2"},
    #{"host": "192.168.1.4", "user": "nviduser", "password": "nVidia64GB", "path": "/home/nviduser/pt3"},
]


def log_metrics(round_id, metrics, filename='nono/step.csv'):
    """
    Appends metrics to a CSV file. Writes header if file does not exist.
    
    Args:
        epoch (int): Current epoch number.
        metrics (tuple): (precision, recall, mAP@50, mAP)
        filename (str): Path to the CSV file.
    """
    file_exists = os.path.exists(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['round', 'precision', 'recall', 'mAP@50', 'mAP'])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'round': str(round_id + 1).zfill(3),
            'precision': f'{metrics[0]:.3f}',
            'recall': f'{metrics[1]:.3f}',
            'mAP@50': f'{metrics[2]:.3f}',
            'mAP': f'{metrics[3]:.3f}'
        })
        f.flush()


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

@allure.feature("Model Training")
@allure.story("Centralized ML Training")
def main():
    start_test_time = time.perf_counter()
    allure.attach(
        f"Test started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_test_time))}",
        name="Test start time",
        attachment_type=allure.attachment_type.TEXT
    )
    
    # Fetch and stream data from Jetsons to local path and log the time taken
    time_fetch_data = time.perf_counter()
    #fetch_and_aggregate() # add path
    allure.attach(
        f"Data fetched in {time.perf_counter() - time_fetch_data:.2f} seconds",
        name="Data Fetch Time",
        attachment_type=allure.attachment_type.TEXT
    )
    time_process_data = time.perf_counter()

    with open(os.path.join("utils", "args.yaml"), errors="ignore") as f:
        params = yaml.safe_load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", default=640, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--local_updates", default=70, type=int)
    args = parser.parse_args()
    params["lr0"] =0.001
    params["lrf"] =1.0
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)

    
    dataset_path = params['dataset_path']
    train_loader = get_concatenated_dataloader(dataset_path, "train", args, params,num_workers=8)
    print("train_loader dataset len: ", len(train_loader.dataset))
    trainer = Trainer(args, params, data_path=dataset_path)
    trainer.train_loader = PersistentDataLoader(train_loader)

    val_clients = {}
    val_clients[dataset_path.split("/")[-1]] = Trainer(args, params, data_path=dataset_path)

    allure.attach(
        f"Data processing took {time.perf_counter() - time_process_data:.2f} seconds",
        name="Data Processing Time",
        attachment_type=allure.attachment_type.TEXT
    )
    allure.attach(
        f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.perf_counter()))}",
        name="Training Start Time",
        attachment_type=allure.attachment_type.TEXT
    )
    for epoch in range(2000):
        print(f"Epoch {epoch + 1}/{2000}")
        train_start = time.perf_counter()
        allure.attach(
            f"Epoch {epoch + 1} started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(train_start))}",
            name=f"Epoch {epoch + 1} Start Time",
            attachment_type=allure.attachment_type.TEXT
        )
        trainer.train()
        for val_client in val_clients:
            name = val_client
            val_clients[val_client].model.load_state_dict(trainer.model.state_dict())
            m_pre, m_rec, map50, mean_ap = val_clients[val_client].validate()
            log_metrics(epoch, [m_pre, m_rec, map50, mean_ap], os.path.join(exp_name,name+'_step.csv'))



if __name__ == "__main__":
    main()
