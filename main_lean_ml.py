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

exp_name = 'lean_ml_fhl_airfield_lr_0.001-500_exp2'
LOCAL_PATH = "/data/aggregated_datasets"
ACROSSER = {
    "host": "100.124.13.41",  # Acrosser IP
    "user": "nviduser",
    "password": "nVidia64GB"
}
JETSONS = [
    {"host": "192.168.1.2", "user": "nviduser", "password": "nVidia64GB", "path": "PATH1"},
    {"host": "192.168.1.3", "user": "nviduser", "password": "nVidia64GB", "path": "PATH2"},
    {"host": "192.168.1.4", "user": "nviduser", "password": "nVidia64GB", "path": "PATH3"},
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


@allure.step("Fetch Data Partitions")
def fetch_and_stream(server_path):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ACROSSER["host"], username=ACROSSER["user"], password=ACROSSER["password"])

    for jetson in JETSONS:
        # Command on Acrosser: scp from Jetson → Server
        cmd = (
            f"scp -r {jetson['user']}@{jetson['host']}:{jetson['path']} "
            f"{ACROSSER['user']}@{ACROSSER['host']}:{server_path}/{jetson['user']}"
        )
        stdin, stdout, stderr = ssh.exec_command(cmd)
        print(stdout.read().decode())
        print(stderr.read().decode())

    ssh.close()


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
    fetch_and_stream(LOCAL_PATH) # add path
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

    #client_names = ['/home/niklas/fedn-ultralytics-tutorial/datasets/dataset_FHL',
    #'/home/niklas/fedn-ultralytics-tutorial/datasets/dataset_Airfield']
    
    dataset_path = params['dataset_path']
    client_names = [os.path.join(dataset_path,'dataset_Airfield')]#,
#                  os.path.join(dataset_path,'dataset_FHL')]

    train_loader = get_concatenated_dataloader(client_names, "train", args, params,num_workers=8)
    print("train_loader dataset len: ", len(train_loader.dataset))
    trainer = Trainer(args, params, data_path=client_names[0])
    trainer.train_loader = PersistentDataLoader(train_loader)

    val_clients = {}
    for client_name in client_names:
        val_clients[client_name.split("/")[-1]] = Trainer(args, params, data_path=client_name)

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