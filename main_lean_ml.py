from dataloader import get_concatenated_dataloader
from trainer import Trainer
import os
from datetime import datetime
import yaml
import argparse
import csv
from trainer import PersistentDataLoader
import paramiko
import allure
import time
from paramiko.proxy import ProxyCommand
from create_wisard_text_complete import fetch_and_aggregate
import torch
from config import settings

exp_name = 'lean_ml_fhl_airfield_lr_0.001-500_exp2'
root = os.getcwd()

class EarlyStoppingMAP:
    def __init__(self, patience=settings["PATIENCE"], min_delta=settings["MIN_DELTA"], verbose=True):
        """
        Args:
            patience (int): Number of epochs to wait for improvement before stopping
            min_delta (float): Minimum improvement in mAP to count as progress
            verbose (bool): Print messages when stopping or saving best model
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.best_state = None  # store model state_dict()

    def __call__(self, current_map, model):
        if self.best_score is None:
            self.best_score = current_map
            self.best_state = model.state_dict()
            if self.verbose:
                print(f"Initial best mAP set at {current_map:.4f}")
        elif current_map < self.best_score + self.min_delta:
            # No significant improvement
            self.counter += 1
            if self.verbose:
                print(f"No improvement in mAP ({current_map:.4f}), patience {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print("Early stopping triggered.")
        else:
            # Improvement found
            self.best_score = current_map
            self.best_state = model.state_dict()
            self.counter = 0
            if self.verbose:
                print(f"New best mAP: {current_map:.4f}")

    def restore_best_weights(self, model):
        """Restore model to the best saved state"""
        if self.best_state:
            model.load_state_dict(self.best_state)
            if self.verbose:
                print(f"Restored model to best mAP = {self.best_score:.4f}")


def log_metrics(round_id, metrics, filename='nono/step.csv', early_stop=False):
    """
    Appends metrics to a CSV file. Writes header if file does not exist.
    
    Args:
        round_id (int): Current epoch/round number.
        metrics (tuple): (precision, recall, mAP@50, mAP)
        filename (str): Path to the CSV file.
        early_stop (bool): Whether early stopping was triggered this epoch.
    """
    file_exists = os.path.exists(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['round', 'precision', 'recall', 'mAP@50', 'mAP', 'timestamp', 'early_stop']
        )
        if not file_exists:
            writer.writeheader()

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        writer.writerow({
            'round': str(round_id + 1).zfill(3),
            'precision': f'{metrics[0]:.3f}',
            'recall': f'{metrics[1]:.3f}',
            'mAP@50': f'{metrics[2]:.3f}',
            'mAP': f'{metrics[3]:.3f}',
            'timestamp': timestamp,
            'early_stop': early_stop
        })
        f.flush()

@allure.feature("Model Training")
@allure.story("Centralized ML Training")
def main():

    with open(os.path.join("utils", "args.yaml"), errors="ignore") as f:
        params = yaml.safe_load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", default=640, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--local_updates", default=settings["LOCAL_UPDATES"]*3, type=int)
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

    early_stopper = EarlyStoppingMAP(patience=settings["PATIENCE"], min_delta=settings["MIN_DELTA"])

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
            m_pre, m_rec, map50, mean_ap = val_clients[val_client].validate(trainer.model)
            early_stopper(mean_ap, trainer.model)
            
            log_metrics(epoch, [m_pre, m_rec, map50, mean_ap], os.path.join(exp_name,name+'_step.csv'),early_stop=early_stopper.should_stop)

            if early_stopper.should_stop:
                print(f"Stopping training at epoch {epoch}")
                break



if __name__ == "__main__":
    main()