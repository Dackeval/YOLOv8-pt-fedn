import argparse
import json
import os
import uuid
import numpy as np
import yaml
from fedn.network.clients.fedn_client import ConnectToApiResult, FednClient
from fedn.utils.helpers.helpers import save_metadata
import time
import allure
import socket
from types import SimpleNamespace


from trainer import Trainer
from fedn_util import extract_weights_from_model, load_weights_into_model
from config import settings

try:
    import config as cfg
    CFG = getattr(cfg, "settings", {})  # your dict
except Exception:
    CFG = {}

class EarlyStoppingMAP:
    def __init__(self, patience=settings["PATIENCE"], min_delta=settings["MIN_DELTA"], verbose=True):
        """
        Args:
            patience (int): Number of rounds to wait for improvement before stopping
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

def nbytes(obj) -> int:
    """Return byte size for path/bytes/BytesIO/stream-like objects."""
    if isinstance(obj, (str, os.PathLike)):
        return os.path.getsize(obj)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return len(obj)
    # BytesIO (and most io.BufferedIOBase) exposes getbuffer()
    gb = getattr(obj, "getbuffer", None)
    if callable(gb):
        return gb().nbytes
    gv = getattr(obj, "getvalue", None)
    if callable(gv):
        return len(gv())
    # Generic readable stream fallback
    read = getattr(obj, "read", None)
    if callable(read):
        pos = obj.tell() if hasattr(obj, "tell") else None
        data = read()
        size = len(data) if data is not None else 0
        if pos is not None and hasattr(obj, "seek"):
            obj.seek(pos)
        return size
    raise TypeError(f"Don't know how to get size of {type(obj)}")

with open(os.path.join("utils", "args.yaml"), errors="ignore") as f:
    params = yaml.safe_load(f)

@allure.feature("Model Training")
@allure.story("FL Training")
class FEDnWrapper:

    def __init__(self,trainer):
        self.trainer = trainer
        self.early_stopper = EarlyStoppingMAP(patience=settings["PATIENCE"], min_delta=settings["MIN_DELTA"])

    @allure.step("Training the model")
    def train(self, weights, client_settings):
        
        # start train timer
        train_start = time.perf_counter()

        old_weights =  [val.cpu().numpy() for _, val in self.trainer.model.state_dict().items()]

        load_weights_into_model(weights, self.trainer.model)
        upd_weights =  [val.cpu().numpy() for _, val in self.trainer.model.state_dict().items()]
        distance = np.sum([np.linalg.norm(a-b) for a,b in zip(old_weights,upd_weights)])
        print("train distance: ", distance)
        print("old state: ",  np.sum([np.linalg.norm(a) for a in old_weights]))
        print("new state: ",  np.sum([np.linalg.norm(a) for a in upd_weights]))

        self.trainer.train()
        out_model = extract_weights_from_model(self.trainer.model)
        metadata = {
            "training_metadata": {
                # num_examples are mandatory
                "num_examples": 1,  # len(train_loader.dataset),
                "batch_size": 32,
                "epochs": 1,
                "lr": self.trainer.optimizer.param_groups[0]["lr"],
            }
        }
        outpath = "temp"
        save_metadata(metadata, outpath)
        with open(outpath + "-metadata", "r") as fh:
            training_metadata = json.loads(fh.read())

        os.unlink(outpath + "-metadata")
        upd_weights =  [val.cpu().numpy() for _, val in self.trainer.model.state_dict().items()]
        print("new state: ",  np.sum([np.linalg.norm(a) for a in upd_weights]))

        # train time elaspsed
        elapsed_time = time.perf_counter() - train_start
        print(f"Training took {elapsed_time:.2f} seconds")
        # size of the model and metadata
        model_size_bytes = nbytes(out_model)
        meta_size_bytes  = len(json.dumps(training_metadata, separators=(",", ":")).encode("utf-8"))
        train_communication_size = model_size_bytes + meta_size_bytes
        print(f"Communication size for training: {train_communication_size} bytes "
            f"(model={model_size_bytes}, meta={meta_size_bytes})")
        # attach the size of the model and metadata to allure report
        allure.attach(
            f"{train_communication_size} bytes",
            name="Training communication size",
            attachment_type=allure.attachment_type.TEXT
        )
        # attach the timing information to allure report
        allure.attach(
            f"Training took {elapsed_time:.2f} seconds",
            name="Training time",
            attachment_type=allure.attachment_type.TEXT
        )
        

        return out_model, training_metadata
    
    @allure.step("Validating the model")
    def validate(self,  weights):
        
        # start validation timer
        train_start = time.perf_counter()

        old_weights =  [val.cpu().numpy() for _, val in self.trainer.model.state_dict().items()]
        load_weights_into_model(weights, self.trainer.model)
        upd_weights =  [val.cpu().numpy() for _, val in self.trainer.model.state_dict().items()]
        distance = np.sum([np.linalg.norm(a-b) for a,b in zip(old_weights,upd_weights)])
        print("val distance: ", distance)
        print("old state: ",  np.sum([np.linalg.norm(a) for a in old_weights]))
        print("new state: ",  np.sum([np.linalg.norm(a) for a in upd_weights]))
        m_pre, m_rec, map50, mean_ap = self.trainer.validate(self.trainer.model)

        performance = {
            "val_precision": m_pre,
            "val_recall": m_rec,
            "val_map50": map50,
            "val_map": mean_ap,
        }

        # Early stopping check
        self.early_stopper(mean_ap, self.trainer.model)
        if self.early_stopper.should_stop:
            print("Early stopping triggered during validation.")
            allure.attach(
                f"Early stopping triggered during validation. Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}",
                name="Early Stopping",
                attachment_type=allure.attachment_type.TEXT
            )


        upd_weights =  [val.cpu().numpy() for _, val in self.trainer.model.state_dict().items()]
        print("new state: ",  np.sum([np.linalg.norm(a) for a in upd_weights]))

        # elapsed time validation
        elapsed_time = time.perf_counter() - train_start
        print(f"Training took {elapsed_time:.2f} seconds")

        # round time 
        validation_complete = time.time()

        validation_metrics_size = len(json.dumps(performance, separators=(",", ":")).encode("utf-8"))
        print(f"Communication size for validation: {validation_metrics_size} bytes")

        allure.attach(
            f"{validation_metrics_size} bytes",
            name="Validation metrics size",
            attachment_type=allure.attachment_type.TEXT
        )
        # attach the timing information to allure report
        allure.attach(
            f"Validation took {elapsed_time:.2f} seconds",
            name="Validation time",
            attachment_type=allure.attachment_type.TEXT
        )
        # attach the performance metrics to allure report
        allure.attach(
            f"Round completed at {validation_complete}",
            name="Round completion time",
            attachment_type=allure.attachment_type.TEXT
        )


        return performance
    

def resolve_data_path(params):
    candidates = [
        os.getenv("DATA_PATH"),
        CFG.get("DATA_PATH"),
        params.get("dataset_path"),
    ]
    for p in candidates:
        if p:
            p = os.path.abspath(p)
            if (os.path.isfile(os.path.join(p, "train.txt")) and
                os.path.isfile(os.path.join(p, "valid.txt"))):
                return p
    raise ValueError("DATA_PATH not set. Put DATA_PATH in config.py settings, "
                     "or export DATA_PATH, or set params['dataset_path'].")


def main():
    start_test_time = time.perf_counter()
    allure.attach(
        f"Test started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_test_time))}",
        name="Test start time",
        attachment_type=allure.attachment_type.TEXT
    )

    project_url = str(settings.get("DISCOVER_HOST"))
    print("project_url: ", project_url)
    client_token = str(settings.get("CLIENT_TOKEN"))
    print("client_token: ", client_token)

    data_path =  str(settings.get("DATA_PATH"))
    name = data_path.split("/")[-1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", default=640, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--epochs", default=settings["ROUNDS"], type=int)
    parser.add_argument("--local_updates", default=settings["LOCAL_UPDATES"], type=int)
    args = parser.parse_args()

    data_path = resolve_data_path(params)
    data_base = os.path.basename(os.path.normpath(data_path))
    unique_name = f"{socket.gethostname()}-{data_base}"
    # pass it explicitly
    trainer = Trainer(args, params, data_path=data_path)

    fednwrapper = FEDnWrapper(trainer)

    fedn_client = FednClient(
        train_callback=fednwrapper.train, validate_callback=fednwrapper.validate
    )

    BASE_URL = "http://" + settings["DISCOVER_HOST"] + ":8092/"   

    def base(url: str) -> str:
        url = (url or "").strip().strip('"').strip("'")
        return url if url.endswith("/") else url + "/"

    url = base(BASE_URL)
    print("API base:", repr(url))  

    fedn_client.set_name("client")  
    fedn_client.set_client_id(str(uuid.uuid4()))

    controller_config = {
        "name": fedn_client.name,
        "client_id": fedn_client.client_id,
        "package": "local",
        "preferred_combiner": "",
    }

    result, _  = fedn_client.connect_to_api(url=url, token=None, json=controller_config)
    assert result == ConnectToApiResult.Assigned, f"Not assigned: {result}"
    
    combiner_config = SimpleNamespace(
        host   = settings["DISCOVER_HOST"],        
        port   = 12080,        
        status = "assigned",
        fqdn = "",
        package = "local",
        ip = "",
        helper_type = ""                      
    )

    ok = fedn_client.init_grpchandler(
        config=combiner_config,
        client_name=fedn_client.name,
        token=None
    )
    assert ok

    fedn_client.run()



if __name__ == "__main__":
    main()
    # training_local()
