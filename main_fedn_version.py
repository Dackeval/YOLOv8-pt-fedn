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

from trainer import Trainer
from fedn_util import extract_weights_from_model, load_weights_into_model




with open(os.path.join("utils", "args.yaml"), errors="ignore") as f:
    params = yaml.safe_load(f)

@allure.feature("Model Training")
@allure.story("FL Training")
class FEDnWrapper:

    def __init__(self,trainer):
        self.trainer = trainer

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
        train_communication_size = os.path.getsize(out_model) + os.path.getsize(training_metadata)
        print(f"Communication size for training: {train_communication_size} bytes")
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
        m_pre, m_rec, map50, mean_ap = self.trainer.validate()

        performance = {
            "val_precision": m_pre,
            "val_recall": m_rec,
            "val_map50": map50,
            "val_map": mean_ap,
        }
        upd_weights =  [val.cpu().numpy() for _, val in self.trainer.model.state_dict().items()]
        print("new state: ",  np.sum([np.linalg.norm(a) for a in upd_weights]))

        # elapsed time validation
        elapsed_time = time.perf_counter() - train_start
        print(f"Training took {elapsed_time:.2f} seconds")

        # round time 
        validation_complete = time.time()

        validation_metrics_size = os.path.getsize(performance)
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









def main():
    start_test_time = time.perf_counter()


    project_url = os.getenv("PROJECT_URL")
    print("project_url: ", project_url)
    client_token = os.getenv("FEDN_AUTH_TOKEN")
    print("client_token: ", client_token)

    data_path = os.getenv("DATA_PATH")
    name = data_path.split("/")[-1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", default=640, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--local_updates", default=100, type=int)
    args = parser.parse_args()

    trainer = Trainer(args, params)
    fednwrapper = FEDnWrapper(trainer)

    fedn_client = FednClient(
        train_callback=fednwrapper.train, validate_callback=fednwrapper.validate
    )


    fedn_client.set_name(name)

    client_id = str(uuid.uuid4())
    #client_id = "214"
    fedn_client.set_client_id(client_id)
    print(client_id)
    controller_config = {
        "name": name,
        "client_id": client_id,
        "package": "local",
        "preferred_combiner": "",
    }

    result, combiner_config = fedn_client.connect_to_api(
        "https://" + project_url + "/", client_token, controller_config
    )

    print("result: ", result)
    print(combiner_config)
    if result != ConnectToApiResult.Assigned:
        print("Failed to connect to API, exiting.")
        exit(1)

    result: bool = fedn_client.init_grpchandler(
        config=combiner_config, client_name=name, token=client_token
    )
    print("result: ", result)
    if not result:
        exit(1)

    fedn_client.run()


if __name__ == "__main__":
    main()
    # training_local()
