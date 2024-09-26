# Installed modules
import argparse
from datetime import datetime

import flwr as fl
from flwr.common import ndarrays_to_parameters
import os

# Own modules
from src import utils
from src.datasets import data_preparation
from src.models import model_utils
from src.flower_strategy import MyStrategy
from src.flower_client import FlowerClient

global conf
global ds_split
global val_ds


# TODO: I don't know how to pass parameters to this function
def client_fn(cid: str) -> fl.client.Client:
    """Prepare flower client from ID (following flower documentation)"""
    client = FlowerClient(int(cid), conf)
    client_train_ds = ds_split[int(cid)]
    client_train_ds = data_preparation.preprocess_data(client_train_ds, conf, shuffle=True)
    # client_train_ds = data_preparation.get_ds_from_np((X_split[int(cid)], Y_split[int(cid)]))
    client.load_data(client_train_ds, val_ds)
    client.init_model()
    return client.to_client()


def train(conf_path=None):
    """Flower training simulation using global config"""
    global ds_split
    global val_ds

    conf["exp_id"] = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join("checkpoints/", conf["exp_id"]), mode=0o777)
    utils.save_config(conf, os.path.join("checkpoints/", conf["exp_id"], "config.yaml"))

    train_ds, val_ds, test_ds = data_preparation.load_data(conf=conf)
    val_ds = data_preparation.preprocess_data(val_ds, conf, shuffle=False)
    # X_val, Y_val = data_preparation.get_np_from_ds(val_ds)
    # X_train, Y_train = data_preparation.get_np_from_ds(train_ds)
    ds_split = data_preparation.split_data(
        train_ds,
        conf
    )

    total_length = sum([len(ds) for ds in ds_split[:conf["num_clients"]]])

    conf["len_total_data"] = total_length

    if conf["wandb"]:
        import wandb
        if "store_id" in conf.keys():
            if conf["store_id"]:
                conf["run_id"] = conf_path.split('/')[-1].split('.')[0]
        wandb.init(
            project="spurious_FL",
            entity="predictive-analytics-lab",
            tags=["federated"],
            config=conf,
            id=conf["exp_id"],
            job_type="train",
            reinit=True
        )

    initial_model = model_utils.init_model(
        conf=conf
    )

    model_utils.print_summary(initial_model)
    ws = model_utils.get_weights(initial_model)
    initial_parameters = ndarrays_to_parameters(
        ws
    )

    # Create FedAvg strategy
    strategy = MyStrategy(
        conf=conf,
        initial_parameters=initial_parameters,  # avoid smaller models as init
        fraction_fit=1.0,  # Sample 10% of available clients for training
        fraction_evaluate=0.000001,  # Sample 5% of available clients for evaluation
        min_fit_clients=1,  # Never sample less than 10 clients for training
        min_evaluate_clients=1,  # Never sample less than 5 clients for evaluation
        # min_available_clients=1, # Wait until at least 75 clients are available
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=conf["num_clients"],
        config=fl.server.ServerConfig(num_rounds=conf["rounds"]),
        strategy=strategy,
        ray_init_args=conf["ray_init_args"],
        client_resources=conf["client_resources"],
    )
    if conf["wandb"]:
        wandb.finish()
        wandb.finish()

    # TODO there is a new, better way of returning with latest model
    model_path = os.path.join(
        "checkpoints/",
        conf["exp_id"],
        "final"
    )
    model = model_utils.init_model(conf=conf, model_path=model_path)
    return model


if __name__ == "__main__":
    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Config path",
        default="config.yaml",
    )
    parser.add_argument(
        "--env_path",
        type=str,
        help="Environment path",
        default="env.yaml",
    )
    args = parser.parse_args()

    conf = utils.load_config(config_path=args.config_path, env_path=args.env_path)
    print(conf)
    train(conf_path=args.config_path)
