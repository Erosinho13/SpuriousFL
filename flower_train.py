# Installed modules
import argparse
from datetime import datetime

import flwr as fl
from flwr.common import ndarrays_to_parameters
import os

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import List, Optional


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


def train(conf, conf_name):
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
                conf["run_id"] = conf_name.split('.')[0]
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
        ray_init_args=conf["machine"]["ray_init_args"],
        client_resources=conf["machine"]["client_resources"],
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


@dataclass
class ClientOptConfig:
    subpop_optimizer: str
    base_optimizer: str
    learning_rate: float
    groupdro_eta: float
    cbloss_beta: float
    focal_gamma: int
    dfr_reg: float
    fex_shallow_model: str
    fex_epochs: int
    fex_balance_classes: bool
    momentum: float


@dataclass
class ServerOptConfig:
    optimizer: str
    learning_rate: float
    beta_1: float
    beta_2: float
    tau: float
    weight_clients: str


@dataclass
class DatasetConfig:
    name: str
    num_targets: int
    num_groups: int
    input_size: Optional[int]


@dataclass
class ModelConfig:
    model_type: str
    norm_layer: str
    pretrained: bool

@dataclass
class EnvironmentConfig:
    root_path: str
    ray_init_args: dict
    client_resources : dict
    CUDA_VISIBLE_DEVICES : str

@dataclass
class Config:
    seed: int
    data_shuffle_seed: Optional[int]
    dirichlet_alpha: float
    batch_size: int
    epochs: int
    rounds: int
    num_clients: int
    norm: bool
    aug_crop: int
    aug_horizontal_flip: bool
    split_mode: str
    client_opt: ClientOptConfig
    server_opt: ServerOptConfig
    wandb: bool
    model_options: ModelConfig
    dataset_options: DatasetConfig
    checkpoint: Optional[str]
    local_training_id: Optional[int]
    machine: Optional[EnvironmentConfig]


cs = ConfigStore.instance()
cs.store(group="job", name="federated_training", node=Config)


@hydra.main(config_path="conf", config_name="federated_training", version_base=None)
def main(cfg: Config):
    global conf
    hydra_cfg = HydraConfig.get()
    conf_name = hydra_cfg.job.config_name
    conf = OmegaConf.to_container(cfg, resolve=True)
    print(conf)
    train(conf, conf_name=conf_name)


if __name__ == "__main__":
    main()
