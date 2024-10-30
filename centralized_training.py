import argparse
import copy
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import hydra
import numpy as np
import torch
import wandb
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from src.config_params import Config

import src.optimizers.optim_utils
from src import utils
from src.datasets import data_preparation
from src.datasets.data_preparation import subsample
from src.datasets.dataset_utils import concat_subsets
from src.models import model_utils
from src.optimizers.dataloaders import WeightedDataLoader
from src.optimizers.subpopbench import ERM, get_sample_weights, get_subpop_optimizer, is_two_stage_optimizer


def test_model(test_loader, model, device, conf, epoch, train_set=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for indeces, images, (labels, groups) in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"{'Test' if not train_set else 'Train'} accuracy: {100 * correct / total}%")
    if conf["wandb"]:
        key = "test_accuracy" if not train_set else "train_accuracy"
        wandb.log({key: 100 * correct / total}, step=epoch)
    if not train_set:
        _, _, group_acc = src.optimizers.optim_utils.evaluate(model, test_loader, conf)
        print(group_acc)
        if conf["wandb"]:
            wandb.log(group_acc, step=epoch)


def train(conf, conf_name=None):
    os.makedirs(os.path.join("checkpoints/", conf["exp_id"]), mode=0o777)
    utils.save_config(conf, os.path.join("checkpoints/", conf["exp_id"], "config.yaml"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, eval_ds, test_ds = data_preparation.load_data(conf=conf)

    if "local_training_id" in conf["dataset_options"].keys():
        if conf["dataset_options"]["local_training_id"] is not None:

            ds_split = data_preparation.split_data(train_ds, conf)
            if conf["dataset_options"]["local_training_id"] == "all":
                # Reconstruct one global dataset if data dropping happened
                total_length = sum([len(ds) for ds in ds_split[: conf["dataset_options"]["num_clients"]]])
                if len(train_ds) != total_length:
                    train_ds = concat_subsets(ds_split, conf["dataset_options"]["num_clients"])
            else:
                # Feature to do local training for one client's data only
                train_ds = ds_split[conf["dataset_options"]["local_training_id"]]
    conf["len_total_data"] = len(train_ds)
    print("Dataset size: ", len(train_ds))

    if conf["wandb"]:
        if "store_id" in conf.keys():
            if conf["store_id"]:
                conf["run_id"] = conf_name.split(".")[0]
        else:
            conf["run_id"] = conf_name.split(".")[0]
        wandb.init(
            project="spurious_FL",
            entity="predictive-analytics-lab",
            tags=["centralized"],
            config=conf,
            id=conf["exp_id"],
            job_type="train",
            reinit=True,
        )

    train_sample_weights = get_sample_weights(train_ds, conf)
    train_loader = WeightedDataLoader(
        dataset=train_ds, weights=train_sample_weights, batch_size=conf["client_opt"]["batch_size"], shuffle=True
    )
    eval_loader = WeightedDataLoader(dataset=eval_ds, batch_size=conf["client_opt"]["batch_size"], shuffle=False)
    test_loader = WeightedDataLoader(dataset=test_ds, batch_size=conf["client_opt"]["batch_size"], shuffle=False)

    # Shallow model training for Forgettable Examples
    if "FEx" in conf["client_opt"]["subpop_optimizer"]:
        #!TODO: train shallow model
        #!TODO: get forgetting accuracies
        train_loader_1 = WeightedDataLoader(dataset=train_ds, weights=None, batch_size=conf["client_opt"]["batch_size"], shuffle=True)
        train_loader_2 = WeightedDataLoader(
            dataset=train_ds, weights=train_sample_weights, batch_size=conf["client_opt"]["batch_size"], shuffle=False
        )
        shallow_conf = copy.deepcopy(conf)
        shallow_conf["model_options"]["model_type"] = conf["client_opt"]["fex_shallow_model"]
        shallow_conf["client_opt"]["subpop_optimizer"] = "ERM"
        shallow_conf["client_opt"]["epochs"] = conf["client_opt"]["fex_epochs"]
        shallow_model = model_utils.init_model(shallow_conf)
        print("Shallow model training for FEx (Forgettable examples)")
        opt = get_subpop_optimizer(shallow_model, train_ds, shallow_conf)
        correct_preds = None
        for epoch in range(shallow_conf["client_opt"]["epochs"]):
            shallow_model.train()
            running_loss = 0.0
            for indeces, images, (labels, groups) in train_loader_1:
                indeces = indeces.to(utils.get_device(shallow_conf))
                images, labels = images.to(utils.get_device(shallow_conf)), labels.to(utils.get_device(shallow_conf))
                groups = groups.to(utils.get_device(shallow_conf))
                opt_out = opt.update((indeces, images, labels, groups), 1)
                loss = opt_out["loss"]
                running_loss += loss
            print(f"Epoch [{epoch + 1}/{shallow_conf['client_opt']['epochs']}], Loss: {running_loss / len(train_loader_1):.4f}")
            # Collect accuracies
            shallow_model.eval()
            with torch.no_grad():
                correct_all = []
                for indeces, images, (labels, groups) in train_loader_2:
                    images, labels = images.to(device), labels.to(device)
                    outputs = shallow_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    correct = (predicted == labels).cpu().numpy().astype(int)
                    correct_all.extend(correct)
                correct_all = np.array(correct_all)[:, np.newaxis]
                if correct_preds is None:
                    correct_preds = correct_all
                else:
                    correct_preds = np.concatenate((correct_preds, correct_all), axis=1)
    else:
        correct_preds = None

    # Loading real model
    model = model_utils.init_model(conf).to(device)
    # model = mobilenet_v2(pretrained=False).to(device)
    model_utils.print_summary(model)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = SGD(model.parameters(), lr=float(conf['client_opt']['learning_rate']),
    #                 momentum=float(conf['client_opt']['momentum']))

    if is_two_stage_optimizer(conf):
        #!TODO: load pretrained weights or train basic ERM
        first_stage_conf = copy.deepcopy(conf)
        first_stage_conf["wandb"] = False
        first_stage_conf["client_opt"]["subpop_optimizer"] = "ERM"
        if conf["checkpoint"] is None:
            print("First stage training with ERM")
            opt = get_subpop_optimizer(model, train_ds, first_stage_conf)
            for epoch in range(first_stage_conf["client_opt"]["epochs"]):
                model.train()
                running_loss = 0.0
                for indeces, images, (labels, groups) in train_loader:
                    indeces = indeces.to(utils.get_device(first_stage_conf))
                    images, labels = images.to(utils.get_device(first_stage_conf)), labels.to(
                        utils.get_device(first_stage_conf)
                    )
                    groups = groups.to(utils.get_device(first_stage_conf))
                    opt_out = opt.update((indeces, images, labels, groups), 1)
                    loss = opt_out["loss"]
                    running_loss += loss
                print(f"Epoch [{epoch + 1}/{first_stage_conf['client_opt']['epochs']}], Loss: {running_loss / len(train_loader):.4f}")
            print("First stage training finished")
        else:
            print("First stage weights from: ", conf["checkpoint"])
            model_path = conf["checkpoint"]
            model_utils.load_model_weights(model, model_path)
        test_model(test_loader, model, device, conf, 0)

    if conf["client_opt"]["subpop_optimizer"] == "DFR" or "FEx" in conf["client_opt"]["subpop_optimizer"]:
        # Subsample for 2nd stage training
        train_ds = subsample(train_ds, conf, accuracies=correct_preds)
        train_loader = WeightedDataLoader(dataset=train_ds, weights=None, batch_size=conf["client_opt"]["batch_size"], shuffle=True)
    print("Dataset size:", len(train_ds))
    opt = get_subpop_optimizer(model, train_loader.dataset, conf)
    if is_two_stage_optimizer(conf):
        print("Trainable parameters:", model_utils.count_params(model, only_trainable=True))
    for epoch in range(conf["client_opt"]["epochs"]):

        model.train()
        running_loss = 0.0
        for indeces, images, (labels, groups) in train_loader:
            indeces = indeces.to(utils.get_device(conf))
            images, labels = images.to(utils.get_device(conf)), labels.to(utils.get_device(conf))
            groups = groups.to(utils.get_device(conf))

            opt_out = opt.update((indeces, images, labels, groups), 1)
            loss = opt_out["loss"]

            running_loss += loss

        print(f"Epoch [{epoch + 1}/{conf['client_opt']['epochs']}], Loss: {running_loss / len(train_loader):.4f}")
        if conf["wandb"]:
            wandb.log({"train_loss": running_loss / len(train_loader)}, step=epoch)

        if (epoch + 1) % 1 == 0:
            test_model(eval_loader, model, device, conf, epoch, train_set=True)

        if (epoch + 1) % 1 == 0:
            test_model(test_loader, model, device, conf, epoch)

    test_model(test_loader, model, device, conf, conf["client_opt"]["epochs"])

    save_path = os.path.join("checkpoints", conf["exp_id"], "final")
    print("Saving model to %s", save_path)
    model_utils.save_model(model, save_path)

    if conf["wandb"]:
        wandb.finish()


cs = ConfigStore.instance()
cs.store(group="job", name="centralized_training", node=Config)

@hydra.main(config_path="conf", config_name="centralized_training", version_base=None)
def main(cfg: Config):
    hydra_cfg = HydraConfig.get()
    conf_name = hydra_cfg.job.config_name
    conf = OmegaConf.to_container(cfg, resolve=True)
    base_start_date = datetime.now()
    # Get the current run number from Hydra and add it as seconds
    try:
        run_number = hydra_cfg.job.num
        start_date = base_start_date + timedelta(seconds=run_number)
    except MissingMandatoryValue:
        start_date = base_start_date 
    conf["exp_id"] = start_date.strftime("%Y%m%d-%H%M%S")
    print(conf)
    train(conf, conf_name=conf_name)


if __name__ == "__main__":
    main()
