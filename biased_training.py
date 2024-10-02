import argparse
from datetime import datetime
import os
import copy
import numpy as np

from src.datasets.dataset_utils import concat_subsets
import torch
import wandb

from src import utils
from src.models import model_utils
from src.datasets import data_preparation
from src.datasets.data_preparation import subsample
from src.optimizers.dataloaders import WeightedDataLoader
import src.optimizers.optim_utils
from src.optimizers.subpopbench import ERM, get_subpop_optimizer, get_sample_weights, is_two_stage_optimizer


def train(conf, conf_path=None):
    conf["exp_id"] = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join("checkpoints/", conf["exp_id"]), mode=0o777)
    utils.save_config(conf, os.path.join("checkpoints/", conf["exp_id"], "config.yaml"))


    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = utils.get_device(conf)

    train_ds, eval_ds, test_ds = data_preparation.load_data(conf=conf)

    if 'local_training_id' in conf.keys():
        if conf["local_training_id"] is not None:
            
            ds_split = data_preparation.split_data(
                    train_ds,
                    conf
                )
            if conf["local_training_id"]=="all":
                # Reconstruct one global dataset if data dropping happened
                total_length = sum([len(ds) for ds in ds_split[:conf["num_clients"]]])
                if len(train_ds)!=total_length:
                    train_ds = concat_subsets(ds_split, conf["num_clients"])
            else:
                # Feature to do local training for one client's data only
                train_ds = ds_split[conf["local_training_id"]]
    conf["len_total_data"] = len(train_ds)
    print("Dataset size: ", len(train_ds))

    # Loading model
    model = model_utils.init_model(conf).to(device)
    # model = mobilenet_v2(pretrained=False).to(device)
    model_utils.print_summary(model)



    # Train ERM for a few epoch
    print("Pre-train with ERM")
    train_loader = WeightedDataLoader(dataset=train_ds, weights=None,
                                      batch_size=conf['batch_size'], shuffle=True)
    erm_conf = copy.deepcopy(conf)
    erm_conf["client_opt"]["subpop_optimizer"] = "ERM"
    erm_conf["epochs"] = 1
    erm_conf["client_opt"]["loss_function"] = "cross_entropy"

    opt = get_subpop_optimizer(model, train_loader.dataset, erm_conf)
    for epoch in range(erm_conf['epochs']):

        model.train()
        running_loss = 0.0
        for indeces, images, (labels, groups) in train_loader:
            images, labels = images.to(device), labels.to(device)
            indeces, groups = indeces.to(device), groups.to(device)

            opt_out = opt.update((indeces, images, labels, groups), 1)
            loss = opt_out["loss"]

            running_loss += loss

        print(f"Epoch [{epoch + 1}/{erm_conf['epochs']}], Loss: {running_loss / len(train_loader):.4f}")

    model.to("cpu")

    # Train the biased classifier
    print("Train biased classifier")
    biased_model = copy.deepcopy(model).to(device)

    biased_conf = copy.deepcopy(conf)
    biased_conf["client_opt"]["subpop_optimizer"] = "CRT"
    biased_conf["client_opt"]["loss_function"] = "generalized_cross_entropy"
    biased_conf["client_opt"]["generalized_cross_entropy_q"] = 0.7
    biased_conf["epochs"] = 1

    opt = get_subpop_optimizer(biased_model, train_loader.dataset, biased_conf)
    for epoch in range(biased_conf['epochs']):

        biased_model.train()
        running_loss = 0.0
        for indeces, images, (labels, groups) in train_loader:
            images, labels = images.to(device), labels.to(device)
            indeces, groups = indeces.to(device), groups.to(device)

            opt_out = opt.update((indeces, images, labels, groups), 1)
            loss = opt_out["loss"]

            running_loss += loss

        print(f"Epoch [{epoch + 1}/{biased_conf['epochs']}], Loss: {running_loss / len(train_loader):.4f}")

    # Get biased predictions
    print("Get biased predictions")
    predictions = {}
    biased_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for indeces, images, (labels, groups) in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = biased_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i, v in zip(indeces.to('cpu').numpy(), np.logical_not((predicted == labels).to('cpu').numpy()).astype(int)):
                predictions[i] = v
    
   # Classify majority-minority

   # print(predictions)
   
   # Train spurious classifier

   # Evaluate predicted N matrix


def main():
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
    train(conf, conf_path=args.config_path)


if __name__ == '__main__':
    main()
