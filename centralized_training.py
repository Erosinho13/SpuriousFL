import argparse
from datetime import datetime
import os
import copy

from src.optimizers.dataloaders import WeightedDataLoader
import src.optimizers.optim_utils
from src.optimizers.subpopbench import ERM, get_subpop_optimizer, get_sample_weights, is_two_stage_optimizer
import torch
import wandb
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from src import utils
from src.models import model_utils
from src.datasets import data_preparation


def test_model(test_loader, model, device, conf, epoch, train_set=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, (labels, groups) in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"{'Test' if not train_set else 'Train'} accuracy: {100 * correct / total}%")
    if conf["wandb"]:
        key = 'test_accuracy' if not train_set else 'train_accuracy'
        wandb.log({key: 100 * correct / total}, step=epoch)
    if not train_set:
        _, _, group_acc = src.optimizers.optim_utils.evaluate(model, test_loader, conf)
        print(group_acc)
        wandb.log(group_acc, step=epoch)


def train(conf, conf_path=None):
    conf["exp_id"] = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join("checkpoints/", conf["exp_id"]), mode=0o777)
    utils.save_config(conf, os.path.join("checkpoints/", conf["exp_id"], "config.yaml"))
    if conf["wandb"]:
        if "store_id" in conf.keys():
            if conf["store_id"]:
                conf["run_id"] = conf_path.split('/')[-1].split('.')[0]
        wandb.init(
            project="spurious_FL",
            entity="predictive-analytics-lab",
            tags=["centralized"],
            config=conf,
            id=conf["exp_id"],
            job_type="train",
            reinit=True
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds, eval_ds, test_ds = data_preparation.load_data(conf=conf)
    train_sample_weights = get_sample_weights(train_ds, conf)
    train_loader = WeightedDataLoader(dataset=train_ds, weights=train_sample_weights,
                                      batch_size=conf['batch_size'], shuffle=True)
    eval_loader = WeightedDataLoader(dataset=eval_ds, batch_size=conf['batch_size'], shuffle=False)
    test_loader = WeightedDataLoader(dataset=test_ds, batch_size=conf['batch_size'], shuffle=False)

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
            for epoch in range(first_stage_conf['epochs']):
                model.train()
                running_loss = 0.0
                for images, (labels, groups) in train_loader:
                    images, labels = images.to(utils.get_device(first_stage_conf)), labels.to(utils.get_device(first_stage_conf))
                    groups = groups.to(utils.get_device(first_stage_conf))
                    opt_out = opt.update((None, images, labels, groups), 1)
                    loss = opt_out["loss"]
                    running_loss += loss
                print(f"Epoch [{epoch + 1}/{first_stage_conf['epochs']}], Loss: {running_loss / len(train_loader):.4f}")
            print("First stage training finished")       
        else:
            print("First stage weights from: ",conf["checkpoint"])
        test_model(test_loader, model, device, conf, 0)
        

    opt = get_subpop_optimizer(model, train_ds, conf)
    for epoch in range(conf['epochs']):

        model.train()
        running_loss = 0.0

        for images, (labels, groups) in train_loader:
            images, labels = images.to(utils.get_device(conf)), labels.to(utils.get_device(conf))
            groups = groups.to(utils.get_device(conf))


            opt_out = opt.update((None, images, labels, groups), 1)
            loss = opt_out["loss"]

            running_loss += loss

        print(f"Epoch [{epoch + 1}/{conf['epochs']}], Loss: {running_loss / len(train_loader):.4f}")
        if conf["wandb"]:
            wandb.log({'train_loss': running_loss / len(train_loader)}, step=epoch)

        if (epoch + 1) % conf['eval_interval'] == 0:
            test_model(eval_loader, model, device, conf, epoch, train_set=True)

        if (epoch + 1) % conf['test_interval'] == 0:
            test_model(test_loader, model, device, conf, epoch)

    test_model(test_loader, model, device, conf, conf['epochs'])

    if conf["wandb"]:
        wandb.finish()


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
