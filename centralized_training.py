import argparse
from datetime import datetime
import os

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
        _, _, group_acc = model_utils.evaluate(model, test_loader, conf)
        print(group_acc)
        wandb.log(group_acc, step=epoch)


def train(conf):
    conf["exp_id"] = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join("checkpoints/", conf["exp_id"]), mode=0o777)
    utils.save_config(conf, os.path.join("checkpoints/", conf["exp_id"], "config.yaml"))
    if conf["wandb"]:
        wandb.init(
            project="spurious_FL",
            entity="predictive-analytics-lab",
            config=conf,
            id=conf["exp_id"],
            job_type="train",
            reinit=True
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds, eval_ds, test_ds = data_preparation.load_data(conf=conf)
    train_loader = DataLoader(dataset=train_ds, batch_size=conf['batch_size'], shuffle=True, drop_last=True)
    eval_loader = DataLoader(dataset=eval_ds, batch_size=conf['batch_size'], shuffle=False)
    test_loader = DataLoader(dataset=test_ds, batch_size=conf['batch_size'], shuffle=False)

    model = model_utils.init_model(conf).to(device)
    # model = mobilenet_v2(pretrained=False).to(device)
    model_utils.print_summary(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=float(conf['client_opt']['learning_rate']),
                    momentum=float(conf['client_opt']['momentum']))

    for epoch in range(conf['epochs']):

        model.train()
        running_loss = 0.0

        for images, (labels, groups) in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

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
    train(conf)


if __name__ == '__main__':
    main()
