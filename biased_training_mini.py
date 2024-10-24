import itertools
import logging

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src import utils
from src.config_params import Config
from src.datasets import data_preparation
from src.models import model_utils
from src.optimizers.dataloaders import InfiniteDataLoader
from src.optimizers.subpopbench import get_base_optimizer


class GCELoss(nn.Module):
    def __init__(self, q: float):
        super().__init__()
        self.q = q

    def forward(self, inputs, targets):
        p = F.softmax(inputs, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        loss = ((1 - Yg.squeeze() ** self.q) / self.q).mean()
        return loss.mean()


def ground_truth_matrix(dataset, n_targets, n_groups, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    y_array, s_array = [], []
    for _, _, (y, s) in tqdm(loader, desc="Ground Truth Matrix"):
        y_array.append(y)
        s_array.append(s)

    y_array = torch.cat(y_array, dim=0)
    s_array = torch.cat(s_array, dim=0)

    m = torch.zeros((n_targets, n_groups), dtype=torch.int64)
    for y, s in itertools.product(range(n_targets), range(n_groups)):
        m[y, s] = torch.sum((y_array == y) & (s_array == s))

    return m


def training(model, loader, optimizer, steps, loss_fn, device):
    model.train()
    for step in tqdm(range(steps + 1), desc="Training Biased Classifier"):
        _, x, (targets, _) = next(loader)
        x = x.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()


def biased_prediction(model, loader, device):
    model.eval()
    featurizer = model.featurizer
    clf = model.classifier

    misclassified, targets, groups, features = [], [], [], []
    for _, x, (y, s) in tqdm(loader, desc="Biased Prediction"):
        x = x.to(device)
        with torch.no_grad():
            feature = featurizer(x)
            logits = clf(feature).cpu()

        prob = F.softmax(logits, dim=1)
        prediction = torch.argmax(prob, dim=1)

        misclassified.append((~(y == prediction)).to(int))
        features.append(feature.cpu())
        targets.append(y)
        groups.append(s)

    misclassified = torch.cat(misclassified)
    targets = torch.cat(targets)
    groups = torch.cat(groups)
    features = torch.vstack(features)
    return TensorDataset(features, misclassified, targets, groups)


def split_by_class(misclf_dataset):
    features, misclassified, targets, groups = misclf_dataset.tensors
    splits = dict()
    for y in torch.unique(targets):
        idx = targets == y
        split = TensorDataset(features[idx], misclassified[idx], targets[idx], groups[idx])

        weights = torch.zeros(len(split))
        for label, count in zip(*torch.unique(misclassified[idx], return_counts=True)):
            weights[misclassified[idx] == label] = len(split) / count

        splits[y] = (split, weights, torch.max(weights))

    return splits


def train_left_right(loader, clf, optimizer, steps, device):
    loss_fn = nn.CrossEntropyLoss()
    clf.train()
    for step in tqdm(range(steps + 1), desc="Training Left-Right classifier"):
        x, targets, *_ = next(loader)
        x = x.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        output = clf(x)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()


def estimate_interaction_matrix(n_targets, n_groups, data_splits: dict, train_idx, lr_clf, loader_cfg, device):
    m = torch.zeros((n_targets, n_groups), dtype=torch.int64)

    for key, value in data_splits.items():
        split = value[0]
        if key == train_idx:
            _, misclassified, *_ = split.tensors
            for s in torch.unique(misclassified):
                idx = misclassified == s
                m[train_idx, s] = torch.sum(idx)
        else:
            loader = torch.utils.data.DataLoader(split, **loader_cfg, shuffle=False)
            pred_labels = []
            for x, *_ in tqdm(loader, desc="Evaluating left/right"):
                x = x.to(device)
                with torch.no_grad():
                    output = lr_clf(x).cpu()
                prob = F.softmax(output, dim=1)
                pred_labels.append(torch.argmax(prob, dim=1))
            pred_labels = torch.cat(pred_labels)
            for s in torch.unique(pred_labels):
                idx = pred_labels == s
                m[key, s] = torch.sum(idx)
    return m


cs = ConfigStore.instance()
cs.store(group="job", name="centralized_training", node=Config)


@hydra.main(config_path="conf", config_name="centralized_training", version_base=None)
def main(cfg: Config):
    conf = OmegaConf.to_container(cfg, resolve=True)

    train_ds, eval_ds, test_ds = data_preparation.load_data(conf=conf)
    gt_int_matrix = ground_truth_matrix(
        train_ds,
        cfg.dataset_options.num_targets,
        cfg.dataset_options.num_groups,
        cfg.client_opt.batch_size,
        cfg.client_opt.num_workers,
    )

    device = utils.get_device(conf)
    model = model_utils.init_model(conf).to(device)
    model_utils.print_summary(model)

    # ==============================================
    logging.info("Pre-train with ERM")
    train_loader = iter(
        InfiniteDataLoader(
            dataset=train_ds,
            weights=None,
            batch_size=cfg.client_opt.batch_size,
            num_workers=cfg.client_opt.num_workers,
        )
    )
    opt = get_base_optimizer(model.parameters(), conf)

    training(
        model,
        train_loader,
        opt,
        cfg.client_opt.biased_trainer_steps,
        GCELoss(cfg.client_opt.generalized_cross_entropy_q),
        device,
    )

    # ==============================================
    logging.info("Get biased predictions")
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        shuffle=False,
        batch_size=cfg.client_opt.batch_size,
        num_workers=cfg.client_opt.num_workers,
    )
    error_dataset = biased_prediction(model, train_loader, device)
    splits = split_by_class(error_dataset)

    # ==============================================
    logging.info("Train Left-Right classifier")

    train_idx = min(splits, key=lambda k: splits[k][2])
    lr_split, weights, _ = splits[train_idx]
    lr_loader = iter(
        InfiniteDataLoader(
            lr_split,
            weights=weights,
            batch_size=cfg.client_opt.batch_size,
            num_workers=cfg.client_opt.num_workers,
        )
    )
    lr_clf = nn.Linear(model.classifier.in_features, 2).to(device)
    opt = get_base_optimizer(lr_clf.parameters(), conf)
    train_left_right(lr_loader, lr_clf, opt, cfg.client_opt.left_right_trainer_steps, device)

    # ==============================================
    logging.info("Estimate interaction matrix")
    est_int_matrix = estimate_interaction_matrix(
        cfg.dataset_options.num_targets,
        cfg.dataset_options.num_groups,
        splits,
        train_idx,
        lr_clf,
        {
            "batch_size": cfg.client_opt.batch_size,
            "num_workers": cfg.client_opt.num_workers,
        },
        device,
    )


if __name__ == "__main__":
    main()
