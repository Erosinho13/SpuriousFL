from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F

import itertools


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