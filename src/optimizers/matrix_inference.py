from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F

import itertools


def ground_truth_matrix(dataset, n_targets, n_groups, batch_size, num_workers=0, verbose=1):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    y_array, s_array = [], []
    iterator = loader
    if verbose>0:
        iterator = tqdm(iterator, desc="Ground Truth Matrix")
    for _, _, (y, s) in iterator:
        y_array.append(y)
        s_array.append(s)

    y_array = torch.cat(y_array, dim=0)
    s_array = torch.cat(s_array, dim=0)

    m = torch.zeros((n_targets, n_groups), dtype=torch.int64)
    for y, s in itertools.product(range(n_targets), range(n_groups)):
        m[y, s] = torch.sum((y_array == y) & (s_array == s))

    y_weights, s_weights = torch.zeros(len(y_array)), torch.zeros(len(s_array))
    for label, count in zip(*torch.unique(y_array, return_counts=True)):
            y_weights[y_array == label] = len(y_array) / count
    for label, count in zip(*torch.unique(s_array, return_counts=True)):
            s_weights[s_array == label] = len(s_array) / count
    y_weights = y_weights / torch.max(y_weights)
    s_weights = s_weights / torch.max(s_weights)
    return m, y_weights, s_weights, y_array, s_array


def training(model, loader, optimizer, steps, loss_fn, device, verbose=1):
    model.train()
    iterator = range(steps + 1)
    if verbose>0:
        iterator = tqdm(iterator, desc="Training Biased Classifier")
    for step in iterator:
        _, x, (targets, _) = next(loader)
        x = x.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()


def biased_prediction(model, loader, device, verbose=1):
    model.eval()
    featurizer = model.featurizer
    clf = model.classifier

    misclassified, targets, groups, features = [], [], [], []
    iterator = loader
    if verbose>0:
        iterator = tqdm(loader, desc="Biased Prediction")
    for _, x, (y, s) in iterator:
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


def biased_binary_prediction(model, loader, device, one_class_id, verbose=1):
    """For One-vs-Rest biased model.
    Expects binary model, but original multiclass loader.
    Follows the `biased_prediction` code
    but returns only with the 'one' class's missclassificaion
    so it can be chained together with the other binary models 
    to build the same misclassified dataset.
    params:
        - one_class_id : the id of the 'one' class to keep"""
    model.eval()
    featurizer = model.featurizer
    clf = model.classifier

    misclassified, targets, groups, features = [], [], [], []
    iterator = loader
    if verbose>0:
        iterator = tqdm(loader, desc="Biased Prediction")
    for _, x, (y, s) in iterator:
        x = x.to(device)
        with torch.no_grad():
            feature = featurizer(x)
            logits = clf(feature).cpu()

        prob = F.softmax(logits, dim=1)
        prediction = torch.argmax(prob, dim=1)
        if len(y==one_class_id)>0:
            misclassified.append((~(1 == prediction)).to(int))
            features.append(feature.cpu()[y==one_class_id])
            targets.append(y[y==one_class_id])
            groups.append(s[y==one_class_id])

    misclassified = torch.cat(misclassified)
    targets = torch.cat(targets)
    groups = torch.cat(groups)
    features = torch.vstack(features)
    return features, misclassified, targets, groups

def concat_error_predictions(prediction_list):
    """Gets the output of the single class binary predictions and put them together to get one TensorDataset"""
    misclassified, targets, groups, features = [], [], [], []
    for predictions in prediction_list:
        m,t,g,f = predictions
        misclassified.append(m)
        targets.append(t)
        groups.append(g)
        features.append(f)
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
        weights = weights / torch.max(weights)
        splits[y] = (split, weights, torch.max(weights))

    return splits


def train_left_right(loader, clf, optimizer, steps, device, verbose=1):
    loss_fn = nn.CrossEntropyLoss()
    clf.train()
    iterator = range(steps + 1)
    if verbose>0:
        iterator = tqdm(iterator, desc="Training Left-Right classifier")
    for step in iterator:
        x, targets, *_ = next(loader)
        x = x.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        output = clf(x)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()


def estimate_interaction_matrix(n_targets, n_groups, data_splits: dict, train_idx, lr_clf, loader_cfg, device, verbose=1):
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
            iterator = loader
            if verbose>0:
                iterator = tqdm(iterator, desc="Evaluating left/right")
            for x, *_ in iterator:
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