from typing import Callable
import numpy as np
from src.optimizers.subpopbench import get_base_optimizer, get_loss, get_subpop_optimizer
from src.utils import get_cpu, get_device, np_to_tensor
import torch

class History:
    def __init__(self):
        self.history = {"loss": [], "accuracy": []}

    def __str__(self):
        return str(self.history)


def fit(model, data, conf, validation_data=None, verbose=0, opt=None):
    model.train()  # switch to training mode
    history = History()

    if opt is None:
        opt = get_subpop_optimizer(model, data.dataset, conf)

    num_steps = count_steps = -1
    if 'num_steps' in conf["client_opt"]:
        if conf["client_opt"]['num_steps'] != -1:
            num_steps = conf["client_opt"]['num_steps']
            count_steps = 0
            conf["client_opt"]["epochs"] = 1000000

    for epoch in range(conf["client_opt"]["epochs"]):

        correct, total, epoch_loss = 0, 0, 0.0

        for i, (indeces, images, (labels, groups)) in enumerate(data):

            indeces = indeces.to(get_device(conf))
            images, labels = images.to(get_device(conf)), labels.to(get_device(conf))
            groups = groups.to(get_device(conf))

            opt_out = opt.update((indeces, images, labels, groups), 1)
            loss = opt_out["loss"]
            minibatch_correct = opt_out["correct"]

            # Metrics
            correct += minibatch_correct
            epoch_loss += loss * images.size(0)
            total += labels.size(0)
            del loss

            if num_steps != -1:
                count_steps += 1
                if count_steps >= num_steps:
                    break

        epoch_loss /= len(data.dataset)
        epoch_acc = correct / total

        if validation_data is not None:
            model.eval()  # validation
            with torch.no_grad():
                correct, total, val_loss = 0, 0, 0.0
                for indeces, images, (labels, groups) in validation_data:
                    indeces = indeces.to(get_device(conf))
                    groups = groups.to(get_device(conf))
                    images, labels = images.to(get_device(conf)), labels.to(get_device(conf))
                    opt_out = opt.update((indeces, images, labels, groups), 1)
                    loss = opt_out["loss"]
                    minibatch_correct = opt_out["correct"]
                    correct += minibatch_correct
                    val_loss += loss * images.size(0)
                    total += labels.size(0)
                    del loss
                val_loss /= len(validation_data.dataset)
                val_acc = correct / total
            model.train()  # switch to training mode

        if verbose > 0:
            v_string = ''
            if validation_data is not None:
                v_string = f" val_loss:{val_loss}, val_acc:{val_acc}"
            print(f"Epoch {epoch + 1}: loss:{epoch_loss:.4f}, acc:{epoch_acc:.4f}" + v_string)

        history.history["loss"].append(epoch_loss)
        history.history["accuracy"].append(epoch_acc)

        if num_steps != -1:
            if count_steps >= num_steps:
                break

    return history

#!TODO update with subpopbench optims
def evaluate(model, data, conf, verbose=0,
             extra_eval_fn: Callable[[
                 torch.Tensor, # indeces
                 torch.Tensor, # images
                 torch.Tensor, # labels
                 torch.Tensor, # groups
                 torch.Tensor, # predicted
                 dict], None]=None): # outdict
    if extra_eval_fn:
        extra_ret = {}
    model.eval()
    loss_fn = get_loss(conf=conf)
    correct, total, loss = 0, 0, 0.0
    label_group_correct, label_group_total = {}, {}
    with torch.no_grad():
        for indeces, images, (labels, groups) in data:
            indeces, images, labels, groups = indeces.to(get_device(conf)), images.to(get_device(conf)), labels.to(get_device(conf)), groups.to(
                get_device(conf))
            outputs = model(images)
            loss += loss_fn(outputs, labels).mean().item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if extra_eval_fn:
                extra_eval_fn(indeces, images, labels, groups, predicted, extra_ret)

            unique_pairs = torch.unique(torch.stack((labels, groups), dim=1), dim=0)
            for label, group in unique_pairs:
                mask = (labels == label) & (groups == group)
                pair_labels = labels[mask]
                pair_predictions = predicted[mask]
                correct_count = (pair_predictions == pair_labels).sum().item()
                total_count = mask.sum().item()

                pair_key = (label.item(), group.item())
                if pair_key not in label_group_total:
                    label_group_total[pair_key] = 0
                    label_group_correct[pair_key] = 0

                label_group_total[pair_key] += total_count
                label_group_correct[pair_key] += correct_count
    loss /= len(data.dataset)
    accuracy = 100 * correct / total
    group_accuracies = {("y" + str(pair[0]) + "g" + str(pair[1])): 100* label_group_correct[pair] / label_group_total[pair]
                        for pair in label_group_total}
    worst_acc = min(group_accuracies.values())
    group_accuracies["worst_group"] = worst_acc
    if extra_eval_fn:
        return loss, accuracy, group_accuracies, extra_ret
    return loss, accuracy, group_accuracies


def predict_numpy(model, X, verbose=0):
    model.eval()
    model.to(get_cpu())
    with torch.no_grad():
        images = np_to_tensor(X)
        # images = batch.to(get_cpu())
        outputs = model(images)
        return outputs


def predict_dataloader(model, dataloader, conf, apply_softmax=True):
    total_outputs = []
    labels_all = None
    model.eval()  # Optional when not using Model Specific layer
    model.to(get_device(conf))
    for images, labels in dataloader:
        l = labels.detach().numpy()
        if labels_all is None:
            labels_all = l
        else:
            labels_all = np.concatenate((labels_all, l))
        images, labels = images.to(get_device(conf)), labels.to(get_device(conf))
        outputs = model(images)
        if apply_softmax:
            outputs = torch.nn.functional.softmax(torch.Tensor(outputs), -1)
        outputs = outputs.to('cpu').detach().numpy()
        total_outputs.extend(outputs)
    return np.array(total_outputs), np.array(labels_all)