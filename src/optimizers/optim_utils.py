import numpy as np
from src.utils import get_cpu, get_device, np_to_tensor
import torch

def get_optimizer(params, conf={}):
    if "client_opt" in conf.keys():
        copt = conf["client_opt"]
        if "optimizer" in copt:
            if copt["optimizer"] == "SGD":
                opt = torch.optim.SGD
            elif copt["optimizer"] == "Adam":
                opt = torch.optim.Adam
            else:
                raise NotImplementedError(f"optimizer not recognized {copt['optimizer']}")
        if "learning_rate" in copt:
            lr = float(copt["learning_rate"])
        else:
            lr = 0.001
        return opt(params, lr=lr)
    return torch.optim.SGD(params, lr=0.001)


def get_loss(conf={}):
    return torch.nn.functional.cross_entropy


class History:
    def __init__(self):
        self.history = {"loss": [], "accuracy": []}

    def __str__(self):
        return str(self.history)


def fit(model, data, conf, validation_data=None, verbose=0):
    model.train()  # switch to training mode
    history = History()
    optimizer = get_optimizer(model.parameters(), conf)
    loss_fn = get_loss(conf)
    for epoch in range(conf["epochs"]):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, (labels, groups) in data:
            images, labels = images.to(get_device(conf)), labels.to(get_device(conf))
            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            del loss, outputs
        epoch_loss /= len(data.dataset)
        epoch_acc = correct / total

        if validation_data is not None:
            model.eval()  # validation
            with torch.no_grad():
                correct, total, val_loss = 0, 0, 0.0
                for images, (labels, groups) in validation_data:
                    images, labels = images.to(get_device(conf)), labels.to(get_device(conf))
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    total += labels.size(0)
                    correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                    del loss, outputs
                val_loss /= len(validation_data.dataset)
                val_acc = correct / total
            model.train()  # switch to training mode
        if verbose > 0:
            v_string = ''
            if validation_data is not None:
                v_string = f" val_loss:{val_loss}, val_acc:{val_acc}"
            print(f"Epoch {epoch + 1}: loss:{epoch_loss}, acc:{epoch_acc}" + v_string)
        history.history["loss"].append(epoch_loss)
        history.history["accuracy"].append(epoch_acc)
    return history


def evaluate(model, data, conf, verbose=0):
    model.eval()
    loss_fn = get_loss(conf)
    correct, total, loss = 0, 0, 0.0
    label_group_correct, label_group_total = {}, {}
    with torch.no_grad():
        for images, (labels, groups) in data:
            images, labels, groups = images.to(get_device(conf)), labels.to(get_device(conf)), groups.to(
                get_device(conf))
            outputs = model(images)
            loss += loss_fn(outputs, labels, reduction="sum").item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

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