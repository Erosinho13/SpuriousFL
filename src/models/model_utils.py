import torch
from typing import List
from collections import OrderedDict
import numpy as np
import os

from .cnn import get_diao_CNN

def get_cpu():
    return torch.device("cpu")


def get_device(conf):
    """Check gpu availability in environment config"""
    if len(conf["CUDA_VISIBLE_DEVICES"]) > 0:
        if "client_resources" in conf.keys() and conf["client_resources"] is not None:
            if "num_gpus" in conf["client_resources"].keys():
                if conf["client_resources"]["num_gpus"] > 0:
                    device = torch.device("cuda")
                    return device
    return get_cpu()


def get_loss(conf={}):
    return torch.nn.functional.cross_entropy


def get_optimizer(params, conf={}):
    return torch.optim.Adam(params, lr=0.001)


def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, weights: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=False)


def load_model_weights(model, model_path):
    model.load_state_dict(torch.load(os.path.join(model_path, "torchmodel.pt")))


def save_model(model, model_path):
    try:
        os.makedirs(os.path.join(model_path))
    except FileExistsError:
        pass # Not nice...
    torch.save(model.state_dict(), os.path.join(model_path,"torchmodel.pt"))


def count_params(model, only_trainable=False):
    if only_trainable:
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        pytorch_total_params = sum(p.numel() for p in model.parameters())
    return pytorch_total_params


def print_summary(model):
    print(model)
    print("Total number of parameters:", count_params(model))
    print("Trainable parameters:", count_params(model, only_trainable=True))


class History:
    def __init__(self):
        self.history = {"loss": [], "accuracy": []}

def fit(model, data, conf, validation_data=None, verbose=0):
    model.train() # switch to training mode
    history = History()
    optimizer = get_optimizer(model.parameters(), conf)
    loss_fn = get_loss(conf)
    for epoch in range(conf["epochs"]):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in data:
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
            model.eval() # validation
            with torch.no_grad():
                correct, total, val_loss = 0, 0, 0.0
                for images, labels in validation_data:
                    images, labels = images.to(get_device(conf)), labels.to(get_device(conf))
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    total += labels.size(0)
                    correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                    del loss, outputs
                val_loss /= len(validation_data.dataset)
                val_acc = correct / total
            model.train() # switch to training mode
        if verbose > 0:
            v_string = ''
            if validation_data is not None:
                v_string = f" val_loss:{val_loss}, val_acc:{val_acc}"
            print(f"Epoch {epoch+1}: loss:{epoch_loss}, acc:{epoch_acc}"+v_string)
        history.history["loss"].append(epoch_loss)
        history.history["accuracy"].append(epoch_acc)
    return history


def evaluate(model, data, conf, verbose=0):
    model.eval()
    loss_fn = get_loss(conf)
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in data:
            images, labels = images.to(get_device(conf)), labels.to(get_device(conf))
            outputs = model(images)
            loss += loss_fn(outputs, labels, reduction="sum").item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(data.dataset)
    accuracy = correct / total
    return loss, accuracy


def np_to_tensor(images):
    return torch.from_numpy(np.transpose(images,(0,3,1,2)))


def predict_numpy(model, X, verbose=0):
    model.eval()
    model.to(get_cpu())
    with torch.no_grad():
        images = np_to_tensor(X)
        #images = batch.to(get_cpu())
        outputs = model(images)
        return outputs


def predict_dataloader(model, dataloader, conf, apply_softmax=True):
    total_outputs = []
    labels_all = None
    model.eval()     # Optional when not using Model Specific layer
    model.to(get_device(conf))
    for images, labels in dataloader:
        l = labels.detach().numpy()
        if labels_all is None:
            labels_all = l
        else:
            labels_all = np.concatenate((labels_all,l))
        images, labels = images.to(get_device(conf)), labels.to(get_device(conf))      
        outputs = model(images)
        if apply_softmax:
            outputs = torch.nn.functional.softmax(torch.Tensor(outputs), -1)
        outputs = outputs.to('cpu').detach().numpy()
        total_outputs.extend(outputs)
    return np.array(total_outputs), np.array(labels_all)


def init_model(conf, model_path=None, weights=None, *args, **kwargs):
    if conf["dataset"]=="CIFAR10":
        input_shape = (3,32,32)
        num_classes=10
    kwargs["input_shape"] = input_shape
    kwargs["num_classes"] = num_classes
    model = get_diao_CNN(*args,**kwargs)
    if model_path is not None:
        load_model_weights(model, model_path)
    if weights is not None:
        set_weights(model, weights)
    model.to(get_device(conf))
    return model
