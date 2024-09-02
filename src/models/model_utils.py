from src.utils import get_device
import torch
from typing import List
from collections import OrderedDict
import numpy as np
import os

from torchvision.models import mobilenet_v2

from .cnn import get_diao_CNN
from .mobilenet import mobilenetv2
from .resnet import get_resnet50


def layer_name(weight_name):
    return '.'.join(weight_name.split('.')[-1])


def get_bn_layers(model):
    return [layer_name(k) for k in model.state_dict().keys() if 'num_batches_tracked' in k]


def get_weights(model):
    # Return model parameters as a list of NumPy ndarrays, excluding parameters of BN layers when using FedBN
    bn_layers = get_bn_layers(model)
    return [val.cpu().numpy() for name, val in model.state_dict().items() if layer_name(name) not in bn_layers]


def set_weights(model, weights: List[np.ndarray]):
    # https://flower.ai/docs/framework/example-fedbn-pytorch-from-centralized-to-federated.html
    bn_layers = get_bn_layers(model)
    keys = [k for k in model.state_dict().keys() if layer_name(k) not in bn_layers]
    params_dict = zip(keys, weights)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=False)


def load_model_weights(model, model_path):
    model.load_state_dict(torch.load(os.path.join(model_path, "torchmodel.pt")))


def save_model(model, model_path):
    try:
        os.makedirs(os.path.join(model_path))
    except FileExistsError:
        pass  # Not nice...
    torch.save(model.state_dict(), os.path.join(model_path, "torchmodel.pt"))


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


def init_model(conf, model_path=None, weights=None, *args, **kwargs):
    if "dataset_options" in conf.keys():
        dataset_mode = conf["dataset_options"]["name"]
    else:
        raise KeyError("Missing dataset options")    
    if dataset_mode == "CIFAR10":
        input_shape = (3, 32, 32)
        num_classes = 10
    elif dataset_mode == "WaterBirds":
        if "dataset_options" in conf.keys() and "input_size" in conf["dataset_options"].keys():
            input_shape = (3, conf["dataset_options"]["input_size"], conf["dataset_options"]["input_size"])
        else:
            input_shape = (3, 32, 32)
        num_classes = 2
    elif dataset_mode == "StackedMNIST":
        input_shape = (3, 32, 32)
        num_classes = conf["dataset_options"]["num_targets"]
    else:
        raise NotImplementedError('Dataset split for dataset ' + dataset_mode + ' not recognized')

    kwargs["input_shape"] = input_shape
    kwargs["num_classes"] = num_classes
    if conf["model_options"]["model_type"] == "CNN":
        model = get_diao_CNN(*args, **kwargs)
    elif conf["model_options"]["model_type"] == "ResNet":
        if 'norm_layer' in conf.keys():
            norm=conf['norm_layer']
        else:
            norm = 'gn'
        model = get_resnet50(num_classes, norm=norm)
    elif conf["model_options"]["model_type"] == "MobileNet":
        pretrained = True
        if 'pretrained' in conf["model_options"].keys():
            pretrained = conf["model_options"]['pretrained']
        if pretrained:
            model = mobilenetv2(num_classes=num_classes, return_features=False)
        else:
            model = mobilenetv2(num_classes=num_classes, return_features=False, pretrained_path=None)
    else:
        raise NotImplementedError("conf['model_type']")
    if model_path is not None:
        load_model_weights(model, model_path)
    if weights is not None:
        set_weights(model, weights)
    model.to(get_device(conf))
    return model
