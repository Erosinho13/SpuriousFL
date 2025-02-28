import yaml
import os
import logging
import numpy as np
import torch
import random
import re

import hashlib
import json
import copy

def load_config(env_path="env.json", config_path="config.json"):
    """Connect config and environment config files"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if env_path is None:
        return config
    with open(env_path, "r") as f:
        env = yaml.safe_load(f)
    if "paths" in config.keys():
        for k, v in config["paths"].items():
            if "root_path" in config.keys():
                if v[: len(config["root_path"])] == config["root_path"]:
                    v = v[len(config["root_path"]):]
            config["paths"][k] = str(os.path.join(env["root_path"], v))
    config.update(env)
    return config


def load_env_config(env_path, config):
    with open(env_path, "r") as f:
        env = yaml.safe_load(f)
    if "paths" in config.keys():
        for k, v in config["paths"].items():
            if "root_path" in config.keys():
                if v[: len(config["root_path"])] == config["root_path"]:
                    v = v[len(config["root_path"]):]
            config["paths"][k] = str(os.path.join(env["root_path"], v))
    config.update(env)
    return config


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


def save_config(conf, path):
    """Save config dict to file"""
    with open(path, 'w') as outfile:
        yaml.dump(conf, outfile, default_flow_style=False)


def get_logger():
    logger = logging.getLogger("spurious-fl")
    logger.setLevel(logging.DEBUG)
    DEFAULT_FORMATTER = logging.Formatter(
        "%(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(DEFAULT_FORMATTER)
    logger.addHandler(console_handler)
    return logger


DEFAULT_LOGGER = get_logger()


def log(*args, **kwargs):
    DEFAULT_LOGGER.log(*args, **kwargs)


def dirichlet_split(
        num_classes, num_clients, dirichlet_alpha=1.0, mode="clients", seed=None
):
    """Dirichlet distribution of the data points,
    with mode 'classes', 1.0 is distributed between num_classes class,
    with 'clients' it is distributed between num_clients clients"""
    if mode == "classes":
        a = num_classes
        b = num_clients
    elif mode == "clients":
        a = num_clients
        b = num_classes
    else:
        raise ValueError(f"unrecognized mode {mode}")
    if np.isscalar(dirichlet_alpha):
        dirichlet_alpha = np.repeat(dirichlet_alpha, a)
    split_norm = np.random.default_rng(seed).dirichlet(dirichlet_alpha, b)
    return split_norm


def get_cpu():
    return torch.device("cpu")


def get_device(conf):
    """Check gpu availability in environment config"""
    if "machine" in conf.keys() and len(conf["machine"]["CUDA_VISIBLE_DEVICES"]) > 0:
        if "client_resources" in conf["machine"].keys() and conf["machine"]["client_resources"] is not None:
            if "num_gpus" in conf["machine"]["client_resources"].keys():
                if conf["machine"]["client_resources"]["num_gpus"] > 0:
                    device = torch.device("cuda")
                    return device
    return get_cpu()


def np_to_tensor(images):
    return torch.from_numpy(np.transpose(images, (0, 3, 1, 2)))


def collect_values_to_2d_array(d):
    """get the 'y<int>g<int>' values from dict and put it into a 2d array"""
    # Create a regular expression to match keys of the form 'y<int>g<int>'
    pattern = re.compile(r'y(\d+)g(\d+)')
    
    # To store the values in a 2D list, we first need to find the max values of y and g to know the size of the array
    max_y = max_g = 0
    coords = []
    
    for key in d.keys():
        match = pattern.match(key)
        if match:
            y_val, g_val = int(match.group(1)), int(match.group(2))
            coords.append((y_val, g_val))
            max_y = max(max_y, y_val)
            max_g = max(max_g, g_val)
    
    # Initialize the 2D list with None or 0 based on your preference
    result = [[None for _ in range(max_g + 1)] for _ in range(max_y + 1)]
    
    # Populate the result array
    for y_val, g_val in coords:
        result[y_val][g_val] = d[f'y{y_val}g{g_val}']
    
    return result


def hash_config(conf: dict, dropkeys=["seed", "exp_id"]) -> str:
    """Generate a unique hash for a given nested config dictionary."""
    config = copy.deepcopy(conf)
    for k in dropkeys:
        if k in config.keys():
            del config[k]
    config_str = json.dumps(config, sort_keys=True)  # Ensure order consistency
    return hashlib.sha256(config_str.encode()).hexdigest()


def get_input_shape(conf, dataset_mode=None):
    """Get data shape from config"""
 
    if dataset_mode is None:
        if "dataset_options" in conf.keys():
            dataset_mode = conf["dataset_options"]["name"]
        else:
            raise KeyError("Missing dataset options")    
        dataset_mode = conf["dataset_options"]["name"]

    if dataset_mode == "CIFAR10":
        input_shape = (3, 32, 32)
    elif dataset_mode == "WaterBirds":
        if "dataset_options" in conf.keys() and "input_size" in conf["dataset_options"].keys():
            input_shape = (3, conf["dataset_options"]["input_size"], conf["dataset_options"]["input_size"])
        else:
            input_shape = (3, 32, 32)
    elif dataset_mode == "StackedMNIST":
        input_shape = (3, 32, 32)
    elif dataset_mode == "Spawrious" or dataset_mode=="FMOW":
        if "dataset_options" in conf.keys() and "input_size" in conf["dataset_options"].keys():
            input_shape = (3, conf["dataset_options"]["input_size"], conf["dataset_options"]["input_size"])
        else:
            input_shape = (3, 224, 224)
    elif dataset_mode == "CMNIST":
            input_shape = (3, 28, 28)
    else:
        raise NotImplementedError('Dataset split for dataset ' + dataset_mode + ' not recognized')
    return input_shape

def adjust_array(v: np.array, q: int, population: np.array) -> np.array:

    """
    adjust array such that the sum is exactly q and no value is higher than the corresponding value in population.

    :param v: array to be adjusted.
    :param q: desired sum.
    :param population: bound for the values of v.

    :return: adjusted array.
    """

    v_int = np.round(v).astype(int)  # Round v to the nearest integer
    for i in range(len(v_int)):
        if v_int[i] > population[i]:
            v_int[i] = population[i]
    diff = q - np.sum(v_int)  # Compute the difference from the desired sum

    while diff > 0:
        new_id = np.random.choice(range(len(v_int)), size=1)[0]
        if v_int[new_id] + 1 > population[new_id]:
            continue
        v_int[new_id] += 1
        diff = q - np.sum(v_int)


    while diff < 0:
        new_id = np.random.choice(range(len(v_int)), size=1)[0]
        v_int[new_id] -= 1
        diff = q - np.sum(v_int)

    return v_int