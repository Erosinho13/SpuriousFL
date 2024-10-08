import yaml
import os
import logging
import numpy as np
import torch
import random


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
