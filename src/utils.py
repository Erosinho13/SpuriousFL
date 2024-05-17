import json
import os
import logging


def load_config(env_path="env.json", config_path="config.json"):
    """Connect config and environment config files"""
    with open(config_path, "r") as f:
        config = json.load(f)
    if env_path is None:
        return config
    with open(env_path, "r") as f:
        env = json.load(f)
    if "paths" in config.keys():
        for k, v in config["paths"].items():
            if "root_path" in config.keys():
                if v[: len(config["root_path"])] == config["root_path"]:
                    v = v[len(config["root_path"]) :]
            config["paths"][k] = str(os.path.join(env["root_path"], v))
    config.update(env)
    return config


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
    DEFAULT_LOGGER.log(*args,**kwargs)