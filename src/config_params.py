
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ClientOptConfig:
    subpop_optimizer: str
    base_optimizer: str
    learning_rate: float
    groupdro_eta: float
    cbloss_beta: float
    focal_gamma: int
    dfr_reg: float
    fex_shallow_model: str
    fex_epochs: int
    fex_balance_classes: bool
    momentum: float
    batch_size: int
    epochs: int


@dataclass
class ServerOptConfig:
    optimizer: str
    learning_rate: float
    beta_1: float
    beta_2: float
    tau: float
    weight_clients: str
    weight_list: Optional[List]


@dataclass
class DatasetConfig:
    name: str
    num_targets: int
    num_groups: int
    input_size: Optional[int]
    data_shuffle_seed: Optional[int]
    dirichlet_alpha: Optional[float]


@dataclass
class ModelConfig:
    model_type: str
    norm_layer: str
    pretrained: Optional[bool]

@dataclass
class EnvironmentConfig:
    root_path: str
    ray_init_args: dict
    client_resources : dict
    CUDA_VISIBLE_DEVICES : str

@dataclass
class Config:
    seed: int
    rounds: int
    num_clients: int
    norm: bool
    aug_crop: int
    aug_horizontal_flip: bool
    split_mode: str
    client_opt: ClientOptConfig
    server_opt: ServerOptConfig
    wandb: bool
    model_options: ModelConfig
    dataset_options: DatasetConfig
    checkpoint: Optional[str]
    local_training_id: Optional[int]
    machine: Optional[EnvironmentConfig]
