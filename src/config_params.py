
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
    biased_trainer_epochs: int
    left_right_trainer_epochs: int
    generalized_cross_entropy_q: float


@dataclass
class ServerOptConfig:
    optimizer: str
    rounds: int
    learning_rate: float
    beta_1: float
    beta_2: float
    tau: float
    weight_clients: str
    participation: str
    client_info: str
    selection_method: str
    weight_list: Optional[List]
    pretrain_rounds: Optional[int]
    num_active_clients: Optional[int]


@dataclass
class DatasetConfig:
    name: str
    num_targets: int
    num_groups: int
    norm: bool
    aug_crop: int
    aug_horizontal_flip: bool
    num_clients: int
    split_mode: str
    input_size: Optional[int]
    data_shuffle_seed: Optional[int]
    dirichlet_alpha: Optional[float]
    local_training_id: Optional[int]
    num_samples_per_class: Optional[int]


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
    client_opt: ClientOptConfig
    server_opt: ServerOptConfig
    wandb: bool
    model_options: ModelConfig
    dataset_options: DatasetConfig
    checkpoint: Optional[str]
    machine: Optional[EnvironmentConfig]
