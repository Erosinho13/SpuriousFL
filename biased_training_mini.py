import logging

import hydra
from src.optimizers.matrix_inference import ground_truth_matrix
from src.optimizers.matrix_inference import training
from src.optimizers.matrix_inference import biased_prediction
from src.optimizers.matrix_inference import split_by_class
from src.optimizers.matrix_inference import train_left_right
from src.optimizers.matrix_inference import estimate_interaction_matrix
import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from src import utils
from src.config_params import Config
from src.datasets import data_preparation
from src.models import model_utils
from src.optimizers.dataloaders import InfiniteDataLoader
from src.optimizers.subpopbench import get_base_optimizer
from src.optimizers.subpopbench import GeneralizedCrossEntropyLoss as GCELoss



cs = ConfigStore.instance()
cs.store(group="job", name="centralized_training", node=Config)


@hydra.main(config_path="conf", config_name="centralized_training", version_base=None)
def main(cfg: Config):
    conf = OmegaConf.to_container(cfg, resolve=True)

    train_ds, eval_ds, test_ds = data_preparation.load_data(conf=conf)
    gt_int_matrix = ground_truth_matrix(
        train_ds,
        cfg.dataset_options.num_targets,
        cfg.dataset_options.num_groups,
        cfg.client_opt.batch_size,
        cfg.client_opt.num_workers,
    )

    device = utils.get_device(conf)
    model = model_utils.init_model(conf).to(device)
    model_utils.print_summary(model)

    # ==============================================
    logging.info("Pre-train with ERM")
    train_loader = iter(
        InfiniteDataLoader(
            dataset=train_ds,
            weights=None,
            batch_size=cfg.client_opt.batch_size,
            num_workers=cfg.client_opt.num_workers,
        )
    )
    opt = get_base_optimizer(model.parameters(), conf)

    training(
        model,
        train_loader,
        opt,
        cfg.client_opt.biased_trainer_steps,
        GCELoss(cfg.client_opt.generalized_cross_entropy_q),
        device,
    )

    # ==============================================
    logging.info("Get biased predictions")
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        shuffle=False,
        batch_size=cfg.client_opt.batch_size,
        num_workers=cfg.client_opt.num_workers,
    )
    error_dataset = biased_prediction(model, train_loader, device)
    splits = split_by_class(error_dataset)

    # ==============================================
    logging.info("Train Left-Right classifier")

    train_idx = min(splits, key=lambda k: splits[k][2])
    lr_split, weights, _ = splits[train_idx]
    lr_loader = iter(
        InfiniteDataLoader(
            lr_split,
            weights=weights,
            batch_size=cfg.client_opt.batch_size,
            num_workers=cfg.client_opt.num_workers,
        )
    )
    lr_clf = nn.Linear(model.classifier.in_features, 2).to(device)
    opt = get_base_optimizer(lr_clf.parameters(), conf)
    train_left_right(lr_loader, lr_clf, opt, cfg.client_opt.left_right_trainer_steps, device)

    # ==============================================
    logging.info("Estimate interaction matrix")
    est_int_matrix = estimate_interaction_matrix(
        cfg.dataset_options.num_targets,
        cfg.dataset_options.num_groups,
        splits,
        train_idx,
        lr_clf,
        {
            "batch_size": cfg.client_opt.batch_size,
            "num_workers": cfg.client_opt.num_workers,
        },
        device,
    )

    logging.info(est_int_matrix.cpu().numpy())


if __name__ == "__main__":
    main()
