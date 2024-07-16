import torch
import copy
import numpy as np

from src.datasets.cifar import CIFAR10, data_transforms_cifar10, cifar_split_data
from src.datasets.waterbrids import WaterBirds, data_transforms_waterbirds, split_data_waterbirds
from src.datasets.stacked_mnist import StackedMNIST, _data_transforms_mnist, split_stackedmnist_data
from src.datasets.subset import SubsetDataset
from src.optimizers.dataloaders import WeightedDataLoader


def load_data(dataset_mode="CIFAR10", val_split=False, val_ratio=0.2, conf={}):
    """Load datasets into dataset object"""
    if "dataset_options" in conf.keys():
        dataset_mode = conf["dataset_options"]["name"]
    else:
        raise KeyError("Missing dataset options")

    if "val_split" in conf.keys():
        val_split = conf["val_split"]
    if "seed" not in conf.keys():
        conf["seed"] = None

    if dataset_mode == "CIFAR10":
        train_tr_list, test_tr_list = data_transforms_cifar10(conf)
        trainset = CIFAR10(
            "./datasets", train=True, transforms=train_tr_list
        )
        testset = CIFAR10(
            "./datasets", train=False, transforms=test_tr_list
        )
        if val_split:
            rand_ids = np.random.default_rng(seed=conf["seed"]).permutation(len(trainset))
            len_val = int(len(trainset) * val_ratio)
            len_train = len(trainset) - len_val
            train_ids = rand_ids[:len_train]
            val_ids = rand_ids[len_train:]
            trainset_reduced = SubsetDataset(trainset, train_ids)
            valset = SubsetDataset(trainset, val_ids)
            return trainset_reduced, valset, testset
        else:
            valset = copy.deepcopy(testset)
        return trainset, valset, testset
    
    if dataset_mode == "WaterBirds":
        train_tr_list, test_tr_list = data_transforms_waterbirds(conf)
        trainset = WaterBirds(
            "./datasets", train=True, transforms=train_tr_list
        )
        testset = WaterBirds(
            "./datasets", train=False, transforms=test_tr_list
        )
        valset = copy.deepcopy(testset)
        return trainset, valset, testset
    
    if dataset_mode == "StackedMNIST":
        ds_opt = conf['dataset_options']
        mean = (0.1307, 0.1307, 0.1307)
        std = (0.3081, 0.3081, 0.3081)
        train_transform, test_transform = _data_transforms_mnist(mean=mean, std=std, norm=True)
        trainset = \
            StackedMNIST(
                root=ds_opt['root'],
                train=True,
                transform=train_transform,
                download=True,
                num_images=ds_opt['num_train_images'],
                num_targets=ds_opt['num_targets'],
                num_groups=ds_opt['num_groups'],
                prevent_targets_shuffling=ds_opt['prevent_class_shuffling'],
                prevent_groups_shuffling=ds_opt['prevent_group_shuffling'],
                dirichlet_targets_alpha=ds_opt['dirichlet_targets_alpha'],
                dirichlet_groups_alpha=ds_opt['dirichlet_groups_alpha'],
                uniform_targets=ds_opt['uniform_targets'],
                uniform_groups=ds_opt['uniform_groups'],
                force_targets=ds_opt['force_targets'],
                force_groups=ds_opt['force_groups'],
                force_targets_proportions=ds_opt['force_targets_proportions'],
                force_groups_proportions=ds_opt['force_groups_proportions'],
                force_proportions=ds_opt['force_proportions']
            )
        testset = \
            StackedMNIST(
                root=ds_opt['root'],
                train=False,
                transform=test_transform,
                download=True,
                num_images=ds_opt['num_test_images'],
                num_targets=ds_opt['num_targets'],
                num_groups=ds_opt['num_groups'],
                prevent_targets_shuffling=ds_opt['prevent_class_shuffling'],
                prevent_groups_shuffling=ds_opt['prevent_group_shuffling'],
                force_targets=trainset.targets,
                force_groups=trainset.groups
            )

        valset = copy.deepcopy(trainset)
        valset.transform = test_transform

        return trainset, valset, testset
    raise NotImplementedError(dataset_mode)


def preprocess_data(data, conf, shuffle=True):
    """From dataset to dataloader in PyTorch
    Transforms, augmentations, etc. are now stored in the dataset"""
    ds = WeightedDataLoader(
        data, batch_size=conf["batch_size"], shuffle=shuffle
    )
    return ds


def split_data(ds, conf):
    if "dataset_options" in conf.keys():
        dataset_mode = conf["dataset_options"]["name"]
    else:
        raise KeyError("Missing dataset options")
    if dataset_mode == 'CIFAR10':
        return cifar_split_data(
            ds,
            conf["num_clients"],
            split_mode=conf["split_mode"],
            mode="clients",
            distribution_seed=conf["seed"],
            shuffle_seed=conf["data_shuffle_seed"],
            dirichlet_alpha=conf["dirichlet_alpha"],
        )
        
    elif dataset_mode == "WaterBirds":
        ds_split = split_data_waterbirds(
            ds,
            conf
        )
    elif dataset_mode == "StackedMNIST":
        ds_split = \
            split_stackedmnist_data(
                ds, conf['split_mode'], conf['num_clients'],
                uniform_proportion=conf['uniform_proportion'] if 'uniform_proportion' in conf.keys() else 0
            )
    else:
        raise NotImplementedError('Dataset split for dataset '+dataset_mode+' not recognized')
    print([len(ds) for ds in ds_split])
    return ds_split
