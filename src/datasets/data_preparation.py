import torchvision
import torch
import copy
import numpy as np

from src.datasets.cifar import CIFAR10, data_transforms_cifar10, cifar_split_data
from src.datasets.waterbrids import WaterBirds, data_transforms_waterbirds, split_data_waterbirds
from src.datasets.subset import SubsetDataset

def load_data(dataset_mode="CIFAR10", val_split=False, val_ratio=0.2, conf={}):
    """Load datasets into dataset object"""
    if "dataset" in conf.keys():
        dataset_mode = conf["dataset"]

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
            val_ids = rand_ids[val_ids:]
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
    raise NotImplementedError(dataset_mode)


def preprocess_data(data, conf, shuffle=True):
    """From dataset to dataloader in PyTorch
    Transforms, augmentations, etc. are now stored in the dataset"""
    ds = torch.utils.data.DataLoader(
        data, batch_size=conf["batch_size"], shuffle=shuffle
    )
    return ds   


def split_data(ds, conf):
    if conf["dataset"]=='CIFAR10':
        ds_split = cifar_split_data(
            ds,
            conf["num_clients"],
            split_mode=conf["split_mode"],
            mode="clients",
            distribution_seed=conf["seed"],
            shuffle_seed=conf["data_shuffle_seed"],
            dirichlet_alpha=conf["dirichlet_alpha"],
        )
        
    elif conf["dataset"]=="WaterBirds":
        ds_split = split_data_waterbirds(
            ds,
            conf
        )
    else:
        dataset = conf['dataset']
        raise NotImplementedError('Dataset split for dataset '+dataset+' not recognized')
    print([len(ds) for ds in ds_split])
    return ds_split

