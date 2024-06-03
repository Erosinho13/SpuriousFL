import random
from collections import defaultdict

import torch
import copy
import numpy as np

from src.datasets.cifar import CIFAR10, data_transforms_cifar10, cifar_split_data
from src.datasets.waterbrids import WaterBirds, data_transforms_waterbirds, split_data_waterbirds
from src.datasets.stacked_mnist import StackedMNIST, _data_transforms_mnist, count_img
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
                force_targets=ds_opt['force_targets'],
                force_groups=ds_opt['force_groups']
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
    if conf["dataset"] == 'CIFAR10':
        return cifar_split_data(
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
    elif conf["dataset"] == "StackedMNIST":
        ds_split = split_data_from_torchvision(ds, conf['split_mode'], conf['num_clients'])
    else:
        dataset = conf['dataset']
        raise NotImplementedError('Dataset split for dataset '+dataset+' not recognized')
    print([len(ds) for ds in ds_split])
    return ds_split

def divide_idx_by_tg_pairs(ds, split_mode):
    if split_mode == 'mode1':
        tg_pairs_indices = {(0, 0): [], (1, 0): [], (0, 1): [], (1, 1): []}
        tg_pairs = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for i in range(len(ds)):
            tg_pairs_indices[tg_pairs[i % 4]].append(i)
    else:
        tg_pairs_indices = defaultdict(lambda: [])
        for i, (_, (target, group)) in enumerate(ds):
            tg_pairs_indices[(target, group)].append(i)
        tg_pairs_indices = dict(tg_pairs_indices)
    return tg_pairs_indices


def get_client_indices_mode1(tg_pairs_indices, ds, num_clients):
    for indices in tg_pairs_indices.values():
        random.shuffle(indices)
    clients_indices = [[] for _ in range(num_clients)]
    cl_id = 0
    tg_pairs = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for i in range(len(ds)):
        clients_indices[cl_id].append(tg_pairs_indices[tg_pairs[i % 4]][i // 4])
        if len(clients_indices[cl_id]) == len(ds) // num_clients:
            cl_id += 1
    return clients_indices


def get_client_indices_mode2(tg_pairs_indices, num_clients):
    def divide_indices_in_groups(indices, num_groups):
        random.shuffle(indices)
        group_size = len(indices) // num_groups
        divided_groups = [indices[i * group_size:(i + 1) * group_size] for i in range(num_groups)]
        remaining_indices = indices[num_groups * group_size:]
        for i, index in enumerate(remaining_indices):
            divided_groups[i % num_groups].append(index)
        return divided_groups

    clients_indices = []
    for k, v in tg_pairs_indices.items():
        clients_indices += divide_indices_in_groups(v, num_clients // 4)

    return clients_indices


def split_data_from_torchvision(ds, split_mode, num_clients):
    clients_datasets = []
    if type(ds).__name__ == 'StackedMNIST':
        if split_mode == 'dirichlet':
            raise NotImplementedError
        if split_mode == 'mode1':
            assert ds.num_targets == 2
            assert ds.num_groups == 2
            assert (ds.uniform_targets and ds.uniform_groups) or ds.train is False
            assert num_clients % 4 == 0
            assert len(ds) % num_clients == 0
            tg_pairs_indices = divide_idx_by_tg_pairs(ds, split_mode)
            clients_indices = get_client_indices_mode1(tg_pairs_indices, ds, num_clients)
        elif split_mode == 'mode2':
            assert ds.num_targets == 2
            assert ds.num_groups == 2
            assert (ds.uniform_targets and ds.uniform_groups) or ds.train is False
            assert num_clients % 4 == 0
            tg_pairs_indices = divide_idx_by_tg_pairs(ds, split_mode)
            clients_indices = get_client_indices_mode2(tg_pairs_indices, num_clients)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    for indices in clients_indices:
        clients_datasets.append(SubsetDataset(ds, indices))

    # assert_list = [count_img(cds) for cds in clients_datasets]  # uncomment to check counts by tg pair per client
    return clients_datasets
