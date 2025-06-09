from src.datasets.celeba import CelebA, data_transforms_celeba, split_data_celeba
from src.datasets.fairface import FairFace, data_transforms_fairface
from src.datasets.fmow import FMOW, data_transforms_fmow
from src.datasets.spawrious import Spawrious, data_transforms_spawrious, split_data_spawrious
from src.datasets.utkface import UTKFace, data_transforms_utkface
import torch
import copy
import numpy as np

from src.datasets.cifar import CIFAR10, data_transforms_cifar10, cifar_split_data
from src.datasets.waterbrids import WaterBirds, data_transforms_waterbirds, split_data_waterbirds
from src.datasets.stacked_mnist import StackedMNIST, _data_transforms_mnist, split_stackedmnist_data
from src.datasets.dataset_utils import SubsetDataset, compute_forgetting, count_groups, select_forgettables
from src.optimizers.dataloaders import WeightedDataLoader
from src.datasets.cmnist import CMNIST, data_transforms_cmnist


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

    if dataset_mode == "Spawrious":
        ds_opt = conf["dataset_options"]
        train_tr_list, test_tr_list = data_transforms_spawrious(conf)
        if "locations" in conf["dataset_options"].keys() and isinstance(conf["dataset_options"]["locations"], list):
            locations = conf["dataset_options"]["locations"]
        else:
            locations = None
        if "breeds" in conf["dataset_options"].keys() and isinstance(conf["dataset_options"]["breeds"], list):
            breeds = conf["dataset_options"]["breeds"]
        else:
            breeds = None
        trainset = Spawrious(
            "./datasets", train=True, transforms=train_tr_list,
            num_groups=ds_opt['num_groups'],
            num_targets=ds_opt['num_targets'],
            num_samples_per_class=ds_opt['num_samples_per_class'],
            locations=locations,
            breeds=breeds
        )
        testset = Spawrious(
            "./datasets", train=False, transforms=test_tr_list,
            num_groups=ds_opt['num_groups'],
            num_targets=ds_opt['num_targets'],
            locations=locations,
            breeds=breeds
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

    if dataset_mode == "CMNIST":
        cmnist_transform = data_transforms_cmnist(conf)
        ds_opt = conf["dataset_options"]
        trainset = CMNIST(
            root="./datasets",
            train=True,
            confounding_factor=ds_opt["confounding_factor"],
            transforms=cmnist_transform,
        )
        testset = CMNIST(
            root="./datasets",
            train=False,
            confounding_factor=0.5,
            transforms=cmnist_transform,
        )
        valset = copy.deepcopy(testset)
        return trainset, valset, testset

    if dataset_mode=="FMOW":
        ds_opt = conf["dataset_options"]
        train_tr_list, test_tr_list = data_transforms_fmow(conf)
        if "categories" in conf["dataset_options"].keys() and isinstance(conf["dataset_options"]["categories"], list):
            categories = conf["dataset_options"]["categories"]
        else:
            categories = None
        if "regions" in conf["dataset_options"].keys() and isinstance(conf["dataset_options"]["regions"], list):
            regions = conf["dataset_options"]["regions"]
        else:
            regions = None
        trainset = FMOW(
            "./datasets", train=True, transforms=train_tr_list,
            num_groups=ds_opt['num_groups'],
            num_targets=ds_opt['num_targets'],
            regions=regions,
            categories=categories
        )
        testset = FMOW(
            "./datasets", train=False, transforms=test_tr_list,
            num_groups=ds_opt['num_groups'],
            num_targets=ds_opt['num_targets'],
            regions=regions,
            categories=categories
        )
        valset = copy.deepcopy(testset)
        return trainset, valset, testset
    if dataset_mode=="CelebA":
        ds_opt = conf["dataset_options"]
        train_tr_list, test_tr_list = data_transforms_celeba(conf)
        group = ds_opt["group_name"]
        target = ds_opt["target_name"]
        trainset = CelebA(
            "./datasets", train=True, transforms=train_tr_list,
            target_attr_name=target,
            group_attr_name=group
        )
        testset = CelebA(
            "./datasets", train=False, transforms=test_tr_list,
            target_attr_name=target,
            group_attr_name=group
        )
        valset = copy.deepcopy(testset)
        return trainset, valset, testset
    if dataset_mode=="UTKFace":
        ds_opt = conf["dataset_options"]
        train_transforms, test_transforms = data_transforms_utkface(conf)
        trainset = UTKFace(
                    "./datasets", train=True, transforms=train_transforms,
                    conf=conf
                )
        testset = UTKFace(
            "./datasets", train=False, transforms=test_transforms,
            conf=conf
        )
        valset = copy.deepcopy(testset)
        return trainset, valset, testset
    if dataset_mode=="FairFace":
        ds_opt = conf["dataset_options"]
        train_transforms, test_transforms = data_transforms_fairface(conf)
        trainset = FairFace(
                    "./datasets", train=True, transforms=train_transforms,
                    conf=conf
                )
        testset = FairFace(
            "./datasets", train=False, transforms=test_transforms,
            conf=conf
        )
        valset = copy.deepcopy(testset)
        return trainset, valset, testset
    raise NotImplementedError(dataset_mode)


def preprocess_data(data, conf, shuffle=True):
    """From dataset to dataloader in PyTorch
    Transforms, augmentations, etc. are now stored in the dataset"""
    ds = WeightedDataLoader(
        data, batch_size=conf["client_opt"]["batch_size"], shuffle=shuffle
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
            conf["dataset_options"]["num_clients"],
            split_mode=conf["dataset_options"]["split_mode"],
            mode="clients",
            distribution_seed=conf["seed"],
            shuffle_seed=conf["dataset_options"]["data_shuffle_seed"],
            dirichlet_alpha=conf["dataset_options"]["dirichlet_alpha"],
        )
        
    elif dataset_mode == "WaterBirds":
        ds_split = split_data_waterbirds(
            ds,
            conf
        )
    elif dataset_mode == "StackedMNIST":
        ds_split = \
            split_stackedmnist_data(
                ds, conf['dataset_options']['split_mode'], conf['dataset_options']['num_clients'],
                uniform_proportion=conf['dataset_options']['uniform_proportion'] if 'uniform_proportion' in conf['dataset_options'].keys() else 0
            )
    elif dataset_mode == "Spawrious" or dataset_mode=="CMNIST" or dataset_mode=="FMOW":
        ds_split = split_data_spawrious(
            ds,
            conf
        )
    elif dataset_mode == "CelebA":
        ds_split = split_data_celeba(
            ds,
            conf
        )
    else:
        raise NotImplementedError('Dataset split for dataset '+dataset_mode+' not recognized')
    print([len(ds) for ds in ds_split])
    return ds_split


def subsample_DFR(ds, subsample_type, *args, metadata=None, random=True, **kwargs):
    """
    Classifier re-training with sub-sampled, group-balanced, held-out(validation) data and l1 regularization.
    Note that when attribute is unavailable in validation data, group-balanced reduces to class-balanced.
    https://openreview.net/pdf?id=Zb6c8A-Fghk
    """
    assert subsample_type in {"group", "class"}
    if not hasattr(ds, "group_sizes") and metadata is None:
        metadata = count_groups(ds, update_ds=False)
    if metadata is None:
        ds_y = ds.y
        ds_s = ds.s
        num_attributes = ds.num_attributes
        num_labels = ds.num_labels
        group_sizes = ds.group_sizes
        class_sizes = ds.class_sizes
        weights_g = ds.weights_g
        weights_y = ds.weights_y
    else:
        ds_y = metadata["y"]
        ds_s = metadata["s"]
        num_attributes = metadata['num_attributes']
        num_labels = metadata['num_labels']
        group_sizes = metadata['group_sizes']
        class_sizes = metadata['class_sizes']
        weights_g = metadata['weights_g']
        weights_y = metadata['weights_y']
    if random:
        perm = torch.randperm(len(ds)).tolist()
    else:
        perm = list(range(len(ds)))
    min_size = min(list(group_sizes)) if subsample_type == "group" else min(list(class_sizes))

    counts_g = [0] * num_attributes * num_labels
    counts_y = [0] * num_labels
    new_idx = []
    for p in perm:
        y, a = ds_y[p], ds_s[p]
        if (subsample_type == "group" and counts_g[num_attributes * int(y) + int(a)] < min_size) or (
                subsample_type == "class" and counts_y[int(y)] < min_size):
            counts_g[num_attributes * int(y) + int(a)] += 1
            counts_y[int(y)] += 1
            new_idx.append(p)

    return SubsetDataset(ds, new_idx)

def subsample_FEx(ds, conf, accuracies, *args, **kwargs):
    """
    Detect forgettable examples from accurate pred list.
    https://arxiv.org/pdf/1911.03861
    """
    #!TODO: balanced filters
    f, u = select_forgettables(accuracies)
    filtered_ids = {}
    filtered_ids['forgettables'] = f
    #filtered_ids['forgettables_b'] = balance_by_class(f)
    filtered_ids['never_learnt'] = u
    #filtered_ids['never_learnt_b'] = balance_by_class(u)

    mode = "forgettables"
    idx = filtered_ids[mode]
    return SubsetDataset(ds, idx)

def subsample(ds, conf, *args, **kwargs):
    """Subsample interesting samples to mitigate spurious correlation
    in 2stage training methods"""
    if conf["client_opt"]["subpop_optimizer"] == "DFR":
        #!TODO: this should be from a held-out validation set
        ret =  subsample_DFR(ds, "group", *args, **kwargs)

    elif "FEx" in conf["client_opt"]["subpop_optimizer"]:
        ret =  subsample_FEx(ds, conf, *args, **kwargs)
        if conf["client_opt"]["fex_balance_classes"]:
            ret = subsample_DFR(ret, "class")
    print("Subsample size:", len(ret))

    print(count_groups(ret)["group_sizes"])
    return ret
