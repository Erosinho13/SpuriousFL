import torchvision
from torch.utils.data import random_split, Dataset
import torch
import numpy as np
from PIL import Image
import copy
import json
import numpy as np

class SubsetDataset(Dataset): # https://discuss.pytorch.org/t/torch-utils-data-dataset-random-split/32209/3 
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)


def load_data(dataset_mode="CIFAR10", val_split=False, val_ratio = 0.2, conf={}):
    """Load datasets into dataset object"""
    if "dataset" in conf.keys():
        dataset_mode = conf["dataset"]

    if "val_split" in conf.keys():
        val_split = conf["val_split"]
    if "seed" not in conf.keys():
        conf["seed"] = None

    if dataset_mode == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(
            "./dump/dataset", train=True, download=True
        )
        testset = torchvision.datasets.CIFAR10(
            "./dump/dataset", train=False, download=True
        )

        if val_split:
            len_val = int(len(trainset) * val_ratio)
            len_train = len(trainset) - len_val
            trainset, valset = random_split(
                trainset,
                [len_train, len_val],
                torch.Generator().manual_seed(conf["seed"]),
            )
            trainset = SubsetDataset(trainset)
            valset = SubsetDataset(valset)
        else:
            valset = copy.deepcopy(testset)
        return trainset, valset, testset
    raise NotImplementedError(dataset_mode)


def preprocess_data(data, conf, shuffle=False):
    """From torch.utils.data.Dataset to DataLoader"""
    add_transforms = []
    add_transforms.append(torchvision.transforms.ToTensor())

    #!TODO check these numbers
    if conf["dataset"]=="CIFAR10":
        add_transforms.append(
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615)
            )
        )
    else:
        raise NotImplementedError("Dataset unknown", conf["dataset"])

    if data.transform is None:
        data.transform = torchvision.transforms.Compose([])
    old_transforms = data.transform.transforms
    new_transforms = old_transforms + add_transforms
    data.transform.transforms = new_transforms

    ds = torch.utils.data.DataLoader(
        data, batch_size=conf["batch_size"], shuffle=shuffle
    )
    return ds


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


def split_data(X, Y, num_clients, 
               split=None,
               split_mode="dirichlet",
               distribution_seed=None,
               shuffle_seed=None,
               sort=True,
               *args, **kwargs):
    """Split data in X,Y between 'num_clients' number of clients"""
    assert len(X) == len(Y)
    classes = np.unique(Y)
    num_classes = len(classes)

    if shuffle_seed is None:
        shuffle_seed = distribution_seed

    if split is None:
        if split_mode == "dirichlet":
            split = dirichlet_split(num_classes, num_clients, seed=distribution_seed, *args, **kwargs)
            if sort:
                column_sums = np.sum(split, axis=0)
                sorted_indices = np.argsort(column_sums)[::-1]
                split = split[:, sorted_indices]
        elif split_mode == "homogen":
            split = [1 / num_clients] * num_clients
            split = [split] * num_classes
            split = np.array(split)
        elif split_mode == "balanced":
            split = dirichlet_split(1, num_clients, seed=distribution_seed, *args, **kwargs)
            split = split.repeat(num_classes, axis=0)
            if sort:
                column_sums = np.sum(split, axis=0)
                sorted_indices = np.argsort(column_sums)[::-1]
                split = split[:, sorted_indices]
        else:
            ValueError(f"Split mode not recognized {split_mode}")
    X_split = None
    Y_split = None
    idx_split = None

    for i, cls in enumerate(classes):
        idx_cls = np.where(Y == cls)[0]
        np.random.default_rng(seed=shuffle_seed).shuffle(idx_cls)
        cls_num_example = len(idx_cls)
        cls_split = np.rint(split[i] * cls_num_example)

        # if rounding error remove it from most populus one
        if sum(cls_split) > cls_num_example:
            max_val = np.max(cls_split)
            max_idx = np.where(cls_split == max_val)[0][0]
            cls_split[max_idx] -= sum(cls_split) - cls_num_example
        cls_split = cls_split.astype(int)
        idx_cls_split = np.split(idx_cls, np.cumsum(cls_split)[:-1])
        if idx_split is None:
            idx_split = idx_cls_split

        else:
            for i in range(len(idx_cls_split)):
                idx_split[i] = np.concatenate([idx_split[i], idx_cls_split[i]], axis=0)

    for i in range(len(idx_split)):
        idx_split[i] = np.sort(idx_split[i])

    X_split = [X[idx] for idx in idx_split]
    Y_split = [Y[idx] for idx in idx_split]
    print([len(y) for y in X_split])
    return X_split, Y_split


def get_np_from_dataset(dataset):
    return np.array(dataset.data), np.array(dataset.targets)


def get_np_from_dataloader(dataloader):
    image_batches = []
    label_batches = []
    for images, labels in dataloader:
        image_batches.append(images.detach().numpy())
        label_batches.append(labels.detach().numpy())
    numpy_labels = np.concatenate(label_batches, axis=0)
    numpy_images = np.concatenate(image_batches, axis=0)
    numpy_images = np.transpose(numpy_images, (0, 2, 3, 1))
    return numpy_images, numpy_labels


def get_np_from_ds(data):
    #!TODO rework this
    #!TODO not working for Subset
    if isinstance(data, torch.utils.data.DataLoader):
        return get_np_from_dataloader(data)
    return get_np_from_dataset(data)

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.data = data[0]
        self.targets = data[1]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        arr = self.data[idx]
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            arr = (arr * 255).astype(np.uint8)
        image = Image.fromarray(arr)
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def get_ds_from_np(data):
    return CustomImageDataset(data, transform=None, target_transform=None)
