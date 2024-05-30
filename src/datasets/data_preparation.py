import torchvision
from torch.utils.data import random_split, Dataset
import torch
from PIL import Image
import copy
import numpy as np

from src.datasets.cifar import CIFAR10, data_transforms_cifar10


class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.dataset[actual_idx]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        if name=="dataset":
            return super().__getattribute__('dataset')
        return getattr(self.dataset, name)



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
            raise ValueError("Code needs update")
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


def preprocess_data(data, conf, shuffle=True):
    """From dataset to dataloader in PyTorch
    Transforms, augmentations, etc. are now stored in the dataset"""
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


def split_data(ds, num_clients,
               split=None,
               split_mode="dirichlet",
               distribution_seed=None,
               shuffle_seed=None,
               sort=True,
               *args, **kwargs):
    """Split data in X,Y between 'num_clients' number of clients"""
    
    y_list = [y for _, (y,_) in ds]
    classes = np.unique(y_list)
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
            
    idx_split = None

    for i, cls in enumerate(classes):
        idx_cls = np.where(y_list == cls)[0]
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

    ds_split = [SubsetDataset(ds, idx) for idx in idx_split]
    print([len(ds) for ds in ds_split])
    return ds_split


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
