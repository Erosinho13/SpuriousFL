import random
from collections import defaultdict

import torchvision
from torch.utils.data import random_split, Dataset
import torch
from PIL import Image
import copy
import numpy as np

from src.datasets.stacked_mnist import StackedMNIST, _data_transforms_mnist, count_img


class SubsetDataset(Dataset):  # https://discuss.pytorch.org/t/torch-utils-data-dataset-random-split/32209/3
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


def load_data(dataset_mode="CIFAR10", val_split=False, val_ratio=0.2, conf={}):
    """Load datasets into dataset object"""
    if "dataset" in conf.keys():
        dataset_mode = conf["dataset"]

    if "val_split" in conf.keys():
        val_split = conf["val_split"]
    if "seed" not in conf.keys():
        conf["seed"] = None

    if dataset_mode == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(
            "./datasets", train=True, download=True
        )
        testset = torchvision.datasets.CIFAR10(
            "./datasets", train=False, download=True
        )

    elif dataset_mode == "StackedMNIST":
        ds_opt = conf['dataset_options']
        mean = (0.1307, 0.1307, 0.1307)
        std = (0.3081, 0.3081, 0.3081)
        train_transform, test_transform = _data_transforms_mnist(mean=mean, std=std, norm=True)
        trainset = StackedMNIST(root=ds_opt['root'], train=True, download=True, transform=train_transform,
                                num_images=ds_opt['num_train_images'],
                                dirichlet_groups_alpha=ds_opt['dirichlet_groups_alpha'],
                                num_targets=ds_opt['num_targets'], max_num_groups=ds_opt['max_num_groups'],
                                prevent_class_shuffling=ds_opt['prevent_class_shuffling'],
                                prevent_group_shuffling=ds_opt['prevent_group_shuffling'],
                                targets=[23, 45], groups=[0, 1],
                                force_balanced_dataset=ds_opt['force_balanced_dataset'])
        testset = StackedMNIST(root=ds_opt['root'], train=False, download=True, transform=test_transform,
                               num_images=ds_opt['num_test_images'], num_targets=ds_opt['num_targets'],
                               max_num_groups=ds_opt['max_num_groups'],
                               prevent_class_shuffling=ds_opt['prevent_class_shuffling'],
                               prevent_group_shuffling=ds_opt['prevent_group_shuffling'],
                               targets=[23, 45], groups=[0, 1], force_balanced_dataset=ds_opt['force_balanced_dataset'])
    else:
        raise NotImplementedError(dataset_mode)

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

def preprocess_data(data, conf, shuffle=False):
    """From torch.utils.data.Dataset to DataLoader"""
    add_transforms = []
    add_transforms.append(torchvision.transforms.ToTensor())

    #!TODO check these numbers
    if conf["dataset"] == "CIFAR10":
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
            assert ds.max_num_groups == 2
            assert ds.force_balanced_dataset is True or ds.train is False
            assert num_clients % 4 == 0
            assert len(ds) % num_clients == 0
            tg_pairs_indices = divide_idx_by_tg_pairs(ds, split_mode)
            clients_indices = get_client_indices_mode1(tg_pairs_indices, ds, num_clients)
        elif split_mode == 'mode2':
            assert ds.num_targets == 2
            assert ds.max_num_groups == 2
            assert ds.force_balanced_dataset is True or ds.train is False
            assert num_clients % 4 == 0
            tg_pairs_indices = divide_idx_by_tg_pairs(ds, split_mode)
            clients_indices = get_client_indices_mode2(tg_pairs_indices, num_clients)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    for indices in clients_indices:
        clients_datasets.append(CustomSubset(ds, indices))

    assert_list = [count_img(cds) for cds in clients_datasets]  # uncomment to check counts by tg pair per client
    return clients_datasets


class CustomSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.dataset[actual_idx]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)