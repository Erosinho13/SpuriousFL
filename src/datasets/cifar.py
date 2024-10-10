from torchvision.datasets import VisionDataset
import torchvision
import numpy as np

from src.utils import dirichlet_split
from src.datasets.dataset_utils import SubsetDataset

class CIFAR10(VisionDataset):
    def __init__(
        self,
        root="./datasets",
        train=True,
        transforms=None,
    ) -> None:
        dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True, transform=transforms
        )
        self.dataset = dataset
        self.root = root


    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        s = 0
        return index, np.asarray(x), (y, s)

    def __getattr__(self, name):
        try:
            return super().__getattribute__('dataset').__getattribute__(name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        

def data_transforms_cifar10(conf={}):
    train_tr_list = [
    ]
    if conf["dataset_options"]["aug_crop"] > 0:
        train_tr_list.append(torchvision.transforms.RandomCrop(32, padding=conf["dataset_options"]['aug_crop']))
    if conf["dataset_options"]["aug_horizontal_flip"]:
        train_tr_list.append(torchvision.transforms.RandomHorizontalFlip())
    train_tr_list.append(torchvision.transforms.ToTensor())
    test_tr_list = [
        torchvision.transforms.ToTensor()
    ]

    if conf["dataset_options"]["norm"]:
        train_tr_list.append(torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615)
            ))
        test_tr_list.append(torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615)
            ))
    return torchvision.transforms.Compose(train_tr_list), torchvision.transforms.Compose(test_tr_list)


def cifar_split_data(ds, num_clients,
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