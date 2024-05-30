from torchvision.datasets import VisionDataset
from wilds import get_dataset
import torchvision
import numpy as np

from src.datasets.subset import SubsetDataset


class WaterBirds(VisionDataset):
    def __init__(
        self,
        root="./datasets",
        train=True,
        transforms=None,
    ) -> None:
        dataset = get_dataset(dataset="waterbirds", download=True, root_dir=root)
        self.subset = dataset.get_subset("train" if train else "test", transform=transforms)
        self.root = root

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int):
        x, y, metadata = self.subset[index]
        s = metadata[0]
        return x, (y, s)

    def __getattr__(self, name):
        try:
            return super().__getattribute__('dataset').__getattribute__(name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


def data_transforms_waterbirds(conf={}):
    train_tr_list = [
        torchvision.transforms.Resize((32,32), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
    ]
    if conf["aug_crop"] > 0:
        train_tr_list.append(torchvision.transforms.RandomCrop(32, padding=conf['aug_crop']))
    if conf["aug_horizontal_flip"]:
        train_tr_list.append(torchvision.transforms.RandomHorizontalFlip())
    train_tr_list.append(torchvision.transforms.ToTensor())
    test_tr_list = [
        torchvision.transforms.Resize((32,32), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.ToTensor()
    ]

    if conf["norm"]:
        pass
    return torchvision.transforms.Compose(train_tr_list), torchvision.transforms.Compose(test_tr_list)




def get_spurious_group_idx(ds):
    """Gets the list of idx for every (y,s) pair for y-label, s-spurious group.
    Expects dataloader with x,(y,s) getter where y and s are scalar ints"""
    d = {}
    for i in range(len(ds)):
        _, (y, s) = ds[i]
        y,s = int(y), int(s)
        if y not in d.keys():
            d[y] = {}
        if s not in d[y].keys():
            d[y][s] = []
        d[y][s].append(i)
    return d


def get_envs(group_ids, split_mode='sameratio', seed=42):
    """Set list of idx for environment
    0 - {(0,0),(1,0)} - birds on land
    1 - {(0,0),(1,1)} - everyone in the expected background
    2 - {(0,1),(1,0)} - everyone in unexpected background
    3 - {(0,1),(1,1)} - birds on water"""

    rng = np.random.default_rng(seed=seed)
    subsets = [[],[],[],[]]
    if split_mode=="sameratio":
        for y in group_ids.keys():
            for s in group_ids[y].keys():
                length = len(group_ids[y][s])//2
                perm = rng.permutation(group_ids[y][s])
                subset1, subset2 = perm[:length], perm[length:]
                if y==0 and s==0:
                    subsets[0].extend(subset1)
                    subsets[1].extend(subset2)
                elif y==0 and s==1:
                    subsets[2].extend(subset1)
                    subsets[3].extend(subset2)
                elif y==1 and s==0:
                    subsets[0].extend(subset1)
                    subsets[2].extend(subset2)
                elif y==1 and s==1:
                    subsets[1].extend(subset1)
                    subsets[3].extend(subset2)
        return subsets
    raise NotImplementedError()


def split_data_waterbirds(ds, conf):
    """Split data in X,Y between 'num_clients' number of clients"""

    if conf["num_clients"]!=4:
        raise NotImplementedError("Only 4 clients for now!")
    
    ids_by_groups = get_spurious_group_idx(ds)
    idx_split = get_envs(ids_by_groups, split_mode='sameratio', seed=conf["seed"])
    ds_split = [SubsetDataset(ds, idx) for idx in idx_split]
    return ds_split
