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
    if "dataset_options" in conf.keys() and "input_size" in conf["dataset_options"].keys():
        input_size = conf["dataset_options"]["input_size"]
    else:
        input_size = 32
    train_tr_list = [

        torchvision.transforms.Resize((int(input_size * (256 / input_size)), int(input_size * (256 / input_size)),), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.CenterCrop(input_size),
    ]
    if conf["aug_crop"] > 0:
        train_tr_list.append(torchvision.transforms.RandomCrop(input_size, padding=conf['aug_crop']))
    if conf["aug_horizontal_flip"]:
        train_tr_list.append(torchvision.transforms.RandomHorizontalFlip())
    train_tr_list.append(torchvision.transforms.ToTensor())
    test_tr_list = [
        torchvision.transforms.Resize((int(input_size * (256 / input_size)), int(input_size * (256 / input_size)),), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.CenterCrop(input_size),
        torchvision.transforms.ToTensor()
    ]

    if conf["norm"]:
        train_tr_list.append(torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        test_tr_list.append(torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
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
    homogen:
      - 1/4 of each
    sameratio:
    0 - {(0,0),(1,0)} - birds on land
    1 - {(0,0),(1,1)} - everyone in the expected background
    2 - {(0,1),(1,0)} - everyone in unexpected background
    3 - {(0,1),(1,1)} - birds on water"""

    rng = np.random.default_rng(seed=seed)
    subsets = [[],[],[],[]]
    if split_mode=="homogen":
        for y in group_ids.keys():
            for s in group_ids[y].keys():
                l = len(group_ids[y][s])//4
                perm = rng.permutation(group_ids[y][s])
                samples_by_envs =  [perm[:l], perm[l:l*2], perm[l*2:l*3], perm[l*3:]]
                for i, x in enumerate(samples_by_envs):
                    subsets[i].extend(x)
        return subsets
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
    raise NotImplementedError("split_mode not recognized")


def split_data_waterbirds(ds, conf):
    """Split data in X,Y between 'num_clients' number of clients"""

    if conf["num_clients"]!=4:
        raise NotImplementedError("Only 4 clients for now!")
    
    ids_by_groups = get_spurious_group_idx(ds)
    idx_split = get_envs(ids_by_groups, split_mode=conf["split_mode"], seed=conf["seed"])
    ds_split = [SubsetDataset(ds, idx) for idx in idx_split]
    return ds_split
