from src.datasets.data_splits import create_subsets_from_list, split_mode_to_matrix
from torchvision.datasets import VisionDataset
from wilds import get_dataset
import torchvision
import numpy as np

from src.datasets.dataset_utils import SubsetDataset, count_groups
from src.datasets.dataset_utils import SubpopDataset, get_spurious_group_idx


class WaterBirds(VisionDataset, SubpopDataset):
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
        return index, x, (y, s)

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
    if conf["dataset_options"]["aug_crop"] > 0:
        train_tr_list.append(torchvision.transforms.RandomCrop(input_size, padding=conf["dataset_options"]['aug_crop']))
    if conf["dataset_options"]["aug_horizontal_flip"]:
        train_tr_list.append(torchvision.transforms.RandomHorizontalFlip())
    train_tr_list.append(torchvision.transforms.ToTensor())
    test_tr_list = [
        torchvision.transforms.Resize((int(input_size * (256 / input_size)), int(input_size * (256 / input_size)),), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.CenterCrop(input_size),
        torchvision.transforms.ToTensor()
    ]

    if conf["dataset_options"]["norm"]:
        train_tr_list.append(torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        test_tr_list.append(torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return torchvision.transforms.Compose(train_tr_list), torchvision.transforms.Compose(test_tr_list)




def get_envs(group_ids, conf):
    """Set list of idx for environment
    homogen:
      - 1/4 of each
    sameratio:
    0 - {(0,0),(1,0)} - birds on land
    1 - {(0,0),(1,1)} - everyone in the expected background
    2 - {(0,1),(1,0)} - everyone in unexpected background
    3 - {(0,1),(1,1)} - birds on water"""

    split_mode=conf["dataset_options"]["split_mode"]
    seed=conf["seed"]

    def findgroup(y, s):
        if y==0 and s==0:
            return 0, 1
        if y==0 and s==1:
            return 2, 3
        if y==1 and s==0:
            return 0, 2
        if y==1 and s==1:
            return 1, 3
    

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
                id1, id2 = findgroup(y, s)
                subsets[id1].extend(subset1)
                subsets[id2].extend(subset2)
        return subsets
    if split_mode=="maxclassbalance":
        # each of the 4 clients have ~0.3 imbalance ratio
        lengths = [[92,-92],[28,-28]]
        for y in [0,1]:
            for s in [0,1]:
                length = lengths[y][s]
                perm = rng.permutation(group_ids[y][s])
                subset1, subset2 = perm[:length], perm[length:]
                id1, id2 = findgroup(y, s)
                subsets[id1].extend(subset1)
                subsets[id2].extend(subset2)
        return subsets
    if split_mode=="balancedclients":
        # 3 small balanced client and the biggest is ~0.25 imbalance
        lengths = [[28,-156],[28,-156]]
        for y in [0,1]:
            for s in [0,1]:
                length = lengths[y][s]
                perm = rng.permutation(group_ids[y][s])
                subset1, subset2 = perm[:length], perm[length:]
                id1, id2 = findgroup(y, s)
                subsets[id1].extend(subset1)
                subsets[id2].extend(subset2)
        return subsets
    client_samples = split_mode_to_matrix(conf)
    if client_samples is not None:
        subsets = create_subsets_from_list(group_ids, client_samples, rng)
        return subsets
    raise NotImplementedError("split_mode not recognized")


def split_data_waterbirds(ds, conf):
    """Split data in X,Y between 'num_clients' number of clients"""

    ids_by_groups = get_spurious_group_idx(ds)
    for k in ids_by_groups.keys():
        for l in ids_by_groups[k].keys():
            print(k,l,len(ids_by_groups[k][l]))
    idx_split = get_envs(ids_by_groups, conf)
    ds_split = [SubsetDataset(ds, idx) for idx in idx_split]
    for ds in ds_split:
        print(count_groups(ds, False, 2, 2)["group_sizes"])
    return ds_split
