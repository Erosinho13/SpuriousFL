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




def get_envs(group_ids, split_mode='sameratio', seed=42):
    """Set list of idx for environment
    homogen:
      - 1/4 of each
    sameratio:
    0 - {(0,0),(1,0)} - birds on land
    1 - {(0,0),(1,1)} - everyone in the expected background
    2 - {(0,1),(1,0)} - everyone in unexpected background
    3 - {(0,1),(1,1)} - birds on water"""
    def findgroup(y, s):
        if y==0 and s==0:
            return 0, 1
        if y==0 and s==1:
            return 2, 3
        if y==1 and s==0:
            return 0, 2
        if y==1 and s==1:
            return 1, 3
    def create_subsets_from_list(group_ids, client_samples, rng=np.random.default_rng()):
        subsets = [[] for _ in range(len(client_samples))]
        for y in [0,1]:
            for s in [0,1]:
                perm = rng.permutation(group_ids[y][s])
                cumulated_idx = 0
                for i,c in enumerate(client_samples):
                    subset = perm[cumulated_idx:cumulated_idx+c[y][s]]
                    subsets[i].extend(subset)
                    cumulated_idx += c[y][s]
        return subsets

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
    if split_mode=="zeroCI_LSC":
        client_samples=[
            [ # Birds on land
                [23,5],
                [23,5]
            ],
            [ # Expected background
                [896,5],
                [5,896]
            ],
            [ # Unexpected background
                [5,23],
                [23,5]
            ],
            [ # Birds on water
                [5,151],
                [5,151]
            ]
        ]
        subsets = create_subsets_from_list(group_ids, client_samples, rng)
        return subsets
    if split_mode=="nCI_LSC":
        client_samples=[
            [ # Birds on land
                [23,5],
                [23,5]
            ],
            [ # Expected background
                [3465,5],
                [5,896]
            ],
            [ # Unexpected background
                [5,23],
                [23,5]
            ],
            [ # Birds on water
                [5,151],
                [5,151]
            ]
        ]
        subsets = create_subsets_from_list(group_ids, client_samples, rng)
        return subsets
    if split_mode=="nDI_nGSC":
        client_samples=[
            [ # Birds on land
                [23,5],
                [23,5]
            ],
            [ # Expected background
                [23,5],
                [5,23]
            ],
            [ # Unexpected background
                [5,23],
                [23,5]
            ],
            [ # Birds on water
                [5,23],
                [5,23]
            ]
        ]
        subsets = create_subsets_from_list(group_ids, client_samples, rng)
        return subsets
    if split_mode=="LCI_LSC_noG":
        client_samples=[
            [ 
                [23,23],
                [5,5]
            ],
            [ 
                [23,5],
                [5,23]
            ],
            [ 
                [5,23],
                [23,5]
            ],
            [ 
                [5,5],
                [23,23]
            ]
        ]
        subsets = create_subsets_from_list(group_ids, client_samples, rng)
        return subsets
    if split_mode=="more_expected_clients":
        client_samples=[
            [ # Birds on land
                [28,0],
                [28,0]
            ],
            [ # Expected background
                [28,0],
                [0,28]
            ],
            [ # Unexpected background
                [0,28],
                [28,0]
            ],
            [ # Birds on water
                [0,28],
                [0,28]
            ]
        ]
        for i in range(35):
            client_samples.append([[28,0],[0,28]])
        subsets = create_subsets_from_list(group_ids, client_samples, rng)
        return subsets
    if split_mode=="CI_GSC_reverse": # global minority is local mayority
        client_samples=[
            [ # mayority land/land bird
                [400,5],
                [5,5]
            ],
            [ # mayority water/water bird
                [5,5],
                [5,400]
            ],
            [ # minority but class majority
                [40,40],
                [5,5]
            ],
            [ # minority but class majority
                [5,5],
                [40,40]
            ]
        ]
        subsets = create_subsets_from_list(group_ids, client_samples, rng)
        return subsets
    raise NotImplementedError("split_mode not recognized")


def split_data_waterbirds(ds, conf):
    """Split data in X,Y between 'num_clients' number of clients"""

    ids_by_groups = get_spurious_group_idx(ds)
    for k in ids_by_groups.keys():
        for l in ids_by_groups[k].keys():
            print(k,l,len(ids_by_groups[k][l]))
    idx_split = get_envs(ids_by_groups, split_mode=conf["dataset_options"]["split_mode"], seed=conf["seed"])
    ds_split = [SubsetDataset(ds, idx) for idx in idx_split]
    assert len(ds_split) == conf["dataset_options"]["num_clients"]
    for ds in ds_split:
        print(count_groups(ds, False, 2, 2)["group_sizes"])
    return ds_split
