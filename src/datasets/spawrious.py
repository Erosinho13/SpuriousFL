from src.datasets.data_splits import create_subsets_from_list, split_mode_to_matrix
from torchvision.datasets import VisionDataset
import numpy as np
import torchvision
import os
import pandas as pd
from PIL import Image

from src.datasets.dataset_utils import SubsetDataset, count_groups
from src.datasets.dataset_utils import SubpopDataset, get_spurious_group_idx

from spawrious.torch import _download_dataset_if_not_available

class Spawrious(VisionDataset, SubpopDataset):
    def __init__(
        self,
        root="./datasets",
        train=True,
        transforms=None,
        num_targets = 4,
        num_groups = 6,
        num_samples_per_class = None,
        locations=None,
        breeds=None
    ) -> None:
        dataset, locations, breeds = get_dataset(root_dir=root, train=train,
                                                 num_targets=num_targets, num_groups=num_groups,
                                                 num_samples_per_class=num_samples_per_class,
                                                 locations=locations,breeds=breeds)
        self.metadata = {"locations":locations, "breeds":breeds}
        self.subset = dataset[["path","breed","location"]].to_numpy()
        self.root = root
        self.transform = transforms

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int):
        x, y, s = self.subset[index]
        x = Image.open(x)
        x = self.transform(x)
        return index, x, (y, s)

    def __getattr__(self, name):
        try:
            return super().__getattribute__('dataset').__getattribute__(name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


def get_image_info(root):
    image_info_list = []

    # Traverse the directory structure
    for location in os.listdir(root):
        location_path = os.path.join(root, location)
        if os.path.isdir(location_path):
            for breed in os.listdir(location_path):
                breed_path = os.path.join(location_path, breed)
                if os.path.isdir(breed_path):
                    for filename in os.listdir(breed_path):
                        if filename.endswith('.png'):
                            image_path = os.path.join(breed_path, filename)
                            image_info_list.append((image_path, location, breed))

    return image_info_list


def get_dataset(root_dir, train=True, test_ratio=0.1, seed=0, num_targets=4, num_groups=6, num_samples_per_class=None, locations=None, breeds=None):
    _download_dataset_if_not_available("entire_dataset", root_dir)
    metadata1 = get_image_info(os.path.join(root_dir,"spawrious224","1"))
    metadata0 = get_image_info(os.path.join(root_dir,"spawrious224","0"))
    metadata = metadata0 + metadata1
    df = pd.DataFrame(metadata, columns=["path","location","breed"])
    # Convert location and breed columns to categorical with predefined order
    df["location"] = pd.Categorical(df["location"], categories=locations, ordered=True)
    df["breed"] = pd.Categorical(df["breed"], categories=breeds, ordered=True)
    df = df.dropna()
    # Factorize based on the predefined categories
    df["location"], location_categories = pd.factorize(df["location"], sort=True)
    df["breed"], breed_categories = pd.factorize(df["breed"], sort=True)
    df = df[df["location"]<num_groups]
    df = df[df["breed"]<num_targets]
    print(location_categories, breed_categories)
    test_set = df.groupby(['location','breed']).apply(lambda x: x.sample(frac=test_ratio, random_state=seed)).droplevel([0,1])
    train_set = df.drop(test_set.index)
    if num_samples_per_class is not None:
        train_set = df.groupby(['location','breed']).apply(lambda x: x.sample(n=num_samples_per_class, random_state=seed)).droplevel([0,1])
    
    if train:
        df_ret = train_set
    else:
        df_ret = test_set
    df_ret = df_ret.reset_index(drop=True)
    return df_ret, locations, breeds


def data_transforms_spawrious(conf={}):
    train_tr_list = [
    ]
    test_tr_list = [
    ]
    if "input_size" in conf["dataset_options"].keys():
        input_size = conf["dataset_options"]["input_size"]
    else:
        input_size = 224
    train_tr_list.append(torchvision.transforms.Resize((input_size, input_size)))
    test_tr_list.append(torchvision.transforms.Resize((input_size, input_size)))
    if conf["dataset_options"]["aug_crop"] > 0:
        train_tr_list.append(torchvision.transforms.RandomCrop(input_size, padding=conf["dataset_options"]['aug_crop']))
    if conf["dataset_options"]["aug_horizontal_flip"]:
        train_tr_list.append(torchvision.transforms.RandomHorizontalFlip())
    train_tr_list.append(torchvision.transforms.ToTensor())
    test_tr_list.append(torchvision.transforms.ToTensor())

    if conf["dataset_options"]["norm"]:
        # ImageNet norms
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
    num_clients=conf["dataset_options"]["num_clients"]


    rng = np.random.default_rng(seed=seed)
    if split_mode=="homogen":
        subsets = [[]]*num_clients
        for y in group_ids.keys():
            for s in group_ids[y].keys():
                l = len(group_ids[y][s])//num_clients
                perm = rng.permutation(group_ids[y][s])
                samples_by_envs = []
                for i in range(num_clients):
                    samples_by_envs.append(perm[i*l:(i+1)*l])
                for i, x in enumerate(samples_by_envs):
                    subsets[i].extend(x)
        return subsets
    if split_mode=="sameratio":
        subsets = [[],[],[],[]]
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
    client_samples = split_mode_to_matrix(conf)
    if client_samples is not None:
        subsets = create_subsets_from_list(group_ids, client_samples, rng)
        return subsets
    raise NotImplementedError("split_mode not recognized")


def split_data_spawrious(ds, conf):
    """Split data in X,Y between 'num_clients' number of clients"""
    
    ids_by_groups = get_spurious_group_idx(ds)
    idx_split = get_envs(ids_by_groups, conf)
    ds_split = [SubsetDataset(ds, idx) for idx in idx_split]
    for ds in ds_split:
        print(count_groups(ds, False, conf["dataset_options"]["num_groups"], conf["dataset_options"]["num_targets"])["group_sizes"])
    return ds_split
