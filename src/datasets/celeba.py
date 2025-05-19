
from src.datasets.data_splits import split_mode_to_matrix
from src.datasets.dataset_utils import SubpopDataset, SubsetDataset, count_groups
import torchvision
import numpy as np
from torchvision.datasets import VisionDataset
import os
from tqdm import tqdm
import pandas as pd


class CelebA(VisionDataset, SubpopDataset):
    def __init__(
        self,
        root="./datasets",
        train=True,
        transforms=None,
        target_attr_name="Wavy_Hair",
        group_attr_name="Male"
    ) -> None:
        if train:
            split = 'train'
        else:
            split = 'test'
        dataset = torchvision.datasets.CelebA(
            root=root, split=split, download=True, transform=transforms, target_type=['attr', 'identity']
        )
        self.dataset = dataset
        self.root = root

        with open(os.path.join(self.root,"celeba/list_attr_celeba.txt"), 'r') as file:
            lines = file.readlines()

        # Get the second line (index 1) and split by spaces
        second_line_values = lines[1].strip().split()
        self.target_attr = second_line_values.index(target_attr_name)
        self.group_attr = second_line_values.index(group_attr_name)

        self.identity_df = None
        self.labels_df = None

    def get_celeba_metadata(self):
        if self.identity_df is not None and self.labels_df is not None:
            return self.labels_df, self.identity_df
        ids = []
        labels = []
        for i in tqdm(range(len(self))):
            _, y, s, identity = self.get_meta(i)
            y = int(y)
            s = int(s)
            ids.append({"index":i, "identity":int(identity)})
            labels.append({"index":i,"target":y,"group":s})
        identity_df = pd.DataFrame(ids)
        labels_df = pd.DataFrame(labels)
        self.labels_df = labels_df
        self.identity_df = identity_df
        return labels_df, identity_df

    def get_meta(self, index:int):
        _, (attrs, indiv) = self.dataset[index]
        y = attrs[self.target_attr]
        s = attrs[self.group_attr]
        return index, y, s, indiv

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, (attrs, _) = self.dataset[index]
        y = attrs[self.target_attr]
        s = attrs[self.group_attr]
        # torchvision already doing the transform
        return index, x, (y, s)

    def __getattr__(self, name):
        try:
            return super().__getattribute__('dataset').__getattribute__(name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


def split_celebA(identity_df, num_celebty, num_clients, seed=None):
    """
    group the data by identity and allocate to different clients
    """
    tot_celebties = np.unique(identity_df["identity"])
    tot_celebties = np.random.default_rng(seed=seed).permutation(tot_celebties)
    celebty_list = tot_celebties[:num_celebty * num_clients]
    data_idx_map = {}
    for cid in range(num_clients):
        selected_celebrity = celebty_list[cid * num_celebty:(cid + 1) * num_celebty]
        data_idx_map[cid] = np.where(identity_df["identity"].isin(selected_celebrity))[0]
    return data_idx_map


def load_split_data(train_ds, conf):
    """Try to get a split with good ratio of group1 vs group2 positive"""
    labels_df, identity_df = train_ds.get_celeba_metadata()
    ratio = 1.0
    seed = conf["seed"]
    i = 0
    while ratio > 0.4:
        data_idx_map = split_celebA(identity_df, conf["dataset_options"]["num_celebrity"], conf["dataset_options"]["num_clients"], conf["seed"])
        seed = np.random.default_rng(seed=seed).integers(1e10)
        i+=1
        if i>1000:
            raise ValueError("Can't generate good split with these parameters")
        group_lst = []
        target_lst = []
        for index in data_idx_map.values():
            group_lst += list(labels_df.values[index][:,2])
            target_lst += list(labels_df.values[index][:,1])
        group_lst = np.array(group_lst)
        target_lst = np.array(target_lst)
        target_idx = np.where(target_lst == 1)[0]
        group_1_positive = np.where((target_lst==1) & (group_lst==1))[0]
        group_0_positive = np.where((target_lst==1) & (group_lst==0))[0]
        ratio = min(len(group_0_positive)/len(target_idx), len(group_1_positive)/len(target_idx))
    return data_idx_map, ratio

def split_celeb_majority(train_ds, conf):
    """Gives each celeb a group by where they have the majority of samples 
    and select celebs from these categories based on celeb distribution matrix."""
    labels_df, identity_df = train_ds.get_celeba_metadata()
    df = identity_df.merge(labels_df)
    # Step 1: Count occurrences of each (identity, target, group) combination
    counts = df.groupby(['identity', 'target', 'group']).size().reset_index(name='count')
    # Step 2: For each identity, get the row with the maximum count
    majority_df = counts.loc[counts.groupby('identity')['count'].idxmax()].reset_index(drop=True)
    # Step 3: Keep only required columns
    majority_df = majority_df[['identity', 'target', 'group']]

    data_split = split_mode_to_matrix(conf)
    data_split = np.array(data_split)

    result = []
    # Make a copy to track unused identities
    unused_df = majority_df.copy()

    for matrix in data_split:
        selected_indices = []

        # For this matrix, we'll consume from unused_df
        current_unused = unused_df.copy()

        num_targets, num_groups = matrix.shape
        for target in range(num_targets):
            for group in range(num_groups):
                n_to_pick = matrix[target, group]
                if n_to_pick > 0:
                    # Filter unused identities for this (target, group)
                    subset = current_unused[(current_unused['target'] == target) & (current_unused['group'] == group)]

                    # Sample without replacement
                    picked = subset.sample(n=min(n_to_pick, len(subset)), random_state=conf["seed"])

                    # Collect picked identities
                    selected_indices.extend(picked['identity'].tolist())

                    # Remove picked identities from the pool
                    current_unused = current_unused.drop(picked.index)

        # Save selected for this matrix
        result.append(selected_indices)

        # Remove all picked identities globally to prevent reuse in next matrix
        unused_df = unused_df.drop(unused_df[unused_df['identity'].isin(selected_indices)].index)
    indexed_ids = [df[df['identity'].isin(identity_list)].index.tolist() for identity_list in result]
    return indexed_ids

def split_data_celeba(ds, conf):
    """
    Torch wrapper for split from AFed paper
    https://arxiv.org/abs/2501.02732
    """

    if conf["dataset_options"]["split_mode"]=="afed":
        data_idx_map, _ = load_split_data(ds, conf)

        ds_split = [SubsetDataset(ds, idx) for idx in data_idx_map.values()]
    else:
        data_idx_map = split_celeb_majority(ds, conf)
        ds_split = [SubsetDataset(ds, idx) for idx in data_idx_map]

    for ds in ds_split:
        print(count_groups(ds, False, conf["dataset_options"]["num_groups"], conf["dataset_options"]["num_targets"])["group_sizes"])
    return ds_split

def data_transforms_celeba(conf):
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