
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

    def get_celeba_metadata(self):
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
        return index, np.asarray(x), (y, s)

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

def split_data_celeba(ds, conf):
    """
    Torch wrapper for split from AFed paper
    https://arxiv.org/abs/2501.02732
    """

    data_idx_map, _ = load_split_data(ds, conf)

    ds_split = [SubsetDataset(ds, idx) for idx in data_idx_map.values()]
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