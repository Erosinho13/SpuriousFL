

from src.datasets.data_splits import split_mode_to_matrix
from src.datasets.dataset_utils import SubpopDataset, SubsetDataset, count_groups
import torchvision
import numpy as np
from torchvision.datasets import VisionDataset
import os
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset, Image
from PIL import Image as PILImage

class FairFace(VisionDataset, SubpopDataset):
    def __init__(
        self,
        root="./datasets",
        train=True,
        transforms=None,
        conf={}
    ) -> None:
        if train:
            split = 'train'
        else:
            split = 'validation'
        dataset = load_dataset("HuggingFaceM4/FairFace", "0.25", split=split, cache_dir=root)

        # Preprocess steps

        def preprocess_labels(example, conf):
            if (conf["num_targets"]==2 and conf["target_name"]=="race") or (conf["num_groups"]==2 and conf["group_name"]=="race"):
                example["race"] = 0 if example["race"]==3 else 1
            age_clusters = 0
            if conf["target_name"]=="age":
                age_clusters = conf["num_targets"]
            if conf["group_name"]=="age":
                age_clusters = conf["num_groups"]
            if age_clusters==2:
                example["age"] = 0 if example["age"]<6 else 1
            return example
                
        dataset = dataset.map(lambda x: preprocess_labels(x,conf["dataset_options"]))
   
        self.transform = transforms  # Save the transform first
        def hf_transform(examples):
            if "image" in examples.keys():
                examples["image"] = [transforms(img) for img in examples["image"]]
            return examples

        dataset.set_transform(hf_transform)
        self.dataset = dataset
        self.root = root
        self.target_attr_name = conf["dataset_options"]["target_name"]
        self.group_attr_name = conf["dataset_options"]["group_name"][0]


        self.identity_df = None
        self.labels_df = None
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        hf_dict = self.dataset[index]
        x = hf_dict["image"]
        y = hf_dict[self.target_attr_name]
        s = hf_dict[self.group_attr_name]
        return index, x, (y, s)

    def __getitems__(self, indices):
        hf_dicts = self.dataset[indices]  # batched fetch
        images = hf_dicts["image"]
        targets = hf_dicts[self.target_attr_name]
        groups = hf_dicts[self.group_attr_name]
        return [(i, img, (y, s)) for i, img, y, s in zip(indices, images, targets, groups)]

    def __getattr__(self, name):
        #print(f"Attribute called: {name}")
        try:
            return super().__getattribute__('dataset').__getattribute__(name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


def data_transforms_fairface(conf):
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


def split_data_fairface(ds, conf):
    raise NotImplementedError()