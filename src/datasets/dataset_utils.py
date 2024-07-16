import torch
from torch.utils.data import Dataset
from abc import abstractmethod

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


def count_groups(ds, update_ds=True):
    ds_y, ds_s= [], []
    for i in range(len(ds)):
        _, (y, s) = ds[i]
        y, s = int(y), int(s)
        ds_y.append(y)
        ds_s.append(s)
    ds_i = list(range(len(ds)))

    weights_g, weights_y = [], []
    num_attributes = len(set(ds_s))
    num_labels = len(set(ds_y))
    group_sizes = [0] * num_attributes * num_labels
    class_sizes = [0] * num_labels

    for i in ds_i:
        group_sizes[num_attributes * ds_y[i] + ds_s[i]] += 1
        class_sizes[ds_y[i]] += 1

    for i in ds_i:
        weights_g.append(len(ds_i) / group_sizes[num_attributes * ds_y[i] + ds_s[i]])
        weights_y.append(len(ds_i) / class_sizes[ds_y[i]])
    if update_ds:
        ds.y = ds_y
        ds.s = ds_s
        ds.num_attributes = num_attributes
        ds.num_labels = num_labels
        ds.group_sizes = group_sizes
        ds.class_sizes = class_sizes
        ds.weights_g = weights_g
        ds.weights_y = weights_y
    return {
        "y": ds_y, 
        "s": ds_s, 
        "num_attributes": num_attributes, 
        "num_labels": num_labels,
        "group_sizes": group_sizes,
        "class_sizes": class_sizes, 
        "weights_g": weights_g, 
        "weights_y": weights_y
    }


def subsample(ds, subsample_type, metadata=None):
    assert subsample_type in {"group", "class"}
    if not hasattr(ds, "group_sizes") and metadata is None:
        metadata = count_groups(ds, update_ds=True)
    if metadata is None:
        ds_y = ds.y
        ds_s = ds.s
        num_attributes = num_attributes
        num_labels = num_labels
        group_sizes = group_sizes
        class_sizes = class_sizes
        weights_g = weights_g
        weights_y = weights_y
    else:
        ds_y = metadata["y"]
        ds_s = metadata["s"]
        num_attributes = metadata['num_attributes']
        num_labels = metadata['num_labels']
        group_sizes = metadata['group_sizes']
        class_sizes = metadata['class_sizes']
        weights_g = metadata['weights_g']
        weights_y = metadata['weights_y']       
    perm = torch.randperm(len(ds)).tolist()
    min_size = min(list(group_sizes)) if subsample_type == "group" else min(list(class_sizes))

    counts_g = [0] * num_attributes * num_labels
    counts_y = [0] * num_labels
    new_idx = []
    for p in perm:
        y, a = ds.y[ds.idx[p]], ds.s[ds.idx[p]]
        if (subsample_type == "group" and counts_g[ds.num_attributes * int(y) + int(a)] < min_size) or (
                subsample_type == "class" and counts_y[int(y)] < min_size):
            counts_g[ds.num_attributes * int(y) + int(a)] += 1
            counts_y[int(y)] += 1
            new_idx.append(ds.idx[p])

    return new_idx

def get_metadata(ds):
    """Return dataset descriptor metadata"""
    if not hasattr(ds, "group_sizes"):
        ds.update_metadata()
    return {
        "num_attributes": ds.num_attributes, 
        "num_labels": ds.num_labels,
        "group_sizes": ds.group_sizes,
        "class_sizes": ds.class_sizes, 
    }

class SubpopDataset:

    @abstractmethod
    def __getitem__(self,idx):
        pass

    def update_metadata(self):
        ds_y, ds_s= [], []
        for i in range(len(self)):
            _, (y, s) = self[i]
            y, s = int(y), int(s)
            ds_y.append(y)
            ds_s.append(s)
        ds_i = list(range(len(self)))

        self.weights_g, self.weights_y = [], []
        self.num_attributes = len(set(ds_s))
        self.num_labels = len(set(ds_y))
        self.group_sizes = [0] * self.num_attributes * self.num_labels
        self.class_sizes = [0] * self.num_labels

        for i in ds_i:
            self.group_sizes[self.num_attributes * ds_y[i] + ds_s[i]] += 1
            self.class_sizes[ds_y[i]] += 1
        for i in ds_i:
            self.weights_g.append(len(self) / self.group_sizes[self.num_attributes * ds_y[i] + ds_s[i]])
            self.weights_y.append(len(self) / self.class_sizes[ds_y[i]])

    def get_weight_g(self, idx):
            _, (y, s) = self[idx]
            return len(self) / self.group_sizes[self.num_attributes * y + s]
    def get_weight_y(self, idx):
            _, (y, _) = self[idx]
            return len(self) / self.class_sizes[y]
    



class SubsetDataset(Dataset, SubpopDataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices


    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.dataset[actual_idx]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        if name == "dataset":
            return super().__getattribute__('dataset')
        return getattr(self.dataset, name)