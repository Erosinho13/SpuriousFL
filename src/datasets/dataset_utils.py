from typing import List
import numpy as np
from torch.utils.data import Dataset
from abc import abstractmethod


class SubpopDataset:

    @abstractmethod
    def __getitem__(self,idx):
        pass

    def update_metadata(self, num_attributes=None, num_labels=None):
        ds_y, ds_s= [], []
        for i in range(len(self)):
            _, _, (y, s) = self[i]
            y, s = int(y), int(s)
            ds_y.append(y)
            ds_s.append(s)
        ds_i = list(range(len(self)))

        self.weights_g, self.weights_y = [], []
        if num_attributes is None:
            self.num_attributes = len(set(ds_s))
        else:
            self.num_attributes = num_attributes
        if num_labels is None:
            self.num_labels = len(set(ds_y))
        else:
            self.num_labels = num_labels
        self.group_sizes = [0] * self.num_attributes * self.num_labels
        self.class_sizes = [0] * self.num_labels

        for i in ds_i:
            self.group_sizes[self.num_attributes * ds_y[i] + ds_s[i]] += 1
            self.class_sizes[ds_y[i]] += 1
        for i in ds_i:
            self.weights_g.append(len(self) / self.group_sizes[self.num_attributes * ds_y[i] + ds_s[i]])
            self.weights_y.append(len(self) / self.class_sizes[ds_y[i]])

    def get_weight_g(self, idx):
            _, _, (y, s) = self[idx]
            return len(self) / self.group_sizes[self.num_attributes * y + s]
    def get_weight_y(self, idx):
            _, _, (y, _) = self[idx]
            return len(self) / self.class_sizes[y]
    

class ModifiedDataset(Dataset, SubpopDataset):
    def __init__(self, dataset, predictions=None, use_groups=False):
        self.dataset = dataset
        self.predicitons = predictions
        self.use_groups = use_groups
    
    def __getitem__(self, index):
        idx, img, (y, s) = self.dataset[index]
        if self.use_groups:
            tmp = y
            y = s 
            s = tmp
        if self.predicitons:
            y = self.predicitons[idx]

        return idx, img, (y, s)

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        if name == "dataset":
            return super().__getattribute__('dataset')
        return getattr(self.dataset, name)   

class OneVsRestDataset(Dataset):
    def __init__(self, base_dataset, target_class):
        """
        Wraps around an existing dataset (including SubsetDataset) for one-vs-rest classification.

        Args:
            base_dataset (Dataset): The dataset with N-class labels.
            target_class (int): The class treated as positive (1), all others as negative (0).
        """
        self.base_dataset = base_dataset
        self.target_class = target_class

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        index, x, (y, s) = self.base_dataset[idx]  # Extract (image, label) from base dataset
        y_binary = 1 if y == self.target_class else 0
        return index, x, (y_binary, s)

    def __getattr__(self, name):
        """Delegate attribute access to the base dataset for compatibility."""
        return getattr(self.base_dataset, name)


class SubsetDataset(Dataset, SubpopDataset):
    def __init__(self, dataset, indices):
        if isinstance(dataset, SubsetDataset):
            self.indices = np.array(indices)
            self.dataset = dataset.dataset
        else:
            self.dataset = dataset
            self.indices = np.array(indices)
        
        
        if hasattr(self.dataset, "__getitems__"):
            self.__getitems__ = self.conditional_getitems

    def conditional_getitems(self, idxs):
        actual_idxs = self.indices[idxs]
        return self.dataset.__getitems__(actual_idxs)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx].astype(list) # Casts to the list-compatible type, eg. int or str
        return self.dataset[actual_idx]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        if name == "dataset":
            return super().__getattribute__('dataset')
        return getattr(self.dataset, name)

def concat_subsets(ds_list: List[SubsetDataset], num_clients:int=0) -> SubsetDataset:
    """Gets a list of SubsetDataset and returns with a subset of the total ids"""
    if num_clients==0:
        num_clients=len(ds_list)
    ds_list = ds_list[:num_clients]
    for i in range(len(ds_list)-1):
        assert ds_list[i].dataset == ds_list[i+1].dataset
    all_indeces = []
    for ds in ds_list:
        all_indeces.extend(ds.indices)
    out_subset = type(ds_list[0])(ds_list[0].dataset, all_indeces)
    return out_subset
    

def get_spurious_group_idx(ds):
    """Gets the list of idx for every (y,s) pair for y-label, s-spurious group.
    Expects dataloader with x,(y,s) getter where y and s are scalar ints"""
    d = {}
    for i in range(len(ds)):
        _, _, (y, s) = ds[i]
        y,s = int(y), int(s)
        if y not in d.keys():
            d[y] = {}
        if s not in d[y].keys():
            d[y][s] = []
        d[y][s].append(i)
    return d


def get_spurious_group_idx_from_df(label_df):
    """Gets the list of idx for every (y,s) pair for y-label, s-spurious group.
    Expects pandas dataframe with 'target', 'group' and 'index' columns."""
    d = {}
    for groupid, df in label_df.groupby(["target","group"]):
        y,s = groupid
        if y not in d.keys():
            d[y]={}
        d[y][s] = list(df.index)
    return d


def count_groups(ds, update_ds=True, num_attributes=None, num_labels=None):
    orig_ids, ds_y, ds_s= [], [], []
    for i in range(len(ds)):
        idx, _, (y, s) = ds[i]
        y, s = int(y), int(s)
        orig_ids.append(int(idx))
        ds_y.append(y)
        ds_s.append(s)
    ds_i = list(range(len(ds)))

    weights_g, weights_y = [], []
    if num_attributes is None:
        num_attributes = len(set(ds_s))
    if num_labels is None:
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
        ds.orig_ds = orig_ids
        ds.y = ds_y
        ds.s = ds_s
        ds.num_attributes = num_attributes
        ds.num_labels = num_labels
        ds.group_sizes = group_sizes
        ds.class_sizes = class_sizes
        ds.weights_g = weights_g
        ds.weights_y = weights_y
    return {
        "orig_ids": orig_ids,
        "y": ds_y, 
        "s": ds_s, 
        "num_attributes": num_attributes, 
        "num_labels": num_labels,
        "group_sizes": group_sizes,
        "class_sizes": class_sizes, 
        "weights_g": weights_g, 
        "weights_y": weights_y
    }


def get_metadata(ds, conf={}):
    """Return dataset descriptor metadata"""
    if "dataset_options" not in conf.keys():
        conf["dataset_options"] = {"num_groups":None, "num_targets":None}
    if not hasattr(ds, "group_sizes"):
        ds.update_metadata(num_attributes=conf["dataset_options"]["num_groups"], num_labels=conf["dataset_options"]["num_targets"])
    return {
        "num_attributes": ds.num_attributes, 
        "num_labels": ds.num_labels,
        "group_sizes": ds.group_sizes,
        "class_sizes": ds.class_sizes, 
    }


def compute_forgetting(accuracies):
    """Counts elements with forgetting events: 
    https://github.dev/sordonia/hans-forgetting/blob/master/utils_forgetting.py"""
    forgetting_max = accuracies.shape[1]
    num_examples = accuracies.shape[0]
    forgetting = np.zeros(num_examples)

    for example in range(num_examples):
        never_learnt = (accuracies[example].sum() == 0)
        if never_learnt:
            forgetting[example] = forgetting_max
        else:
            num_forgetting_events = 0
            last_acc = 0
            num_present = 0
            for current_acc in accuracies[example]:
                if current_acc == -1:
                    break
                if current_acc == 0 and last_acc == 1:
                    num_forgetting_events += 1
                num_present += 1
                last_acc = current_acc
            assert num_present == accuracies.shape[1]
            forgetting[example] = num_forgetting_events

    most_forgotten_id = np.argsort(forgetting)[::-1]
    most_forgotten_count = np.take(forgetting, most_forgotten_id)
    return np.where(forgetting > 0)[0], forgetting, forgetting_max


def select_forgettables(accuracies):
    f, c, m = compute_forgetting(accuracies)
    c_hard = c[c>0]
    ids_by_forgettings = sorted(list(enumerate(c_hard)), key=lambda x: x[1], reverse=True)
    ids_by_forgettings = [index for index, value in ids_by_forgettings]
    f = f[ids_by_forgettings]
    never_learnt = np.where(c == m)[0]
    forgettables = f
    return forgettables, never_learnt
