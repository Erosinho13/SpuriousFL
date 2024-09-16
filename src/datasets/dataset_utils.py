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
            _, (y, s) = self[i]
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


def count_groups(ds, update_ds=True, num_attributes=None, num_labels=None):
    ds_y, ds_s= [], []
    for i in range(len(ds)):
        _, (y, s) = ds[i]
        y, s = int(y), int(s)
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
