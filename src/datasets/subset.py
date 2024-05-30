from torch.utils.data import Dataset


class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.dataset[actual_idx]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        if name=="dataset":
            return super().__getattribute__('dataset')
        return getattr(self.dataset, name)
