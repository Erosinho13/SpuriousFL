from torchvision.datasets import VisionDataset
from wilds import get_dataset


class WaterBirds(VisionDataset):
    def __init__(
        self,
        root: str,
        train=True,
        transforms=None,
    ) -> None:
        dataset = get_dataset(dataset="waterbirds", download=True, root_dir=root)
        self.subset = dataset.get_subset("train" if train else "test", transform=transforms)

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int):
        x, y, metadata = self.subset[index]
        s = metadata[0]
        return x, (y, s)
