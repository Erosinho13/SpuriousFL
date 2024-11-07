import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import MNIST, VisionDataset

from src.datasets.dataset_utils import SubpopDataset


class CMNIST(VisionDataset, SubpopDataset):
    def __init__(
        self,
        root="./datasets",
        train=True,
        confounding_factor: float = 0.5,
        transforms=None,
    ) -> None:
        self.root = root
        self.transform = transforms
        self.subset = MNIST(root=root, train=train, download=True)
        self.subset.targets = (self.subset.targets >= 5).to(torch.int64)
        self.subset.data = torch.einsum("ijk,c -> ijkc", self.subset.data, torch.tensor([1, 1, 1], dtype=torch.uint8))

        rng = np.random.RandomState(42)

        def generate_attr(targets, alpha):
            attr = torch.zeros_like(targets)
            for y, a in zip([0, 1], [alpha, 1 - alpha]):
                idx = torch.where(targets == y)[0]
                s = rng.binomial(1, a, len(idx))
                attr[idx] = torch.from_numpy(s)
            return attr

        self.subset.attr = generate_attr(self.subset.targets, confounding_factor)
        mask = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.uint8)
        self.subset.data = torch.einsum("ijkc,ic -> ijkc", self.subset.data, mask[self.subset.attr])

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        img, target, attr = self.subset.data[index], int(self.subset.targets[index]), int(self.subset.attr[index])
        img = Image.fromarray(img.numpy())
        img = self.transform(img)
        return index, img, (target, attr)


def data_transforms_cmnist(conf={}):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
        ]
    )
