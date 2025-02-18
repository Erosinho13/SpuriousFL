
from torchvision.datasets import VisionDataset
from src.datasets.dataset_utils import SubsetDataset, SubpopDataset
from PIL import Image
import torchvision


class FMOW(VisionDataset, SubpopDataset):
    def __init__(
        self,
        root="./datasets",
        train=True,
        transforms=None,
        num_targets = 62,
        num_groups = 5,
    ) -> None:
        NotImplementedError()

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int):
        x, y, s = self.subset[index]
        x = Image.open(x)
        x = self.transforms(x)
        return index, x, (y, s)

    def __getattr__(self, name):
        try:
            return super().__getattribute__('dataset').__getattribute__(name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


def data_transforms_fmow(conf={}):
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
