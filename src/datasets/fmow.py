
from torchvision.datasets import VisionDataset
from src.datasets.dataset_utils import SubsetDataset, SubpopDataset
from PIL import Image
import torchvision
import pandas as pd
import os

class FMOW(VisionDataset, SubpopDataset):
    def __init__(
        self,
        root="./datasets",
        train=True,
        transforms=None,
        num_targets = 36,
        num_groups = 2,
        categories = None,
        regions = None
    ) -> None:
        dataset, regions, categories = get_dataset(root_dir=root, train=train,
                                                 num_targets=num_targets, num_groups=num_groups,
                                                 categories=categories,regions=regions)
        self.metadata = {"regions":regions, "categories":categories}
        self.subset = dataset[["path","category","region"]].to_numpy()
        self.root = root
        self.transforms = transforms

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


def get_dataset(root_dir, train=True, num_targets=32, num_groups=2, categories=None, regions=None):
    rgb_path = os.path.join(root_dir,"fmow_v1.1","rgb_metadata.csv")
    if not os.path.exists(rgb_path):
        print("Dataset missing")
        download_dataset(root_dir)
    
    rgb_data = pd.read_csv(os.path.join(root_dir,"fmow_v1.1","rgb_metadata.csv"))
    ccmap = pd.read_csv(os.path.join(root_dir,"fmow_v1.1","country_code_mapping.csv"))
    if train:
        rgb_data = rgb_data[rgb_data.split=="train"][["split","country_code","category"]]
    else:
        rgb_data = rgb_data[rgb_data.split=="test"][["split","country_code","category"]]
    df = pd.merge(left=rgb_data, right=ccmap, left_on="country_code", right_on="alpha-3")[["split","country_code","category","region"]]
    assert len(categories)>=num_targets
    assert len(regions)>=num_groups
    categories = categories[:num_targets]
    regions = regions[:num_groups]
    df = df[df.region.isin(regions)&df.category.isin(categories)]
    if not train:
        # Balanced sampling for testing
        df = df.groupby(["category","region"]).sample(100, random_state=1)
        assert len(df)==100*len(categories)*len(regions)

    df = df.sample(frac=1, random_state=1)
    df = df.drop(columns=["split"]).reset_index().rename(columns={"index":"path"})
    df["path"] = df["path"].astype(str)
    df["path"] = root_dir+"/fmow_v1.1/images/rgb_img_"+df["path"]+".png"
    return df, regions, categories


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


def download_dataset(root_path):
    """
    Download FMOW dataset into 'fmow_v1.1:
    Put images in a single folder images/ with filenames: rgb_img_<ID>.png
    create rgb_metadata.csv with image ID as index and required columns: ["split","country_code","category"]
    create country_code_mapping.csv with required columns: ["alpha-3","region"]
    """
    raise NotImplementedError("Dataset download not implemented!")