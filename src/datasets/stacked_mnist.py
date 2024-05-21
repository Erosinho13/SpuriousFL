import torch
import random
import numpy as np

from collections import defaultdict

from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from PIL import Image
from global_variables import DATASETS_ROOT
from torchvision.transforms.functional import normalize


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


def _data_transforms_mnist(mean, std, norm=True):
    train_tr_list = [
        v2.Pad(padding=2),
        v2.RandomResize(32, 40),
        v2.RandomCrop((32, 32)),
        v2.ToTensor()
    ]
    test_tr_list = [
        v2.Pad(padding=2),
        v2.ToTensor()
    ]
    if norm:
        train_tr_list.append(v2.Normalize(mean=mean, std=std))
        test_tr_list.append(v2.Normalize(mean=mean, std=std))
    return v2.Compose(train_tr_list), v2.Compose(test_tr_list)


class StackedMNIST(MNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, num_images: int = 60000,
                 dirichlet_groups_alpha: float = 1.0, num_classes: int = 120, max_num_groups: int = 6,
                 prevent_class_shuffling: bool = False, prevent_group_shuffling: bool = False):
        super().__init__(root=root, train=train, transform=transform, target_transform=target_transform,
                         download=download)

        assert dirichlet_groups_alpha >= 0
        assert 2 <= num_classes <= 120
        assert 1 <= max_num_groups <= 6

        self.num_images = num_images
        self.num_classes = num_classes
        self.max_num_groups = max_num_groups

        # original: original stacked MNIST label, integer (123 != 321)
        # unordered: unordered stacked MNIST label, human friendly, string (123 = 321, label: '123')
        # target: integer associated to each unordered label, from 0 to max number of combinations - 1
        self.target_to_unordered = {}
        self.unordered_to_target = {}
        self.target_to_original = defaultdict(lambda: [])
        self.original_to_target = {}
        self.map_targets(prevent_class_shuffling, prevent_group_shuffling)  # fill the dictionaries above

        # array to count the images per class in the test set, needed to build a balanced test set
        self.last_test_original_id = np.zeros(len(self.original_to_target.keys())).astype(np.uint8)

        # divide original mnist data per class
        self.mnist_label_to_img = [[] for _ in range(10)]
        for img, label in zip(self.data, self.targets):
            self.mnist_label_to_img[label].append(img)

        self.index = []  # list of pairs (class, id) of images from self.mnist_label_to_img

        # get dirichlet probabilities to sample the groups
        proportions = None
        if dirichlet_groups_alpha > 0:
            proportions = np.random.dirichlet(np.repeat(dirichlet_groups_alpha, self.max_num_groups))

        id_label = 0
        for i in range(self.num_images):

            # sample a group, i.e. original (stacked mnist) label
            if dirichlet_groups_alpha == 0:  # each class is associated with one single group
                original_label = self.target_to_original[id_label][0]
            else:
                if train:
                    original_label = \
                        random.choices(self.target_to_original[id_label][:self.max_num_groups], proportions, k=1)[0]
                else:
                    original_label = self.target_to_original[id_label][self.last_test_original_id[id_label]]
                    self.last_test_original_id[id_label] += 1
                    if self.last_test_original_id[id_label] == self.max_num_groups:
                        self.last_test_original_id[id_label] = 0

            # randomly sample 3 images for the corresponding original_label original mnist classes
            original_label_list = [int(i) for i in list(str(original_label).zfill(3))]  # list of original label digits
            img_id_list = []  # list of ids of original mnist images for the current RGB stacked mnist image
            for j in original_label_list:
                img_id_list.append(random.randint(0, len(self.mnist_label_to_img[j]) - 1))

            self.index.append((
                (original_label_list[0], img_id_list[0]), # R channel
                (original_label_list[1], img_id_list[1]), # G channel
                (original_label_list[2], img_id_list[2])  # B channel
            ))

            id_label += 1
            if id_label == self.num_classes:  # max_num_classes = Bin(10, 3) = 120
                id_label = 0

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        img = np.zeros((28, 28, 3), dtype=np.uint8)
        target = 0
        for i in range(3):
            idx = self.index[index]
            img_, target_ = self.mnist_label_to_img[idx[i][0]][idx[i][1]], idx[i][0]
            img[:, :, i] = img_
            target += target_ * 10 ** (2 - i)
        original_target = target
        target = self.original_to_target[target]

        img = Image.fromarray(img, mode="RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, (target, original_target)

    def mean(self):
        return np.round(self.data.float().mean(axis=(0, 1, 2)) / 255, 4)

    def std(self):
        return np.round(self.data.float().std(axis=(0, 1, 2)) / 255, 4)

    def map_targets(self, prevent_class_shuffling, prevent_group_shuffling):
        new = 0
        original_classes = list(range(1000))
        if not prevent_class_shuffling:
            random.shuffle(original_classes)
        for i in original_classes:
            list_unordered_target = [str(j) for j in sorted([int(i) for i in list(str(i).zfill(3))])]
            if len(list_unordered_target) != len(set(list_unordered_target)):
                continue  # ignore values with digits repetition, e.g. 133
            str_unordered_target = ''.join(list_unordered_target)
            if str_unordered_target not in set(self.target_to_unordered.values()) and new < self.num_classes:
                self.target_to_unordered[new] = str_unordered_target
                self.unordered_to_target[str_unordered_target] = new
                new += 1
            if str_unordered_target in self.unordered_to_target.keys():
                self.target_to_original[self.unordered_to_target[str_unordered_target]].append(i)
                self.original_to_target[i] = self.unordered_to_target[str_unordered_target]
        self.target_to_original = dict(self.target_to_original)
        if not prevent_group_shuffling:
            for v in self.target_to_original.values():  # shuffle the groups
                random.shuffle(v)


class Denormalize(object):

    def __init__(self, mean, std):
        if mean is not None and std is not None:
            self._mean = -np.array(mean) / std
            self._std = 1 / np.array(std)

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1, 1, 1)) / self._std.reshape(-1, 1, 1)
        return normalize(tensor, self._mean, self._std)


def plot_sample_image(sample_image):
    denorm = Denormalize(mean=mean, std=std)
    img = (denorm(sample_image[0]) * 255).permute(1, 2, 0).numpy().astype(np.uint8)
    fig, ax = plt.subplots()
    ax.imshow(v2.ToPILImage()(img))
    ax.margins(0)
    plt.axis('off')
    plt.show()


def count_img(ds):
    # The groups are the original labels. This dict is structured in this way: {012: {012: 3, 021: 6, ...}}
    count_images_per_groups_per_label = defaultdict(lambda: {})

    for _, (target, original_target) in ds:
        unordered_target = ds.target_to_unordered[target]
        if original_target not in count_images_per_groups_per_label[unordered_target].keys():
            count_images_per_groups_per_label[unordered_target][original_target] = 0
        else:
            count_images_per_groups_per_label[unordered_target][original_target] += 1

    return count_images_per_groups_per_label


if __name__ == '__main__':

    seed = 0
    sample_id = 25
    dirichlet_groups_alpha = 1.0
    num_train_images = 60000
    num_test_images = 10000
    num_classes = 2
    max_num_groups = 2
    prevent_class_shuffling = False
    prevent_group_shuffling = True

    set_seed(seed)
    mean = (0.1307, 0.1307, 0.1307)
    std = (0.3081, 0.3081, 0.3081)
    train_transform, test_transform = _data_transforms_mnist(mean=mean, std=std, norm=True)
    train_data = StackedMNIST(root=DATASETS_ROOT, train=True, download=True, transform=train_transform,
                              num_images=num_train_images, dirichlet_groups_alpha=dirichlet_groups_alpha,
                              num_classes=num_classes, max_num_groups=max_num_groups,
                              prevent_class_shuffling=prevent_class_shuffling,
                              prevent_group_shuffling=prevent_group_shuffling)
    test_data = StackedMNIST(root=DATASETS_ROOT, train=False, download=True, transform=test_transform,
                             num_images=num_test_images, num_classes=num_classes, max_num_groups=max_num_groups)

    sample_image = train_data[sample_id]
    print(f"Target: {sample_image[1][0]}, "
          f"Unordered_target: {train_data.target_to_unordered[sample_image[1][0]]}, "
          f"Original_target: {sample_image[1][1]}")
    # R_channel = torch.Tensor.numpy(sample_image[0])[0]
    # G_channel = torch.Tensor.numpy(sample_image[0])[1]
    # B_channel = torch.Tensor.numpy(sample_image[0])[2]
    plot_sample_image(sample_image)

    count_images_per_groups_per_label_train = count_img(train_data)
    count_images_per_groups_per_label_test = count_img(test_data)
    pass
