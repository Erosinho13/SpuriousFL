import math
import random
import numpy as np

from matplotlib import pyplot as plt, gridspec
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from PIL import Image
from torchvision.transforms.functional import normalize

from src.utils import set_seed


def get_mnist_images_by_target(data, targets):
    # divide original mnist data per class
    mnist_images_by_target = [[] for _ in range(10)]
    for img, target in zip(data, targets):
        mnist_images_by_target[target].append(img)
    return mnist_images_by_target


train_mnist_dataset = MNIST(root='datasets/', train=True, download=True)
train_mnist_target_to_img = get_mnist_images_by_target(train_mnist_dataset.data, train_mnist_dataset.targets)

test_mnist_dataset = MNIST(root='datasets/', train=False, download=True)
test_mnist_target_to_img = get_mnist_images_by_target(test_mnist_dataset.data, test_mnist_dataset.targets)


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

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 num_images: int = 60000,
                 num_targets: int = 100,
                 num_groups: int = 10,
                 prevent_targets_shuffling: bool = False,
                 prevent_groups_shuffling: bool = False,
                 dirichlet_targets_alpha: float = 1.0,
                 dirichlet_groups_alpha: float = 1.0,
                 uniform_targets: bool = False,
                 uniform_groups: bool = False,
                 force_targets: list[int] = None,
                 force_groups: list[int] = None,
                 force_targets_proportions: list[float] = None,
                 force_groups_proportions: list[float] = None,
                 force_proportions: list[list[float]] = None):

        super().__init__(root=root, train=train, transform=transform, target_transform=target_transform,
                         download=download)

        assert dirichlet_targets_alpha >= 0
        assert dirichlet_groups_alpha >= 0
        assert 2 <= num_targets <= 100
        assert 1 <= num_groups <= 10
        if force_proportions is not None:
            assert np.array(force_proportions).shape[0] == num_targets
            assert np.array(force_proportions).shape[1] == num_groups
            assert math.isclose(np.array(force_proportions).sum(), 1.0, rel_tol=1e-9)
            assert force_targets_proportions is None
            assert force_groups_proportions is None
        if force_targets_proportions is not None:
            assert len(force_targets_proportions) == num_targets
            assert math.isclose(sum(force_targets_proportions), 1.0, rel_tol=1e-9)
        if force_groups_proportions is not None:
            assert len(force_groups_proportions) == num_groups
            assert math.isclose(sum(force_groups_proportions), 1.0, rel_tol=1e-9)

        self.num_images = num_images
        self.num_targets = num_targets
        self.num_groups = num_groups
        self.dirichlet_targets_alpha = dirichlet_targets_alpha
        self.dirichlet_groups_alpha = dirichlet_groups_alpha
        self.uniform_targets = uniform_targets
        self.uniform_groups = uniform_groups
        self.force_targets_proportions = force_targets_proportions
        self.force_groups_proportions = force_groups_proportions

        self.targets, self.groups = self.get_targets_and_groups(prevent_targets_shuffling, prevent_groups_shuffling)
        self.update_targets_and_groups(force_targets, force_groups)
        self.target_ids = {value: idx for idx, value in enumerate(self.targets)}
        self.group_ids = {value: idx for idx, value in enumerate(self.groups)}

        self.index = []  # list of 3 RGB pairs (class, id) of images from mnist_target_to_img

        # get proportions to sample each (target, group) couple
        self.proportions = self.get_proportions(self.dirichlet_targets_alpha,
                                                self.dirichlet_groups_alpha,
                                                self.num_targets,
                                                self.num_groups,
                                                uniform_targets=self.uniform_targets,
                                                uniform_groups=self.uniform_groups,
                                                force_targets_proportions=force_targets_proportions,
                                                force_groups_proportions=force_groups_proportions,
                                                force_proportions=force_proportions)
        self.num_images_per_tg_pair = np.rint(self.num_images * self.proportions).astype(int)
        self.num_images_per_tg_pair = self.round_num_images(self.num_images_per_tg_pair, self.num_images)

        for i in range(self.num_images_per_tg_pair.shape[0]):
            for j in range(self.num_images_per_tg_pair.shape[1]):

                remaining_images = self.num_images_per_tg_pair[i, j]
                group = self.groups[j]
                target = self.targets[i]
                original_target_list = [group] + [int(i) for i in list(str(target).zfill(2))]
                selected_indices = set()  # prevents same images generation

                while remaining_images > 0:

                    while True:
                        img_id_list = []  # list of ids of original mnist images for the current RGB stacked mnist image
                        for k in original_target_list:
                            img_id_list.append(random.randint(0, len(train_mnist_target_to_img[k]) - 1))
                        if tuple(img_id_list) not in selected_indices:
                            selected_indices.add(tuple(img_id_list))
                            break

                    self.index.append((
                        (original_target_list[0], img_id_list[0]),  # R channel
                        (original_target_list[1], img_id_list[1]),  # G channel
                        (original_target_list[2], img_id_list[2])  # B channel
                    ))

                    remaining_images -= 1

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):

        img = np.zeros((28, 28, 3), dtype=np.uint8)

        red_img = train_mnist_target_to_img[self.index[index][0][0]][self.index[index][0][1]]
        group = self.index[index][0][0]

        green_img = train_mnist_target_to_img[self.index[index][1][0]][self.index[index][1][1]]
        blue_img = train_mnist_target_to_img[self.index[index][2][0]][self.index[index][2][1]]
        target = 10 * self.index[index][1][0] + self.index[index][2][0]

        img[:, :, 0] = red_img
        img[:, :, 1] = green_img
        img[:, :, 2] = blue_img

        img = Image.fromarray(img, mode="RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, (self.target_ids[target], self.group_ids[group])

    @property
    def mean(self):
        return np.round(self.data.float().mean(axis=(0, 1, 2)) / 255, 4)

    @property
    def std(self):
        return np.round(self.data.float().std(axis=(0, 1, 2)) / 255, 4)

    def get_targets_and_groups(self, _prevent_targets_shuffling, _prevent_groups_shuffling):
        _targets = list(range(100))
        _groups = list(range(10))
        if not _prevent_targets_shuffling:
            random.shuffle(_targets)
        if not _prevent_groups_shuffling:
            random.shuffle(_groups)
        _targets = _targets[:self.num_targets]
        _groups = _groups[:self.num_groups]
        return _targets, _groups

    def update_targets_and_groups(self, force_targets, force_groups):
        if force_targets is not None:
            assert len(force_targets) == self.num_targets
            assert 0 <= all(force_targets) < 100
            self.targets = force_targets

        if force_groups is not None:
            assert len(force_groups) == self.num_groups
            assert 0 <= all(force_groups) < 10
            self.groups = force_groups

    @staticmethod
    def get_dirichlet_distribution(alpha, num_elements_1, num_elements_2, uniform=False):
        if uniform:
            dirichlet = 1 / num_elements_2 * np.ones(num_elements_2)
            dirichlet = np.tile(dirichlet, (num_elements_1, 1))
        elif alpha == 0:
            dirichlet = np.zeros((num_elements_1, num_elements_2))
            dirichlet[:, 0] = np.ones(num_elements_1)
        else:
            dirichlet = np.tile(np.random.dirichlet(np.repeat(alpha, num_elements_2)), (num_elements_1, 1))
        return dirichlet

    def get_proportions(self, targets_alpha, groups_alpha, num_targets, num_groups, uniform_targets=False,
                        uniform_groups=False, force_targets_proportions=None, force_groups_proportions=None,
                        force_proportions=None):
        """

        """

        if force_proportions is not None and self.train:
            return np.array(force_proportions)

        if force_targets_proportions is None:
            target_proportions = self.get_dirichlet_distribution(targets_alpha, num_groups, num_targets,
                                                                 uniform=uniform_targets)
        else:
            target_proportions = np.tile(np.array(force_targets_proportions), (num_groups, 1))

        if force_groups_proportions is None:
            group_proportions = self.get_dirichlet_distribution(groups_alpha, num_targets, num_groups,
                                                                uniform=uniform_groups)
        else:
            group_proportions = np.tile(np.array(force_groups_proportions), (num_targets, 1))

        proportions = target_proportions.T * group_proportions

        return proportions

    @staticmethod
    def round_num_images(num_images_per_tg_pair, num_images):
        current_sum = np.sum(num_images_per_tg_pair)
        difference = num_images - current_sum

        while difference != 0:
            if difference > 0:
                i, j = np.unravel_index(np.random.randint(num_images_per_tg_pair.size), num_images_per_tg_pair.shape)
                num_images_per_tg_pair[i, j] += 1
                difference -= 1
            else:
                i, j = np.unravel_index(np.random.randint(num_images_per_tg_pair.size), num_images_per_tg_pair.shape)
                if num_images_per_tg_pair[i, j] > 0:  # Ensure we don't go below zero
                    num_images_per_tg_pair[i, j] -= 1
                    difference += 1

        return num_images_per_tg_pair


class Denormalize(object):

    def __init__(self, mean, std):
        if mean is not None and std is not None:
            self._mean = -np.array(mean) / std
            self._std = 1 / np.array(std)

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1, 1, 1)) / self._std.reshape(-1, 1, 1)
        return normalize(tensor, self._mean, self._std)


def plot_sample_image(sample_image, mean, std):
    denorm = Denormalize(mean=mean, std=std)
    img = (denorm(sample_image[0]) * 255).permute(1, 2, 0).numpy().astype(np.uint8)
    fig, ax = plt.subplots()
    ax.imshow(v2.ToPILImage()(img))
    ax.margins(0)
    plt.axis('off')
    plt.show()


def count_img(data):
    count_images_per_groups_per_target = np.zeros((data.num_groups, data.num_targets))
    for img, (target, group) in data:
        count_images_per_groups_per_target[group][target] += 1
    return count_images_per_groups_per_target


def plot_heatmap(count_images_per_groups_per_target, data, title, write_annotations=True):

    row_sums = count_images_per_groups_per_target.sum(axis=1)
    col_sums = count_images_per_groups_per_target.sum(axis=0)

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])

    ax0 = plt.subplot(gs[1, 0])
    im = ax0.imshow(count_images_per_groups_per_target, cmap='viridis', aspect='auto')

    ax0.set_xticks(np.arange(data.num_targets + 1) - 0.5, minor=True)
    ax0.set_yticks(np.arange(data.num_groups + 1) - 0.5, minor=True)
    ax0.grid(which='minor', color='white', linestyle='-', linewidth=1)
    ax0.tick_params(which='minor', size=0)

    ax0.set_xticks(np.arange(data.num_groups))
    ax0.set_xticklabels(data.groups)
    ax0.set_yticks(np.arange(data.num_targets))
    ax0.set_yticklabels(data.targets)
    ax0.set_xlabel('Targets')
    ax0.set_ylabel('Groups')

    if write_annotations:
        for i in range(data.num_groups):
            for j in range(data.num_targets):
                value = count_images_per_groups_per_target[i, j]
                color = 'white' if value > count_images_per_groups_per_target.max() / 2 else 'black'
                ax0.text(j, i, str(int(value)), ha='center', va='center', color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    ax1 = plt.subplot(gs[1, 1], sharey=ax0)
    ax1.barh(np.arange(data.num_groups), row_sums, align='center', color='blue')
    ax1.set_yticks(np.arange(data.num_groups))
    ax1.set_yticklabels(data.groups)
    ax1.set_xlabel('Tot images by groups')
    ax1.invert_yaxis()

    ax2 = plt.subplot(gs[0, 0], sharex=ax0)
    ax2.bar(np.arange(data.num_targets), col_sums, align='center', color='red')
    ax2.set_xticks(np.arange(data.num_targets))
    ax2.set_xticklabels(data.targets)
    ax2.set_ylabel('Tot images by target')

    plt.tight_layout()
    plt.title(title)
    plt.show()


def main():

    seed = 0
    root = '/home/efani/DATASETS'
    sample_id = 25
    num_train_images = 60000
    num_test_images = 10000

    num_targets = 2
    num_groups = 2
    dirichlet_targets_alpha = 10.0
    dirichlet_groups_alpha = 1.0
    uniform_targets = False
    uniform_groups = False
    force_targets = [23, 45]
    force_groups = [0, 1]
    prevent_targets_shuffling = False
    prevent_groups_shuffling = False

    force_targets_proportions = None  # [0.5, 0.5]
    force_groups_proportions = None  # [0.9, 0.1]
    force_proportions = [[0.45, 0.05], [0.05, 0.45]]

    write_annotations = True

    set_seed(seed)
    mean = (0.1307, 0.1307, 0.1307)
    std = (0.3081, 0.3081, 0.3081)
    train_transform, test_transform = _data_transforms_mnist(mean=mean, std=std, norm=True)

    train_data = StackedMNIST(
        root=root,
        train=True,
        transform=train_transform,
        download=True,
        num_images=num_train_images,
        num_targets=num_targets,
        num_groups=num_groups,
        prevent_targets_shuffling=prevent_targets_shuffling,
        prevent_groups_shuffling=prevent_groups_shuffling,
        dirichlet_targets_alpha=dirichlet_targets_alpha,
        dirichlet_groups_alpha=dirichlet_groups_alpha,
        uniform_targets=uniform_targets,
        uniform_groups=uniform_groups,
        force_targets=force_targets,
        force_groups=force_groups,
        force_targets_proportions=force_targets_proportions,
        force_groups_proportions=force_groups_proportions,
        force_proportions=force_proportions
    )
    test_data = StackedMNIST(
        root=root,
        train=False,
        transform=test_transform,
        download=True,
        num_images=num_test_images,
        num_targets=num_targets,
        num_groups=num_groups,
        prevent_targets_shuffling=prevent_targets_shuffling,
        prevent_groups_shuffling=prevent_groups_shuffling,
        uniform_targets=True,
        uniform_groups=True,
        force_targets=force_targets,
        force_groups=force_groups,
    )

    sample_image = train_data[sample_id]
    print(f"Target: {train_data.targets[sample_image[1][0]]}, "
          f"Group: {train_data.groups[sample_image[1][1]]}, "
          f"Original target: {train_data.groups[sample_image[1][1]]}"
          f"{str(train_data.targets[sample_image[1][0]]).zfill(2)}, "
          f"Target id: {sample_image[1][0]}, "
          f"Group id: {sample_image[1][1]}, ")
    # R_channel = torch.Tensor.numpy(sample_image[0])[0]
    # G_channel = torch.Tensor.numpy(sample_image[0])[1]
    # B_channel = torch.Tensor.numpy(sample_image[0])[2]
    plot_sample_image(sample_image, mean, std)

    count_images_per_groups_per_target_train = count_img(train_data)
    count_images_per_groups_per_target_test = count_img(test_data)

    plot_heatmap(count_images_per_groups_per_target_train, train_data, 'Train dataset',
                 write_annotations=write_annotations)
    plot_heatmap(count_images_per_groups_per_target_test, test_data, 'Test dataset',
                 write_annotations=write_annotations)


if __name__ == '__main__':
    main()