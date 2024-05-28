import random
import numpy as np

from matplotlib import pyplot as plt, gridspec
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from PIL import Image
from torchvision.transforms.functional import normalize

from src.utils import set_seed


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
                 force_groups_proportions: list[float] = None):

        super().__init__(root=root, train=train, transform=transform, target_transform=target_transform,
                         download=download)

        assert dirichlet_targets_alpha >= 0
        assert dirichlet_groups_alpha >= 0
        assert 2 <= num_targets <= 100
        assert 1 <= num_groups <= 10
        if force_targets_proportions is not None:
            assert len(force_targets_proportions) == num_targets
            assert sum(force_targets_proportions) == 1.0
        if force_groups_proportions is not None:
            assert len(force_groups_proportions) == num_groups
            assert sum(force_groups_proportions) == 1.0

        num_original_mnist_targets = 10
        self.mnist_target_to_img = self.get_mnist_images_by_target(num_original_mnist_targets)

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

        self.index = []  # list of 3 RGB pairs (class, id) of images from self.mnist_target_to_img

        # get dirichlet proportions to sample each (target, group) couple
        proportions = self.dirichlet_proportions(self.dirichlet_targets_alpha,
                                                 self.dirichlet_groups_alpha,
                                                 self.num_targets,
                                                 self.num_groups,
                                                 uniform_targets=self.uniform_targets,
                                                 uniform_groups=self.uniform_groups)
        num_images_per_tg_pair = self.num_images * proportions

        target = 0
        group_id = np.zeros(self.num_targets, dtype=np.uint8)
        target_id = np.zeros(self.num_groups, dtype=np.uint8)

        for i in range(self.num_images):
            # sample a group, i.e. original (stacked mnist) target
            if dirichlet_groups_alpha == 0:  # each class is associated with one single group
                group = self.groups[0]
            elif not train or self.force_balanced_dataset:
                group = self.groups[group_id[target]]
            else:
                group = random.choices(self.groups[:self.num_groups], groups_proportions, k=1)[0]

            # sample a target, i.e. new (stacked mnist) target
            if dirichlet_targets_alpha == 0:  # each class is associated with one single target
                target = self.targets[0]
            elif not train or self.force_balanced_dataset:
                target = self.targets[target_id[group]]
            else:
                target = random.choices(self.targets[:self.num_targets], targets_proportions, k=1)[0]

            # randomly sample 3 images for the corresponding group and target
            # list of original target digits
            original_target_list = [group] + [int(i) for i in list(str(self.targets[target]).zfill(2))]
            img_id_list = []  # list of ids of original mnist images for the current RGB stacked mnist image
            for j in original_target_list:
                img_id_list.append(random.randint(0, len(self.mnist_target_to_img[j]) - 1))

            self.index.append((
                (original_target_list[0], img_id_list[0]), # R channel
                (original_target_list[1], img_id_list[1]), # G channel
                (original_target_list[2], img_id_list[2])  # B channel
            ))

            group_id[target] += 1
            if group_id[target] == self.num_groups:
                group_id[target] = 0

            target_id[group] += 1
            if target_id[group] == self.num_targets:
                target_id[group] = 0

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):

        img = np.zeros((28, 28, 3), dtype=np.uint8)

        red_img = self.mnist_target_to_img[self.index[index][0][0]][self.index[index][0][1]]
        group = self.index[index][0][0]

        green_img = self.mnist_target_to_img[self.index[index][1][0]][self.index[index][1][1]]
        blue_img = self.mnist_target_to_img[self.index[index][2][0]][self.index[index][2][1]]
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

    def mean(self):
        return np.round(self.data.float().mean(axis=(0, 1, 2)) / 255, 4)

    def std(self):
        return np.round(self.data.float().std(axis=(0, 1, 2)) / 255, 4)

    def get_mnist_images_by_target(self, num_original_mnist_targets):
        # divide original mnist data per class
        mnist_images_by_target = [[] for _ in range(num_original_mnist_targets)]
        for img, target in zip(self.data, self.targets):
            mnist_images_by_target[target].append(img)
        return mnist_images_by_target

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
            dirichlet = 1/num_elements_2 * np.ones(num_elements_2)
            dirichlet = np.tile(dirichlet, (num_elements_1, 1))
        elif alpha == 0:
            dirichlet = np.zeros((num_elements_1, num_elements_2))
            dirichlet[:, 0] = np.ones(num_elements_1)
        else:
            dirichlet = np.tile(np.random.dirichlet(np.repeat(alpha, num_elements_2)), (num_elements_1, 1))
        return dirichlet

    def dirichlet_proportions(self, targets_alpha, groups_alpha, num_targets, num_groups,
                              uniform_targets=False, uniform_groups=False):
        target_dirichlet = self.get_dirichlet_distribution(targets_alpha, num_groups, num_targets,
                                                           uniform=uniform_targets)
        group_dirichlet = self.get_dirichlet_distribution(groups_alpha, num_targets, num_groups,
                                                          uniform=uniform_groups)
        proportions = target_dirichlet.T * group_dirichlet
        return proportions


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

    num_targets = 6
    num_groups = 3
    dirichlet_targets_alpha = 10.0
    dirichlet_groups_alpha = 0.0
    uniform_targets = False
    uniform_groups = True
    force_targets = None  # [23, 45]
    force_groups = None  # [0, 1]
    prevent_targets_shuffling = False
    prevent_groups_shuffling = False
    force_targets_proportions=None
    force_groups_proportions=None

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
        force_groups_proportions=force_groups_proportions
    )
    # test_data = StackedMNIST(root=root, train=False, transform=test_transform, download=True,
    #                          num_images=num_test_images, dirichlet_targets_alpha=dirichlet_targets_alpha,
    #                          dirichlet_groups_alpha=dirichlet_groups_alpha, num_targets=num_targets,
    #                          num_groups=num_groups, prevent_targets_shuffling=prevent_targets_shuffling,
    #                          prevent_groups_shuffling=prevent_groups_shuffling, force_targets=targets,
    #                          force_groups=groups, force_balanced_dataset=force_balanced_dataset)
    #
    # sample_image = train_data[sample_id]
    # print(f"Target: {train_data.targets[sample_image[1][0]]}, "
    #       f"Group: {train_data.groups[sample_image[1][1]]}, "
    #       f"Original target: {train_data.groups[sample_image[1][1]]}"
    #       f"{str(train_data.targets[sample_image[1][0]]).zfill(2)}, "
    #       f"Target id: {sample_image[1][0]}, "
    #       f"Group id: {sample_image[1][1]}, ")
    # # R_channel = torch.Tensor.numpy(sample_image[0])[0]
    # # G_channel = torch.Tensor.numpy(sample_image[0])[1]
    # # B_channel = torch.Tensor.numpy(sample_image[0])[2]
    # plot_sample_image(sample_image)
    #
    # count_images_per_groups_per_target_train = count_img(train_data)
    # count_images_per_groups_per_target_test = count_img(test_data)
    #
    # plot_heatmap(count_images_per_groups_per_target_train, train_data, 'Train dataset',
    #              write_annotations=write_annotations)
    # plot_heatmap(count_images_per_groups_per_target_test, test_data, 'Test dataset',
    #              write_annotations=write_annotations)


if __name__ == '__main__':
    main()
