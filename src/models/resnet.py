import torch.nn as nn
from timm.models import resnet50_gn
from torchvision.models import resnet50, ResNet50_Weights

from src.models.abstract_model import AbstractModel


class ResNet50(AbstractModel):
    def __init__(self, num_classes, norm="gn"):
        super(ResNet50, self).__init__()

        if norm == "gn":
            base_model = resnet50_gn(pretrained=True)
        elif norm == "bn":
            base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise NotImplementedError("not recognized norm mode")
        num_ftrs = base_model.fc.in_features
        base_model.fc = nn.Identity()
        self.featurizer = base_model
        self.classifier = nn.Linear(num_ftrs, num_classes)


def get_resnet50_gn(num_classes):
    model = resnet50_gn(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def get_resnet50_bn(num_classes):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def get_resnet50(num_classes, norm="gn"):
    return ResNet50(num_classes, norm)


def replace_batchnorm_with_groupnorm(model, num_groups=4):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            setattr(
                model,
                name,
                nn.GroupNorm(num_groups=num_groups, num_channels=module.num_features),
            )
        else:
            replace_batchnorm_with_groupnorm(module, num_groups)


def set_track_running_stats_false(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
