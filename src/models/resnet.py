import torch.nn as nn
from timm.models import resnet50_gn
from torchvision.models import  resnet50, ResNet50_Weights


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

def get_resnet50(num_classes, norm='gn'):
    if norm=='gn':
        return get_resnet50_gn(num_classes)
    elif norm=='bn':
        return get_resnet50_bn(num_classes)
    else:
        raise NotImplementedError("not recognized norm mode")

def replace_batchnorm_with_groupnorm(model, num_groups=4):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            setattr(model, name, nn.GroupNorm(num_groups=num_groups, num_channels=module.num_features))
        else:
            replace_batchnorm_with_groupnorm(module, num_groups)


def set_track_running_stats_false(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
