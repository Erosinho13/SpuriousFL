import torch.nn as nn
from torchvision import  models

def get_resnet50(num_classes):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    set_track_running_stats_false(model)
    return model


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
