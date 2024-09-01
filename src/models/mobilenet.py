from functools import partial

import torch
from timm.models.efficientnet import mobilenetv2_100
from torch import nn


def mobilenetv2(num_classes, pretrained_path='pretrained_ckpt/pt_mobilenetv2_100-4065ukrp/pretrained_imagenet.pth',
                return_features=False, force_num_classes=False):
    norm_layer = partial(torch.nn.GroupNorm, num_groups=8, eps=1e-6)
    model = mobilenetv2_100(pretrained=False, norm_layer=norm_layer, drop_rate=0.2, drop_path_rate=0.2)
    # force this size to match pretrained size
    model.classifier = torch.nn.Linear(in_features=1280, out_features=2028 if not force_num_classes else num_classes)
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location='cpu')
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            model.load_state_dict(state_dict['server_model_state_dict'], strict=True)
        print("Reloading pretrained weights from", pretrained_path)
    model.classifier = torch.nn.Linear(in_features=1280, out_features=num_classes)
    if return_features:
        modules = list(model.children())[:-1]
        model = torch.nn.Sequential(*modules)
    model.featurizer = nn.Sequential(*list(model.children())[:-1])
    return model
