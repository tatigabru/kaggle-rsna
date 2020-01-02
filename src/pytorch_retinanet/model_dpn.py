from collections import OrderedDict

import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo

from pretrainedmodels.models import dpn
from .model import RetinaNet, RetinaNetEncoder


class DPNEncoder(RetinaNetEncoder):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = dpn.DPN(**kwargs)
        layer_masks = ['conv2_', 'conv3_', 'conv4_', 'conv5_bn_ac']

        self.layer_names = [''] * 4
        self.fpn_sizes = [0] * 4

        for name, module in self.encoder.features.named_children():
            print(name)
            for level, mask in enumerate(layer_masks):
                if name.startswith(mask):
                    self.layer_names[level] = name
                    try:
                        self.fpn_sizes[level] = module.out_channels
                        print(level, self.fpn_sizes[level])
                    except AttributeError:
                        pass

        self.fpn_sizes = [336, 704, 1552, 2688]
        print(self.layer_names)
        print(self.fpn_sizes)

    def forward(self, inputs):
        results = [0] * 4

        img_batch = inputs
        x = torch.cat([img_batch, img_batch, img_batch], dim=1)

        for name, module in self.encoder.features.named_children():
            x = module(x)
            if name in self.layer_names:
                results[self.layer_names.index(name)] = x

        results = [
            torch.cat(x, dim=1) if isinstance(x, tuple) else x
            for x in results
        ]

        return results


def dpn92(num_classes, pretrained=True, **kwargs):
    """Constructs a DPN model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    encoder = DPNEncoder(
        num_init_features=64, k_r=96, groups=32,
        k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
        num_classes=1000, test_time_pool=True)

    if pretrained:
        encoder.encoder.load_state_dict(model_zoo.load_url(
            dpn.pretrained_settings['dpn92']['imagenet+5k']['url'], model_dir='models'), strict=False)

    model = RetinaNet(encoder=encoder, num_classes=num_classes, **kwargs)
    return model


# dpn92(num_classes=1, pretrained=True)
