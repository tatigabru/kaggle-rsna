from collections import OrderedDict

import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo

from pretrainedmodels.models import senet
from torchsummary import summary
from .model import RetinaNet


class SeResNetXtEncoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        block = senet.SEResNeXtBottleneck
        # layers = [3, 4, 23, 3]
        groups = 32
        reduction = 16
        inplanes = 64
        downsample_kernel_size = 1
        downsample_padding = 0

        self.inplanes = inplanes

        layer0_modules = [
            ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', nn.BatchNorm2d(inplanes)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True))
        ]
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))

        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )

        self.fpn_sizes = [
            self.layer1[layers[0] - 1].conv3.out_channels,
            self.layer2[layers[1] - 1].conv3.out_channels,
            self.layer3[layers[2] - 1].conv3.out_channels,
            self.layer4[layers[3] - 1].conv3.out_channels
        ]

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        img_batch = inputs

        x = torch.cat([img_batch, img_batch, img_batch], dim=1)
        x = self.layer0(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x1, x2, x3, x4


def se_resnext101(num_classes, pretrained=False, dropout=0.5, fold=0):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    encoder = SeResNetXtEncoder(layers=[3, 4, 23, 3])
    if pretrained == 'imagenet':
        encoder.load_state_dict(model_zoo.load_url(
            senet.pretrained_settings['se_resnext101_32x4d']['imagenet']['url'], model_dir='models'), strict=False)

    if pretrained == 'nih':
        weights = {
            0: 'checkpoints/pretrained/se_resnext101_nih_dr0_fold_0/se_resnext101_nih_dr0_006.pt',
            1: 'checkpoints/pretrained/se_resnext101_nih_dr0_fold_1/se_resnext101_nih_dr0_004.pt',
            2: 'checkpoints/pretrained/se_resnext101_nih_dr0_fold_2/se_resnext101_nih_dr0_004.pt',
            3: 'checkpoints/pretrained/se_resnext101_nih_dr0_fold_3/se_resnext101_nih_dr0_004.pt',
        }
        print('load', weights[fold])
        encoder.load_state_dict(torch.load(weights[fold]), strict=False)

    model = RetinaNet(encoder=encoder, num_classes=num_classes, dropout_cls=dropout, dropout_global_cls=dropout)
    return model


def se_resnext50(num_classes, pretrained=False, dropout=0.5):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    encoder = SeResNetXtEncoder(layers=[3, 4, 6, 3])
    if pretrained == 'imagenet':
        encoder.load_state_dict(model_zoo.load_url(
            senet.pretrained_settings['se_resnext50_32x4d']['imagenet']['url'], model_dir='models'), strict=False)

    model = RetinaNet(encoder=encoder, num_classes=num_classes, dropout_cls=dropout, dropout_global_cls=dropout)
    return model

# if __name__ == '__main__':
# encoder = SeResNetXtEncoder()
# encoder.cuda()
# summary(encoder, (1, 512, 512))
