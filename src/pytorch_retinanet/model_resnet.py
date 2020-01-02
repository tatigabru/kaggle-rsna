import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import time
import torch.utils.model_zoo as model_zoo
from .utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from .anchors import Anchors
from . import losses
from .lib.nms.pth_nms import pth_nms
from .model import RetinaNet, RetinaNetEncoder

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ResNetEncoder(RetinaNetEncoder):
    def __init__(self, block, layers):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            self.fpn_sizes = [
                self.layer1[layers[0]-1].conv2.out_channels,
                self.layer2[layers[1]-1].conv2.out_channels,
                self.layer3[layers[2]-1].conv2.out_channels,
                self.layer4[layers[3]-1].conv2.out_channels
            ]
        elif block == Bottleneck:
            self.fpn_sizes = [
                self.layer1[layers[0]-1].conv3.out_channels,
                self.layer2[layers[1]-1].conv3.out_channels,
                self.layer3[layers[2]-1].conv3.out_channels,
                self.layer4[layers[3]-1].conv3.out_channels
            ]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        img_batch = inputs

        x = torch.cat([img_batch, img_batch, img_batch], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x1, x2, x3, x4


def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    encoder = ResNetEncoder(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        encoder.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='models'), strict=False)
    model = RetinaNet(encoder=encoder, num_classes=num_classes)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    encoder = ResNetEncoder(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        encoder.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='models'), strict=False)
    model = RetinaNet(encoder=encoder, num_classes=num_classes)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    encoder = ResNetEncoder(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        encoder.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='models'), strict=False)

    model = RetinaNet(encoder=encoder, num_classes=num_classes, **kwargs)
    return model

def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    encoder = ResNetEncoder(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        encoder.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='models'), strict=False)

    model = RetinaNet(encoder=encoder, num_classes=num_classes, **kwargs)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    encoder = ResNetEncoder(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        encoder.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='models'), strict=False)

    model = RetinaNet(encoder=encoder, num_classes=num_classes, **kwargs)
    return model
