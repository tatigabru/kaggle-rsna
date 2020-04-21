# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 18:01:28 2019

Models

"""
from torch import nn

import pytorch_retinanet.dataloader
import pytorch_retinanet.model
import pytorch_retinanet.model_incresv2
import pytorch_retinanet.model_nasnet_mobile
import pytorch_retinanet.model_pnasnet
import pytorch_retinanet.model_resnet
import pytorch_retinanet.model_se_resnext
import pytorch_retinanet.model_xception
from config import IMG_SIZE


class ModelInfo:
    """
    Initialises main model parameters    
    """

    def __init__(self, factory: nn.Module, args: dict, batch_size: int, dataset_args: dict, use_sgd: bool=False, img_size: int=IMG_SIZE):
        """
        Args:
            factory : base model architectures
            args : a dictionary with model arguments
            dataset_args : a dictionary with model arguments    
            batch_size: batch size
            img_size: image size to use in training   
        """
        self.factory = factory
        self.args = args
        self.batch_size = batch_size
        self.dataset_args = dataset_args
        self.img_size = img_size
        self.use_sgd = use_sgd


# dictionary of models with parameters
MODELS = {
    "resnet101_512": ModelInfo(
        factory=pytorch_retinanet.model_resnet.resnet101,
        args=dict(num_classes=1, pretrained=True),
        img_size=512,
        batch_size=6,
        dataset_args=dict(augmentation_level=20),
    ),
    "resnet152_512": ModelInfo(
        factory=pytorch_retinanet.model_resnet.resnet152, 
        args=dict(num_classes=1, pretrained=True), 
        img_size=512, 
        batch_size=4, 
        dataset_args=dict(augmentation_level=20),
    ),
    "se_resnext101_512": ModelInfo(
        factory=pytorch_retinanet.model_se_resnext.se_resnext101,
        args=dict(num_classes=1, pretrained="imagenet"),
        img_size=512,
        batch_size=3,
        dataset_args=dict(),
    ),
    "se_resnext101_dr_512": ModelInfo(
        factory=pytorch_retinanet.model_se_resnext.se_resnext101,
        args=dict(num_classes=1, pretrained="imagenet", dropout=0.5),
        img_size=512,
        batch_size=4,
        dataset_args=dict(augmentation_level=20),
    ),
    "se_resnext101_dr0.75_512": ModelInfo(
        factory=pytorch_retinanet.model_se_resnext.se_resnext101,
        args=dict(num_classes=1, pretrained="imagenet", dropout=0.75),
        img_size=512,
        batch_size=6,
        dataset_args=dict(augmentation_level=20),
    ),
    'se_resnext101_dr0.75_512_aug1': ModelInfo(
        factory=pytorch_retinanet.model_se_resnext.se_resnext101,
        args=dict(num_classes=1, pretrained='imagenet', dropout=0.75),
        img_size=512,
        batch_size=6,
        dataset_args=dict(augmentation_level=1)
    ),
    'se_resnext101_dr0.75_512_aug10': ModelInfo(
        factory=pytorch_retinanet.model_se_resnext.se_resnext101,
        args=dict(num_classes=1, pretrained='imagenet', dropout=0.75),
        img_size=512,
        batch_size=6,
        dataset_args=dict(augmentation_level=10)
    ),
    'se_resnext101_dr0.75_512_basic_rotations': ModelInfo(
        factory=pytorch_retinanet.model_se_resnext.se_resnext101,
        args=dict(num_classes=1, pretrained='imagenet', dropout=0.75),
        img_size=512,
        batch_size=6,
        dataset_args=dict(augmentation_level=10)
    ),
    'se_resnext101_dr0.75_512_aug21': ModelInfo(
        factory=pytorch_retinanet.model_se_resnext.se_resnext101,
        args=dict(num_classes=1, pretrained='imagenet', dropout=0.75),
        img_size=512,
        batch_size=6,
        dataset_args=dict(augmentation_level=21)
    ),
    "se_resnext101_dr_512_without_pretrained": ModelInfo(
        factory=pytorch_retinanet.model_se_resnext.se_resnext101,
        args=dict(num_classes=1, pretrained=False, dropout=0.5),
        img_size=512,
        batch_size=4,
        dataset_args=dict(augmentation_level=20),
    ),
    "se_resnext101_512_bs12": ModelInfo(
        factory=pytorch_retinanet.model_se_resnext.se_resnext101,
        args=dict(num_classes=1, pretrained="imagenet"),
        img_size=512,
        batch_size=12,
        dataset_args=dict(),
    ),
    "se_resnext101_512_bs12_aug20": ModelInfo(
        factory=pytorch_retinanet.model_se_resnext.se_resnext101,
        args=dict(num_classes=1, pretrained="imagenet"),
        img_size=512,
        batch_size=12,
        dataset_args=dict(augmentation_level=20),
    ),
    "se_resnext101_512_sgd": ModelInfo(
        factory=pytorch_retinanet.model_se_resnext.se_resnext101,
        args=dict(num_classes=1, pretrained="imagenet", dropout=0.5),
        img_size=512,
        batch_size=4,
        use_sgd=True,
        dataset_args=dict(augmentation_level=15),
    ),
    "se_resnext101_256": ModelInfo(
        factory=pytorch_retinanet.model_se_resnext.se_resnext101,
        args=dict(num_classes=1, pretrained="imagenet"),
        img_size=256,
        batch_size=12,
        dataset_args=dict(),
    ),
    "resnet34_256": ModelInfo(
        factory=pytorch_retinanet.model_resnet.resnet34, 
        args=dict(num_classes=1, pretrained=True), 
        img_size=256, 
        batch_size=32, 
        dataset_args=dict(),
    ),
    "pnas_512": ModelInfo(
        factory=pytorch_retinanet.model_pnasnet.pnasnet5large,
        args=dict(num_classes=1, pretrained=True),
        img_size=512,
        batch_size=4,
        dataset_args=dict(),
    ),
    "pnas_512_dr": ModelInfo(
        factory=pytorch_retinanet.model_pnasnet.pnasnet5large,
        args=dict(num_classes=1, pretrained=True, dropout=0.5),
        img_size=512,
        batch_size=2,
        dataset_args=dict(augmentation_level=20),
    ),
    "pnas_512_bs12": ModelInfo(
        factory=pytorch_retinanet.model_pnasnet.pnasnet5large,
        args=dict(num_classes=1, pretrained=True),
        img_size=512,
        batch_size=8,
        dataset_args=dict(),
    ),
    "pnas_256_aug20": ModelInfo(
        factory=pytorch_retinanet.model_pnasnet.pnasnet5large,
        args=dict(num_classes=1, pretrained=True),
        img_size=256,
        batch_size=8,
        dataset_args=dict(augmentation_level=20),
    ),
    "inc_resnet_v2_512": ModelInfo(
        factory=pytorch_retinanet.model_incresv2.inceptionresnetv2,
        args=dict(num_classes=1, pretrained=True),
        img_size=512,
        batch_size=4,
        dataset_args=dict(augmentation_level=20),
    ),
    "inc_resnet_v2_512_dr": ModelInfo(
        factory=pytorch_retinanet.model_incresv2.inceptionresnetv2,
        args=dict(num_classes=1, pretrained=True, dropout_cls=0.6, dropout_global_cls=0.6),
        img_size=512,
        batch_size=4,
        dataset_args=dict(augmentation_level=20),
    ),
    "inc_resnet_v2_256": ModelInfo(
        factory=pytorch_retinanet.model_incresv2.inceptionresnetv2,
        args=dict(num_classes=1, pretrained=True),
        img_size=256,
        batch_size=16,
        dataset_args=dict(augmentation_level=20),
    ),
    "resnet50_512": ModelInfo(
        factory=pytorch_retinanet.model_resnet.resnet50,
        args=dict(num_classes=1, pretrained=True, dropout_cls=0.5, dropout_global_cls=0.5),
        img_size=512,
        batch_size=12,
        dataset_args=dict(augmentation_level=15),
    ),
    "se_resnext50_512": ModelInfo(
        factory=pytorch_retinanet.model_se_resnext.se_resnext50,
        args=dict(num_classes=1, pretrained="imagenet", dropout=0.5),
        img_size=512,
        batch_size=8,
        dataset_args=dict(augmentation_level=20),
    ),
    "se_resnext50_512_dr0.8": ModelInfo(
        factory=pytorch_retinanet.model_se_resnext.se_resnext50,
        args=dict(num_classes=1, pretrained="imagenet", dropout=0.8),
        img_size=512,
        batch_size=8,
        dataset_args=dict(augmentation_level=20),
    ),
    "nasnet_mobile_512": ModelInfo(
        factory=pytorch_retinanet.model_nasnet_mobile.nasnet_mobile_model,
        args=dict(num_classes=1, pretrained=True, dropout_cls=0.5, dropout_global_cls=0.5, use_l2_features=True),
        img_size=512,
        batch_size=8,
        dataset_args=dict(augmentation_level=20),
    ),
    "xception_512_dr": ModelInfo(
        factory=pytorch_retinanet.model_xception.xception_model,
        args=dict(num_classes=1, pretrained=True, dropout_cls=0.6, dropout_global_cls=0.6),
        img_size=512,
        batch_size=6,
        dataset_args=dict(augmentation_level=20),
    ),
}
