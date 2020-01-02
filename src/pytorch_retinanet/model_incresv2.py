import torch
import torch.utils.model_zoo as model_zoo

from . import model_inc_resnet_v2_aligned
from .model import RetinaNet, RetinaNetEncoder


class InceptionResnetV2Encoder(RetinaNetEncoder):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = model_inc_resnet_v2_aligned.inceptionresnetv2(num_classes=1001, pretrained='imagenet+background')

        self.fpn_sizes = [192, 320, 1088, 1536]
        print(self.fpn_sizes)

    def forward(self, inputs):
        x = torch.cat([inputs, inputs, inputs], dim=1)
        res = []

        x = self.encoder.conv2d_1a(x)
        # print(x.shape)
        x = self.encoder.conv2d_2a(x)
        # print(x.shape)
        x = self.encoder.conv2d_2b(x)
        # print(x.shape)
        x = self.encoder.maxpool_3a(x)
        # print(x.shape)
        x = self.encoder.conv2d_3b(x)
        # print(x.shape)
        x = self.encoder.conv2d_4a(x)
        # print(x.shape)
        res.append(x)
        x = self.encoder.maxpool_5a(x)
        # print(x.shape)
        x = self.encoder.mixed_5b(x)
        # print(x.shape)
        x = self.encoder.repeat(x)
        # print(x.shape)
        res.append(x)
        x = self.encoder.mixed_6a(x)
        # print(x.shape)
        x = self.encoder.repeat_1(x)
        # print(x.shape)
        res.append(x)
        x = self.encoder.mixed_7a(x)
        # print(x.shape)
        x = self.encoder.repeat_2(x)
        # print(x.shape)
        x = self.encoder.block8(x)
        # print(x.shape)
        x = self.encoder.conv2d_7b(x)
        # print(x.shape)
        res.append(x)

        # print([r.shape for r in res])

        # print(x_stem_0.shape, x_cell_3.shape, x_cell_7.shape, x_cell_11.shape)
        return res


def inceptionresnetv2(num_classes, pretrained=True, **kwargs):
    """Constructs a DPN model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    encoder = InceptionResnetV2Encoder()

    if pretrained:
        encoder.encoder.load_state_dict(model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth',
                                                           model_dir='models'),
                                        strict=False)

    model = RetinaNet(encoder=encoder, num_classes=num_classes, **kwargs)
    return model
