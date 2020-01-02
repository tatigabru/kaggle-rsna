import torch
import torch.utils.model_zoo as model_zoo

import pretrainedmodels.models
from pretrainedmodels.models import nasnet_mobile
from .model import RetinaNet, RetinaNetEncoder


class NasnetMobileEncoder(RetinaNetEncoder):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nasnet_mobile.NASNetAMobile(**kwargs)

        self.fpn_sizes = [44, 264, 528, 1056]
        print(self.fpn_sizes)

    def forward(self, inputs):
        x = torch.cat([inputs, inputs, inputs], dim=1)

        x_conv0 = self.encoder.conv0(x)
        x_stem_0 = self.encoder.cell_stem_0(x_conv0)
        x_stem_1 = self.encoder.cell_stem_1(x_conv0, x_stem_0)

        x_cell_0 = self.encoder.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.encoder.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.encoder.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.encoder.cell_3(x_cell_2, x_cell_1)

        x_reduction_cell_0 = self.encoder.reduction_cell_0(x_cell_3, x_cell_2)

        x_cell_6 = self.encoder.cell_6(x_reduction_cell_0, x_cell_3)
        x_cell_7 = self.encoder.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.encoder.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.encoder.cell_9(x_cell_8, x_cell_7)

        x_reduction_cell_1 = self.encoder.reduction_cell_1(x_cell_9, x_cell_8)

        x_cell_12 = self.encoder.cell_12(x_reduction_cell_1, x_cell_9)
        x_cell_13 = self.encoder.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.encoder.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.encoder.cell_15(x_cell_14, x_cell_13)

        # for name, val in locals().items():
        #     if name.startswith('x_cell_') or name.startswith('x_'):
        #         print(name, val.shape)
        #
        # print(x_stem_0.shape, x_cell_3.shape, x_cell_9.shape, x_cell_15.shape)

        return x_stem_0, x_cell_3, x_cell_9, x_cell_15


def nasnet_mobile_model(num_classes, pretrained=True, **kwargs):
    """Constructs a nasnet mobile model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    encoder = NasnetMobileEncoder(num_classes=1000)

    if pretrained:
        encoder.encoder.load_state_dict(
            model_zoo.load_url(
                pretrainedmodels.models.nasnet_mobile.pretrained_settings['nasnetamobile']['imagenet']['url'],
                model_dir='models'),
            strict=True)

    model = RetinaNet(encoder=encoder, num_classes=num_classes, **kwargs)
    return model
