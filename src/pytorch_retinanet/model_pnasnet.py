import torch
import torch.utils.model_zoo as model_zoo

from pretrainedmodels.models import pnasnet
from .model import RetinaNet, RetinaNetEncoder


class PNasnetEncoder(RetinaNetEncoder):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = pnasnet.PNASNet5Large(num_classes=1001)

        self.fpn_sizes = [270, 1080, 2160, 4320]
        print(self.fpn_sizes)

    def forward(self, inputs):
        x = torch.cat([inputs, inputs, inputs], dim=1)
        x_conv_0 = self.encoder.conv_0(x)
        x_stem_0 = self.encoder.cell_stem_0(x_conv_0)
        x_stem_1 = self.encoder.cell_stem_1(x_conv_0, x_stem_0)
        x_cell_0 = self.encoder.cell_0(x_stem_0, x_stem_1)
        x_cell_1 = self.encoder.cell_1(x_stem_1, x_cell_0)
        x_cell_2 = self.encoder.cell_2(x_cell_0, x_cell_1)
        x_cell_3 = self.encoder.cell_3(x_cell_1, x_cell_2)
        x_cell_4 = self.encoder.cell_4(x_cell_2, x_cell_3)
        x_cell_5 = self.encoder.cell_5(x_cell_3, x_cell_4)
        x_cell_6 = self.encoder.cell_6(x_cell_4, x_cell_5)
        x_cell_7 = self.encoder.cell_7(x_cell_5, x_cell_6)
        x_cell_8 = self.encoder.cell_8(x_cell_6, x_cell_7)
        x_cell_9 = self.encoder.cell_9(x_cell_7, x_cell_8)
        x_cell_10 = self.encoder.cell_10(x_cell_8, x_cell_9)
        x_cell_11 = self.encoder.cell_11(x_cell_9, x_cell_10)

        # print(x_stem_0.shape, x_cell_3.shape, x_cell_7.shape, x_cell_11.shape)

        return x_stem_0, x_cell_3, x_cell_7, x_cell_11


def pnasnet5large(num_classes, pretrained=True, dropout=0.0):
    """Constructs a DPN model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    encoder = PNasnetEncoder()

    if pretrained:
        encoder.encoder.load_state_dict(model_zoo.load_url(
            pnasnet.pretrained_settings['pnasnet5large']['imagenet+background']['url'], model_dir='models'), strict=False)

    model = RetinaNet(encoder=encoder, num_classes=num_classes, dropout_cls=dropout, dropout_global_cls=dropout)
    return model
