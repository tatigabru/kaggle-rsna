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

def nms(dets, thresh):
    "Dispatch to either CPU or GPU NMS implementations.\
    Accept dets as tensor"""
    return pth_nms(dets, thresh)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class PyramidFeatures(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, feature_size=256, use_l2_features=True):
        super(PyramidFeatures, self).__init__()
        self.use_l2_features = use_l2_features
        
        # upsample C5 to get P5 from the FPN paper
        self.P5_1           = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1           = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2           = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P3 elementwise to C2
        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):

        C2, C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)
        
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        P3_x = self.P3_2(P3_x)

        if self.use_l2_features:
            P2_x = self.P2_1(C2)
            P2_x = P2_x + P3_upsampled_x
            P2_x = self.P2_2(P2_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        if self.use_l2_features:
            return [P2_x, P3_x, P4_x, P5_x, P6_x, P7_x]
        else:
            return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors*4, kernel_size=3, padding=1)

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256, dropout=0.5):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors*num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

        self.dropout = dropout

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        if self.dropout > 0:
            out = F.dropout(out, self.dropout, self.training)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class GlobalClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_classes=80, feature_size=256, dropout=0.5):
        super().__init__()

        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, dilation=1, padding=0)
        self.fc = nn.Linear(feature_size*2, num_classes)
        self.output_act = nn.LogSoftmax(dim=-1)

        self.dropout = dropout

    def forward(self, x):
        out = F.max_pool2d(x, 2)
        out = self.conv1(out)
        out = F.relu(out)

        # if self.dropout > 0:
        #     out = F.dropout(out, self.dropout, self.training)

        avg_pool = F.avg_pool2d(out, out.shape[2:])
        max_pool = F.max_pool2d(out, out.shape[2:])
        avg_max_pool = torch.cat((avg_pool, max_pool), 1)
        out = avg_max_pool.view(avg_max_pool.size(0), -1)

        if self.dropout > 0:
            out = F.dropout(out, self.dropout, self.training)

        out = self.fc(out)
        out = self.output_act(out)

        return out


class RetinaNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fpn_sizes = []

    def forward(self, x):
        """
        :param x: input tensor
        :return: x1, x2, x3, x4 layer outputs
        """
        raise NotImplementedError()


class RetinaNet(nn.Module):
    def __init__(self, encoder: RetinaNetEncoder,
                 num_classes,
                 dropout_cls=0.5,
                 dropout_global_cls=0.5,
                 use_l2_features=True):
        super(RetinaNet, self).__init__()

        print('dropout_cls', dropout_cls, '   dropout_global_cls', dropout_global_cls)

        # self.encoder = encoder
        fpn_sizes = encoder.fpn_sizes
        self.use_l2_features = use_l2_features

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3], use_l2_features=use_l2_features)

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes, dropout=dropout_cls)
        self.globalClassificationModel = GlobalClassificationModel(fpn_sizes[-1], num_classes=3, feature_size=256, dropout=dropout_global_cls)
        self.globalClassificationLoss = nn.NLLLoss()

        if use_l2_features:
            pyramid_levels = [2, 3, 4, 5, 6, 7]
        else:
            pyramid_levels = [3, 4, 5, 6, 7]

        self.anchors = Anchors(pyramid_levels=pyramid_levels)

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.encoder = encoder

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def freeze_bn(self):
        """Freeze BatchNorm layers."""
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def freeze_encoder(self):
        self.encoder.eval()
        # correct version, but keep original as model has been trained this way
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def boxes(self, img_batch, regression, classification, global_classification, anchors):
        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

        scores = torch.max(classification, dim=2, keepdim=True)[0]

        scores_over_thresh = (scores > 0.025)[0, :, 0]

        if scores_over_thresh.sum() == 0:
            # no boxes to NMS, just return
            return [torch.zeros(0), global_classification, torch.zeros(0, 4)]
        else:
            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            # use very low threshold of 0.05 as boxes should not overlap
            anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 0.05)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, global_classification, transformed_anchors[0, anchors_nms_idx, :]]

    def forward(self, inputs, return_loss, return_boxes, return_raw=False):
        if return_loss:
            img_batch, annotations, global_annotations = inputs
        else:
            img_batch = inputs

        x1, x2, x3, x4 = self.encoder.forward(img_batch)

        features = self.fpn([x1, x2, x3, x4])

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        global_classification = self.globalClassificationModel(x4)

        anchors = self.anchors(img_batch)

        if return_raw:
            return [regression, classification, torch.exp(global_classification), anchors]

        res = []

        if return_loss:
            res += list(self.focalLoss(classification, regression, anchors, annotations))
            res += [self.globalClassificationLoss(global_classification, global_annotations)]

        if return_boxes:
            res += self.boxes(img_batch=img_batch,
                              regression=regression,
                              classification=classification,
                              global_classification=global_classification,
                              anchors=anchors)

        return res

