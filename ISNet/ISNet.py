import torch.nn as nn
import torch
import torch.nn.functional as F

from .fpn import PyramidFeatures_with_Atrous
from .fcn import FCNHead
from .resnet import build_backbone
from .utils import joint_loss

class My_BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, use_deformConv=True,
                 use_bn=True, use_act=True, bias=True, act_type='ReLU'):
        super(My_BasicConv2d, self).__init__()
        self.use_bn = use_bn
        self.use_act = use_act
        self.act_type = act_type
        if use_deformConv:
            self.conv =DeformConv2dPack(in_planes, out_planes,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, bias=False)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=bias)
        if use_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        if use_act:
            if act_type == 'ReLU':
                self.act = nn.ReLU(inplace=True)
            elif act_type == 'Sigmoid':
                self.act = nn.Sigmoid()
            else:
                raise NotImplementedError

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_act:
            x = self.act(x)
        return x


class ISNet(nn.Module):

    def __init__(self, num_classes,  output_stride=16, model_depth=50):

        super(ISNet, self).__init__()
        self.inplanes = 64
        self.output_stride = output_stride

        self.backbone = build_backbone(backbone='resnet', output_stride=output_stride, BatchNorm=nn.BatchNorm2d,
                                           model_depth=model_depth, pre_trained=True)
        fpn_sizes = [256, 512, 1024, 2048]

        self.fpn_with_atrous = PyramidFeatures_with_Atrous(fpn_sizes[1], fpn_sizes[2], fpn_sizes[3],
                                                           feature_size=fpn_sizes[0], output_stride=output_stride)

        self.edge_conv1 = My_BasicConv2d(fpn_sizes[0], int(fpn_sizes[0] / 2), kernel_size=1, use_deformConv=False)
        self.edge_conv2 = My_BasicConv2d(int(fpn_sizes[0] / 2), int(fpn_sizes[0] / 2), kernel_size=3, padding=1, use_deformConv=False)
        self.edge_conv3 = nn.Conv2d(int(fpn_sizes[0] / 2), 1, kernel_size=3, padding=1)

        self.fcn = FCNHead(in_channels=fpn_sizes[0], out_channels=int(fpn_sizes[0] / 2), num_classes=num_classes,
                           num_layers=2, with_norm='batch_norm', upsample_rate=4, output_stride=output_stride)

        self.loss_cal = joint_loss

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, data, label=None, edge=None):

        x = self.backbone.conv1(data)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x1 = self.backbone.layer1(x)

        x2 = self.backbone.layer2(x1)

        x3 = self.backbone.layer3(x2)

        x4 = self.backbone.layer4(x3)

        x_edge = self.edge_conv1(x1)
        edge_guidance_feat = self.edge_conv2(x_edge)
        edge_pre = self.edge_conv3(edge_guidance_feat)
        edge_pre = F.interpolate(edge_pre, scale_factor=4,
                                 mode='bilinear', align_corners=True)

        fpn_p3, fpn_p4, fpn_p5 = self.fpn_with_atrous([x2, x3, x4, x1])

        fcn_res = self.fcn(fpn_p3=fpn_p3, fpn_p4=fpn_p4, fpn_p5=fpn_p5, edge_guidance=edge_guidance_feat)

        if self.training:
            fcn_loss = self.loss_cal(fcn_res['fcn_output'], label)

            if self.edge_guidance:
                edge_loss = torch.nn.BCEWithLogitsLoss()(edge_pre, edge)
                return fcn_loss, edge_loss

            return fcn_loss

        else:

            fcn_output = fcn_res['fcn_output']

            return fcn_output