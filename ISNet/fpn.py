import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import DeformConv2d as DeformConv, RoIAlign, DeformConv2dPack


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


class PyramidFeatures_with_Atrous(nn.Module):

    def __init__(self, C3_size, C4_size, C5_size, feature_size=256, output_stride=16):
        super(PyramidFeatures_with_Atrous, self).__init__()

        self.output_stride = output_stride

        dilations = [6, 12, 18]

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=1, padding=dilations[0], dilation=dilations[0], bias=True)
        self.P5_1_bn = nn.BatchNorm2d(feature_size)
        self.P5_1_act = nn.ReLU(inplace=True)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1, bias=True)

        self.P5_2_bn = nn.BatchNorm2d(feature_size)
        self.P5_2_act = nn.ReLU(inplace=True)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=3, stride=1, padding=dilations[1], dilation=dilations[1], bias=True)
        self.P4_1_bn = nn.BatchNorm2d(feature_size)
        self.P4_1_act = nn.ReLU(inplace=True)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1, bias=True)

        self.P4_2_bn = nn.BatchNorm2d(feature_size)
        self.P4_2_act = nn.ReLU(inplace=True)
        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=3, stride=1, padding=dilations[2], dilation=dilations[2], bias=True)
        self.P3_1_bn = nn.BatchNorm2d(feature_size)
        self.P3_1_act = nn.ReLU(inplace=True)

        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1, bias=True)

        self.P3_2_bn = nn.BatchNorm2d(feature_size)
        self.P3_2_act = nn.ReLU(inplace=True)

        self.P4_edge_1 = My_BasicConv2d(in_planes=feature_size, out_planes=feature_size, kernel_size=3, stride=1,
                                        padding=1, use_deformConv=True, use_bn=False, use_act=False, act_type='ReLU')

        self.P3_edge_1 = My_BasicConv2d(in_planes=feature_size, out_planes=feature_size, kernel_size=3, stride=1,
                                        padding=1, use_deformConv=True, use_bn=False, use_act=False, act_type='ReLU')

        self.P5_upsampled_conv = My_BasicConv2d(in_planes=feature_size, out_planes=feature_size, kernel_size=3, stride=1,
                                         padding=1, use_deformConv=True, use_bn=False, use_act=False)

        self.P4_upsampled_conv = My_BasicConv2d(in_planes=feature_size, out_planes=feature_size, kernel_size=3, stride=1,
                                         padding=1, use_deformConv=True, use_bn=False, use_act=False)

        self._init_weight()

    def forward(self, inputs):

        C3, C4, C5, edge_guidance = inputs
        print(C3.shape, C4.shape, C5.shape, edge_guidance.shape)

        edge_4 = F.interpolate(edge_guidance, size=(C4.size(2), C4.size(3)), mode='bilinear', align_corners=False)
        edge_4 = self.P4_edge_1(edge_4)

        edge_3 = F.interpolate(edge_guidance, size=(C3.size(2), C3.size(3)), mode='bilinear', align_corners=False)
        edge_3 = self.P3_edge_1(edge_3)

        P5_x = self.P5_1_act(self.P5_1_bn(self.P5_1(C5)))
        P5_upsampled_x = F.interpolate(P5_x, size=(C4.size(2), C4.size(3)), mode='bilinear', align_corners=False)

        P5_upsampled_x = self.P5_upsampled_conv(P5_upsampled_x)
        P5_out = self.P5_2_act(self.P5_2_bn(self.P5_2(P5_x)))

        P4_x = self.P4_1_act(self.P4_1_bn(self.P4_1(C4)))
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = F.interpolate(P4_x, size=(C3.size(2), C3.size(3)), mode='bilinear', align_corners=False)
        P4_x = P4_x + edge_4
        P4_upsampled_x = self.P4_upsampled_conv(P4_upsampled_x)
        P4_out = self.P4_2_act(self.P4_2_bn(self.P4_2(P4_x)))

        P3_x = self.P3_1_act(self.P3_1_bn(self.P3_1(C3)))
        P3_x = P3_x + P4_upsampled_x
        P3_x = P3_x + edge_3
        P3_out = self.P3_2_act(self.P3_2_bn(self.P3_2(P3_x)))

        return P3_out, P4_out, P5_out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, DeformConv):
                nn.init.kaiming_normal_(m.weight.data)