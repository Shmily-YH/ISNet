import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import DeformConv2d as DeformConv, DeformConv2dPack
from torch.nn import BatchNorm2d

from .utils import SelfTrans


class FCNSubNet(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers, dilation=1, with_norm='none'):
        super(FCNSubNet, self).__init__()

        assert with_norm in ['none', 'batch_norm', 'group_norm']
        assert num_layers >= 2
        self.num_layers = num_layers

        if with_norm == 'batch_norm':
            norm = BatchNorm2d
        elif with_norm == 'group_norm':
            def group_norm(in_channel):
                return nn.GroupNorm(32, in_channel)
            norm = group_norm
        else:
            norm = None

        conv_pack = DeformConv2dPack

        self.conv = nn.ModuleList()
        for i in range(num_layers):
            conv = []
            if i == num_layers - 2:
                conv.append(conv_pack(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                             stride=1, padding=dilation, dilation=dilation))
                in_channels = out_channels
            else:
                conv.append(
                    conv_pack(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1,
                                     padding=dilation, dilation=dilation))
            if with_norm != 'none':
                conv.append(norm(in_channels))
            conv.append(nn.ReLU(inplace=True))

            self.conv.append(nn.Sequential(*conv))

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.fill_(0)
                m.bias.data.fill_(0)
            elif isinstance(m, DeformConv):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.conv[i](x)
        return x


class FCNHead(nn.Module):

    def __init__(self, in_channels, out_channels, num_classes, num_layers, with_norm='none', upsample_rate=4,
                output_stride=16):
        super(FCNHead, self).__init__()
        self.upsample_rate = upsample_rate
        self.output_stride = output_stride

        self.fcn_subnet_3 = FCNSubNet(in_channels, out_channels, num_layers, with_norm=with_norm)
        self.attention_3_1 = SelfTrans(n_head=1, n_mix=2, d_model=out_channels, d_k=out_channels, d_v=out_channels, pooling=False)
        self.attention_3_2 = SelfTrans(n_head=1, n_mix=2, d_model=out_channels, d_k=out_channels, d_v=out_channels, pooling=False)

        self.fcn_subnet_4 = FCNSubNet(in_channels, out_channels, num_layers, with_norm=with_norm)
        self.attention_4_1 = SelfTrans(n_head=1, n_mix=2, d_model=out_channels, d_k=out_channels, d_v=out_channels, pooling=False)
        self.attention_4_2 = SelfTrans(n_head=1, n_mix=2, d_model=out_channels, d_k=out_channels, d_v=out_channels, pooling=False)

        self.fcn_subnet_5 = FCNSubNet(in_channels, out_channels, num_layers, with_norm=with_norm)
        self.attention_5_1 = SelfTrans(n_head=1, n_mix=2, d_model=out_channels, d_k=out_channels, d_v=out_channels, pooling=False)
        self.attention_5_2 = SelfTrans(n_head=1, n_mix=2, d_model=out_channels, d_k=out_channels, d_v=out_channels, pooling=False)

        self.score = nn.Sequential(nn.Conv2d(out_channels*3 + out_channels, out_channels*2, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels*2),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   nn.Conv2d(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels*2),
                                   nn.ReLU(),
                                   nn.Dropout(0.1),
                                   nn.Conv2d(out_channels*2, num_classes, kernel_size=1, stride=1))

        self._init_weight()

    def forward(self, fpn_p3=None, fpn_p4=None, fpn_p5=None, edge_guidance=None):

        upsample_size = (fpn_p3.size(2) * 2, fpn_p3.size(3) * 2)

        fpn_p3 = self.fcn_subnet_3(fpn_p3)
        fpn_p3 = self.attention_3_1(fpn_p3)
        fpn_p3 = self.attention_3_2(fpn_p3)

        fpn_p4 = self.fcn_subnet_4(fpn_p4)
        fpn_p4 = self.attention_4_1(fpn_p4)
        fpn_p4 = self.attention_4_2(fpn_p4)

        fpn_p5 = self.fcn_subnet_5(fpn_p5)
        fpn_p5 = self.attention_5_1(fpn_p5)
        fpn_p5 = self.attention_5_2(fpn_p5)

        fpn_p3 = F.interpolate(fpn_p3, size=upsample_size, mode='bilinear', align_corners=False)
        fpn_p4 = F.interpolate(fpn_p4, size=upsample_size, mode='bilinear', align_corners=False)
        fpn_p5 = F.interpolate(fpn_p5, size=upsample_size, mode='bilinear', align_corners=False)

        feat = torch.cat([fpn_p3, fpn_p4, fpn_p5, edge_guidance], dim=1)

        score = self.score(feat)

        ret = {'fcn_score': score}
        if self.upsample_rate != 1:
            output = F.interpolate(score, None, self.upsample_rate, mode='bilinear', align_corners=False)
            ret.update({'fcn_output': output})

        return ret

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()