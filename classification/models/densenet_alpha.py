import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from pruner.alpha_op import AlphaLayer
from models.densenet_us import USBasicBlock, USDenseBlock, USDenseNet, TransitionBlock
from utils.options_dm import args
from utils.utils import *

class AlphaBasicBlock(USBasicBlock):
    def __init__(self, in_planes, out_planes, dropRate=0.0, n_block=-1):
        super().__init__(in_planes, out_planes, dropRate, n_block=n_block)
        self.alpha = AlphaLayer(in_planes, n_block, prob_type=args.prob_type)
        self.alpha_training = False
        self.cut_num = None

    def forward(self, x):
        out = self.relu(self.bn1(x))

        if self.alpha_training:
            out = self.alpha(out)
        else:
            ### init mask
            if self.mask is None or self.mask.size() != out.size():
                self.mask = torch.ones(out.size()).cuda()
            else:
                self.mask = torch.ones_like(self.mask)

            if self.cut_num is None:
                ### cut connection
                input_ch = 24 + self.n_block * 144
                num_connection = 1 + ((out.shape[1] - input_ch) // 12)
                cut_connection = int(num_connection * (1 - self.width_mult))

                if cut_connection > 0:
                    cut_num = input_ch + (cut_connection - 1) * 12
                    self.mask[:, : cut_num] = 0
                    out = out * self.mask
            else:
                if self.cut_num == 0:
                    pass
                else:
                    self.mask[:, : self.cut_num] = 0
                out = out * self.mask

        out = self.conv1(out)

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

    def cal_flops(self, in_height, in_width, in_ch=None):
        # conv1
        conv_flops, out_height, out_width = conv_compute_flops(self.conv1, in_height,
                                            in_width, e_in_ch=in_ch)

        return conv_flops, out_height, out_width

    def expected_flops(self, in_height, in_width):
        # conv1
        e_in_channel = self.alpha.expected_channel()
        e_conv_flops, out_height, out_width = conv_compute_flops(self.conv1, in_height,
                                            in_width, e_in_channel)

        return e_conv_flops, out_height, out_width

class AlphaDenseNet(USDenseNet):
    def __init__(self, block, depth, num_classes, growth_rate=12,
                 reduction=1, dropRate=0.0):
        # super(AlphaDenseNet, self).__init__()

        # def alpha(channels):
        #     return AlphaLayer(channels, min_width, max_width, width_offset, prob_type)
        # global Alpha
        # Alpha = alpha
        super(AlphaDenseNet, self).__init__(block, depth, num_classes, growth_rate,
                 reduction, dropRate)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)

    def set_alpha_training(self, training):
        self.alpha_training = training
        for m in self.modules():
            if isinstance(m, (AlphaBasicBlock)):
                m.alpha_training = training

    def cal_flops(self, input_size=32):
        flopses = []
        # conv1
        total_flops, out_height, out_width = conv_compute_flops(self.conv1, input_size, input_size)
        flopses.append(total_flops)

        for m in self.modules():
            # BasicBlock
            if isinstance(m, AlphaBasicBlock):
                flops, out_height, out_width = m.cal_flops(out_height, out_width)
                flopses.append(flops)
                total_flops += flops
            # transitionBlock
            elif isinstance(m, TransitionBlock):
                flops, out_height, out_width = m.cal_flops(out_height, out_width)
                flopses.append(flops)
                total_flops += flops

        # fc
        flops = fc_compute_flops(self.fc)
        flopses.append(flops)
        total_flops += flops

        return total_flops / 1e6, np.array(flopses) / 1e6 ### M

    def expected_flops(self, in_height, in_width):
        # conv1
        total_flops, out_height, out_width = conv_compute_flops(
            self.conv1, in_height, in_width)

        for m in self.modules():
            # BasicBlock
            if isinstance(m, AlphaBasicBlock):
                flops, out_height, out_width = m.expected_flops(out_height, out_width)
                total_flops += flops
            # transitionBlock
            elif isinstance(m, TransitionBlock):
                flops, out_height, out_width = m.cal_flops(out_height, out_width)
                total_flops += flops
        # fc
        flops = fc_compute_flops(self.fc)
        total_flops += flops
        return total_flops / 1e6 ### M

    def expected_sampling(self):
        flopses = []
        # conv1
        total_flops, out_height, out_width = conv_compute_flops(
            self.conv1, 32, 32)
        flopses.append(total_flops)

        for m in self.modules():
            # BasicBlock
            if isinstance(m, AlphaBasicBlock):
                m.cut_num, input_ch, expected = m.alpha.expected_sampling()
                print(m.alpha.channels, m.cut_num, expected)
                m.alpha_training = False
                ## cal flops
                flops, out_height, out_width = m.cal_flops(out_height, out_width, in_ch=input_ch)
                flopses.append(flops)
                total_flops += flops
                # transitionBlock
            elif isinstance(m, TransitionBlock):
                flops, out_height, out_width = m.cal_flops(out_height, out_width)
                flopses.append(flops)
                total_flops += flops

        # fc
        flops = fc_compute_flops(self.fc)
        flopses.append(flops)
        total_flops += flops

        return np.array(flopses) / 1e6, total_flops/1e6

def densenet40_alpha(depth=40, classes=10, growth_rate=12, reduction=1, dropRate=0):
    return AlphaDenseNet(AlphaBasicBlock, depth, classes, growth_rate, reduction, dropRate)