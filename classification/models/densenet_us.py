import math
import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import *

class USBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0, n_block=-1):
        super(USBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

        ### slim part -hxy
        self.width_mult = 1
        self.n_block = n_block
        self.mask = None

    def forward(self, x):
        out = self.relu(self.bn1(x))

        ### init mask
        if self.mask is None or self.mask.size() != out.size():
            self.mask = torch.ones(out.size()).cuda()
        else:
            self.mask = torch.ones_like(self.mask)

        ### cut connection
        input_ch = 24 + self.n_block * 144
        num_connection = 1 + ((out.shape[1] - input_ch) // 12)
        cut_connection = int(num_connection * (1 - self.width_mult))

        if cut_connection > 0:
            cut_num = input_ch + (cut_connection - 1) * 12
            self.mask[:, : cut_num] = 0
            out = out * self.mask
            # out[ : , : cut_num] = 0

        out = self.conv1(out)

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate


    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

    def cal_flops(self, in_height, in_width):
        # conv1
        flops, out_height, out_width = conv_compute_flops(self.conv1, in_height, in_width)
        return flops, out_height/2, out_width/2


class USDenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0, n_block=-1):
        super(USDenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate, n_block)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate, n_block):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes + i * growth_rate, growth_rate, dropRate, n_block))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class USDenseNet(nn.Module):
    def __init__(self, block=None, depth=40, num_classes=10, growth_rate=12,
                 reduction=1, dropRate=0.0):
        super(USDenseNet, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if block == None:
            block = USBasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.dense1 = USDenseBlock(n, in_planes, growth_rate, block, dropRate, n_block=0)
        in_planes = int(in_planes + n * growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes * reduction))
        # 2nd block
        self.dense2 = USDenseBlock(n, in_planes, growth_rate, block, dropRate, n_block=1)
        in_planes = int(in_planes + n * growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes * reduction))
        # 3rd block
        self.dense3 = USDenseBlock(n, in_planes, growth_rate, block, dropRate, n_block=2)
        in_planes = int(in_planes + n * growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)


class BasicBlock_rd(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0, n_block=-1, cut_connection=0):
        super(BasicBlock_rd, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv1 = DConv2d(in_planes, out_planes, kernel_size=3, stride=1,
        #                        padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

        ### slim part -hxy
        self.in_channels_max = in_planes
        self.out_channels_max = out_planes
        self.n_block = n_block
        self.cut_connection = cut_connection

    def forward(self, x):
        ### cut connection
        x1 = x.clone()
        input_ch = 24 + self.n_block * 144
        if self.cut_connection > 0:
            cut_num = input_ch + (self.cut_connection - 1) * 12
            x1[:, : int(cut_num)] = 0

        out = self.conv1(self.relu(self.bn1(x1)))
        # out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        return torch.cat([x, out], 1)


class DenseBlock_rd(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0, n_block=-1, cut_connection=None):
        super(DenseBlock_rd, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate, n_block, cut_connection)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate, n_block, cut_connection):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes + i * growth_rate, growth_rate, dropRate, n_block, cut_connection[i]))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DenseNet_rd(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=1, dropRate=0.0, cut_connection=None):
        super(DenseNet_rd, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        block = BasicBlock_rd
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.dense1 = DenseBlock_rd(n, in_planes, growth_rate, block, dropRate, n_block=0,
                                    cut_connection=cut_connection[0:12])
        in_planes = int(in_planes + n * growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes * reduction))
        # 2nd block
        self.dense2 = DenseBlock_rd(n, in_planes, growth_rate, block, dropRate, n_block=1,
                                    cut_connection=cut_connection[12:24])
        in_planes = int(in_planes + n * growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes * reduction))
        # 3rd block
        self.dense3 = DenseBlock_rd(n, in_planes, growth_rate, block, dropRate, n_block=2,
                                    cut_connection=cut_connection[24:36])
        in_planes = int(in_planes + n * growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)


def make_divisible(v, divisor=12, min_value=12):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def densenet40_us(depth=40, classes=10, growth_rate=12, reduction=1, dropRate=0):
    return USDenseNet(USBasicBlock,depth, classes, growth_rate, reduction, dropRate)


def densenet_40_rd(depth=40, classes=10, growth_rate=12, redution=1, dropRate=0, cut_connection_list=None):
    return DenseNet_rd(depth, classes, growth_rate, redution, dropRate, cut_connection=cut_connection_list)
