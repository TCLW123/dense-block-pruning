import torch.nn as nn
import numpy as np

## Dynamic conv2d hxy
class DConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, bias=False):
        super(DConv2d, self).__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias)
        # self.depthwise = depthwise
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult = 1
        self.stride = stride
        self.padding = padding

        # self.us = us
        # self.ratio = ratio
    def init_mask(self):
        self.mask = np.ones([self.in_channels_ma, self.out_channels_max, 3, 3])
        self.mask[:, :self.in_channels, :, :] = 0

    def forward(self, input):
        self.in_channels = make_divisible(self.in_channels_max * self.width_mult)
        self.init_mask()
        weight = self.weight * self.mask
        y = nn.Conv2d(input, weight, self.stride, self.padding)
        return y

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