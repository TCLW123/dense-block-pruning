import numpy as np
import random
import torch
import torch.nn as nn
from utils.options_dm import args

class AlphaLayer(nn.Module):
    def __init__(self, channels,n_block ,prob_type='exp'):
        super(AlphaLayer, self).__init__()
        assert prob_type in ['exp', 'sigmoid']
        assert n_block >= 0

        self.prob_type = prob_type
        self.channels = channels ### length of alpha layer

        #ch_indice 每条连接对应的channel数 如 [24, 12, 12, 12, 12]
        self.input_ch = 24 + n_block * 144
        if self.channels == self.input_ch:
            self.num_groups = 0
            self.min_ch = self.input_ch
            self.alpha = None
        elif self.channels > self.input_ch:
            self.min_ch = 12
            self.group_size = 12
            if self.channels - self.min_ch - self.input_ch == 0:
                self.num_groups = 1
            else:
                self.num_groups = int((self.channels - self.min_ch - self.input_ch) / self.group_size + 1)
            # self.alpha = nn.Parameter(torch.zeros(self.num_groups))
            self.alpha = nn.Parameter(torch.ones(self.num_groups) * args.init_alpha)

    def forward(self, x):
        size_x = x.size()

        if self.num_groups == 0:
            return x

        prob = self.get_marginal_prob().view(self.num_groups, 1)
        tp_x = x.transpose(0, 1).contiguous()
        # tp_x = x.transpose(0, 1)
        tp_group_x1 = tp_x[:self.input_ch]
        tp_group_x1 = tp_group_x1 * prob[0]
        input_min_ch = self.input_ch + self.min_ch

        if size_x[1] == input_min_ch:
            x = torch.cat((tp_group_x1, tp_x[-self.min_ch:]),0).transpose(0,1)
            # x = torch.cat((tp_group_x1, tp_x[-self.min_ch:]),0).transpose(0, 1).contiguous
        else:
            tp_group_x = tp_x[self.input_ch:-self.min_ch]
            size_tp_group = tp_group_x.size()
            num_groups = size_tp_group[0] // self.group_size
            tp_group_x = tp_group_x.view(num_groups, -1) * prob[1:num_groups+1]
            tp_group_x = tp_group_x.view(size_tp_group)

            x = torch.cat((torch.cat((tp_group_x1, tp_group_x),0),tp_x[-self.min_ch:]), 0).transpose(0, 1)
            # x = torch.cat(tp_group_x1, tp_group_x, tp_x[-self.min_ch:]).transpose(0, 1).contiguous
        return x

    def get_marginal_prob(self):
        ### get condition_prob
        if self.prob_type == 'exp':
            self.alpha.data.clamp_(min=0.)
            p = torch.exp(-self.apha)
        elif self.prob_type == 'sigmoid':
            p = torch.sigmoid(self.alpha)
        else:
            return NotImplementedError

        marginal_prob = torch.cumprod(torch.flip(p, dims=[0]),dim=0)
        return torch.flip(marginal_prob, dims=[0])

    def _get_ch_indice(self, offset, max_ch):
        num_offset = (max_ch - self.input_ch - self.min_ch) / offset
        indice = []
        indice.append(self.input_ch)
        for i in range(int(num_offset + 1)):
            if i == num_offset:
                indice.append(self.min_ch)
            else:
                indice.append(offset)
        return indice

    def expected_channel(self):
        if self.num_groups == 0:
            return self.min_ch
        marginal_prob = self.get_marginal_prob()
        expected_channel_1 = marginal_prob[0] * self.input_ch
        if self.num_groups > 1:
            expected_channel = torch.sum(marginal_prob[1:]) * self.group_size
            return expected_channel_1 + expected_channel + self.min_ch
        else:
            return expected_channel_1 + self.min_ch

    def expected_sampling(self):
        expected = self.expected_channel()
        ch_indice = self._get_ch_indice(offset=12, max_ch=self.channels)
        ch_indice = np.array(ch_indice)
        candidate = np.cumsum(np.flip(ch_indice))
        if len(candidate) < 2:
            return 0, candidate[0], self.channels - expected
        idx = np.argmin([abs(ch - expected) for ch in candidate])
        cut_num = self.channels - candidate[idx]
        return cut_num, candidate[idx], self.channels - int(expected.cpu().detach().numpy())