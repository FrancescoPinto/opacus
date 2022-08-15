import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
#they use this one (appendix for pytorch): https://arxiv.org/pdf/2101.08692.pdf
class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,gain=None, eps=1e-4):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1,1,1))
        else:
            self.gain = None
        self.eps = eps
    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1, 2, 3],keepdims=True)
        var = torch.var(self.weight, axis=[1, 2, 3],keepdims=True)
        weight = (self.weight - mean) / (var * fan_in + self.eps) ** 0.5
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias,
                            self.stride, self.padding,
                            self.dilation, self.groups)
