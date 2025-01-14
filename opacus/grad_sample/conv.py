#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
from opacus.utils.tensor_utils import unfold2d, unfold3d
from opt_einsum import contract

from .utils import register_grad_sampler
from opacus.layers.weight_scaled_conv import WSConv2d


@register_grad_sampler([nn.Conv1d, nn.Conv2d, nn.Conv3d, WSConv2d])
def compute_conv_grad_sample(
    layer: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d,WSConv2d],
    activations: torch.Tensor,
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for convolutional layers

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    n = activations.shape[0]
    # get activations and backprops in shape depending on the Conv layer
    if type(layer) == nn.Conv2d or type(layer) == WSConv2d:
        activations = unfold2d(
            activations,
            kernel_size=layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
            dilation=layer.dilation,
        )
    elif type(layer) == nn.Conv1d:
        activations = activations.unsqueeze(-2)  # add the H dimension
        # set arguments to tuples with appropriate second element
        activations = torch.nn.functional.unfold(
            activations,
            kernel_size=(1, layer.kernel_size[0]),
            padding=(0, layer.padding[0]),
            stride=(1, layer.stride[0]),
            dilation=(1, layer.dilation[0]),
        )
    elif type(layer) == nn.Conv3d:
        activations = unfold3d(
            activations,
            kernel_size=layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
            dilation=layer.dilation,
        )
    backprops = backprops.reshape(n, -1, activations.shape[-1])

    ret = {}
    if layer.weight.requires_grad:
        # n=batch_sz; o=num_out_channels; p=(num_in_channels/groups)*kernel_sz
        grad_sample = contract("noq,npq->nop", backprops, activations)
        # rearrange the above tensor and extract diagonals.
        grad_sample = grad_sample.view(
            n,
            layer.groups,
            -1,
            layer.groups,
            int(layer.in_channels / layer.groups),
            np.prod(layer.kernel_size),
        )
        grad_sample = contract("ngrg...->ngr...", grad_sample).contiguous()
        shape = [n] + list(layer.weight.shape)
        ret[layer.weight] = grad_sample.view(shape)

    if layer.bias is not None and layer.bias.requires_grad:
        ret[layer.bias] = torch.sum(backprops, dim=2)

    return ret
