# The functions for computing individual gradients are borrowed from https://github.com/pytorch/opacus

from typing import Union

import numpy as np
import torch
from torch import nn
from torch.functional import F

def _compute_conv_grad_sample(
    # for some reason pyre doesn't understand that
    # nn.Conv1d and nn.modules.conv.Conv1d is the same thing
    # pyre-ignore[11]
    layer: Union[nn.Conv2d, nn.Conv1d],
    A: torch.Tensor,
    B: torch.Tensor,
    batch_dim: int = 0,
    gpu_id=None
) -> None:
    """
    Computes per sample gradients for convolutional layers
    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """
    n = A.shape[0]
    layer_type = "Conv2d"
    # get A and B in shape depending on the Conv layer
    A = torch.nn.functional.unfold(
        A, layer.kernel_size, padding=layer.padding, stride=layer.stride
    )
    B = B.reshape(n, -1, A.shape[-1])

    # n=batch_sz; o=num_out_channels; p=(num_in_channels/groups)*kernel_sz
    grad_sample = torch.einsum("noq,npq->nop", B, A)
    # rearrange the above tensor and extract diagonals.
    grad_sample = grad_sample.view(
        n,
        layer.groups,
        -1,
        layer.groups,
        int(layer.in_channels / layer.groups),
        np.prod(layer.kernel_size),
    )
    grad_sample = torch.einsum("ngrg...->ngr...", grad_sample).contiguous()
    shape = [n] + list(layer.weight.shape)
    layer.weight.grad_sample = grad_sample.view(shape)

def _compute_linear_grad_sample(
    layer: nn.Linear, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0, gpu_id=None
) -> None:
    """
    Computes per sample gradients for ``nn.Linear`` layer
    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """
    layer.weight.grad_sample = torch.einsum("n...i,n...j->n...ij", B, A)
