import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from enum import Enum
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from typing import Tuple
from types import MethodType
from typing import Optional, Dict, Callable
from lors.base.linear import Linear

from accelerate import PartialState
from collections import namedtuple
from .utils import SparsifyFn, GradFn

Fns = namedtuple("Fns", ["sparsify_fn", "grad_fn"])
Params = namedtuple("Params", ["lora_B", "scale", "sparse_decay"])

class LorsFn(Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor, 
        weight: torch.Tensor,
        bias: torch.Tensor,
        lora_A: torch.Tensor,
        params: Params,
        fns: Fns,
    ) -> torch.Tensor:
        output_shape = x.shape[:-1] + (-1,)
        x = x.view(-1, x.shape[-1])
        sparse_weight = fns.sparsify_fn(weight, params.scale)
        y = x.mm(sparse_weight.t()).view(output_shape)
        if bias is not None:
            y.add_(bias)
        ctx.save_for_backward(x, sparse_weight)
        ctx.fns = fns
        ctx.params = params
        return y

    @staticmethod
    @once_differentiable
    def backward(
        ctx, 
        d_y: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        x, sparse_weight = ctx.saved_tensors
        fns: Fns = ctx.fns
        params: Params = ctx.params
                
        x_shape = x.shape
        grad_output_shape = d_y.shape
        x = x.view(-1, x_shape[-1])
        d_y = d_y.view(-1, grad_output_shape[-1])
        d_w = d_y.t().mm(x)
        
        fns.grad_fn.update_d_w(d_w, sparse_weight, params.sparse_decay)
        fns.grad_fn.update_lora_B(params.lora_B, d_w)

        d_x = d_weight = d_bias = d_lora_A = None
        if ctx.needs_input_grad[0]:
            d_x = d_y.mm(sparse_weight)

        if ctx.needs_input_grad[1]:
            d_weight = d_w
        if ctx.needs_input_grad[2]:
            d_bias = d_y.sum(dim=0)
        if ctx.needs_input_grad[3]:
            d_lora_A = params.lora_B.t().mm(d_w)
        if d_x is not None:
            d_x = d_x.view(*x_shape)
        return d_x, d_weight, d_bias, d_lora_A, None, None
