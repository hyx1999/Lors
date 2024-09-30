import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from typing import Tuple
from abc import ABC
from types import MethodType
from typing import Optional, Dict, Callable
from spft.base.linear import Linear

from accelerate import PartialState


class _SpGaloreFn(Function):
    state = PartialState()
    
    @staticmethod
    @torch.no_grad()
    def update_grad(d_w: torch.Tensor, lora_B: torch.Tensor) -> torch.Tensor:
        # W = B @ A
        # dA = B^T @ dW
        # B @ dA = B @ B^T @ dW
        @_SpGaloreFn.state.on_main_process
        def decompose_d_w(d_w: torch.Tensor, rank: int):
            _, Q = torch.linalg.eigh(d_w @ d_w.t())
            return Q[:, :rank]
        r = lora_B.shape[-1]
        Q = decompose_d_w(d_w, r)
        lora_B.copy_(Q)
    
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor, 
        weight: torch.Tensor, 
        bias: torch.Tensor,
        lora_A: torch.Tensor, 
        lora_B: torch.Tensor,
        alpha: float,
        forward_step: int,
        update_freq: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight, lora_B)
        ctx.forward_step = forward_step
        ctx.update_freq = update_freq
        output_shape = x.shape[:-1] + (-1,)
        x = x.view(-1, x.shape[-1])
        weight.addmm_(lora_B, lora_A, alpha=alpha)
        y = x.mm(weight.t()).view(output_shape)
        if bias is not None:
            y.add_(bias)
        lora_A.zero_()
        return y

    @staticmethod
    @once_differentiable
    def backward(
        ctx, 
        d_y: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        x, weight, lora_B = ctx.saved_tensors
                
        x_shape = x.shape
        grad_output_shape = d_y.shape
        x = x.view(-1, x_shape[-1])
        d_y = d_y.view(-1, grad_output_shape[-1])
        d_w = d_y.t().mm(x)
        
        if ctx.forward_step % ctx.update_freq == 0:
            _SpGaloreFn.update_grad(d_w, lora_B)

        d_x = d_weight = d_bias = d_lora_A = None
        if ctx.needs_input_grad[0]:
            d_x = d_y.mm(weight)

        if ctx.needs_input_grad[1]:
            d_weight = d_w
        if ctx.needs_input_grad[2]:
            d_bias = d_y.sum(dim=0)
        if ctx.needs_input_grad[3]:
            d_lora_A = lora_B.t().mm(d_w)
        if d_x is not None:
            d_x = d_x.view(*x_shape)
        return d_x, d_weight, d_bias, d_lora_A, None, None, None


class SpGaloreFnLinear(Linear):
    
    def __init__(self, 
        # linear
        in_features: int, 
        out_features: int, 
        bias: bool = False, 
        device=None, 
        dtype=None,
        # galore
        r: int = 16,
        alpha: float = 16,
        update_freq: int = 128,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.empty(out_features, r, dtype=dtype, device=device))
        self.alpha = alpha / r
        self.forward_step = 0
        self.update_freq = update_freq
        self._reset_lora_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _SpGaloreFn.apply(
            x, 
            self.weight, 
            self.bias,
            self.lora_A, 
            self.lora_B, 
            self.alpha,
            self.forward_step,
            self.update_freq,
        )
        self.forward_step += 1
        return y
    
    def _reset_lora_parameters(self) -> None:
        nn.init.zeros_(self.lora_A)
        nn.init.zeros_(self.lora_B)

    @torch.no_grad()
    def merge_adapter(self):
        if hasattr(self, "lora_A") and hasattr(self, "lora_B"):
            self.weight.data += (self.lora_B @ self.lora_A)
            delattr(self, "lora_A")
            delattr(self, "lora_B")
            def forward_patch(self, x: torch.Tensor) -> torch.Tensor:
                return F.linear(x, self.weight, self.bias)
            self.forward = MethodType(forward_patch, self)
