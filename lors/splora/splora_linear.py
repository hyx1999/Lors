import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Function
from typing import Tuple
from abc import ABC
from types import MethodType
from typing import Optional, Dict, Callable
from lors.base.linear import Linear


class SpLoraFn(Function):
        
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor, 
        weight: torch.Tensor, 
        bias: torch.Tensor,
        lora_A: torch.Tensor, 
        lora_B: torch.Tensor,
        scaling: float,
        training: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight, lora_A, lora_B)
        ctx.scaling = scaling
        output_shape = x.shape[:-1] + (-1,)
        x = x.view(-1, x.shape[-1])
        mask = (weight != 0)
        weight.addmm_(lora_B, lora_A, alpha=scaling).mul_(mask)
        output = x.mm(weight.t()).view(output_shape)
        if bias is not None:
            output.add_(bias)
        if not training:
            weight.addmm_(lora_B, lora_A, alpha=-scaling).mul_(mask)
        return output

    @staticmethod
    def backward(ctx, 
        grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        x, weight, lora_A, lora_B = ctx.saved_tensors
        scaling = ctx.scaling

        mask = (weight != 0)
        
        x_shape = x.shape
        grad_output_shape = grad_output.shape
        x = x.view(-1, x_shape[-1])
        grad_output = grad_output.view(-1, grad_output_shape[-1])
        _grad_weight = grad_output.t().mm(x).mul_(mask)

        grad_x = grad_weight = grad_bias = grad_lora_A = grad_lora_B = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output.mm(weight)

        weight.addmm_(lora_B, lora_A, alpha=-scaling).mul_(mask)

        if ctx.needs_input_grad[1]:
            grad_weight = _grad_weight
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)
        if ctx.needs_input_grad[3]:
            grad_lora_A = _grad_weight.t().mm(lora_B).t()
        if ctx.needs_input_grad[4]:
            grad_lora_B = _grad_weight.mm(lora_A.t())
        if grad_x is not None:
            grad_x = grad_x.view(*x_shape)
        return grad_x, grad_weight, grad_bias, grad_lora_A, grad_lora_B, None, None


class SpLoraLinear(Linear):
    
    def __init__(self, 
        # linear
        in_features: int, 
        out_features: int, 
        bias: bool = False, 
        device=None, 
        dtype=None,
        # lora
        r: int = 16,
        lora_alpha: float = 16,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.empty(out_features, r, dtype=dtype, device=device))
        self.scaling = lora_alpha / r
        self._reset_lora_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SpLoraFn.apply(
            x, 
            self.weight, 
            self.bias,
            self.lora_A, 
            self.lora_B, 
            self.scaling,
            self.training,
        )
    
    def _reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    @torch.no_grad()
    def set_mask(self):
        pass

    @torch.no_grad()
    def merge_adapter(self):
        if hasattr(self, "lora_A"):
            mask = (self.weight != 0)
            self.weight.data += (self.lora_B @ self.lora_A) * mask
            delattr(self, "lora_A")
            delattr(self, "lora_B")
            def forward_patch(self, x: torch.Tensor) -> torch.Tensor:
                return F.linear(x, self.weight, self.bias)
            self.forward = MethodType(forward_patch, self)
