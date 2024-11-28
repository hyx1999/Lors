import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
import deepspeed
from enum import Enum
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from typing import Tuple
from types import MethodType
from typing import Optional, Dict, Callable
from lors.base.linear import Linear

from .autograd import LorsFn, Fns, Params
from .utils import SparsifyFn, GradFn, ScaleFn

class LorsLinear(Linear):
    
    def __init__(self,
        # linear
        in_features: int, 
        out_features: int, 
        bias: bool = False, 
        device=None, 
        dtype=None,
        # lors
        r: int = 16,
        alpha: float = 16,
        scale_mtm: float = 0.95,
        sparse_decay: float = 1e-4,
        # sparsify & grad_update
        scale_fn: ScaleFn = None,
        sparsify_fn: SparsifyFn = None,
        grad_fn: GradFn = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.empty(out_features, r, dtype=dtype, device=device))
        self.scale = nn.Parameter(torch.empty(in_features, dtype=dtype, device=device))
        self.lora_alpha = alpha / r
        self.scale_mtm = scale_mtm
        self.sparse_decay = sparse_decay
        self.scale_fn = scale_fn
        self.sparsify_fn = sparsify_fn
        self.grad_fn = grad_fn
        self._reset_lora_parameters()

    def _reset_lora_parameters(self) -> None:
        nn.init.zeros_(self.lora_A)
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
        nn.init.zeros_(self.scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.scale_fn.update_scale(self.scale, x, self.scale_mtm)
        y = LorsFn.apply(
            x,
            self.weight, 
            self.bias,
            self.lora_A, 
            Params(scale=self.scale, sparse_decay=self.sparse_decay),
            Fns(sparsify_fn=self.sparsify_fn, grad_fn=self.grad_fn)
        )
        return y

    @torch.no_grad()
    def update_adapters(self):
        self.weight.addmm_(self.lora_B, self.lora_A, alpha=self.lora_alpha)
        self.lora_A.zero_()

    @torch.no_grad()
    def merge_adapter(self):
        if hasattr(self, "lora_A") and hasattr(self, "lora_B"):
            self.weight.data += self.lora_alpha * (self.lora_B @ self.lora_A)
            delattr(self, "lora_A")
            delattr(self, "lora_B")
            def forward_patch(self, x: torch.Tensor) -> torch.Tensor:
                return F.linear(x, self.weight, self.bias)
            self.forward = MethodType(forward_patch, self)
