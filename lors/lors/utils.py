import torch
import torch.nn as nn
import torch.distributed as dist
import deepspeed
from ..lors_fn import (
    sparsify_6by8,
    sparsify_3by4,
    sparsify_2by4,
)

class ScaleFn:
    
    def __init__(self):
        self.steps = 0
        self.substeps = 0

    def update_step(self):
        self.steps += 1
        self.substeps = 0

    def update_substep(self):
        self.substeps += 1

    @torch.no_grad()
    def update_scale(self, 
        scale: torch.Tensor, 
        x: torch.Tensor,
        scale_mtm: float,
    ):
        # x: [B, L, D]
        x = x.view((-1, x.shape[-1])).abs().mean(dim=0)
        dist.all_reduce(x, dist.ReduceOp.AVG)
        scale.mul_(scale_mtm).add_((1.0 - scale_mtm) * x)  


class SparsifyFn:
    
    def __init__(self, 
        update_freq: int,
    ):
        self.update_freq = update_freq
        self.steps = 0
        self.substeps = 0
        self.stage = 0

    def update_step(self):
        self.steps += 1
        self.substeps = 0
        if self.steps % self.update_freq == 0:
            self.stage += 1

    def update_substep(self):
        self.substeps += 1
    
    @torch.no_grad()
    def prune(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if self.stage == 0:
            return sparsify_6by8(x, scale)
        elif self.stage == 1:
            return sparsify_3by4(x, scale)
        else:
            return sparsify_2by4(x, scale)


class GradFn:

    def __init__(self, 
        update_freq: int,
    ):
        self.update_freq = update_freq
        self.steps = 0
        self.substeps = 0

    def update_step(self):
        self.steps += 1
        self.substeps = 0

    def update_substep(self):
        self.substeps += 1

    @torch.no_grad()
    def update_d_w(self, 
        d_w: torch.Tensor,
        sparse_weight: torch.Tensor,
        sparse_decay: float,
    ) -> None:
        mask = (sparse_weight == 0)
        d_w.add_(sparse_decay * mask * sparse_weight)

    def decompose_d_w(A: torch.Tensor, r: int):  # min |A - Q Q^T A| => B = Q
        m = A.shape[1]
        P = torch.randn(m, r)
        Q, _ = torch.linalg.qr(A @ P)
        return Q

    @torch.no_grad()
    def update_lora_B(self, 
        lora_B: torch.Tensor,
        d_w: torch.Tensor, 
    ) -> None:
        """
        W = B @ A
        dA = B^T @ dW
        B @ dA = B @ B^T @ dW
        """
        if self.steps % self.update_freq != 0 or self.substeps != 0:
            return
        r = lora_B.shape[-1]
        with deepspeed.zero.GatheredParameters(lora_B, modifier_rank=0):
            if deepspeed.comm.get_rank() == 0:
                Q = self.decompose_d_w(d_w, r)
                lora_B.copy_(Q)
