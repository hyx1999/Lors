import torch
import deepspeed
from accelerate import PartialState

state = PartialState()

def decompose_d_w(A: torch.Tensor, r: int):  # min |A - Q Q^T A| => B = Q
    n, m = A.shape
    P = torch.randn(m, r)
    Q, _ = torch.linalg.qr(A @ P)
    return Q

@torch.no_grad()
def update_lora_B(d_w: torch.Tensor, lora_B: torch.Tensor) -> torch.Tensor:
    # W = B @ A
    # dA = B^T @ dW
    # B @ dA = B @ B^T @ dW
    r = lora_B.shape[-1]
    with deepspeed.zero.GatheredParameters(lora_B, modifier_rank=0):
        if deepspeed.comm.get_rank() == 0:
            Q = decompose_d_w(d_w, r)
            lora_B.copy_(Q)
