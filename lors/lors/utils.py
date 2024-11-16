import torch
from accelerate import PartialState

state = PartialState()

@torch.no_grad()
def update_lora_B(d_w: torch.Tensor, lora_B: torch.Tensor) -> torch.Tensor:
    # W = B @ A
    # dA = B^T @ dW
    # B @ dA = B @ B^T @ dW
    state.on_main_process
    def decompose_d_w(A: torch.Tensor, r: int):  # min |A - Q Q^T A| => B = Q
        n, m = A.shape
        P = torch.randn(m, r)
        Q, _ = torch.linalg.qr(A @ P)
        return Q
    r = lora_B.shape[-1]
    Q = decompose_d_w(d_w, r)
    lora_B.copy_(Q)
