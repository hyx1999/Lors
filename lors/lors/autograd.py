import torch
from typing import Optional
from ..lors_ops import sparsify_2by4

class SparsifyFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor):  # type: ignore[override]
        x = sparsify_2by4(x)
        return x

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        # We just return grad_out, since we just use STE - straight through estimation
        return grad_out

@torch._dynamo.allow_in_graph
def sparsify(
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Sparsifies a dense tensor into a semi-structured tensor, according to the algo and backend passed.
    """
    x = SparsifyFunc.apply(x)
    return x
