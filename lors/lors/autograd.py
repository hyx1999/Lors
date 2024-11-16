import torch
from typing import Optional

class SparsifyFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor):  # type: ignore[override]
        x_shape = x.shape
        x = x.unfold(-1, 4, 4)
        _, index = torch.topk(x.abs(), k=2, dim=-1, largest=False)
        x.scatter_(dim=-1, index=index, value=0)
        x = x.view(x_shape)
        return x

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        # We just return grad_out, since we just use STE - straight through estimation
        return grad_out

@torch._dynamo.allow_in_graph
def sparsify(
    x: torch.Tensor,
    order: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Sparsifies a dense tensor into a semi-structured tensor, according to the algo and backend passed.
    """
    x = SparsifyFunc.apply(x, order)
    return x
