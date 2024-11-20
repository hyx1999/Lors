import torch
import lors_kernels

__all__ = [
    'sparsify_2by4',
]

def sparsify_2by4(x: torch.Tensor) -> torch.Tensor:
    """
    Sparsifies the input tensor according to the 2:4 sparsity pattern. The input tensor must be two-dimensional.
    The sparsification process will be applied along dimension 1.

    Args:
        x (torch.Tensor): The input tensor to be sparsified.

    Returns:
        torch.Tensor: The resulting sparse tensor after applying the 2:4 sparsity pattern.
    """
    assert len(x.shape) == 2 and x.shape[1] % 4 == 0
    return lors_kernels.sparsify_2by4(x)
