import torch
import lors_kernels
from typing import Optional

__all__ = [
    'sparsify_6by8',
    'sparsify_3by4',
    'sparsify_2by4',
]

def sparsify_6by8(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sparsifies the input tensor according to the 2:4 sparsity pattern. The input tensor must be two-dimensional.
    The sparsification process will be applied along dimension 1.

    Args:
        x (torch.Tensor): The input tensor to be sparsified.

    Returns:
        torch.Tensor: The resulting sparse tensor after applying the 2:4 sparsity pattern.
    """
    assert len(x.shape) == 2 and x.shape[1] % 8 == 0
    if scale is None:
        scale = torch.ones((x.shape[1],), dtype=x.dtype, device=x.device)
    return lors_kernels.sparsify_6by8(x, scale)


def sparsify_3by4(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sparsifies the input tensor according to the 2:4 sparsity pattern. The input tensor must be two-dimensional.
    The sparsification process will be applied along dimension 1.

    Args:
        x (torch.Tensor): The input tensor to be sparsified.

    Returns:
        torch.Tensor: The resulting sparse tensor after applying the 2:4 sparsity pattern.
    """
    assert len(x.shape) == 2 and x.shape[1] % 4 == 0
    if scale is None:
        scale = torch.ones((x.shape[1],), dtype=x.dtype, device=x.device)
    return lors_kernels.sparsify_3by4(x, scale)


def sparsify_2by4(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sparsifies the input tensor according to the 2:4 sparsity pattern. The input tensor must be two-dimensional.
    The sparsification process will be applied along dimension 1.

    Args:
        x (torch.Tensor): The input tensor to be sparsified.

    Returns:
        torch.Tensor: The resulting sparse tensor after applying the 2:4 sparsity pattern.
    """
    assert len(x.shape) == 2 and x.shape[1] % 4 == 0
    if scale is None:
        scale = torch.ones((x.shape[1],), dtype=x.dtype, device=x.device)
    return lors_kernels.sparsify_2by4(x, scale)
