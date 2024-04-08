from typing import Optional

import torch
import math
def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute mean of masked values by soft blending.

    Args:
        x (types.Array): Input array of shape (...,).
        mask (types.Array): Mask array in [0, 1]. Shape will be broadcast to
            match x.

    Returns:
        types.Array: Masked mean of x of shape ().
        :param x:
        :param mask:
        :param eps:
    """
    if mask is None:
        return x.mean()

    mask = torch.broadcast_to(mask, x.shape)
    return (x * mask).sum() / mask.sum().clip(eps)  # type: ignore


def compute_psnr(img0: torch.Tensor, img1: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute PSNR between two images.

    Args:
        img0 (jnp.ndarray): An image of shape (H, W, 3) in float32.
        img1 (jnp.ndarray): An image of shape (H, W, 3) in float32.
        mask (Optional[jnp.ndarray]): An optional foreground mask of shape (H,
            W, 1) in float32 {0, 1}. The metric is computed only on the pixels
            with mask == 1.

    Returns:
        jnp.ndarray: PSNR in dB of shape ().
    """
    mse = (img0 - img1) ** 2
    return -10.0 / math.log(10) * torch.log(masked_mean(mse, mask))