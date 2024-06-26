import torch
from typing import Tuple


import torch
from typing import Tuple


def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Reshape xq and xk to complex format
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Ensure `freqs_cis` is expanded to align with 21 keypoints and 4 sources
    # Assuming freqs_cis is of shape [max_length, freqs_dim]
    # Add extra dimensions to match the 21 keypoints and 4 sources
    freqs_cis = freqs_cis[:, None, None, :]  # Add singleton dimensions
    
    # Apply rotary embedding using complex multiplication
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    
    # Return the output tensors with the original data type
    return xq_out.type_as(xq), xk_out.type_as(xk)