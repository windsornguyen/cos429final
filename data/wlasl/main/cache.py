import torch
from typing import List, Tuple
from dataclasses import dataclass

from xformers.ops.fmha.attn_bias import (
    AttentionBias,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalMask,
)


@dataclass
class RotatingCacheInputMetadata:
    # rope absolute positions
    positions: torch.Tensor
    # which elements in the sequences need to be cached
    to_cache_mask: torch.Tensor
    # how many elements are cached per sequence
    cached_elements: torch.Tensor
    # where tokens should go in the cache
    cache_positions: torch.Tensor

    # if prefill, use block diagonal causal mask
    # else use causal with padded key mask
    prefill: bool
    mask: AttentionBias
    seqlens: List[int]


def interleave_list(l1: List[torch.Tensor], l2: List[torch.Tensor]):
    assert len(l1) == len(l2)
    return [v for pair in zip(l1, l2, strict=True) for v in pair]


def unrotate(cache: torch.Tensor, seqlen: int) -> torch.Tensor:
    assert cache.ndim == 5  # (W, K, H, D)
    position = seqlen % cache.shape[0]
    if seqlen < cache.shape[0]:
        return cache[:seqlen]
    elif position == 0:
        return cache
    else:
        return torch.cat([cache[position:], cache[:position]], dim=0)


class CacheView:
    def __init__(
        self,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        metadata: RotatingCacheInputMetadata,
        kv_seqlens: torch.Tensor,
    ):
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.kv_seqlens = kv_seqlens
        self.metadata = metadata

    def update(self, xk: torch.Tensor, xv: torch.Tensor):
        """
        to_cache_mask masks the last [sliding_window] tokens in each sequence
        """
        batch_size, seqlen_sum, num_keypoints, n_kv_heads, head_dim = xk.shape
        flat_cache_k = self.cache_k.view(batch_size, -1, num_keypoints, n_kv_heads, head_dim)
        flat_cache_v = self.cache_v.view(batch_size, -1, num_keypoints, n_kv_heads, head_dim)

        to_cache_mask = self.metadata.to_cache_mask.view(batch_size, seqlen_sum, -1, 1, 1)
        flat_xk = xk[to_cache_mask].view(-1, num_keypoints, n_kv_heads, head_dim)
        flat_xv = xv[to_cache_mask].view(-1, num_keypoints, n_kv_heads, head_dim)

        cache_positions = self.metadata.cache_positions.view(batch_size, -1)
        flat_cache_k.scatter_(1, cache_positions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_keypoints, n_kv_heads, head_dim), flat_xk)
        flat_cache_v.scatter_(1, cache_positions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, num_keypoints, n_kv_heads, head_dim), flat_xv)

    def interleave_kv(
        self, xk: torch.Tensor, xv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Interleave cached keys and values with new keys and values.
        """
        assert xk.ndim == xv.ndim == 5  # (B, T, K, H, D)
        assert xk.shape == xv.shape

        batch_size, seqlen_sum, num_keypoints, n_kv_heads, head_dim = xk.shape

        if all([s == 0 for s in self.metadata.seqlens]):
            # No cache to interleave
            return xk, xv

        # Use the same device as xk (assuming xk and xv are on the same device)
        device = xk.device

        # Move cache tensors to the same device as xk
        cache_k = self.cache_k.to(device)[:batch_size, :, :, :, :].view(batch_size, -1, num_keypoints, n_kv_heads, head_dim)
        cache_v = self.cache_v.to(device)[:batch_size, :, :, :, :].view(batch_size, -1, num_keypoints, n_kv_heads, head_dim)

        # Create a mask to identify positions where the cache should be used
        cache_mask = torch.cat([torch.ones(s, dtype=torch.bool, device=device) if s > 0 else torch.zeros(seqlen, dtype=torch.bool, device=device) for s, seqlen in zip(self.kv_seqlens, self.metadata.seqlens)])
        cache_mask = cache_mask.view(batch_size, seqlen_sum, 1, 1, 1)

        # Interleave cached and new keys/values based on the cache mask
        interleaved_k = torch.where(cache_mask, cache_k, xk)
        interleaved_v = torch.where(cache_mask, cache_v, xv)

        return interleaved_k, interleaved_v


    @property
    def sliding_window(self):
        return self.cache_k.shape[1]

    @property
    def key(self) -> torch.Tensor:
        return self.cache_k[: len(self.kv_seqlens)]

    @property
    def value(self) -> torch.Tensor:
        return self.cache_v[: len(self.kv_seqlens)]

    @property
    def prefill(self):
        return self.metadata.prefill

    @property
    def mask(self):
        return self.metadata.mask


class RotatingBufferCache:
    """
    This is an example that implements a less naive rotating buffer cache, allowing for variable length sequences.
    Allocated cache is rectangular which is wasteful (see PagedAttention for better mechanisms)
    """

    def __init__(
        self,
        n_layers: int,
        max_batch_size: int,
        sliding_window: int,
        n_kv_heads: int,
        head_dim: int,
        num_keypoints: int,
    ):
        self.sliding_window = sliding_window
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.num_keypoints = num_keypoints

        self.cache_k = torch.empty(
            (n_layers, max_batch_size, sliding_window, num_keypoints, n_kv_heads, head_dim)
        )
        self.cache_v = torch.empty(
            (n_layers, max_batch_size, sliding_window, num_keypoints, n_kv_heads, head_dim)
        )
        # holds the valid length for each batch element in the cache
        self.kv_seqlens = None

    def get_view(
        self, layer_id: int, metadata: RotatingCacheInputMetadata
    ) -> CacheView:
        return CacheView(
            self.cache_k[layer_id], self.cache_v[layer_id], metadata, self.kv_seqlens
        )

    def reset(self):
        self.kv_seqlens = None

    def init_kvseqlens(self, batch_size: int):
        self.kv_seqlens = torch.zeros(
            (batch_size,), device=self.device, dtype=torch.long
        )

    @property
    def device(self):
        return self.cache_k.device

    def to(self, device: torch.device, dtype: torch.dtype):
        self.cache_k = self.cache_k.to(device=device, dtype=dtype)
        self.cache_v = self.cache_v.to(device=device, dtype=dtype)

        return self

    def update_seqlens(self, seqlens: List[int]):
        self.kv_seqlens += torch.tensor(seqlens, device=self.device, dtype=torch.long)

    def get_input_metadata(self, seqlens: List[int]) -> RotatingCacheInputMetadata:
        """
        inpput = seqlens [5,7,2] // seqpos [0, 1, 3] // sliding_window 3
        --> only cache last 3 tokens in each sequence
        - to_cache_mask = [0 0 1 1 1 | 0 0 0 0 1 1 1 | 1 1]
        - cached_elements = [3 | 3 | 2]
        --> absolute positions are used for rope
        - positions = [0 1 2 3 4 | 1 2 3 4 5 6 7 | 3 4]
        --> cache positions are positions cache_masked, modulo sliding_window + batch_idx * sliding_window
        - cache_positions = [2 0 1 | 5 3 4 | 6 7]
        """
        if self.kv_seqlens is None:
            self.init_kvseqlens(len(seqlens))
        assert (
            len(seqlens) == len(self.kv_seqlens)
        ), f'Batch size is {len(self.kv_seqlens)}, got {len(seqlens)}, did you forget to reset cache?'
        seqpos = self.kv_seqlens.tolist()

        assert len(seqlens) > 0, seqlens
        masks = [
            [x >= seqlen - self.sliding_window for x in range(seqlen)]
            for seqlen in seqlens
        ]
        to_cache_mask = torch.tensor(
            sum(masks, []), device=self.device, dtype=torch.bool
        )
        cached_elements = torch.tensor(
            [sum(mask) for mask in masks], device=self.device, dtype=torch.long
        )
        positions = torch.cat(
            [
                torch.arange(pos, pos + seqlen)
                for pos, seqlen in zip(seqpos, seqlens, strict=True)
            ]
        ).to(device=self.device, dtype=torch.long)
        batch_idx = torch.tensor(
            sum([[i] * seqlen for i, seqlen in enumerate(seqlens)], []),
            device=self.device,
            dtype=torch.long,
        )
        cache_positions = (
            positions % self.sliding_window + batch_idx * self.sliding_window
        )

        first_prefill = seqpos[0] == 0
        subsequent_prefill = any(seqlen > 1 for seqlen in seqlens)
        if first_prefill:
            assert all([pos == 0 for pos in seqpos]), seqpos
            mask = BlockDiagonalCausalMask.from_seqlens(seqlens).make_local_attention(
                self.sliding_window
            )
        elif subsequent_prefill:
            mask = BlockDiagonalMask.from_seqlens(
                q_seqlen=seqlens,
                kv_seqlen=[
                    s + cached_s.clamp(max=self.sliding_window).item()
                    for (s, cached_s) in zip(seqlens, self.kv_seqlens, strict=True)
                ],
            ).make_local_attention_from_bottomright(self.sliding_window)
        else:
            mask = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=seqlens,
                kv_padding=self.sliding_window,
                kv_seqlen=(self.kv_seqlens + cached_elements)
                .clamp(max=self.sliding_window)
                .tolist(),
            )

        return RotatingCacheInputMetadata(
            positions=positions,
            to_cache_mask=to_cache_mask,
            cached_elements=cached_elements,
            cache_positions=cache_positions[to_cache_mask],
            prefill=first_prefill or subsequent_prefill,
            mask=mask,
            seqlens=seqlens,
        )