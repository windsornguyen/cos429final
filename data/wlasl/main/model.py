import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from sklearn.preprocessing import LabelEncoder
from simple_parsing.helpers import Serializable
from xformers.ops.fmha import memory_efficient_attention
from moe import MoeArgs, MoeLayer
from cache import (
    BlockDiagonalMask,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    CacheView,
    RotatingBufferCache,
    RotatingCacheInputMetadata,
)
from rope import precompute_freqs_cis, apply_rotary_emb
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class MoeArgs:
    num_experts: int
    num_experts_per_tok: int


class SimpleInputMetadata:
    def __init__(self, positions: torch.Tensor):
        self.positions = positions

    @staticmethod
    def from_seqlens(seqlens: List[int], device: torch.device) -> 'SimpleInputMetadata':
        return SimpleInputMetadata(
            positions=torch.cat([torch.arange(0, seqlen) for seqlen in seqlens]).to(
                device=device, dtype=torch.long
            )
        )


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


class Attention(nn.Module):
    def __init__(self, args: dict):
        super().__init__()
        self.args = args
        self.n_heads = args['n_heads']
        self.head_dim = args['head_dim']
        self.n_kv_heads = args['n_kv_heads']
        self.repeats = self.n_heads // self.n_kv_heads
        self.scale = self.args['head_dim']**-0.5
        self.wq = nn.Linear(args['dim'], args['n_heads'] * args['head_dim'], bias=False)
        self.wk = nn.Linear(args['dim'], args['n_kv_heads'] * args['head_dim'], bias=False)
        self.wv = nn.Linear(args['dim'], args['n_kv_heads'] * args['head_dim'], bias=False)
        self.wo = nn.Linear(args['n_heads'] * args['head_dim'], args['dim'], bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional['CacheView'],
    ) -> torch.Tensor:
        batch_size, seqlen_sum, num_keypoints, hidden_dim = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(batch_size, seqlen_sum, num_keypoints, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seqlen_sum, num_keypoints, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seqlen_sum, num_keypoints, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if cache is None:
            key, val = xk, xv
        elif cache.prefill:
            key, val = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
        else:
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            key = key.view(
                batch_size, seqlen_sum, num_keypoints, cache.sliding_window, self.n_kv_heads, self.head_dim
            )
            val = val.view(
                batch_size, seqlen_sum, num_keypoints, cache.sliding_window, self.n_kv_heads, self.head_dim
            )

        key, val = repeat_kv(key, val, self.repeats, dim=4)
        
        # Reshape tensors to compatible format for memory_efficient_attention
        xq = xq.view(batch_size, seqlen_sum * num_keypoints, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seqlen_sum * num_keypoints, self.n_heads, self.head_dim).transpose(1, 2)
        val = val.view(batch_size, seqlen_sum * num_keypoints, self.n_heads, self.head_dim).transpose(1, 2)

        output = memory_efficient_attention(
            xq.float(), key.float(), val.float(), None if cache is None else cache.mask
        )

        output = output.transpose(1, 2).contiguous().view(batch_size, seqlen_sum, num_keypoints, self.n_heads * self.head_dim)
        out = self.wo(output)

        return out

class FeedForward(nn.Module):
    def __init__(self, args: dict):
        super().__init__()
        self.w1 = nn.Linear(args['dim'], args['hidden_dim'], bias=False, dtype=torch.float32)
        self.w2 = nn.Linear(args['hidden_dim'], args['dim'], bias=False, dtype=torch.float32)
        self.w3 = nn.Linear(args['dim'], args['hidden_dim'], bias=False, dtype=torch.float32)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, args: dict):
        super().__init__()
        self.n_heads = args['n_heads']
        self.dim = args['dim']
        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args['dim'], eps=args['norm_eps'])
        self.ffn_norm = RMSNorm(args['dim'], eps=args['norm_eps'])
        self.args = args

        if args['moe'] is not None:
            self.feed_forward = MoeLayer(
                experts=[FeedForward(args=args) for _ in range(args['moe']['num_experts'])],
                gate=nn.Linear(args['dim'], args['moe']['num_experts'], bias=False, dtype=torch.float32),
                moe_args=MoeArgs(**args['moe']),
            )
        else:
            self.feed_forward = FeedForward(args=args)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, cache: Optional['CacheView']
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cis, cache)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 84):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(8_192.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1) # Get the sequence length
        pe = self.pe[:seq_len, :].unsqueeze(0).unsqueeze(2)
        return x + pe


class Transformer(nn.Module):
    def __init__(self, args: dict):
        super().__init__()
        self.args = argparse.Namespace(**args)
        self.num_glosses = args['num_glosses']
        self.n_layers = args['n_layers']
        self._precomputed_freqs_cis: Optional[torch.Tensor] = None

        self.left_pos_embedding = nn.Linear(2, args['pos_emb_dim'], dtype=torch.float32)
        self.right_pos_embedding = nn.Linear(2, args['pos_emb_dim'], dtype=torch.float32)
        self.left_disp_embedding = nn.Linear(2, args['disp_emb_dim'], dtype=torch.float32)
        self.right_disp_embedding = nn.Linear(2, args['disp_emb_dim'], dtype=torch.float32)

        self.fusion_layer = nn.Linear(
            2 * (args['pos_emb_dim'] + args['disp_emb_dim']), args['dim'], dtype=torch.float32
        )

        self.temporal_encoding = PositionalEncoding(args['dim'])

        self.norm = RMSNorm(args['dim'], eps=args['norm_eps'])
        self.output = nn.Linear(args['dim'], args['num_glosses'], dtype=torch.float32)

        self.layers = nn.ModuleList(
            [TransformerBlock(args=args) for _ in range(args['n_layers'])]
        )

    @property
    def freqs_cis(self) -> torch.Tensor:
        if self._precomputed_freqs_cis is None:
            self._precomputed_freqs_cis = precompute_freqs_cis(
                self.args.head_dim, 8_192.0, self.args.rope_theta
            )
        return self._precomputed_freqs_cis

    def forward(
        self,
        left_pos: torch.Tensor,
        right_pos: torch.Tensor,
        left_disp: torch.Tensor,
        right_disp: torch.Tensor,
        seqlens: List[int],
        cache: Optional[RotatingBufferCache] = None,
    ) -> torch.Tensor:
        left_pos_emb = self.left_pos_embedding(left_pos.to(torch.float32))
        right_pos_emb = self.right_pos_embedding(right_pos.to(torch.float32))
        left_disp_emb = self.left_disp_embedding(left_disp.to(torch.float32))
        right_disp_emb = self.right_disp_embedding(right_disp.to(torch.float32))

        fused_emb = self.fusion_layer(
            torch.cat(
                [left_pos_emb, right_pos_emb, left_disp_emb, right_disp_emb], dim=-1
            )
        )

        h = fused_emb + self.temporal_encoding(fused_emb)

        freqs_cis = self.freqs_cis[: max(seqlens)]
        for local_layer_id, layer in enumerate(self.layers):
            if cache is not None:
                metadata = cache.get_input_metadata(seqlens)
                cache_view = cache.get_view(local_layer_id, metadata)
            else:
                cache_view = None
            h = layer(h, freqs_cis, cache_view)

        if cache is not None:
            cache.update_seqlens(seqlens)

        h = self.norm(h)
        outs = self.output(h)

        return outs.float()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Unpack the batch data
        left_pos, right_pos, left_disp, right_disp, glosses = batch
        seqlens = [len(lp) for lp in left_pos]

        # Forward pass through the model
        outputs = self(left_pos, right_pos, left_disp, right_disp, seqlens)

        # Compute cross-entropy loss for testing
        loss = nn.functional.cross_entropy(outputs, glosses)

        # Get the most likely gloss predictions
        _, predicted_glosses = torch.max(outputs, dim=1)

        
        # Log test loss and accuracy
        self.log('train_loss', loss)
        self.log('train_accuracy', torch.sum(predicted_glosses == glosses) / glosses.size(0))

        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack the batch data
        left_pos, right_pos, left_disp, right_disp, glosses = batch
        seqlens = [len(lp) for lp in left_pos]

        # Forward pass through the model
        outputs = self(left_pos, right_pos, left_disp, right_disp, seqlens)

        # Compute cross-entropy loss for testing
        loss = nn.functional.cross_entropy(outputs, glosses)

        # Get the most likely gloss predictions
        _, predicted_glosses = torch.max(outputs, dim=1)

        # Log test loss and accuracy
        self.log('val_loss', loss)
        self.log('val_accuracy', torch.sum(predicted_glosses == glosses) / glosses.size(0))

        return loss
    
    def test_step(self, batch, batch_idx):
        # Unpack the batch data
        left_pos, right_pos, left_disp, right_disp, glosses = batch
        seqlens = [len(lp) for lp in left_pos]

        # Forward pass through the model
        outputs = self(left_pos, right_pos, left_disp, right_disp, seqlens)

        # Compute cross-entropy loss for testing
        loss = nn.functional.cross_entropy(outputs, glosses)

        # Get the most likely gloss predictions
        _, predicted_glosses = torch.max(outputs, dim=1)

        # Log test loss and accuracy
        self.log('test_loss', loss)
        self.log('test_accuracy', torch.sum(predicted_glosses == glosses) / glosses.size(0))

        return loss


    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def freqs_cis(self) -> torch.Tensor:
        if self._precomputed_freqs_cis is None:
            theta = self.args.rope_theta
            if theta is None:
                theta = 8_192.0 if self.args.sliding_window is None else 8_192.0
            self._precomputed_freqs_cis = precompute_freqs_cis(
                self.args.head_dim, 16_384.0, theta
            )
        if self._precomputed_freqs_cis.device != self.device:
            self._precomputed_freqs_cis = self._precomputed_freqs_cis.to(
                device=self.device
            )
        return self._precomputed_freqs_cis
