import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import logging
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
class ModelArgs(Serializable):
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int

    max_batch_size: int = 0

    # For rotary embeddings. If not set, will be infered from sliding window.
    rope_theta: Optional[float] = None
    # If this is set, use sliding window attention rotating cache.
    sliding_window: Optional[int] = None
    # If this is set, we will use MoE layers instead of dense layers.
    moe: Optional[MoeArgs] = None

    # Added parameters for ASL-to-text task
    pos_emb_dim: int = 128
    disp_emb_dim: int = 128
    num_glosses: int = 2000 # Will be more after we add ASL Citizen

@dataclass
class SimpleInputMetadata:
    # rope absolute positions
    positions: torch.Tensor

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
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.head_dim: int = args.head_dim
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[CacheView],
    ) -> torch.Tensor:
        seqlen_sum, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)
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
                seqlen_sum * cache.sliding_window, self.n_kv_heads, self.head_dim
            )
            val = val.view(
                seqlen_sum * cache.sliding_window, self.n_kv_heads, self.head_dim
            )

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        output = memory_efficient_attention(
            xq, key, val, None if cache is None else cache.mask
        )

        return self.wo(output.view(seqlen_sum, self.n_heads * self.head_dim))


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

        self.feed_forward: nn.Module
        if args.moe is not None:
            self.feed_forward = MoeLayer(
                experts=[FeedForward(args=args) for _ in range(args.moe.num_experts)],
                gate=nn.Linear(args.dim, args.moe.num_experts, bias=False),
                moe_args=args.moe,
            )
        else:
            self.feed_forward = FeedForward(args=args)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, cache: Optional[CacheView]
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cis, cache)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class Transformer(pl.LightningModule):
    def __init__(
        self,
        args: ModelArgs,
    ):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args
        self.num_glosses = args.num_glosses
        self.n_layers = args.n_layers
        self._precomputed_freqs_cis: Optional[torch.Tensor] = None
        
        # Embedding layers for hand positions and displacements
        self.left_pos_embedding = nn.Linear(21 * 2, args.pos_emb_dim)
        self.right_pos_embedding = nn.Linear(21 * 2, args.pos_emb_dim)
        self.left_disp_embedding = nn.Linear(21 * 2, args.disp_emb_dim)
        self.right_disp_embedding = nn.Linear(21 * 2, args.disp_emb_dim)
        
        # Fusion layer to combine embeddings
        self.fusion_layer = nn.Linear(2 * (args.pos_emb_dim + args.disp_emb_dim), args.dim)
        
        # Temporal encoding
        self.temporal_encoding = PositionalEncoding(args.dim)
        
        # Output layers
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.num_glosses)
        
        # Initialize transformer layers
        self.layers = nn.ModuleList([TransformerBlock(args=args) for _ in range(args.n_layers)])

    def forward(
        self,
        left_pos: torch.Tensor,
        right_pos: torch.Tensor,
        left_disp: torch.Tensor,
        right_disp: torch.Tensor,
        seqlens: List[int],
        cache: Optional[RotatingBufferCache] = None,
    ) -> torch.Tensor:
        # Compute embeddings for hand positions and displacements
        left_pos_emb = self.left_pos_embedding(left_pos)
        right_pos_emb = self.right_pos_embedding(right_pos)
        left_disp_emb = self.left_disp_embedding(left_disp)
        right_disp_emb = self.right_disp_embedding(right_disp)
        
        # Fuse embeddings
        fused_emb = self.fusion_layer(torch.cat([left_pos_emb, right_pos_emb, left_disp_emb, right_disp_emb], dim=-1))
        
        # Add temporal encoding
        h = fused_emb + self.temporal_encoding(fused_emb)
        
        # Transformer layers
        freqs_cis = self.freqs_cis[:max(seqlens)]
        for local_layer_id, layer in enumerate(self.layers):
            if cache is not None:
                cache_view = cache.get_view(local_layer_id, seqlens)
            else:
                cache_view = None
            h = layer(h, freqs_cis, cache_view)

        if cache is not None:
            cache.update_seqlens(seqlens)
        
        # Final normalization and output
        h = self.norm(h)
        outs = self.output(h)
        
        return outs.float()
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        left_pos, right_pos, left_disp, right_disp, glosses = batch
        seqlens = [len(lp) for lp in left_pos]
        outputs = self(left_pos, right_pos, left_disp, right_disp, seqlens)
        loss = nn.functional.cross_entropy(outputs.view(-1, self.num_glosses), glosses.view(-1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        left_pos, right_pos, left_disp, right_disp, glosses = batch
        seqlens = [len(lp) for lp in left_pos] 
        outputs = self(left_pos, right_pos, left_disp, right_disp, seqlens)
        loss = nn.functional.cross_entropy(outputs.view(-1, self.num_glosses), glosses.view(-1))
        self.log('val_loss', loss, prog_bar=True)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def freqs_cis(self) -> torch.Tensor:
        # We cache freqs_cis but need to take care that it is on the right device
        # and has the right dtype (complex64). The fact that the dtype is different
        # from the module's  dtype means we cannot register it as a buffer
        if self._precomputed_freqs_cis is None:
            # If no sliding window, assume a larger seqlen
            theta = self.args.rope_theta
            if theta is None:
                theta = 1000000.0 if self.args.sliding_window is None else 10000.0
            # theta = 10000.
            self._precomputed_freqs_cis = precompute_freqs_cis(
                self.args.head_dim, 128_000, theta
            )
        if self._precomputed_freqs_cis.device != self.device:
            self._precomputed_freqs_cis = self._precomputed_freqs_cis.to(
                device=self.device
            )
        return self._precomputed_freqs_cis

    def forward_partial(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        cache: Optional[RotatingBufferCache] = None,
    ) -> torch.Tensor:
        """Local forward pass.

        If doing pipeline parallelism, this will return the activations of the last layer of this stage.
        For the last stage, this will return the normalized final embeddings.
        """
        assert (
            len(seqlens) <= self.args.max_batch_size
        ), f'Max batch size is {self.args.max_batch_size}, got batch size of {len(seqlens)}'
        (num_toks,) = input_ids.shape
        assert sum(seqlens) == num_toks, (sum(seqlens), num_toks)
        if cache is not None:
            input_metadata = cache.get_input_metadata(seqlens)
        else:
            input_metadata = SimpleInputMetadata.from_seqlens(seqlens, self.device)

        if self.pipeline_rank == 0:
            assert self.tok_embeddings is not None
            h = self.tok_embeddings(input_ids)
        else:
            h = torch.empty(
                num_toks, self.args.dim, device=self.device, dtype=self.dtype
            )
            torch.distributed.recv(h, src=self.pipeline_rank - 1)

        freqs_cis = self.freqs_cis[input_metadata.positions]

        for local_layer_id, layer in enumerate(self.layers.values()):
            if cache is not None:
                assert input_metadata is not None
                cache_view = cache.get_view(local_layer_id, input_metadata)
            else:
                cache_view = None
            h = layer(h, freqs_cis, cache_view)

        if cache is not None:
            cache.update_seqlens(seqlens)
        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            torch.distributed.send(h, dst=self.pipeline_rank + 1)
            return h
        else:
            # Last rank has a final normalization step.
            assert self.norm is not None
            return self.norm(h)


# Mixture of Experts (MoE)
@dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int


# Rotating Buffer Cache
class RotatingBufferCache:
    def __init__(
        self,
        n_layers: int,
        max_batch_size: int,
        sliding_window: int,
        n_kv_heads: int,
        head_dim: int,
    ):
        self.sliding_window = sliding_window
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.cache_k = torch.empty(
            (n_layers, max_batch_size, sliding_window, n_kv_heads, head_dim)
        )
        self.cache_v = torch.empty(
            (n_layers, max_batch_size, sliding_window, n_kv_heads, head_dim)
        )
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


# ASL Vision Transformer
class ASLVisionTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_layers = args.n_layers
        self._precomputed_freqs_cis: Optional[torch.Tensor] = None

        # Positional encoding
        # Use RoPE

        # Input embedding
        self.input_embedding = nn.Linear(args.input_dim, args.dim)

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(args=args) for _ in range(args.n_layers)]
        )

        # Rotating buffer cache
        if args.sliding_window is not None:
            self.cache = RotatingBufferCache(
                n_layers=args.n_layers,
                max_batch_size=args.max_batch_size,
                sliding_window=args.sliding_window,
                n_kv_heads=args.n_kv_heads,
                head_dim=args.head_dim,
            )
        else:
            self.cache = None

        # Output layer
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.num_classes)

    def forward(self, x: torch.Tensor, seqlens: List[int]) -> torch.Tensor:
        # Input embedding
        x = self.input_embedding(x)

        # Positional encoding
        x = self.pos_encoder(x)

        # Transformer layers with rotating buffer cache
        if self.cache is not None:
            metadata = self.cache.get_input_metadata(seqlens)
            self.cache.update_seqlens(seqlens)
            for i, layer in enumerate(self.layers):
                x = layer(
                    x, freqs_cis=self.freqs_cis, cache=self.cache.get_view(i, metadata)
                )
        else:
            for layer in self.layers:
                x = layer(x, freqs_cis=self.freqs_cis, cache=None)

        # Output layer
        x = self.norm(x)
        x = x.mean(dim=1)  # Average pooling
        output = self.output(x)

        return output

    @property
    def freqs_cis(self) -> torch.Tensor:
        if self._precomputed_freqs_cis is None:
            theta = self.args.rope_theta
            if theta is None:
                theta = 1000000.0 if self.args.sliding_window is None else 10000.0
            self._precomputed_freqs_cis = precompute_freqs_cis(
                self.args.head_dim, 128_000, theta
            )
        if self._precomputed_freqs_cis.device != self.device:
            self._precomputed_freqs_cis = self._precomputed_freqs_cis.to(
                device=self.device
            )
        return self._precomputed_freqs_cis


def prepare_data(data_list, max_length):
    X, y = [], []
    label_encoder = LabelEncoder()
    all_words = [data['word'] for data in data_list]
    labels = label_encoder.fit_transform(all_words)
    label_dict = dict(zip(all_words, labels, strict=True))

    for data in data_list:
        word = data['word']
        positions = data['positions']
        lefts = [point for sublist in positions['leftpositions'] for point in sublist]
        rights = [point for sublist in positions['rightpositions'] for point in sublist]
        features = lefts + rights
        if len(features) < max_length:
            features += [(0, 0)] * (max_length - len(features))
        else:
            features = features[:max_length]  # Truncate if longer than max_length
        features_flat = [item for sublist in features for item in sublist]
        X.append(features_flat)
        y.append(label_dict[word])

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
        label_encoder,
    )


class SignLanguageDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, batch_size, max_length):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.max_length = max_length

    def setup(self, stage=None):
        self.train_X, self.train_y, _ = prepare_data(self.train_data, self.max_length)
        self.val_X, self.val_y, _ = prepare_data(self.val_data, self.max_length)
        self.test_X, self.test_y, _ = prepare_data(self.test_data, self.max_length)

    def train_dataloader(self):
        train_dataset = torch.utils.data.TensorDataset(self.train_X, self.train_y)
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        val_dataset = torch.utils.data.TensorDataset(self.val_X, self.val_y)
        return torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        test_dataset = torch.utils.data.TensorDataset(self.test_X, self.test_y)
        return torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size)


class SignLanguageClassifier(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_encoder_layers,
        num_classes,
        dropout,
        max_length,
    ):
        super().__init__()
        self.model = ASLVisionTransformer(
            input_dim,
            d_model,
            nhead,
            num_encoder_layers,
            num_classes,
            dropout,
            max_length,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        self.log(
            'train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Log gradient norm
        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm**0.5
        self.log('grad_norm', grad_norm, on_step=True, prog_bar=True, logger=True)
