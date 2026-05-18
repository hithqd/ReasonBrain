"""QFormer aligning MLLM hidden states to the diffusion conditioning space.

The paper uses a 6-layer transformer with 77 learnable query tokens. Given the
``r`` learnable image tokens that LLaVA outputs (their hidden states ``V``),
the QFormer produces a fixed-length sequence ``\\hat V`` that is suitable as
context for the diffusion transformer.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class _QFormerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                               batch_first=True)
        self.norm_cross_q = nn.LayerNorm(dim)
        self.norm_cross_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                                batch_first=True)
        hidden = int(dim * mlp_ratio)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                kv_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # self-attention over the queries
        h = self.norm_self(q)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        q = q + h
        # cross-attention onto MLLM hidden states
        h = self.cross_attn(
            self.norm_cross_q(q),
            self.norm_cross_kv(kv),
            self.norm_cross_kv(kv),
            key_padding_mask=kv_mask,
            need_weights=False,
        )[0]
        q = q + h
        # FFN
        q = q + self.ffn(self.norm_ffn(q))
        return q


class QFormer(nn.Module):
    """Cross-attention based projector MLLM hidden states -> diffusion cond."""

    def __init__(self,
                 mllm_hidden_dim: int,
                 hidden_dim: int = 768,
                 num_layers: int = 6,
                 num_queries: int = 77,
                 num_heads: int = 12,
                 dropout: float = 0.0):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.input_proj = nn.Linear(mllm_hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            _QFormerBlock(hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, mllm_hidden: torch.Tensor,
                mllm_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            mllm_hidden: [B, T, C_mllm]  hidden states of the r MLLM tokens.
            mllm_mask  : [B, T]          True == padded.

        Returns:
            [B, num_queries, hidden_dim]
        """
        B = mllm_hidden.size(0)
        kv = self.input_proj(mllm_hidden)
        q = self.queries.expand(B, -1, -1)
        for blk in self.blocks:
            q = blk(q, kv, kv_mask=mllm_mask)
        return self.out_norm(q)
