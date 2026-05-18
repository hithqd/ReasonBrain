"""ID-Controller used by the textual branch of FRCE.

Given a sequence of entity ("object") tokens extracted from the hypothetical
instruction, the ID-Controller injects fine-grained visual semantics into
these tokens via cross-attention onto the visual cue tensor ``R_V``.

    R_T = FFN( CrossAttn(Q = O,  K, V = R_V) )

We stack ``num_layers`` such blocks so the controller can refine entity
representations multiple times.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class _CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                          batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                kv_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        h, _ = self.attn(qn, kvn, kvn, key_padding_mask=kv_mask,
                         need_weights=False)
        q = q + h
        q = q + self.ff(self.norm_ff(q))
        return q


class IDController(nn.Module):
    """Cross-attention controller mapping object tokens onto visual cues."""

    def __init__(self, dim: int, num_heads: int = 8, num_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            _CrossAttnBlock(dim=dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(dim)

    def forward(self,
                object_tokens: torch.Tensor,
                visual_context: torch.Tensor,
                object_mask: Optional[torch.Tensor] = None,
                visual_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            object_tokens : [B, N_o, D]
            visual_context: [B, N_v, D]
            object_mask   : [B, N_o]  -- currently unused (kept for API symmetry).
            visual_mask   : [B, N_v]  -- True == padded position.

        Returns:
            [B, N_o, D]
        """
        del object_mask  # cross-attention only needs the kv mask.
        h = object_tokens
        for layer in self.layers:
            h = layer(h, visual_context, kv_mask=visual_mask)
        return self.out_norm(h)
