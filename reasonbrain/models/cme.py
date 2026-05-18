"""Cross-Modal Enhancer (CME).

Implements §4.3 of the paper. The MLLM compresses the multimodal context into
``r`` learnable tokens whose hidden states are then projected to the diffusion
conditioning space via a QFormer. This compression inevitably drops fine
details, so the CME *re-injects* them via two symmetric branches:

    * Visual-oriented enhancer:
          F1 = CrossAttn(Q, V_hat)                 # Q is learnable
          F2 = CrossAttn(E_I(I), R_V)              # local visual cues
          \bar V = LN(F1 + F2)                      # residual fusion
          \bar R_V = CrossAttn(\bar V, [F2, R_V])   # final visual cond
    * Text-oriented enhancer: identical, replacing
          E_I(I) -> E_T(H)  and  R_V -> R_T.

Each enhancer is built of ``num_blocks`` mixed cross-attention blocks (the
paper uses 5). The implementation below is a clean PyTorch translation that
keeps the math but stays parameter-light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class _CrossAttn(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                          batch_first=True)

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                kv_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        qn = self.norm_q(q)
        kn = self.norm_kv(kv)
        out, _ = self.attn(qn, kn, kn, key_padding_mask=kv_mask,
                           need_weights=False)
        return q + out


class _FFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


class _MixedBlock(nn.Module):
    """One mixed cross-attention block from §4.3.

    Inputs:
        q        : learnable / refined queries          [B, Nq, D]
        v_hat    : MLLM-derived features                [B, Nv, D]
        e_modal  : modality-specific features           [B, Ne, D]
                   (image patch tokens for visual branch,
                    text token embeddings for text branch)
        r_modal  : modality-specific FRCE cues          [B, Nr, D]
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn_q_vhat = _CrossAttn(dim, num_heads, dropout)
        self.attn_e_r = _CrossAttn(dim, num_heads, dropout)
        self.attn_fuse = _CrossAttn(dim, num_heads, dropout)
        self.norm_fuse = nn.LayerNorm(dim)
        self.ffn = _FFN(dim, dropout=dropout)

    def forward(self,
                q: torch.Tensor,
                v_hat: torch.Tensor,
                e_modal: torch.Tensor,
                r_modal: torch.Tensor,
                v_hat_mask: Optional[torch.Tensor] = None,
                e_mask: Optional[torch.Tensor] = None,
                r_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # F1 : Q attends to V_hat (MLLM features)
        f1 = self.attn_q_vhat(q, v_hat, kv_mask=v_hat_mask)
        # F2 : E_modal attends to R_modal (FRCE cues)
        f2 = self.attn_e_r(e_modal, r_modal, kv_mask=r_mask)
        # Fuse: residual norm of F1 with F2 (project F2 onto F1 via attention)
        f1_aug = self.attn_fuse(f1, f2, kv_mask=e_mask)
        v_bar = self.norm_fuse(f1_aug)
        v_bar = self.ffn(v_bar)
        return v_bar, f2


# ---------------------------------------------------------------------------
# Per-modality enhancer (visual OR text)
# ---------------------------------------------------------------------------
class ModalityEnhancer(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_blocks: int,
                 num_queries: int, dropout: float = 0.0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.blocks = nn.ModuleList([
            _MixedBlock(dim=dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.final_attn = _CrossAttn(dim, num_heads, dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self,
                v_hat: torch.Tensor,
                e_modal: torch.Tensor,
                r_modal: torch.Tensor,
                v_hat_mask: Optional[torch.Tensor] = None,
                e_mask: Optional[torch.Tensor] = None,
                r_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the enhanced modality-specific embedding ``\\bar R_modal``."""
        B = v_hat.size(0)
        q = self.queries.expand(B, -1, -1)
        f2_last: Optional[torch.Tensor] = None
        for blk in self.blocks:
            q, f2_last = blk(
                q, v_hat, e_modal, r_modal,
                v_hat_mask=v_hat_mask, e_mask=e_mask, r_mask=r_mask,
            )
        # final cross-attention: \bar V attends to [F2, R_modal]
        assert f2_last is not None
        ctx = torch.cat([f2_last, r_modal], dim=1)
        if e_mask is not None or r_mask is not None:
            mE = e_mask if e_mask is not None else torch.zeros(
                B, f2_last.size(1), dtype=torch.bool, device=v_hat.device)
            mR = r_mask if r_mask is not None else torch.zeros(
                B, r_modal.size(1), dtype=torch.bool, device=v_hat.device)
            ctx_mask = torch.cat([mE, mR], dim=1)
        else:
            ctx_mask = None
        r_bar = self.final_attn(q, ctx, kv_mask=ctx_mask)
        return self.final_norm(r_bar)


# ---------------------------------------------------------------------------
# Full CME = visual enhancer + text enhancer
# ---------------------------------------------------------------------------
@dataclass
class CMEOutput:
    visual: torch.Tensor   # \bar R_V  [B, Nq, D]
    text: torch.Tensor     # \bar R_T  [B, Nq, D]


class CME(nn.Module):
    """Cross-Modal Enhancer — two symmetric branches."""

    def __init__(self, dim: int = 1024, num_heads: int = 16,
                 num_blocks: int = 5, num_queries: int = 77,
                 dropout: float = 0.0):
        super().__init__()
        self.visual_enhancer = ModalityEnhancer(
            dim=dim, num_heads=num_heads, num_blocks=num_blocks,
            num_queries=num_queries, dropout=dropout,
        )
        self.text_enhancer = ModalityEnhancer(
            dim=dim, num_heads=num_heads, num_blocks=num_blocks,
            num_queries=num_queries, dropout=dropout,
        )

    def forward(self,
                v_hat: torch.Tensor,
                image_feats: torch.Tensor,
                text_feats: torch.Tensor,
                r_v: torch.Tensor,
                r_t: torch.Tensor,
                v_hat_mask: Optional[torch.Tensor] = None,
                image_mask: Optional[torch.Tensor] = None,
                text_mask: Optional[torch.Tensor] = None,
                rv_mask: Optional[torch.Tensor] = None,
                rt_mask: Optional[torch.Tensor] = None) -> CMEOutput:
        bar_rv = self.visual_enhancer(
            v_hat=v_hat, e_modal=image_feats, r_modal=r_v,
            v_hat_mask=v_hat_mask, e_mask=image_mask, r_mask=rv_mask,
        )
        bar_rt = self.text_enhancer(
            v_hat=v_hat, e_modal=text_feats, r_modal=r_t,
            v_hat_mask=v_hat_mask, e_mask=text_mask, r_mask=rt_mask,
        )
        return CMEOutput(visual=bar_rv, text=bar_rt)
