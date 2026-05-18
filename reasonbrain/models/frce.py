"""Fine-grained Reasoning Cue Extraction (FRCE) module.

This module implements §4.2 of the paper. It exposes two branches:

* **Visual reasoning cues** combining
    * a *local* MAE-style patch encoder  -> R_local
    * a *global* SAM-region encoder      -> R_global
  fused into a single visual cue tensor R_V.

* **Textual reasoning cues** that
    * tokenize entity names extracted from the hypothetical instruction
    * apply an ID-Controller (cross-attention + FFN) conditioned on R_V
      to produce a textual cue tensor R_T.

The image / text encoders are intentionally lightweight wrappers around the
HuggingFace CLIP backbone so that other backbones (DINOv2, EVA-CLIP, ...) can
be swapped in with minimal changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .id_controller import IDController


# ---------------------------------------------------------------------------
# Lightweight transformer block used by both branches
# ---------------------------------------------------------------------------
class _TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Local patch extractor (MAE-style on top of CLIP patch features)
# ---------------------------------------------------------------------------
class PatchExtractor(nn.Module):
    """Refines CLIP patch tokens with a few transformer blocks."""

    def __init__(self, in_dim: int, out_dim: int, num_layers: int = 4,
                 num_heads: int = 8):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, out_dim)
        self.blocks = nn.ModuleList([
            _TransformerBlock(out_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: [B, N, C_in]  (CLIP patch features without the CLS token)

        Returns:
            [B, N, out_dim]
        """
        h = self.proj_in(patch_tokens)
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)


# ---------------------------------------------------------------------------
# Global region extractor (SAM mask-pooled features)
# ---------------------------------------------------------------------------
class RegionExtractor(nn.Module):
    """Aggregates per-region features derived from SAM masks."""

    def __init__(self, in_dim: int, out_dim: int, num_layers: int = 2,
                 num_heads: int = 8, max_regions: int = 64):
        super().__init__()
        self.max_regions = max_regions
        self.proj_in = nn.Linear(in_dim, out_dim)
        self.blocks = nn.ModuleList([
            _TransformerBlock(out_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, region_feats: torch.Tensor,
                region_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            region_feats: [B, R, C_in]
            region_mask:  [B, R]  with True for *padded* (invalid) regions.

        Returns:
            [B, R, out_dim]
        """
        h = self.proj_in(region_feats)
        for blk in self.blocks:
            h = blk(h, key_padding_mask=region_mask)
        return self.norm(h)


# ---------------------------------------------------------------------------
# Full FRCE
# ---------------------------------------------------------------------------
@dataclass
class FRCEOutput:
    """Container for FRCE outputs."""

    visual_cues: torch.Tensor            # [B, N_v, D]
    text_cues: torch.Tensor              # [B, N_t, D]
    visual_mask: Optional[torch.Tensor]  # [B, N_v] (True == pad)
    text_mask: Optional[torch.Tensor]    # [B, N_t]


class FRCE(nn.Module):
    """Fine-grained Reasoning Cue Extraction module.

    The module expects pre-encoded CLIP features (patch tokens + global CLS
    embeddings) plus SAM region features.  Splitting feature extraction
    keeps this module decoupled from concrete encoders.
    """

    def __init__(self,
                 patch_in_dim: int = 1024,
                 region_in_dim: int = 256,
                 text_in_dim: int = 768,
                 out_dim: int = 1024,
                 patch_layers: int = 4,
                 region_layers: int = 2,
                 id_controller_layers: int = 2,
                 num_heads: int = 8,
                 max_objects: int = 16):
        super().__init__()
        self.out_dim = out_dim
        self.max_objects = max_objects

        # ---- visual branch ----
        self.patch_extractor = PatchExtractor(
            in_dim=patch_in_dim, out_dim=out_dim,
            num_layers=patch_layers, num_heads=num_heads,
        )
        self.region_extractor = RegionExtractor(
            in_dim=region_in_dim, out_dim=out_dim,
            num_layers=region_layers, num_heads=num_heads,
        )

        # Learnable type embeddings let downstream attention distinguish
        # local vs. global tokens after concatenation.
        self.local_type_emb = nn.Parameter(torch.zeros(1, 1, out_dim))
        self.global_type_emb = nn.Parameter(torch.zeros(1, 1, out_dim))
        nn.init.trunc_normal_(self.local_type_emb, std=0.02)
        nn.init.trunc_normal_(self.global_type_emb, std=0.02)

        # ---- textual branch ----
        self.text_proj_in = nn.Linear(text_in_dim, out_dim)
        self.id_controller = IDController(
            dim=out_dim, num_heads=num_heads,
            num_layers=id_controller_layers,
        )

    # ------------------------------------------------------------------
    def forward(self,
                patch_tokens: torch.Tensor,
                region_feats: torch.Tensor,
                region_mask: Optional[torch.Tensor],
                object_tokens: torch.Tensor,
                object_mask: Optional[torch.Tensor] = None) -> FRCEOutput:
        """
        Args:
            patch_tokens : [B, N_p, C_patch] CLIP patch features (no CLS).
            region_feats : [B, R,   C_reg]   SAM-pooled region features.
            region_mask  : [B, R]            True = padded region.
            object_tokens: [B, N_o, C_text]  text embeddings for entity names.
            object_mask  : [B, N_o]          True = padded entity.

        Returns:
            FRCEOutput.
        """
        B = patch_tokens.size(0)

        # ---- visual branch ----
        r_local = self.patch_extractor(patch_tokens)               # [B, N_p, D]
        r_global = self.region_extractor(region_feats, region_mask)  # [B, R, D]

        r_local = r_local + self.local_type_emb
        r_global = r_global + self.global_type_emb

        # Build mask for the concatenated visual cue sequence.
        if region_mask is not None:
            local_mask = torch.zeros(B, r_local.size(1), dtype=torch.bool,
                                     device=r_local.device)
            visual_mask = torch.cat([local_mask, region_mask], dim=1)
        else:
            visual_mask = None
        r_v = torch.cat([r_local, r_global], dim=1)                # [B, N_v, D]

        # ---- textual branch ----
        obj = self.text_proj_in(object_tokens)
        r_t = self.id_controller(
            object_tokens=obj,
            visual_context=r_v,
            object_mask=object_mask,
            visual_mask=visual_mask,
        )                                                          # [B, N_o, D]

        return FRCEOutput(
            visual_cues=r_v,
            text_cues=r_t,
            visual_mask=visual_mask,
            text_mask=object_mask,
        )

    # ------------------------------------------------------------------
    # Convenience helper that wires SAM masks -> region feats.
    # ------------------------------------------------------------------
    @staticmethod
    def masked_pool(feature_map: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Average-pool ``feature_map`` over each binary mask.

        Args:
            feature_map: [B, C, H, W]
            masks:       [B, R, H, W]  (float / bool)

        Returns:
            [B, R, C]
        """
        B, C, H, W = feature_map.shape
        R = masks.size(1)
        feat = feature_map.view(B, 1, C, H * W)
        m = masks.view(B, R, 1, H * W).float()
        weights = m.sum(dim=-1).clamp_min(1e-6)
        pooled = (feat * m).sum(dim=-1) / weights
        return pooled  # [B, R, C]
