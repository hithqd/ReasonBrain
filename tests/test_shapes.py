"""Shape-level smoke tests for the small ReasonBrain sub-modules.

These tests deliberately avoid loading LLaVA / FLUX so they run quickly
on CPU.
"""

from __future__ import annotations

import torch

from reasonbrain.models.cme import CME
from reasonbrain.models.frce import FRCE
from reasonbrain.models.qformer import QFormer


# ---------------------------------------------------------------------------
def test_frce_shapes():
    B, Np, Cp = 2, 257, 1024
    R, Cr = 8, 256
    No, Ct = 5, 768

    frce = FRCE(patch_in_dim=Cp, region_in_dim=Cr,
                text_in_dim=Ct, out_dim=512,
                patch_layers=1, region_layers=1, id_controller_layers=1)
    patch = torch.randn(B, Np, Cp)
    region = torch.randn(B, R, Cr)
    region_mask = torch.zeros(B, R, dtype=torch.bool)
    obj = torch.randn(B, No, Ct)
    out = frce(patch, region, region_mask, obj)
    assert out.visual_cues.shape == (B, Np + R, 512)
    assert out.text_cues.shape == (B, No, 512)


# ---------------------------------------------------------------------------
def test_qformer_shape():
    B, T, C = 2, 32, 4096
    qformer = QFormer(mllm_hidden_dim=C, hidden_dim=256, num_layers=2,
                       num_queries=77, num_heads=4)
    h = torch.randn(B, T, C)
    out = qformer(h)
    assert out.shape == (B, 77, 256)


# ---------------------------------------------------------------------------
def test_cme_shape():
    B, D = 2, 512
    cme = CME(dim=D, num_heads=4, num_blocks=2, num_queries=16)
    v_hat = torch.randn(B, 16, D)
    img = torch.randn(B, 32, D)
    text = torch.randn(B, 8, D)
    r_v = torch.randn(B, 24, D)
    r_t = torch.randn(B, 6, D)
    out = cme(v_hat, img, text, r_v, r_t)
    assert out.visual.shape == (B, 16, D)
    assert out.text.shape == (B, 16, D)


if __name__ == "__main__":
    test_frce_shapes()
    test_qformer_shape()
    test_cme_shape()
    print("All shape tests passed.")
