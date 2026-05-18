"""Top-level ReasonBrain model.

Wires together:
    image / text encoders (CLIP)
    SAM segmenter for region features
        ↓
    FRCE   ->  R_V, R_T
        ↓
    MLLM (LLaVA + LoRA)         -> r learnable query hidden states V
        ↓
    QFormer (6 × 77 queries)    -> \hat V
        ↓
    CME (visual + text)         -> \bar R_V, \bar R_T
        ↓
    FLUX denoiser               -> predicted noise (training) / image (inference)

The forward signature is intentionally kept simple: training code passes
already-prepared tensors; the inference pipeline (in
:mod:`reasonbrain.inference.pipeline`) takes care of pre-processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cme import CME, CMEOutput
from .frce import FRCE, FRCEOutput
from .mllm_wrapper import MLLMOutput, MLLMWrapper
from .qformer import QFormer
from .flux_wrapper import FluxWrapper


# ---------------------------------------------------------------------------
@dataclass
class ReasonBrainOutput:
    pred_noise: torch.Tensor
    target_noise: torch.Tensor
    timesteps: torch.Tensor
    mllm_logits: Optional[torch.Tensor]
    image_token_labels: Optional[torch.Tensor]


# ---------------------------------------------------------------------------
class ReasonBrain(nn.Module):
    """End-to-end ReasonBrain model.

    Sub-modules are **already-constructed** to allow the trainer to manage
    device placement / dtype individually (LLaVA / FLUX are large and may
    live on different devices in a sharded setup).
    """

    def __init__(self,
                 mllm: MLLMWrapper,
                 flux: FluxWrapper,
                 frce: FRCE,
                 qformer: QFormer,
                 cme: CME,
                 cond_proj: Optional[nn.Module] = None):
        super().__init__()
        self.mllm = mllm
        self.flux = flux
        self.frce = frce
        self.qformer = qformer
        self.cme = cme

        # Projects QFormer / CME outputs (hidden_dim) to FLUX cond dim.
        self.cond_proj = cond_proj or nn.Linear(
            qformer.queries.shape[-1], flux.cond_dim, bias=False,
        )
        self.cme_proj = nn.Linear(cme.visual_enhancer.queries.shape[-1],
                                  flux.cond_dim, bias=False)

    # ==================================================================
    # Forward (training)
    # ==================================================================
    def forward(self, batch: Dict[str, torch.Tensor]) -> ReasonBrainOutput:
        """One training step.

        ``batch`` keys (all tensors batched):
            src_pixels    : [B, 3, H, W]  source image (normalized to [-1,1])
            tgt_pixels    : [B, 3, H, W]  target image
            mllm_input_ids: [B, L]
            mllm_attn_mask: [B, L]
            mllm_pixels   : [B, 3, h, w]  CLIP preprocessing of source image
            mllm_labels   : [B, L]        labels for L_MLLM (with -100 ignore)
            patch_tokens  : [B, N_p, C]   CLIP patch tokens of source image
            region_feats  : [B, R, C_r]   SAM-pooled regions of source image
            region_mask   : [B, R]        True == pad
            object_tokens : [B, N_o, C_t] text tokens of extracted objects
            object_mask   : [B, N_o]
            image_tokens  : [B, N_i, C_t] CLIP image-features sequence used by CME
            text_tokens   : [B, N_t, C_t] CLIP text-features sequence used by CME
        """
        # ---- 1. FRCE ----
        cues: FRCEOutput = self.frce(
            patch_tokens=batch["patch_tokens"],
            region_feats=batch["region_feats"],
            region_mask=batch.get("region_mask"),
            object_tokens=batch["object_tokens"],
            object_mask=batch.get("object_mask"),
        )

        # ---- 2. MLLM ----
        mllm_out: MLLMOutput = self.mllm(
            input_ids=batch["mllm_input_ids"],
            attention_mask=batch["mllm_attn_mask"],
            pixel_values=batch["mllm_pixels"],
            labels=batch.get("mllm_labels"),
        )

        # ---- 3. QFormer -> \hat V ----
        v_hat = self.qformer(mllm_out.query_hidden)            # [B, Q, D]

        # ---- 4. CME -> \bar R_V, \bar R_T ----
        cme_out: CMEOutput = self.cme(
            v_hat=v_hat,
            image_feats=batch["image_tokens"],
            text_feats=batch["text_tokens"],
            r_v=cues.visual_cues,
            r_t=cues.text_cues,
            rv_mask=cues.visual_mask,
            rt_mask=cues.text_mask,
        )

        cond_visual = self.cme_proj(cme_out.visual)
        cond_text = self.cme_proj(cme_out.text)

        # ---- 5. Diffusion training step ----
        with torch.no_grad():
            tgt_lat = self.flux.encode_image(batch["tgt_pixels"])
            src_lat = self.flux.encode_image(batch["src_pixels"])

        noise = torch.randn_like(tgt_lat)
        bsz = tgt_lat.size(0)
        timesteps = torch.randint(
            0, self.flux.scheduler.config.num_train_timesteps, (bsz,),
            device=tgt_lat.device, dtype=torch.long,
        )
        # Flow-matching noised sample: z_t = (1 - sigma) * x_0 + sigma * eps
        sigmas = self.flux.scheduler.sigmas.to(tgt_lat.device)[timesteps].view(-1, 1, 1, 1)
        z_t = (1.0 - sigmas) * tgt_lat + sigmas * noise

        pred = self.flux.predict_noise(
            z_t=z_t, timesteps=timesteps.to(tgt_lat.dtype),
            cond_visual=cond_visual, cond_text=cond_text,
            image_latents=src_lat,
        )

        # Flow-matching target = (noise - latent)
        target = noise - tgt_lat

        return ReasonBrainOutput(
            pred_noise=pred,
            target_noise=target,
            timesteps=timesteps,
            mllm_logits=mllm_out.image_token_logits,
            image_token_labels=batch.get("mllm_labels"),
        )

    # ==================================================================
    # Inference
    # ==================================================================
    @torch.no_grad()
    def edit(self, batch: Dict[str, torch.Tensor],
             num_inference_steps: int = 28,
             guidance_scale: float = 3.5,
             generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Edit an image conditioned on a hypothetical instruction."""
        cues = self.frce(
            patch_tokens=batch["patch_tokens"],
            region_feats=batch["region_feats"],
            region_mask=batch.get("region_mask"),
            object_tokens=batch["object_tokens"],
            object_mask=batch.get("object_mask"),
        )
        mllm_out = self.mllm(
            input_ids=batch["mllm_input_ids"],
            attention_mask=batch["mllm_attn_mask"],
            pixel_values=batch["mllm_pixels"],
        )
        v_hat = self.qformer(mllm_out.query_hidden)
        cme_out = self.cme(
            v_hat=v_hat,
            image_feats=batch["image_tokens"],
            text_feats=batch["text_tokens"],
            r_v=cues.visual_cues,
            r_t=cues.text_cues,
            rv_mask=cues.visual_mask,
            rt_mask=cues.text_mask,
        )
        cond_visual = self.cme_proj(cme_out.visual)
        cond_text = self.cme_proj(cme_out.text)

        src_lat = self.flux.encode_image(batch["src_pixels"])
        z = self.flux.generate(
            cond_visual=cond_visual, cond_text=cond_text,
            image_latents=src_lat,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return self.flux.decode_latents(z)


# ---------------------------------------------------------------------------
# Convenience: build a ReasonBrain from a configuration dict
# ---------------------------------------------------------------------------
def build_reasonbrain(cfg: Dict[str, Any]) -> ReasonBrain:
    """Construct a ReasonBrain model from a parsed YAML config."""
    mcfg = cfg["model"]

    mllm = MLLMWrapper(
        pretrained=mcfg["mllm"]["pretrained"],
        num_image_tokens=mcfg["mllm"]["num_image_tokens"],
        num_query_tokens=mcfg["mllm"]["num_query_tokens"],
        lora_r=mcfg["mllm"]["lora"]["r"],
        lora_alpha=mcfg["mllm"]["lora"]["alpha"],
        lora_dropout=mcfg["mllm"]["lora"]["dropout"],
        lora_target_modules=mcfg["mllm"]["lora"]["target_modules"],
    )
    flux = FluxWrapper(
        pretrained=mcfg["diffusion"]["pretrained"],
        freeze_transformer=mcfg["diffusion"]["freeze_transformer"],
    )
    frce = FRCE(
        patch_in_dim=mcfg["frce"]["patch_dim"],
        region_in_dim=mcfg["frce"]["region_dim"],
        out_dim=mcfg["frce"]["out_dim"],
        patch_layers=mcfg["frce"]["patch_layers"],
        region_layers=mcfg["frce"]["region_layers"],
        id_controller_layers=mcfg["frce"]["id_controller_layers"],
        max_objects=mcfg["frce"]["max_objects"],
    )
    qformer = QFormer(
        mllm_hidden_dim=mllm.hidden_size,
        hidden_dim=mcfg["qformer"]["hidden_dim"],
        num_layers=mcfg["qformer"]["num_layers"],
        num_queries=mcfg["qformer"]["num_queries"],
        num_heads=mcfg["qformer"]["num_heads"],
    )
    cme = CME(
        dim=mcfg["cme"]["hidden_dim"],
        num_heads=mcfg["cme"]["num_heads"],
        num_blocks=mcfg["cme"]["num_blocks"],
        num_queries=mcfg["qformer"]["num_queries"],
        dropout=mcfg["cme"]["dropout"],
    )
    return ReasonBrain(mllm=mllm, flux=flux, frce=frce,
                       qformer=qformer, cme=cme)
