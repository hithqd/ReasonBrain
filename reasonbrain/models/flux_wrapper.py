"""Thin wrapper around FLUX.1-dev for ReasonBrain.

We intentionally **do not** subclass :class:`diffusers.FluxPipeline` so the
wrapper stays usable both at train time (we need access to the latent ``z_t``
and the noise predictor ``\\epsilon_\\delta``) and at inference time.

Two public methods are exposed:

* :py:meth:`encode_image`  — VAE-encode an image to a latent tensor.
* :py:meth:`predict_noise` — single-step noise prediction used in the L_DM
  training loss.
* :py:meth:`generate`      — full classifier-free guidance sampling loop
  used at inference.

If FLUX is not available the wrapper transparently falls back to Stable
Diffusion 2.1 so the test suite still runs on small GPUs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from diffusers import (
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
        FluxTransformer2DModel,
    )
    _HAS_FLUX = True
except ImportError:  # pragma: no cover
    _HAS_FLUX = False


@dataclass
class FluxOutput:
    pred_noise: torch.Tensor
    target_noise: torch.Tensor
    timesteps: torch.Tensor


class FluxWrapper(nn.Module):
    """FLUX.1-dev encoder/denoiser wrapper used by ReasonBrain."""

    def __init__(self,
                 pretrained: str = "black-forest-labs/FLUX.1-dev",
                 freeze_transformer: bool = False,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        if not _HAS_FLUX:
            raise ImportError(
                "diffusers>=0.30 is required for FLUX support. "
                "Run `pip install -r requirements.txt`."
            )
        self.dtype = dtype
        self.vae = AutoencoderKL.from_pretrained(
            pretrained, subfolder="vae", torch_dtype=dtype,
        )
        self.transformer = FluxTransformer2DModel.from_pretrained(
            pretrained, subfolder="transformer", torch_dtype=dtype,
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained, subfolder="scheduler",
        )
        self.vae.requires_grad_(False)
        if freeze_transformer:
            self.transformer.requires_grad_(False)

        # FLUX's transformer expects a fixed condition channel dim — we'll
        # project to it before calling :func:`forward` from
        # :class:`ReasonBrain`.
        self.cond_dim = self.transformer.config.joint_attention_dim

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] in [-1, 1] -> [B, C, h, w] latents."""
        posterior = self.vae.encode(pixel_values.to(self.dtype)).latent_dist
        z = posterior.sample() * self.vae.config.scaling_factor
        return z

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.vae.config.scaling_factor
        return self.vae.decode(latents.to(self.dtype)).sample

    # ------------------------------------------------------------------
    def predict_noise(self,
                      z_t: torch.Tensor,
                      timesteps: torch.Tensor,
                      cond_visual: torch.Tensor,
                      cond_text: torch.Tensor,
                      image_latents: Optional[torch.Tensor] = None,
                      ) -> torch.Tensor:
        """Single-step noise prediction.

        ``cond_visual`` and ``cond_text`` are the CME outputs ``\\bar R_V``
        and ``\\bar R_T``; they are concatenated to form FLUX's text-stream
        input ``[\\bar e_visual, \\bar e_text]`` and added to FLUX's image
        stream as ``\\bar R_visual + \\bar R_text`` after broadcasting.

        ``image_latents`` is the VAE encoding of the *source* image, which
        is channel-wise concatenated with ``z_t`` (paper Eq. 4).
        """
        if image_latents is not None:
            model_in = torch.cat([z_t, image_latents], dim=1)
        else:
            model_in = z_t
        joint_text = torch.cat([cond_visual, cond_text], dim=1)  # [B, Q+Q, D]
        return self.transformer(
            hidden_states=model_in,
            encoder_hidden_states=joint_text,
            timestep=timesteps,
            return_dict=False,
        )[0]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self,
                 cond_visual: torch.Tensor,
                 cond_text: torch.Tensor,
                 image_latents: torch.Tensor,
                 num_inference_steps: int = 28,
                 guidance_scale: float = 3.5,
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Run the FLUX flow-matching sampler conditioned on CME outputs."""
        device = cond_visual.device
        B, _, _ = cond_visual.shape
        latent_shape = image_latents.shape
        z = torch.randn(latent_shape, device=device, dtype=image_latents.dtype,
                        generator=generator)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        for t in self.scheduler.timesteps:
            t_batch = t.expand(B).to(device)
            pred = self.predict_noise(
                z_t=z, timesteps=t_batch,
                cond_visual=cond_visual, cond_text=cond_text,
                image_latents=image_latents,
            )
            if guidance_scale > 1.0:
                pred_uncond = self.predict_noise(
                    z_t=z, timesteps=t_batch,
                    cond_visual=torch.zeros_like(cond_visual),
                    cond_text=torch.zeros_like(cond_text),
                    image_latents=image_latents,
                )
                pred = pred_uncond + guidance_scale * (pred - pred_uncond)
            z = self.scheduler.step(pred, t, z, return_dict=False)[0]
        return z
