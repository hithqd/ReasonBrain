"""End-to-end ReasonBrain inference pipeline.

The pipeline takes a (source image, hypothetical instruction) pair and
returns the edited image as a :class:`PIL.Image.Image`.

It owns the heavy preprocessors (CLIP image / text encoder + LLaVA
processor) so the model only sees ready-to-use tensors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image

from ..data.transforms import build_transforms
from ..models.reasonbrain import ReasonBrain, build_reasonbrain
from ..utils.config import load_config


# ---------------------------------------------------------------------------
class ReasonBrainPipeline:
    def __init__(self,
                 model: ReasonBrain,
                 cfg: dict,
                 device: Optional[torch.device] = None):
        self.model = model.eval()
        self.cfg = cfg
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.diffusion_tf, self.clip_tf = build_transforms(
            cfg["data"]["image_size"])

        # CLIP image + text encoders for cue extraction
        from transformers import CLIPModel, CLIPProcessor
        clip_id = cfg["model"]["frce"]["image_encoder"]
        self.clip_processor = CLIPProcessor.from_pretrained(clip_id)
        self.clip_model = CLIPModel.from_pretrained(clip_id).to(self.device).eval()

    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, ckpt_dir: Union[str, Path],
                        config_path: Union[str, Path, None] = None,
                        device: Optional[torch.device] = None
                        ) -> "ReasonBrainPipeline":
        ckpt_dir = Path(ckpt_dir)
        config_path = config_path or (ckpt_dir.parent.parent / "configs/default.yaml")
        cfg = load_config(str(config_path))
        model = build_reasonbrain(cfg)
        state_path = ckpt_dir / "reasonbrain.pt"
        if state_path.exists():
            sd = torch.load(state_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                print(f"[ReasonBrain] missing keys: {len(missing)}")
            if unexpected:
                print(f"[ReasonBrain] unexpected keys: {len(unexpected)}")
        return cls(model, cfg, device=device)

    # ------------------------------------------------------------------
    def _prepare_batch(self, image: Image.Image, instruction: str) -> dict:
        device = self.device
        src_t = self.diffusion_tf(image).unsqueeze(0).to(device)

        clip_inputs = self.clip_processor(images=image, return_tensors="pt")
        clip_pixel = clip_inputs["pixel_values"].to(device)
        with torch.no_grad():
            vision_out = self.clip_model.vision_model(
                clip_pixel, output_hidden_states=True)
            patch_tokens = vision_out.last_hidden_state[:, 1:]
            image_tokens = vision_out.last_hidden_state

            instr_inputs = self.clip_processor(
                text=[instruction], padding=True, truncation=True,
                return_tensors="pt",
            )
            instr_inputs = {k: v.to(device) for k, v in instr_inputs.items()}
            text_out = self.clip_model.text_model(**instr_inputs,
                                                  output_hidden_states=True)
            text_tokens = text_out.last_hidden_state

        # Object tokens — extract nouns with spaCy if available, else use a
        # single dummy token (the CME branch will largely rely on R_T anyway).
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            objs = [c.text for c in nlp(instruction).noun_chunks][
                : self.model.frce.max_objects]
        except Exception:
            objs = [instruction]

        if not objs:
            objs = [instruction]
        obj_inputs = self.clip_processor(text=objs, padding=True,
                                         truncation=True, return_tensors="pt")
        obj_inputs = {k: v.to(device) for k, v in obj_inputs.items()}
        with torch.no_grad():
            obj_out = self.clip_model.text_model(**obj_inputs)
            obj_tokens = obj_out.pooler_output.unsqueeze(0)  # [1, N_o, C]

        # Build dummy region features (replace with SAM cache if available).
        max_regions = 64
        region_feats = torch.zeros(1, max_regions, 256, device=device)
        region_mask = torch.ones(1, max_regions, dtype=torch.bool, device=device)

        # LLaVA tokenisation
        n_q = self.model.mllm.num_query_tokens
        tail = "".join(f"[Q_{i}]" for i in range(n_q))
        prompt = f"USER: <image>\n{instruction}\nASSISTANT:{tail}"
        llava = self.model.mllm.processor(
            text=prompt, images=image, return_tensors="pt",
        )
        llava = {k: v.to(device) for k, v in llava.items()}

        return {
            "src_pixels": src_t,
            "tgt_pixels": src_t,                # placeholder, unused at infer
            "mllm_input_ids": llava["input_ids"],
            "mllm_attn_mask": llava["attention_mask"],
            "mllm_pixels": llava["pixel_values"],
            "mllm_labels": None,
            "patch_tokens": patch_tokens,
            "image_tokens": image_tokens,
            "region_feats": region_feats,
            "region_mask": region_mask,
            "object_tokens": obj_tokens,
            "object_mask": torch.zeros(1, obj_tokens.size(1),
                                       dtype=torch.bool, device=device),
            "text_tokens": text_tokens,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def __call__(self,
                 image: Union[str, Path, Image.Image],
                 instruction: str,
                 num_inference_steps: Optional[int] = None,
                 guidance_scale: Optional[float] = None,
                 seed: Optional[int] = None) -> Image.Image:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        batch = self._prepare_batch(image, instruction)

        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(seed)

        pixels = self.model.edit(
            batch,
            num_inference_steps=num_inference_steps
                or self.cfg["inference"]["num_inference_steps"],
            guidance_scale=guidance_scale
                or self.cfg["inference"]["guidance_scale"],
            generator=gen,
        )
        # pixels: [1, 3, H, W] in [-1, 1]
        img = (pixels.clamp(-1, 1) + 1.0) / 2.0
        arr = (img[0].permute(1, 2, 0).cpu().float().numpy() * 255).round().astype("uint8")
        return Image.fromarray(arr)

    # ------------------------------------------------------------------
    def to(self, device) -> "ReasonBrainPipeline":
        self.device = torch.device(device)
        self.model.to(self.device)
        self.clip_model.to(self.device)
        return self
