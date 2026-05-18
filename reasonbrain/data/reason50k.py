"""PyTorch ``Dataset`` and collator for Reason50K.

The on-disk layout is described in the README. Each sample yields raw image
tensors and the instruction string; the actual feature extraction (CLIP,
SAM, MLLM tokenisation) is performed by :class:`Reason50KCollator` so it
can be batched and (optionally) cached.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import build_transforms


# ---------------------------------------------------------------------------
@dataclass
class Reason50KSample:
    sample_id: str
    category: str
    src_image: torch.Tensor              # diffusion-normalized [-1,1]
    tgt_image: torch.Tensor
    src_image_clip: torch.Tensor         # CLIP-normalized
    instruction: str
    objects: List[str]
    src_image_pil: Image.Image           # kept for SAM (raw RGB pixels)


# ---------------------------------------------------------------------------
class Reason50K(Dataset):
    """Map-style dataset over a Reason50K jsonl manifest."""

    def __init__(self,
                 root: str,
                 split_file: str,
                 image_size: int = 512):
        super().__init__()
        self.root = Path(root)
        manifest_path = self.root / split_file
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Reason50K manifest not found at {manifest_path}."
            )
        with manifest_path.open() as f:
            self.records: List[Dict[str, Any]] = [json.loads(l) for l in f if l.strip()]
        self.diffusion_tf, self.clip_tf = build_transforms(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Reason50KSample:
        rec = self.records[idx]
        src_pil = Image.open(self.root / rec["src_image"]).convert("RGB")
        tgt_pil = Image.open(self.root / rec["tgt_image"]).convert("RGB")
        return Reason50KSample(
            sample_id=rec["id"],
            category=rec["category"],
            src_image=self.diffusion_tf(src_pil),
            tgt_image=self.diffusion_tf(tgt_pil),
            src_image_clip=self.clip_tf(src_pil),
            instruction=rec["instruction"],
            objects=rec.get("objects", []),
            src_image_pil=src_pil,
        )


# ---------------------------------------------------------------------------
class Reason50KCollator:
    """Collator that turns raw samples into model-ready tensor batches.

    The heavy preprocessors (CLIP image / text encoders, SAM, LLaVA
    processor) live here so each worker can run them on CPU before the
    main process receives the batch.  Once the model runs on GPU it only
    needs cheap tensor operations.

    For brevity we do **not** ship SAM inference inside the collator —
    region features are passed in as cached `.pt` files. The dataset
    construction script handles their generation. If they are not
    available, dummy zero tensors are produced (useful for tests).
    """

    def __init__(self,
                 clip_image_encoder,
                 clip_text_encoder,
                 clip_processor,
                 llava_processor,
                 num_image_tokens: int,
                 num_query_tokens: int,
                 max_objects: int = 16,
                 max_regions: int = 64):
        self.clip_image_encoder = clip_image_encoder
        self.clip_text_encoder = clip_text_encoder
        self.clip_processor = clip_processor
        self.llava_processor = llava_processor
        self.num_image_tokens = num_image_tokens
        self.num_query_tokens = num_query_tokens
        self.max_objects = max_objects
        self.max_regions = max_regions

    # ------------------------------------------------------------------
    def __call__(self, batch: List[Reason50KSample]) -> Dict[str, torch.Tensor]:
        B = len(batch)
        device = next(self.clip_image_encoder.parameters()).device

        # ---- (1) stack diffusion images ----
        src_pixels = torch.stack([s.src_image for s in batch])
        tgt_pixels = torch.stack([s.tgt_image for s in batch])

        # ---- (2) CLIP image features ----
        clip_inputs = self.clip_processor(
            images=[s.src_image_pil for s in batch],
            return_tensors="pt",
        )["pixel_values"].to(device)
        with torch.no_grad():
            vision_out = self.clip_image_encoder(clip_inputs,
                                                 output_hidden_states=True)
            patch_tokens = vision_out.last_hidden_state[:, 1:]  # drop CLS
            image_tokens = vision_out.last_hidden_state          # CLS + patches

        # ---- (3) CLIP text features for objects ----
        # Pad objects to ``max_objects``.
        all_obj_texts: List[str] = []
        obj_mask = torch.ones(B, self.max_objects, dtype=torch.bool)
        for i, s in enumerate(batch):
            objs = s.objects[: self.max_objects]
            for j, name in enumerate(objs):
                all_obj_texts.append(name)
                obj_mask[i, j] = False
            # pad with empty strings
            for _ in range(self.max_objects - len(objs)):
                all_obj_texts.append("")

        text_inputs = self.clip_processor(
            text=all_obj_texts, padding=True, truncation=True,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_out = self.clip_text_encoder(**text_inputs,
                                              output_hidden_states=True)
            pooled = text_out.pooler_output                       # [B*N_o, C]
        object_tokens = pooled.view(B, self.max_objects, -1)

        # Full instruction embeddings (for CME text branch).
        instr_inputs = self.clip_processor(
            text=[s.instruction for s in batch], padding=True, truncation=True,
            return_tensors="pt",
        )
        instr_inputs = {k: v.to(device) for k, v in instr_inputs.items()}
        with torch.no_grad():
            instr_out = self.clip_text_encoder(**instr_inputs,
                                               output_hidden_states=True)
            text_tokens = instr_out.last_hidden_state

        # ---- (4) Region features (placeholder if SAM cache not present) ----
        region_feats = torch.zeros(B, self.max_regions, 256)
        region_mask = torch.ones(B, self.max_regions, dtype=torch.bool)

        # ---- (5) LLaVA / MLLM tokenisation ----
        prompts = []
        for s in batch:
            tail = "".join(f"[Q_{i}]" for i in range(self.num_query_tokens))
            prompts.append(f"USER: <image>\n{s.instruction}\nASSISTANT:{tail}")
        llava_inputs = self.llava_processor(
            text=prompts, images=[s.src_image_pil for s in batch],
            padding=True, truncation=True, return_tensors="pt",
        )

        # ---- (6) MLLM labels: predict [IMG_i] tokens at the very end ----
        #   We supervise the LM to emit ``num_image_tokens`` special tokens
        #   that mark the region where the visual prediction will be aligned.
        labels = llava_inputs["input_ids"].clone()
        labels[labels == self.llava_processor.tokenizer.pad_token_id] = -100

        return {
            "src_pixels": src_pixels,
            "tgt_pixels": tgt_pixels,
            "mllm_input_ids": llava_inputs["input_ids"],
            "mllm_attn_mask": llava_inputs["attention_mask"],
            "mllm_pixels": llava_inputs["pixel_values"],
            "mllm_labels": labels,
            "patch_tokens": patch_tokens,
            "image_tokens": image_tokens,
            "region_feats": region_feats,
            "region_mask": region_mask,
            "object_tokens": object_tokens,
            "object_mask": obj_mask,
            "text_tokens": text_tokens,
        }
