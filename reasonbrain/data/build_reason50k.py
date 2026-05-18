"""Reason50K (re)construction pipeline.

This script reproduces the *reverse-generation* strategy of Section 3 of the
ReasonBrain paper:

    1.  Sample (category, scene description) seeds.
    2.  Call GPT to produce a hypothetical instruction H,
        a target caption C_t, and a list of entities O.
    3.  Synthesize the target image with a Text-to-Image diffusion model.
    4.  Use IP-Adapter (image-conditioned T2I) to draft several
        candidate *source* images that depict the pre-edit scene.
    5.  Score each candidate with GPT + CLIP + LPIPS and keep the best.
    6.  Write everything to disk as JSONL + images.

The dependencies on ``openai`` and IP-Adapter are *only* required at
construction time, never at training/inference time. The expensive calls
are batched and resumable: a sample is skipped if its target file already
exists on disk.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm


# ---------------------------------------------------------------------------
def _safe_import_openai():
    try:
        from openai import OpenAI
        return OpenAI
    except ImportError as e:  # pragma: no cover
        raise ImportError("openai>=1.30 is required for dataset construction.") from e


def _safe_import_spacy():
    try:
        import spacy
        return spacy
    except ImportError as e:  # pragma: no cover
        raise ImportError("spacy>=3.7 is required for dataset construction.") from e


# ---------------------------------------------------------------------------
@dataclass
class SampleDraft:
    sample_id: str
    category: str
    instruction: str
    target_caption: str
    objects: List[str]


# ---------------------------------------------------------------------------
def generate_instruction(client, system_prompt: str, seed: Dict,
                          model: str = "gpt-4o",
                          temperature: float = 0.8) -> Optional[Dict]:
    """Call GPT to produce (instruction, target_caption, objects)."""
    response = client.chat.completions.create(
        model=model, temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",
             "content": (
                 f"Category: {seed['category']}\n"
                 f"Scene description: {seed['scene']}\n"
                 f"Return JSON with keys instruction, target_caption, objects."
             )},
        ],
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
def synthesize_target(pipe, prompt: str, steps: int, guidance: float,
                       generator=None):
    out = pipe(prompt, num_inference_steps=steps,
               guidance_scale=guidance, generator=generator).images[0]
    return out


def synthesize_source_candidates(ip_adapter_pipe, target_image, instruction: str,
                                  num: int, steps: int, guidance: float):
    """Use an IP-Adapter pipeline to draft source candidates."""
    prompt = f"Pre-edit version: scene before '{instruction}'."
    images = ip_adapter_pipe(
        prompt, image=target_image, num_images_per_prompt=num,
        num_inference_steps=steps, guidance_scale=guidance,
    ).images
    return images


# ---------------------------------------------------------------------------
def filter_best_source(target_image, candidates,
                       clip_score_fn, lpips_fn,
                       clip_threshold: float,
                       lpips_min: float, lpips_max: float):
    """Pick the highest-scoring candidate respecting LPIPS bounds."""
    scored = []
    for cand in candidates:
        clip_score = float(clip_score_fn(cand, target_image))
        lp = float(lpips_fn(cand, target_image))
        if clip_score < clip_threshold:
            continue
        if not (lpips_min <= lp <= lpips_max):
            continue
        # Reward image-text alignment; penalize too-similar / too-different pairs.
        score = clip_score - 0.5 * abs(lp - (lpips_min + lpips_max) / 2.0)
        scored.append((score, cand))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


# ---------------------------------------------------------------------------
def build_dataset(seeds_path: str, out_dir: str, cfg: Dict) -> None:
    """Top-level pipeline. ``cfg`` mirrors ``configs/data.yaml``."""
    OpenAI = _safe_import_openai()
    spacy = _safe_import_spacy()
    nlp = spacy.load("en_core_web_sm")

    client = OpenAI()  # requires OPENAI_API_KEY in env

    # Lazy imports – heavy.
    from diffusers import StableDiffusionXLPipeline

    t2i_pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg["t2i"]["model"], torch_dtype="auto"
    ).to("cuda")
    # IP-Adapter is loaded via diffusers' load_ip_adapter helper.
    t2i_pipe.load_ip_adapter(
        cfg["t2i"]["ip_adapter"], subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.safetensors",
    )

    # CLIP + LPIPS for filtering
    import lpips
    import clip as openai_clip
    clip_model, clip_preprocess = openai_clip.load("ViT-L/14", device="cuda")
    lpips_model = lpips.LPIPS(net="alex").to("cuda")

    def clip_score(a, b):
        import torch
        a_t = clip_preprocess(a).unsqueeze(0).to("cuda")
        b_t = clip_preprocess(b).unsqueeze(0).to("cuda")
        with torch.no_grad():
            fa = clip_model.encode_image(a_t)
            fb = clip_model.encode_image(b_t)
            return torch.nn.functional.cosine_similarity(fa, fb).item()

    def lpips_score(a, b):
        import torch, numpy as np
        def to_tensor(img):
            arr = np.asarray(img).astype("float32") / 127.5 - 1.0
            return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to("cuda")
        with torch.no_grad():
            return lpips_model(to_tensor(a), to_tensor(b)).item()

    out_dir = Path(out_dir)
    (out_dir / "images" / "src").mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "tgt").mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "train.jsonl"

    with open(seeds_path) as f, manifest_path.open("a") as out_f:
        for line in tqdm(f, desc="seeds"):
            seed = json.loads(line)
            sid = seed["id"]
            tgt_path = out_dir / "images" / "tgt" / f"{sid}.png"
            src_path = out_dir / "images" / "src" / f"{sid}.png"
            if tgt_path.exists() and src_path.exists():
                continue

            ann = generate_instruction(client, cfg["gpt"]["system_prompt"], seed,
                                       model=cfg["gpt"]["model"],
                                       temperature=cfg["gpt"]["temperature"])
            if ann is None:
                continue

            instruction = ann["instruction"]
            target_caption = ann["target_caption"]
            objects = ann.get("objects", [])
            if not objects:  # fall back to spaCy entities
                doc = nlp(instruction)
                objects = [chunk.text for chunk in doc.noun_chunks][:12]

            tgt_img = synthesize_target(
                t2i_pipe, target_caption,
                steps=cfg["t2i"]["steps"],
                guidance=cfg["t2i"]["guidance_scale"],
            )
            tgt_img.save(tgt_path)

            cands = synthesize_source_candidates(
                t2i_pipe, tgt_img, instruction,
                num=cfg["t2i"]["num_candidates"],
                steps=cfg["t2i"]["steps"],
                guidance=cfg["t2i"]["guidance_scale"],
            )
            best = filter_best_source(
                tgt_img, cands,
                clip_score_fn=clip_score, lpips_fn=lpips_score,
                clip_threshold=cfg["filter"]["clip_score_threshold"],
                lpips_min=cfg["filter"]["lpips_min"],
                lpips_max=cfg["filter"]["lpips_max"],
            )
            if best is None:
                continue
            best.save(src_path)

            record = {
                "id": sid,
                "category": seed["category"],
                "src_image": f"images/src/{sid}.png",
                "tgt_image": f"images/tgt/{sid}.png",
                "instruction": instruction,
                "target_caption": target_caption,
                "objects": objects,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
