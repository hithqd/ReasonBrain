"""Standard image-editing metrics: CLIP-T, CLIP-I, DINO, LPIPS.

All functions accept :class:`PIL.Image.Image` instances (or lists thereof)
and return Python floats so they can be averaged trivially over a test set.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence, Union

import torch
from PIL import Image


ImageLike = Union[Image.Image, str]


# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _clip(model_id: str = "openai/clip-vit-large-patch14"):
    from transformers import CLIPModel, CLIPProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_id)
    return model, proc, device


@lru_cache(maxsize=1)
def _dino(model_id: str = "facebook/dinov2-base"):
    from transformers import AutoImageProcessor, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(model_id)
    return model, proc, device


@lru_cache(maxsize=1)
def _lpips_model():
    import lpips
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return lpips.LPIPS(net="alex").to(device).eval(), device


# ---------------------------------------------------------------------------
def _load(img: ImageLike) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.open(img).convert("RGB")


# ---------------------------------------------------------------------------
def clip_image_similarity(a: ImageLike, b: ImageLike) -> float:
    """Cosine similarity between CLIP image embeddings of ``a`` and ``b``."""
    model, proc, device = _clip()
    a, b = _load(a), _load(b)
    inputs = proc(images=[a, b], return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        sim = (feats[0] * feats[1]).sum().item()
    return float(sim)


def clip_text_image_similarity(image: ImageLike, text: str) -> float:
    model, proc, device = _clip()
    image = _load(image)
    inputs = proc(images=image, text=text, return_tensors="pt",
                  padding=True, truncation=True).to(device)
    with torch.no_grad():
        img_f = model.get_image_features(pixel_values=inputs["pixel_values"])
        txt_f = model.get_text_features(input_ids=inputs["input_ids"],
                                        attention_mask=inputs["attention_mask"])
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
        sim = (img_f * txt_f).sum().item()
    return float(sim)


def dino_similarity(a: ImageLike, b: ImageLike) -> float:
    model, proc, device = _dino()
    a, b = _load(a), _load(b)
    inputs = proc(images=[a, b], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
        cls = out.last_hidden_state[:, 0]
        cls = cls / cls.norm(dim=-1, keepdim=True)
        sim = (cls[0] * cls[1]).sum().item()
    return float(sim)


def lpips_distance(a: ImageLike, b: ImageLike) -> float:
    import torchvision.transforms as T
    model, device = _lpips_model()
    tf = T.Compose([T.Resize(256, antialias=True), T.CenterCrop(256), T.ToTensor()])
    a_t = (tf(_load(a)) * 2 - 1).unsqueeze(0).to(device)
    b_t = (tf(_load(b)) * 2 - 1).unsqueeze(0).to(device)
    with torch.no_grad():
        d = model(a_t, b_t).item()
    return float(d)
