"""Image transforms for ReasonBrain.

We use two parallel pipelines:

* ``diffusion_transform`` (for FLUX VAE input): resize → center-crop → ``[-1, 1]``.
* ``clip_transform`` (for CLIP / SAM): standard CLIP normalization.
"""

from __future__ import annotations

from typing import Tuple

import torchvision.transforms as T


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_transforms(image_size: int = 512) -> Tuple[T.Compose, T.Compose]:
    """Return (diffusion_transform, clip_transform)."""

    diffusion = T.Compose([
        T.Resize(image_size, antialias=True),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    clip = T.Compose([
        T.Resize(224, antialias=True),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
    return diffusion, clip
