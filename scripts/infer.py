"""Run inference with a trained ReasonBrain checkpoint.

Usage:
    python scripts/infer.py \\
        --config configs/default.yaml \\
        --ckpt   outputs/reasonbrain/last \\
        --src_image examples/glass.png \\
        --instruction "What if this glass was dropped onto the floor?" \\
        --out edited.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasonbrain.inference.pipeline import ReasonBrainPipeline


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--src_image", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--out", default="edited.png")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pipe = ReasonBrainPipeline.from_pretrained(args.ckpt, config_path=args.config)
    img = pipe(args.src_image, args.instruction,
               num_inference_steps=args.steps,
               guidance_scale=args.guidance,
               seed=args.seed)
    img.save(args.out)
    print(f"Saved edit to {args.out}")


if __name__ == "__main__":
    main()
