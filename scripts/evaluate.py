"""Evaluate a trained ReasonBrain checkpoint on Reason50K's test split.

Usage:
    python scripts/evaluate.py \\
        --config configs/default.yaml \\
        --ckpt outputs/reasonbrain/last \\
        --test data/reason50k/test.jsonl \\
        --metrics clip_t clip_i dino lpips
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasonbrain.evaluation.metrics import (
    clip_image_similarity, clip_text_image_similarity,
    dino_similarity, lpips_distance,
)
from reasonbrain.inference.pipeline import ReasonBrainPipeline


_METRIC_FNS = {
    "clip_t": lambda edit, gt, txt: clip_text_image_similarity(edit, txt),
    "clip_i": lambda edit, gt, txt: clip_image_similarity(edit, gt),
    "dino":   lambda edit, gt, txt: dino_similarity(edit, gt),
    "lpips":  lambda edit, gt, txt: lpips_distance(edit, gt),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--metrics", nargs="+", default=["clip_t", "clip_i", "dino", "lpips"])
    ap.add_argument("--limit", type=int, default=None,
                    help="If given, evaluate at most this many samples.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pipe = ReasonBrainPipeline.from_pretrained(args.ckpt, config_path=args.config)

    data_root = Path(args.test).parent
    records = [json.loads(l) for l in Path(args.test).read_text().splitlines() if l.strip()]
    if args.limit is not None:
        records = records[: args.limit]

    scores: dict[str, list[float]] = {m: [] for m in args.metrics}
    for rec in tqdm(records, desc="evaluating"):
        src = data_root / rec["src_image"]
        gt = Image.open(data_root / rec["tgt_image"]).convert("RGB")
        edit = pipe(src, rec["instruction"])
        for m in args.metrics:
            scores[m].append(_METRIC_FNS[m](edit, gt, rec["instruction"]))

    print("\n=== Evaluation results ===")
    for m, vals in scores.items():
        print(f"{m:>10s}: mean={statistics.mean(vals):.4f}  "
              f"std={statistics.pstdev(vals):.4f}  n={len(vals)}")


if __name__ == "__main__":
    main()
