"""Drive the Reason50K (re)construction pipeline.

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/build_dataset.py \\
        --config configs/data.yaml \\
        --seeds data/seeds.jsonl \\
        --out data/reason50k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasonbrain.data.build_reason50k import build_dataset
from reasonbrain.utils.config import load_config


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", required=True,
                    help="JSONL file with seed scene descriptions.")
    ap.add_argument("--out", default=None,
                    help="Override out_dir from the config.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = args.out or cfg["out_dir"]
    build_dataset(seeds_path=args.seeds, out_dir=out_dir, cfg=cfg)


if __name__ == "__main__":
    main()
