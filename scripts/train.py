"""Train ReasonBrain on Reason50K.

Usage:
    accelerate launch scripts/train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``reasonbrain`` importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import CLIPModel, CLIPProcessor

from reasonbrain.data.reason50k import Reason50K, Reason50KCollator
from reasonbrain.models.reasonbrain import build_reasonbrain
from reasonbrain.training.trainer import ReasonBrainTrainer
from reasonbrain.utils.config import load_config
from reasonbrain.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger("train")
    logger.info(f"Loaded config from {args.config}")

    # ---- model ----
    logger.info("Building ReasonBrain model ...")
    model = build_reasonbrain(cfg)

    # ---- preprocessors (used by the collator) ----
    logger.info("Loading CLIP for the data collator ...")
    clip_id = cfg["model"]["frce"]["image_encoder"]
    clip_processor = CLIPProcessor.from_pretrained(clip_id)
    clip = CLIPModel.from_pretrained(clip_id).eval().requires_grad_(False)

    collator = Reason50KCollator(
        clip_image_encoder=clip.vision_model,
        clip_text_encoder=clip.text_model,
        clip_processor=clip_processor,
        llava_processor=model.mllm.processor,
        num_image_tokens=cfg["model"]["mllm"]["num_image_tokens"],
        num_query_tokens=cfg["model"]["mllm"]["num_query_tokens"],
        max_objects=cfg["model"]["frce"]["max_objects"],
    )

    # ---- data ----
    train_ds = Reason50K(
        root=cfg["data"]["root"],
        split_file=cfg["data"]["train_jsonl"],
        image_size=cfg["data"]["image_size"],
    )
    val_ds = None
    try:
        val_ds = Reason50K(
            root=cfg["data"]["root"],
            split_file=cfg["data"]["val_jsonl"],
            image_size=cfg["data"]["image_size"],
        )
    except FileNotFoundError:
        logger.warning("No val split found — running training only.")

    # ---- trainer ----
    trainer = ReasonBrainTrainer(cfg, model, collator,
                                 train_dataset=train_ds, val_dataset=val_ds)
    trainer.train()


if __name__ == "__main__":
    main()
