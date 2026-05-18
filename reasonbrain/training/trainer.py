
"""Accelerate-based trainer.

The trainer is intentionally compact: it expects an already-built model and
collator, hooks them into 🤗 Accelerate for distributed / mixed-precision
training, and persists checkpoints under ``cfg.output_dir``.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.reasonbrain import ReasonBrain
from .losses import total_loss
from .optim import build_optimizer, build_scheduler, trainable_parameters


# ---------------------------------------------------------------------------
class ReasonBrainTrainer:
    def __init__(self, cfg: Dict[str, Any], model: ReasonBrain,
                 collator, train_dataset, val_dataset=None):
        self.cfg = cfg
        self.model = model
        self.collator = collator

        # --- accelerate ----
        self.accelerator = Accelerator(
            mixed_precision=cfg["train"].get("mixed_precision", "bf16"),
            gradient_accumulation_steps=cfg["train"].get(
                "gradient_accumulation_steps", 1),
            log_with="wandb",
            project_dir=cfg["output_dir"],
        )
        set_seed(cfg.get("seed", 42))

        # --- data ----
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["train"]["batch_size"],
            shuffle=True,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=cfg["data"]["pin_memory"],
            collate_fn=collator,
            drop_last=True,
        )
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=cfg["train"]["batch_size"],
                shuffle=False,
                num_workers=cfg["data"]["num_workers"],
                pin_memory=cfg["data"]["pin_memory"],
                collate_fn=collator,
            )
        else:
            self.val_loader = None

        # --- optim ----
        params = trainable_parameters(model)
        self.optimizer = build_optimizer(params, cfg["optim"])
        self.scheduler = build_scheduler(self.optimizer, cfg["optim"])

        (self.model, self.optimizer, self.train_loader, self.scheduler
         ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler,
        )
        if self.val_loader is not None:
            self.val_loader = self.accelerator.prepare(self.val_loader)

        self.output_dir = Path(cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                cfg.get("run_name", "reasonbrain"), config=cfg,
            )

        self.global_step = 0

    # ------------------------------------------------------------------
    def train(self) -> None:
        cfg_t = self.cfg["train"]
        max_steps = self.cfg["optim"]["max_steps"]
        w_mllm = cfg_t["loss_weights"].get("mllm", 1.0)
        w_dm = cfg_t["loss_weights"].get("dm", 1.0)

        progress = tqdm(total=max_steps, disable=not self.accelerator.is_main_process,
                         desc="train")
        running = {"total": 0.0, "l_dm": 0.0, "l_mllm": 0.0}
        ema = 0.99

        self.model.train()
        while self.global_step < max_steps:
            for batch in self.train_loader:
                with self.accelerator.accumulate(self.model):
                    out = self.model(batch)
                    losses = total_loss(out, w_mllm=w_mllm, w_dm=w_dm)
                    loss = losses["total"]
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            trainable_parameters(self.model),
                            self.cfg["optim"].get("grad_clip", 1.0),
                        )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                # ---- logging ----
                for k, v in losses.items():
                    running[k] = ema * running.get(k, 0.0) + (1 - ema) * float(v)

                if self.global_step % cfg_t["log_every"] == 0:
                    if self.accelerator.is_main_process:
                        log = {f"train/{k}": running[k] for k in running}
                        log["train/lr"] = self.scheduler.get_last_lr()[0]
                        self.accelerator.log(log, step=self.global_step)
                        progress.set_postfix({k: f"{v:.4f}" for k, v in log.items()})

                # ---- checkpoint ----
                if self.global_step > 0 and self.global_step % cfg_t["ckpt_every"] == 0:
                    self.save(f"step_{self.global_step}")

                self.global_step += 1
                progress.update(1)
                if self.global_step >= max_steps:
                    break

        self.save("last")
        if self.accelerator.is_main_process:
            self.accelerator.end_training()

    # ------------------------------------------------------------------
    def save(self, name: str) -> None:
        if not self.accelerator.is_main_process:
            return
        path = self.output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        # Save only trainable parameters to keep checkpoints compact.
        state = {k: v.detach().cpu()
                 for k, v in unwrapped.state_dict().items()
                 if v.requires_grad or "lora" in k.lower()}
        torch.save(state, path / "reasonbrain.pt")
