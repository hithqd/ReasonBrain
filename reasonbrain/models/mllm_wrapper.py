"""Thin wrapper around the LLaVA-v1.1-7B multimodal LLM.

This wrapper:
  * loads LLaVA + its vision tower (CLIP-L/14),
  * extends the vocabulary with ``num_image_tokens`` special ``[IMG_i]``
    tokens (paper uses 32),
  * exposes ``num_query_tokens`` learnable embeddings that are appended to
    the input sequence and whose final hidden states become the diffusion
    condition ``V``,
  * wraps the LLM in PEFT-LoRA so only a small subset of parameters is
    trained.

If the official LLaVA weights are not available, any HuggingFace
``LlavaForConditionalGeneration`` checkpoint (e.g. ``llava-hf/llava-1.5-7b-hf``)
can be plugged in by changing ``pretrained``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

try:  # newer transformers expose a unified LLaVA class
    from transformers import (
        AutoProcessor,
        LlavaForConditionalGeneration,
    )
    _HAS_LLAVA = True
except ImportError:  # pragma: no cover
    LlavaForConditionalGeneration = None  # type: ignore
    AutoProcessor = None  # type: ignore
    _HAS_LLAVA = False

try:
    from peft import LoraConfig, get_peft_model
    _HAS_PEFT = True
except ImportError:  # pragma: no cover
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore
    _HAS_PEFT = False


# ---------------------------------------------------------------------------
@dataclass
class MLLMOutput:
    """Outputs of a forward pass through the MLLM wrapper."""

    # hidden states of the appended learnable query tokens [B, r, C_mllm]
    query_hidden: torch.Tensor
    # vocab logits for the extra image tokens (used by L_MLLM)
    image_token_logits: Optional[torch.Tensor]
    # raw outputs in case caller needs them
    last_hidden_state: torch.Tensor


# ---------------------------------------------------------------------------
class MLLMWrapper(nn.Module):
    """LLaVA wrapper exposing exactly what ReasonBrain needs."""

    def __init__(self,
                 pretrained: str = "liuhaotian/LLaVA-7b-v1",
                 num_image_tokens: int = 32,
                 num_query_tokens: int = 32,
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.05,
                 lora_target_modules: Optional[List[str]] = None,
                 dtype: torch.dtype = torch.bfloat16,
                 device_map: Optional[str] = None):
        super().__init__()
        if not _HAS_LLAVA:
            raise ImportError(
                "transformers>=4.43 with LLaVA support is required. "
                "Run `pip install -r requirements.txt`."
            )
        if not _HAS_PEFT:
            raise ImportError(
                "peft>=0.11 is required. Run `pip install -r requirements.txt`."
            )

        self.num_image_tokens = num_image_tokens
        self.num_query_tokens = num_query_tokens

        # ---- Load processor + base model ----
        self.processor = AutoProcessor.from_pretrained(pretrained)
        base = LlavaForConditionalGeneration.from_pretrained(
            pretrained, torch_dtype=dtype, device_map=device_map,
        )

        # ---- Extend the tokenizer with new special tokens ----
        tokenizer = self.processor.tokenizer
        img_tokens = [f"[IMG_{i}]" for i in range(num_image_tokens)]
        query_tokens = [f"[Q_{i}]" for i in range(num_query_tokens)]
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": img_tokens + query_tokens}
        )
        if added > 0:
            base.resize_token_embeddings(len(tokenizer))

        # Record the token ids
        self.image_token_ids = torch.tensor(
            tokenizer.convert_tokens_to_ids(img_tokens), dtype=torch.long,
        )
        self.query_token_ids = torch.tensor(
            tokenizer.convert_tokens_to_ids(query_tokens), dtype=torch.long,
        )

        # ---- Apply LoRA ----
        targets = lora_target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]
        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=targets,
        )
        self.model = get_peft_model(base, lora_cfg)

        # The vision tower / projector remain frozen by default; only LoRA
        # parameters + the newly added token embeddings are trainable. The
        # embedding row for new tokens needs gradients explicitly.
        self._enable_new_token_grads()

        # Cache hidden size for downstream modules.
        self.hidden_size = base.config.text_config.hidden_size

    # ------------------------------------------------------------------
    def _enable_new_token_grads(self) -> None:
        """Make the new-token embedding rows trainable while keeping the rest frozen."""
        embedding = self.model.get_input_embeddings()
        # Re-enable grads on the embedding weight (PEFT freezes everything by default).
        embedding.weight.requires_grad_(True)
        # Mask gradients of pre-existing tokens via a hook so we don't drift them.
        orig_vocab = embedding.weight.shape[0] - self.num_image_tokens - self.num_query_tokens
        mask = torch.zeros_like(embedding.weight)
        mask[orig_vocab:] = 1.0
        self._embed_grad_mask = mask  # buffer-like, kept on CPU until first use

        def _hook(grad: torch.Tensor) -> torch.Tensor:
            if self._embed_grad_mask.device != grad.device:
                self._embed_grad_mask = self._embed_grad_mask.to(grad.device)
            return grad * self._embed_grad_mask

        embedding.weight.register_hook(_hook)

    # ------------------------------------------------------------------
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                pixel_values: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> MLLMOutput:
        """Run a forward pass.

        Args:
            input_ids     : [B, L] token ids — the caller must ensure that the
                            last ``num_query_tokens`` positions are the
                            ``[Q_i]`` placeholders.
            attention_mask: [B, L]
            pixel_values  : [B, 3, H, W] image pixels (CLIP preprocessing).
            labels        : optional [B, L] for the MLLM next-token loss.

        Returns:
            MLLMOutput
        """
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = out.hidden_states[-1]                      # [B, L, C]
        # the last ``num_query_tokens`` positions correspond to [Q_i]
        query_hidden = last_hidden[:, -self.num_query_tokens:]    # [B, r, C]
        return MLLMOutput(
            query_hidden=query_hidden,
            image_token_logits=out.logits if labels is not None else None,
            last_hidden_state=last_hidden,
        )

    # ------------------------------------------------------------------
    def build_inputs(self,
                     image: "torch.Tensor | None",
                     instruction: str,
                     device: torch.device) -> dict:
        """Convenience for inference: tokenize an instruction and append the
        ``[Q_i]`` placeholders so that the LLM emits ``r`` hidden states for
        them."""
        prompt = f"USER: <image>\n{instruction}\nASSISTANT:"
        # append the learnable query tokens at the end
        prompt += "".join(f"[Q_{i}]" for i in range(self.num_query_tokens))
        encoded = self.processor(
            images=image, text=prompt, return_tensors="pt",
        )
        return {k: v.to(device) for k, v in encoded.items()}
