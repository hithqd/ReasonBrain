"""Loss functions used by ReasonBrain.

Two terms (§4.4 of the paper):

* **L_MLLM** — next-token log-likelihood on the ``[IMG_i]`` tokens. We rely
  on Hugging Face's built-in ``labels`` mechanism, so the logit is already
  computed by :class:`MLLMWrapper` and only the cross-entropy needs to be
  recomputed here when ``mllm_logits`` is provided.
* **L_DM**   — squared-error between the FLUX noise prediction and the
  flow-matching target ``(noise - target_latent)``.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ..models.reasonbrain import ReasonBrainOutput


# ---------------------------------------------------------------------------
def diffusion_loss(output: ReasonBrainOutput) -> torch.Tensor:
    """Mean-squared error in the diffusion latent space."""
    return F.mse_loss(output.pred_noise.float(),
                      output.target_noise.float())


# ---------------------------------------------------------------------------
def mllm_loss(output: ReasonBrainOutput,
              ignore_index: int = -100) -> Optional[torch.Tensor]:
    """Causal-LM cross-entropy on the ``[IMG_i]`` token slots."""
    if output.mllm_logits is None or output.image_token_labels is None:
        return None
    logits = output.mllm_logits[..., :-1, :].contiguous()
    labels = output.image_token_labels[..., 1:].contiguous()
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
    )


# ---------------------------------------------------------------------------
def total_loss(output: ReasonBrainOutput,
               w_mllm: float = 1.0, w_dm: float = 1.0) -> Dict[str, torch.Tensor]:
    """Return the weighted total loss and its components."""
    l_dm = diffusion_loss(output)
    l_mllm = mllm_loss(output)
    components = {"l_dm": l_dm}
    if l_mllm is not None:
        components["l_mllm"] = l_mllm
        total = w_mllm * l_mllm + w_dm * l_dm
    else:
        total = w_dm * l_dm
    components["total"] = total
    return components
