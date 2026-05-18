"""Model components for the ReasonBrain framework.

The heavy ``ReasonBrain`` model and its builders are imported lazily so that
``from reasonbrain.models import FRCE`` does not require LLaVA / FLUX to be
installed.
"""

from .cme import CME
from .frce import FRCE
from .id_controller import IDController
from .qformer import QFormer

__all__ = ["FRCE", "CME", "QFormer", "IDController", "ReasonBrain", "build_reasonbrain"]


def __getattr__(name):  # pragma: no cover - thin lazy loader
    if name in ("ReasonBrain", "build_reasonbrain"):
        from .reasonbrain import ReasonBrain, build_reasonbrain
        return {"ReasonBrain": ReasonBrain,
                "build_reasonbrain": build_reasonbrain}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
