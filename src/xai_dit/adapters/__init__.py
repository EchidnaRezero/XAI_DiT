"""Model adapters. ``base`` is torch-free; concrete adapters (torch) are imported lazily.

``from xai_dit.adapters import TokenMap`` works without PyTorch; ``AnimaAdapter``,
``MOSSImageAdapter``, ``Krea2Adapter`` and :func:`get_adapter` import torch on first use.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .base import (
    BRANCH_LABEL,
    CAPTURE_FIELDS,
    COND,
    INTERVENTION_HOOKS,
    UNCOND,
    AdapterCapability,
    CapturedTensors,
    CaptureSpec,
    InterventionSpec,
    ModelAdapter,
    SeqLayout,
    StepContext,
    StepWindow,
    TokenAlignmentError,
    TokenMap,
    TokenSpan,
    as_spans,
)

if TYPE_CHECKING:
    from .anima import AnimaAdapter
    from .krea2 import Krea2Adapter
    from .moss_image import MOSSImageAdapter

#: Canonical model name -> (module, class).
ADAPTERS: dict[str, tuple[str, str]] = {
    "anima": ("anima", "AnimaAdapter"),
    "moss_image": ("moss_image", "MOSSImageAdapter"),
    "krea2": ("krea2", "Krea2Adapter"),
}
_ALIASES = {"moss": "moss_image", "moss-image": "moss_image", "mossimage": "moss_image",
            "krea2_raw": "krea2", "krea-2": "krea2", "krea_2": "krea2"}
_LAZY = {cls: mod for mod, cls in ADAPTERS.values()}


def canonical_model_name(model_name: str) -> str:
    key = model_name.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in ADAPTERS:
        raise KeyError(f"unknown model {model_name!r}; expected one of {sorted(ADAPTERS)}")
    return key


def get_adapter(model_name: str, diffusion_model: Any = None, **kwargs: Any) -> ModelAdapter:
    """Instantiate the adapter for ``model_name`` (optionally bound to a DiT / ModelPatcher)."""
    mod, cls = ADAPTERS[canonical_model_name(model_name)]
    return getattr(importlib.import_module(f".{mod}", __name__), cls)(diffusion_model, **kwargs)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BRANCH_LABEL", "CAPTURE_FIELDS", "COND", "INTERVENTION_HOOKS", "UNCOND",
    "AdapterCapability", "CapturedTensors", "CaptureSpec", "InterventionSpec", "ModelAdapter",
    "SeqLayout", "StepContext", "StepWindow", "TokenAlignmentError", "TokenMap", "TokenSpan",
    "as_spans", "ADAPTERS", "AnimaAdapter", "Krea2Adapter", "MOSSImageAdapter",
    "canonical_model_name", "get_adapter",
]
