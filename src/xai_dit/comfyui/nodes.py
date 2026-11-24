"""Thin ComfyUI custom-node wrapper around :class:`~xai_dit.core.session.XAISession` + adapters.

Nodes never touch model internals directly: hooks are applied to a ``ModelPatcher`` clone via
``add_object_patch`` (restored by ComfyUI on unpatch), and the fixed CFG batching policy is a
``CALC_COND_BATCH`` wrapper on that clone.

* ``XAIDiTSession``  MODEL -> (MODEL, XAI_SESSION): attach an adapter + streaming img→txt heatmap.
* ``XAIDiTHeatmap``  (XAI_SESSION, LATENT) -> IMAGE: block-mean heatmap for selected text slots.

The heavy logic lives in ``xai_dit.core``; this module only parses widgets and routes objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch

from xai_dit.adapters import CAPTURE_FIELDS, CaptureSpec, ModelAdapter, StepWindow, canonical_model_name, get_adapter
from xai_dit.adapters._comfy_hooks import cfg_batching_wrapper
from xai_dit.core.heatmap import ImgToTxtHeatmap
from xai_dit.core.session import XAISession

try:  # optional: only present inside a ComfyUI process
    import comfy.patcher_extension as _patcher_extension
except ImportError:  # pragma: no cover - exercised only outside ComfyUI
    _patcher_extension = None

CATEGORY = "xai_dit"
MODEL_NAMES = ["auto", "anima", "moss_image", "krea2"]
_CFG_KEY = "xai_dit_cfg_batching"
_DIT_CLASS_TO_MODEL = {"Anima": "anima", "MOSSImage": "moss_image", "SingleStreamDiT": "krea2"}


@dataclass
class XAISessionHandle:
    """Value carried on the ``XAI_SESSION`` socket."""

    adapter: ModelAdapter
    session: XAISession
    heatmap: Optional[ImgToTxtHeatmap] = None
    meta: dict[str, Any] = field(default_factory=dict)


def parse_int_list(text: str) -> Optional[list[int]]:
    """``""`` -> ``None`` (all); ``"3,5,8-10"`` -> ``[3, 5, 8, 9, 10]`` (non-negative, ranges inclusive)."""
    text = (text or "").strip()
    if not text:
        return None
    out: list[int] = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def detect_model_name(model: Any) -> str:
    dm = model.model.diffusion_model
    name = _DIT_CLASS_TO_MODEL.get(type(dm).__name__)
    if name is None:
        raise ValueError(f"unsupported diffusion model {type(dm).__name__}; pick model_name explicitly")
    return name


def image_grid(n_img: int, latent: torch.Tensor) -> tuple[int, int]:
    """Token grid ``(h, w)`` for ``n_img`` image tokens given the latent (patch 1 or 2)."""
    lh, lw = latent.shape[-2], latent.shape[-1]
    for p in (1, 2):
        h, w = -(-lh // p), -(-lw // p)
        if h * w == n_img:
            return h, w
    raise ValueError(f"cannot map {n_img} image tokens onto a {lh}x{lw} latent")


class XAIDiTSession:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "model_name": (MODEL_NAMES, {"default": "auto"}),
                "heatmap": ("BOOLEAN", {"default": True}),
                "branch": (["cond", "uncond"], {"default": "cond"}),
                "blocks": ("STRING", {"default": "", "tooltip": "e.g. '6-20' (empty = all blocks)"}),
                "step_start": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "step_end": ("INT", {"default": -1, "min": -1, "max": 10000, "tooltip": "-1 = until the last step"}),
                "cfg_batching": (["split", "batched"], {"default": "split"}),
            },
            "optional": {
                "capture_fields": ("STRING", {"default": "", "tooltip": f"comma list of {', '.join(CAPTURE_FIELDS)}"}),
                "query_chunk": ("INT", {"default": 1024, "min": 16, "max": 65536}),
            },
        }

    RETURN_TYPES = ("MODEL", "XAI_SESSION")
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(self, model: Any, model_name: str, heatmap: bool, branch: str, blocks: str, step_start: int,
              step_end: int, cfg_batching: str, capture_fields: str = "", query_chunk: int = 1024) -> tuple[Any, XAISessionHandle]:
        name = detect_model_name(model) if model_name == "auto" else canonical_model_name(model_name)
        adapter = get_adapter(name, model)
        block_ids = parse_int_list(blocks)
        window = StepWindow(start=step_start, end=None if step_end < 0 else step_end)
        fields = frozenset(f.strip() for f in (capture_fields or "").split(",") if f.strip())
        capture = CaptureSpec(fields, blocks=block_ids, window=window, branches=frozenset({branch})) if fields else None
        session = XAISession(adapter.capability, capture=capture, query_chunk=query_chunk)
        hm = None
        if heatmap:
            hm = ImgToTxtHeatmap(branch=branch)
            session.add_observer(hm.factory, blocks=block_ids, window=window)

        m = model.clone()
        for path, fn in adapter.object_patches(session).items():
            m.add_object_patch(path, fn)
        if _patcher_extension is not None:
            m.add_wrapper_with_key(_patcher_extension.WrappersMP.CALC_COND_BATCH, _CFG_KEY, cfg_batching_wrapper(cfg_batching))
        return m, XAISessionHandle(adapter, session, hm, {"model": name, "cfg_batching": cfg_batching})


class XAIDiTHeatmap:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "xai_session": ("XAI_SESSION",),
                "samples": ("LATENT", {"tooltip": "sampler output (orders this node after sampling)"}),
                "token_indices": ("STRING", {"default": "", "tooltip": "DiT text slots, e.g. '3-5' (empty = all)"}),
                "blocks": ("STRING", {"default": "", "tooltip": "subset of recorded blocks (empty = all)"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = CATEGORY

    def render(self, xai_session: XAISessionHandle, samples: dict, token_indices: str, blocks: str) -> tuple[torch.Tensor]:
        hm = xai_session.heatmap
        if hm is None or not hm.maps:
            raise ValueError("no heatmap recorded: enable 'heatmap' on XAIDiTSession and run a sampler first")
        maps = hm.block_mean(parse_int_list(blocks))  # [B, 1|H, N, L]
        cols: Sequence[int] = parse_int_list(token_indices) or range(maps.shape[-1])
        m = maps[..., list(cols)].mean(-1).mean(1)  # [B, N]
        h, w = image_grid(m.shape[-1], samples["samples"])
        m = m.reshape(-1, h, w)
        lo = m.amin(dim=(-2, -1), keepdim=True)
        hi = m.amax(dim=(-2, -1), keepdim=True)
        m = (m - lo) / (hi - lo).clamp_min(1e-12)
        return (m.unsqueeze(-1).expand(-1, -1, -1, 3).float().cpu().contiguous(),)


NODE_CLASS_MAPPINGS = {"XAIDiTSession": XAIDiTSession, "XAIDiTHeatmap": XAIDiTHeatmap}
NODE_DISPLAY_NAME_MAPPINGS = {"XAIDiTSession": "XAI DiT Session", "XAIDiTHeatmap": "XAI DiT Heatmap"}
