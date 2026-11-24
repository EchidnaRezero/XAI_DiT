"""Direction 01 heatmaps (``docs/specs/01_heatmap.md``).

* E1-1 ExploringMMDiT: streaming ``img_to_txt`` (cond rows only, all keys in the softmax
  denominator), average-rank Top-K block selection, sub-token word maps, 2D Gaussian
  smoothing (``k=7, sigma=1``), spatial-max-relative thresholding.
* E1-2 ConceptAttention: scale-free ``o_c o_x^T`` scores, concept-axis softmax with
  ``per_layer`` / ``after_mean`` order, Krea 2 ``pre_gate`` vs ``post_gate`` readout,
  one-directional concept-stream attention.
* E1-3 Seg4Diff: per-head (pre-GQA-repeat) V-norm ratio, symmetric KL, and the
  span-mass-preserving 1D text-axis blur (:class:`~xai_dit.core.session.TextAxisBlur`).

Pure PyTorch: no ``comfy.*`` imports.
"""

from __future__ import annotations

from typing import Literal, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from ..adapters.base import SeqLayout, TokenSpan
from .layer_selection import rankdata
from .session import (
    AttentionCall,
    ProbObserver,
    RunningMap,
    TextAxisBlur,
    chunked_attention,
    gaussian_kernel_1d,
    index_mask,
    slice_overlap,
)

Tensor = torch.Tensor

__all__ = [
    "img_to_txt", "ImgToTxtHeatmap", "average_rank_top_k", "fallback_middle_blocks", "word_map",
    "gaussian_kernel_2d", "gaussian_smooth_2d", "relative_threshold", "mask_metrics",
    "concept_scores", "concept_softmax", "ConceptAccumulator", "attention_readout",
    "prompt_output_space_scores", "concept_stream_attention", "mean_pool_span",
    "v_norm_ratio", "symmetric_kl", "TextAxisBlur", "text_axis_blur_factory", "span_mass",
]

# =========================================================================== E1-1 ExploringMMDiT


def img_to_txt(probs: Tensor, layout: SeqLayout) -> Tensor:
    """Slice ``Q_img × K_txt`` from full-row probabilities ``[B, H, Nq, Nk] -> [B, H, N, L]``.

    ``[img|txt]``: ``A[:, :, :N, N:N+L]``; ``[txt|img]``: ``A[:, :, T:T+N, :T]``; cross: all.
    """
    return probs[:, :, layout.q_img, layout.k_txt]


class _ImgToTxtObserver(ProbObserver):
    def __init__(self, target: RunningMap, layout: SeqLayout, rows: Sequence[int], head_mean: bool,
                 group: object, device: torch.device) -> None:
        self.target, self.layout, self.head_mean, self.group = target, layout, head_mean, group
        self.rows = torch.tensor(list(rows), device=device)
        self.buf: Optional[Tensor] = None

    def __call__(self, probs: Tensor, q_sl: slice, k_sl: slice) -> None:
        qo, ko = slice_overlap(q_sl, self.layout.q_img), slice_overlap(k_sl, self.layout.k_txt)
        if qo is None or ko is None:
            return
        p = probs.index_select(0, self.rows)[:, :, qo[0], ko[0]].float()
        if self.head_mean:
            p = p.mean(1, keepdim=True)
        if self.buf is None:
            self.buf = torch.zeros(p.shape[0], p.shape[1], self.layout.n_img, self.layout.n_txt,
                                   dtype=torch.float32, device=p.device)
        self.buf[:, :, qo[1], ko[1]] = p

    def finish(self) -> None:
        if self.buf is not None:
            self.target.add(self.buf, group=self.group)
            self.buf = None


class ImgToTxtHeatmap:
    """Per-block streaming ``img_to_txt`` maps for one CFG branch (default ``cond``).

    Register with ``session.add_observer(hm.factory, blocks=..., window=...)``. Each map is
    ``[B_branch, 1 or H, N, L]``. ``mode="mean"`` averages over steps/calls.
    Cond slicing happens *before* the head mean (fixes ``dit_attn_viz.py:L151`` CFG mixing).
    """

    def __init__(self, *, branch: Literal["cond", "uncond"] = "cond", head_mean: bool = True,
                 mode: Literal["mean", "group_mean_sum"] = "mean", device: torch.device | str = "cpu") -> None:
        self.branch, self.head_mean, self.mode, self.device = branch, head_mean, mode, device
        self.maps: dict[int, RunningMap] = {}

    def factory(self, call: AttentionCall) -> Optional[ProbObserver]:
        rows = call.ctx.rows(self.branch)
        if not rows:
            return None
        rm = self.maps.setdefault(call.block, RunningMap(self.mode, device=self.device))
        return _ImgToTxtObserver(rm, call.layout, rows, self.head_mean, call.ctx.step_index, call.device)

    def block_mean(self, blocks: Optional[Sequence[int]] = None) -> Tensor:
        """Mean map over ``blocks`` (default: every recorded block)."""
        ids = sorted(self.maps) if blocks is None else list(blocks)
        return torch.stack([self.maps[b].value() for b in ids]).mean(0)


def average_rank_top_k(metrics: Mapping[str, Sequence[float]], higher_is_better: Mapping[str, bool],
                       k: int = 5) -> list[int]:
    """Top-``k`` block indices by *average rank* across metrics (rank 1 = best; ties averaged).

    E.g. ``{"bce": ..., "soft_miou": ..., "mse": ...}`` with ``{"soft_miou": True}``.
    """
    ranks = []
    for name, vals in metrics.items():
        v = torch.as_tensor(vals, dtype=torch.float64)
        ranks.append(rankdata(-v if higher_is_better.get(name, False) else v))
    mean_rank = torch.stack(ranks).mean(0)
    order = torch.argsort(mean_rank, stable=True)
    return sorted(order[:k].tolist())


def fallback_middle_blocks(num_blocks: int, keep: float = 0.6) -> list[int]:
    """Pre-calibration fallback: middle ``keep`` fraction (drop first/last 20% noisy blocks)."""
    drop = round(num_blocks * (1 - keep) / 2)
    return list(range(drop, num_blocks - drop))


def word_map(maps: Tensor, token_positions: Sequence[int] | TokenSpan) -> Tensor:
    """Mean over the sub-token text columns of one word/tag: ``[..., N, L] -> [..., N]``."""
    idx = list(token_positions.indices()) if isinstance(token_positions, TokenSpan) else list(token_positions)
    return maps[..., idx].mean(-1)


def gaussian_kernel_2d(kernel_size: int = 7, sigma: float = 1.0, device=None, dtype=torch.float32) -> Tensor:
    """Normalised 2D Gaussian (separable outer product == ``ExploringMMDiT/utils/common.py``)."""
    g = gaussian_kernel_1d(kernel_size, sigma, device, dtype)
    k = torch.outer(g, g)
    return k / k.sum()


def gaussian_smooth_2d(maps: Tensor, h: int, w: int, kernel_size: int = 7, sigma: float = 1.0) -> Tensor:
    """Smooth flattened spatial maps ``[..., h*w]`` with zero padding ``k // 2`` (repo match)."""
    lead = maps.shape[:-1]
    x = maps.reshape(-1, 1, h, w)
    k = gaussian_kernel_2d(kernel_size, sigma, x.device, x.dtype).view(1, 1, kernel_size, kernel_size)
    return F.conv2d(x, k, padding=kernel_size // 2).reshape(*lead, h * w)


def relative_threshold(maps: Tensor, theta: float, eps: float = 0.0) -> Tensor:
    """Binary mask ``maps / spatial_max > theta`` (theta is relative to the max, in ``[0, 1]``)."""
    return maps / (maps.amax(-1, keepdim=True) + eps) > theta


def mask_metrics(attn: Tensor, mask: Tensor, eps: float = 1e-7) -> dict[str, float]:
    """BCE / soft IoU / MSE of a max-normalised map vs a binary mask.

    Mirrors ``ExploringMMDiT/utils/dit_attn_viz.py:evaluate_attention_map`` (L492-502).
    """
    a = (attn / (attn.max() + eps)).float()
    m = mask.float()
    inter = (a * m).sum()
    union = a.sum() + m.sum() - inter
    return {"bce": F.binary_cross_entropy(a.clamp(0, 1), m).item(),
            "soft_miou": (inter / (union + eps)).item(),
            "mse": F.mse_loss(a, m).item()}


# =========================================================================== E1-2 ConceptAttention


def concept_scores(o_c: Tensor, o_x: Tensor) -> Tensor:
    """Scale-free ``einsum("brd,bnd->brn")`` (no ``1/sqrt(D)``), ``[B, R, D] × [B, N, D] -> [B, R, N]``."""
    return torch.einsum("brd,bnd->brn", o_c.float(), o_x.float())


def concept_softmax(scores: Tensor, order: Literal["per_layer", "after_mean"] = "per_layer",
                    temperature: float = 1.0) -> Tensor:
    """Concept-axis softmax over stacked scores ``[Lyr, B, R, N] -> [B, R, N]``.

    ``per_layer`` (paper Eq. 11): mean over layers of ``softmax_R``.
    ``after_mean`` (``heatmap_utils.py``): ``softmax_R(mean / temperature)``.
    """
    if order == "per_layer":
        return torch.softmax(scores.float(), dim=-2).mean(0)
    if order == "after_mean":
        return torch.softmax(scores.float().mean(0) / temperature, dim=-2)
    raise ValueError(f"unknown order {order!r}")


class ConceptAccumulator:
    """Streaming ConceptAttention over (layer, step) calls without storing the score stack."""

    def __init__(self, order: Literal["per_layer", "after_mean"] = "per_layer", temperature: float = 1.0,
                 device: torch.device | str = "cpu") -> None:
        if order not in ("per_layer", "after_mean"):
            raise ValueError(f"unknown order {order!r}")
        self.order, self.temperature = order, temperature
        self.acc = RunningMap("mean", device=device)

    def add(self, scores: Tensor) -> None:
        """``scores [B, R, N]`` from :func:`concept_scores` for one (layer, step)."""
        self.acc.add(torch.softmax(scores.float(), dim=-2) if self.order == "per_layer" else scores)

    def value(self) -> Tensor:
        v = self.acc.value()
        return v if self.order == "per_layer" else torch.softmax(v / self.temperature, dim=-2)


def attention_readout(ctx: Tensor, gate_logits: Optional[Tensor] = None,
                      mode: Literal["pre_gate", "post_gate"] = "pre_gate") -> Tensor:
    """Krea 2 readout: ``pre_gate`` = ``ctx``; ``post_gate`` = ``ctx * sigmoid(gate_logits)``.

    ``gate_logits`` is the raw ``attn.gate(x)`` output (pre-sigmoid, ``krea2/model.py:106``).
    """
    if mode == "pre_gate":
        return ctx
    if gate_logits is None:
        raise ValueError("post_gate readout needs gate_logits")
    return ctx * torch.sigmoid(gate_logits)


def prompt_output_space_scores(o: Tensor, layout: SeqLayout, prompt_span: TokenSpan) -> Tensor:
    """Mode (1) ``prompt_output_space``: joint pre-``Wo`` output ``o [B, Nq, D]`` → ``[B, |S|, N]``.

    Uses existing prompt-token outputs (template tokens excluded via ``prompt_span``); 0 extra SDPA.
    """
    if not layout.is_joint:
        raise ValueError("prompt_output_space requires joint attention (Anima: N/A)")
    return concept_scores(o[:, layout.txt_queries(prompt_span)], o[:, layout.q_img])


def concept_stream_attention(q_c: Tensor, k_x: Tensor, v_x: Tensor, k_c: Tensor, v_c: Tensor,
                             scale: Optional[float] = None) -> Tensor:
    """One-directional concept attention ``SDPA(q_c, [k_x; k_c], [v_x; v_c])`` (Eq. 7) with GQA.

    ``q_c [B, Hq, R, d]``, ``k_x/v_x [B, Hkv, Nx, d]`` (image + prompt keys), ``k_c/v_c [B, Hkv, R, d]``.
    Image/prompt queries never see concept keys, so generation is unperturbed.
    """
    return chunked_attention(q_c, torch.cat([k_x, k_c], 2), torch.cat([v_x, v_c], 2), scale=scale)


def mean_pool_span(hidden: Tensor, span: TokenSpan) -> Tensor:
    """Mean-pool a multi-subtoken concept span ``[..., T, D] -> [..., D]`` (template tokens excluded by caller)."""
    return hidden[..., span.start:span.end, :].mean(-2)


# =========================================================================== E1-3 Seg4Diff


def v_norm_ratio(v_txt: Tensor, v_img: Tensor, *, num_kv_heads: Optional[int] = None,
                 mode: Literal["per_head", "concat"] = "per_head") -> dict[str, Tensor]:
    """Mean token L2 norm of prompt vs image values, and their ratio ``||V_prompt|| / ||V_img||``.

    Inputs ``[B, Hkv, T, d]`` **pre GQA repeat** (Krea 2: 12 KV heads), already sliced to the
    prompt-content span (no ``</s>`` / pad / template) and image tokens. On Anima pass
    ``cross_attn.v`` and ``self_attn.v`` (two projections → proxy ratio).
    ``per_head``: per-head norms averaged; ``concat``: norm of the concatenated heads.
    """
    if num_kv_heads is not None and (v_txt.shape[1] != num_kv_heads or v_img.shape[1] != num_kv_heads):
        raise ValueError("pass V before GQA repeat (repeating inflates concatenated norms)")

    def norms(v: Tensor) -> Tensor:
        v = v.float()
        if mode == "per_head":
            return v.norm(dim=-1).mean(dim=(0, 2))  # [Hkv]
        if mode == "concat":
            return v.transpose(1, 2).flatten(2).norm(dim=-1).mean().reshape(1)
        raise ValueError(f"unknown mode {mode!r}")

    t, i = norms(v_txt), norms(v_img)
    return {"txt_norm": t.mean(), "img_norm": i.mean(), "ratio": t.mean() / i.mean(), "per_head_ratio": t / i}


def symmetric_kl(p: Tensor, q: Tensor, eps: float = 1e-12, dim: int = -1) -> Tensor:
    """``0.5 Σ (p - q)(log p - log q)`` with ``clamp(min=eps)`` (Seg4Diff unsupervised merge)."""
    p, q = p.clamp_min(eps), q.clamp_min(eps)
    return 0.5 * ((p - q) * (p.log() - q.log())).sum(dim)


def text_axis_blur_factory(span: TokenSpan, *, kernel_size: int = 5, sigma: float = 9.0,
                           strength: float = 1.0, heads: Optional[Sequence[int]] = None,
                           num_heads: Optional[int] = None, preserve_mass: bool = True):
    """``attn_override`` factory for :class:`InterventionSpec` (``params={"override": ...}``).

    Blurs image-query rows along the text-key axis inside the prompt-content ``span`` only.
    """
    def make(layout: SeqLayout, batch: Optional[Tensor]) -> TextAxisBlur:
        head_mask = None
        if heads is not None:
            if num_heads is None:
                raise ValueError("num_heads is required with heads")
            head_mask = index_mask(num_heads, list(heads))
        return TextAxisBlur.for_layout(layout, span, batch=batch, heads=head_mask, kernel_size=kernel_size,
                                       sigma=sigma, strength=strength, preserve_mass=preserve_mass)

    return make


def span_mass(probs: Tensor, key_slice: slice) -> Tensor:
    """Per-row probability mass inside a key slice ``[..., Nk] -> [...]``."""
    return probs[..., key_slice].sum(-1)
