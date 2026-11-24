"""Direction 02 layer selection (``docs/specs/02_layer_selection.md``).

* E2-1 StableFlow vitality: ``vitality(l) = 1 - mean_k cos(M_full, M_-l)`` on DINOv2 CLS
  embeddings (range ``[0, 2]``), elbow-first / top-20% fallback selection, disjoint-half
  Spearman stability, prompt-level paired bootstrap 95% CI.
* E2-2 LocalizingKnowledgeDiT ``Cont^(l)(κ)``: ``(1/N) Σ_i ||Wo concat_h(Σ_{j∈I_κ} A_ij v_j)||``
  computed by *calling* the adapter's ``wo`` module (quantization-safe), with Krea 2
  ``raw_mvo`` / ``gated_mvo``; streaming over query/key chunks; Top/Bottom/Random-20%.

True block bypass itself is a model-side hook (``InterventionSpec("true_block_bypass")``,
see :meth:`XAISession.bypass_mode`). Pure PyTorch: no ``comfy.*`` imports.
"""

from __future__ import annotations

import math
from typing import Callable, Literal, Optional, Sequence

import torch

from ..adapters.base import SeqLayout, TokenSpan, as_spans
from .session import AttentionCall, ProbObserver, RunningMap, grouped_pv, slice_overlap, union_mask

Tensor = torch.Tensor
Module = Callable[[Tensor], Tensor]

# =========================================================================== statistics


def rankdata(x: Tensor) -> Tensor:
    """1-based ranks of a 1D tensor with ties averaged (``scipy.stats.rankdata`` 'average')."""
    x = x.flatten().double()
    s, order = torch.sort(x, stable=True)
    _, inv, counts = torch.unique_consecutive(s, return_inverse=True, return_counts=True)
    ends = counts.cumsum(0).double()
    avg = (ends - counts.double() + 1 + ends) / 2
    r = torch.empty_like(x)
    r[order] = avg[inv]
    return r


def spearman_rho(a: Tensor, b: Tensor) -> float:
    """Spearman rank correlation (Pearson on average ranks)."""
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = ra.norm() * rb.norm()
    return float((ra @ rb) / denom) if denom > 0 else float("nan")


def paired_bootstrap_ci(per_prompt: Tensor, *, n_boot: int = 2000, alpha: float = 0.05,
                        generator: Optional[torch.Generator] = None) -> tuple[Tensor, Tensor, Tensor]:
    """Prompt-level paired bootstrap over the last axis ``[..., k]``.

    The same resampled prompt indices are used for every leading entry (e.g. all layers), so
    between-layer comparisons stay paired. Returns ``(mean, lo, hi)`` of the per-entry mean.
    """
    x = per_prompt.detach().double().cpu()
    k = x.shape[-1]
    idx = torch.randint(0, k, (n_boot, k), generator=generator)
    boots = x[..., idx].mean(-1)  # [..., n_boot]
    q = torch.tensor([alpha / 2, 1 - alpha / 2], dtype=torch.float64)
    lo, hi = torch.quantile(boots, q, dim=-1)
    return x.mean(-1), lo, hi


def split_half_spearman(per_prompt: Tensor, *, halves: Optional[tuple[Sequence[int], Sequence[int]]] = None,
                        n_splits: int = 1, generator: Optional[torch.Generator] = None) -> Tensor:
    """Spearman ρ between layer scores from two *disjoint* prompt halves (e.g. 8+8 of k=16).

    ``per_prompt [L, k]``. Pass ``halves`` explicitly or draw ``n_splits`` random disjoint splits.
    """
    k = per_prompt.shape[-1]
    if halves is not None:
        a, b = list(halves[0]), list(halves[1])
        if set(a) & set(b):
            raise ValueError("halves must be disjoint")
        splits = [(a, b)]
    else:
        splits = []
        for _ in range(n_splits):
            p = torch.randperm(k, generator=generator).tolist()
            splits.append((p[:k // 2], p[k // 2:2 * (k // 2)]))
    return torch.tensor([spearman_rho(per_prompt[:, a].mean(-1), per_prompt[:, b].mean(-1)) for a, b in splits])


# =========================================================================== E2-1 StableFlow vitality


def cosine_vitality(full: Tensor, bypassed: Tensor) -> Tensor:
    """Per-prompt vitality ``1 - cos(M_full, M_-l)``: ``full [k, D]``, ``bypassed [L, k, D]`` -> ``[L, k]``.

    Identical → 0, orthogonal → 1, opposite → 2.
    """
    cos = torch.nn.functional.cosine_similarity(bypassed.double(), full.double().unsqueeze(0), dim=-1)
    return 1.0 - cos


def select_top_fraction(scores: Tensor, frac: float = 0.2) -> list[int]:
    """Top ``round(frac * L)`` (at least 1) layers by score."""
    n = max(1, round(frac * scores.numel()))
    return sorted(torch.argsort(scores.double(), descending=True, stable=True)[:n].tolist())


def select_elbow(scores: Tensor, *, min_distance: float = 0.1, lo: Optional[Tensor] = None,
                 hi: Optional[Tensor] = None) -> Optional[list[int]]:
    """Kneedle elbow on the descending score curve; ``None`` if no elbow is detected.

    The knee is the point farthest below the chord of the normalised curve; layers before it
    are selected. With bootstrap bounds ``lo``/``hi`` the elbow only counts if CI-separated
    (``min lo[selected] > max hi[rest]``).
    """
    s = scores.double()
    L = s.numel()
    if L < 3:
        return None
    order = torch.argsort(s, descending=True, stable=True)
    y = s[order]
    span = y[0] - y[-1]
    if span <= 0:
        return None
    yn = (y - y[-1]) / span
    xn = torch.linspace(0, 1, L, dtype=torch.float64)
    dist = (1 - xn) - yn
    knee = int(torch.argmax(dist))
    if dist[knee] < min_distance or knee == 0:
        return None
    sel = order[:knee]
    if lo is not None and hi is not None:
        rest = order[knee:]
        if not bool(lo[sel].min() > hi[rest].max()):
            return None
    return sorted(sel.tolist())


def select_vital_layers(scores: Tensor, *, frac: float = 0.2, lo: Optional[Tensor] = None,
                        hi: Optional[Tensor] = None, min_distance: float = 0.1) -> tuple[list[int], str]:
    """E2-1 rule: elbow first; if not detected, fixed top-``frac`` (rounded). Returns ``(layers, rule)``."""
    elbow = select_elbow(scores, min_distance=min_distance, lo=lo, hi=hi)
    return (elbow, "elbow") if elbow is not None else (select_top_fraction(scores, frac), f"top_{frac:g}")


# =========================================================================== E2-2 Cont^(l)


def _concat_heads(o: Tensor) -> Tensor:
    """``[B, Hq, N, d] -> [B, N, Hq*d]`` (channel ``h*d + dd`` = head ``h``, dim ``dd``)."""
    return o.transpose(1, 2).flatten(2)


def contribution_from_context(o: Tensor, wo: Module, *, gate_logits: Optional[Tensor] = None,
                              g1: Optional[Tensor] = None, wo_dtype: Optional[torch.dtype] = None) -> dict[str, Tensor]:
    """Apply ``wo`` (module call) to a per-query concept context ``o [B, N, Hq*d]``.

    Returns ``raw_mvo [B] = mean_i ||wo(o_i)||`` and, if ``gate_logits`` is given (Krea 2),
    ``gated_mvo [B] = mean_i ||g1 ⊙ wo(sigmoid(gate_logits_i) ⊙ o_i)||`` (exact residual delta).
    Norms are taken in float32.
    """
    dt = wo_dtype or o.dtype
    out = {"raw_mvo": wo(o.to(dt)).float().norm(dim=-1).mean(-1)}
    if gate_logits is not None:
        y = wo((o.float() * torch.sigmoid(gate_logits.float())).to(dt)).float()
        if g1 is not None:
            y = y * g1.float()
        out["gated_mvo"] = y.norm(dim=-1).mean(-1)
    return out


def attention_contribution(a: Tensor, v: Tensor, wo: Module, **kwargs) -> dict[str, Tensor]:
    """Non-streaming ``Cont``: ``a [B, Hq, N, |I|]`` (img rows, concept columns of the *full*
    softmax), ``v [B, Hkv, |I|, d]`` (pre GQA repeat). Sums ``A v`` over ``I`` before concatenating
    heads, then calls ``wo``. See :func:`contribution_from_context` for kwargs."""
    return contribution_from_context(_concat_heads(grouped_pv(a.float(), v.float())), wo, **kwargs)


class _ContObserver(ProbObserver):
    def __init__(self, tracker: ContributionTracker, call: AttentionCall, rows: Sequence[int]) -> None:
        self.tracker, self.call = tracker, call
        self.layout: SeqLayout = call.layout
        dev = call.device
        self.rows = torch.tensor(list(rows), device=dev)
        self.key_mask = union_mask(self.layout.n_key, [self.layout.txt_keys(s) for s in tracker.spans], dev)
        self.v = call.v.index_select(0, self.rows).float()
        B, Hq, d = len(rows), call.q.shape[1], call.v.shape[-1]
        self.buf = torch.zeros(B, Hq, self.layout.n_img, d, dtype=torch.float32, device=dev)

    def __call__(self, probs: Tensor, q_sl: slice, k_sl: slice) -> None:
        qo = slice_overlap(q_sl, self.layout.q_img)
        km = self.key_mask[k_sl]
        if qo is None or not bool(km.any()):
            return
        p = probs.index_select(0, self.rows)[:, :, qo[0]][..., km].float()
        self.buf[:, :, qo[1]] += grouped_pv(p, self.v[:, :, k_sl][:, :, km])

    def finish(self) -> None:
        def img_rows(t: Optional[Tensor]) -> Optional[Tensor]:
            # per-token extras ([B, Nq, C]) are restricted to image queries; broadcast ones kept as-is
            if t is None:
                return None
            t = t.index_select(0, self.rows)
            return t[:, self.layout.q_img] if t.ndim >= 2 and t.shape[1] == self.layout.n_query else t

        ex = self.call.extras
        res = contribution_from_context(_concat_heads(self.buf), self.tracker.wo_for_block(self.call.block),
                                        gate_logits=img_rows(ex.get("gate_logits")), g1=img_rows(ex.get("g1")),
                                        wo_dtype=self.call.v.dtype)
        for name, val in res.items():
            self.tracker.maps.setdefault((name, self.call.block), RunningMap("mean")).add(val)


class ContributionTracker:
    """Streaming per-block ``Cont^(l)(κ)`` for one concept κ (text spans), averaged over steps.

    Register with ``session.add_observer(tracker.factory)``; the adapter must pass
    ``extras={"gate_logits": ..., "g1": ...}`` on Krea 2 for ``gated_mvo``.
    Never materialises full probabilities: only ``A[img, I_κ] @ V[I_κ]`` per chunk.
    """

    def __init__(self, spans: TokenSpan | Sequence[TokenSpan], wo_for_block: Callable[[int], Module],
                 branch: Literal["cond", "uncond"] = "cond") -> None:
        self.spans = as_spans(spans)
        self.wo_for_block, self.branch = wo_for_block, branch
        self.maps: dict[tuple[str, int], RunningMap] = {}

    def factory(self, call: AttentionCall) -> Optional[ProbObserver]:
        rows = call.ctx.rows(self.branch)
        return _ContObserver(self, call, rows) if rows else None

    def scores(self, kind: Literal["raw_mvo", "gated_mvo"] = "raw_mvo") -> dict[int, Tensor]:
        """Per-block ``Cont`` (``[B_branch]``) averaged over all recorded steps."""
        return {blk: rm.value() for (name, blk), rm in sorted(self.maps.items(), key=lambda kv: kv[0][1])
                if name == kind}


def top_bottom_random(scores: Tensor, frac: float = 0.2,
                      generator: Optional[torch.Generator] = None) -> dict[str, list[int]]:
    """Top / Bottom / Random ``round(frac*L)`` block sets (Random is the matched-size control)."""
    L = scores.numel()
    n = max(1, round(frac * L))
    order = torch.argsort(scores.double(), descending=True, stable=True)
    return {"top": sorted(order[:n].tolist()), "bottom": sorted(order[-n:].tolist()),
            "random": sorted(torch.randperm(L, generator=generator)[:n].tolist())}


def relative_depth_blocks(num_blocks: int, fractions: Sequence[float] = (0.25, 0.5, 0.75)) -> list[int]:
    """Map relative depths to block indices by flooring (28 → [7, 14, 21], 42 → [10, 21, 31])."""
    return [min(num_blocks - 1, math.floor(f * num_blocks)) for f in fractions]
