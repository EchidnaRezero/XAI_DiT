"""Direction 03 tag influence & causal interventions (``docs/specs/03_tag_influence.md``,
``docs/experiment_design.md`` §E3).

* E3-0 Same-State effects at a fixed state ``x_σ``: ``direct_effect`` (logit knockout) vs
  ``total_effect`` (token-ID/length-matched neutral substitute), raw ``||Δv||`` and
  ``|Δσ_i| ||Δv||``, CFG-combined output, and the rollout bridge ``bridge_corr``.
* E3-1 TextGen K/V patch cache (text or image slice, KV-head groups, ``i >= start``).
* E3-2 PCI Euler prefix-cache controller, ``τ_q`` on ``τ := σ``, reference-normalised score.
* E3-3 TACA ``σ >= 0.97`` logit scaling spec and App. A Listing 2 fast path.
* E3-4 DiT-ST attention-change boundary (Δσ-normalised / uniform-grid curvature) and SAP
  stage-prompt bundle schedule.
* Offline BF16 conditioning cache (hash-keyed).

Pure PyTorch: no ``comfy.*`` imports. Step index ``i`` always counts from the noise end.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from ..adapters.base import (
    CfgScope,
    InterventionSpec,
    SeqLayout,
    StepContext,
    StepWindow,
    TokenMap,
    TokenSpan,
    as_spans,
)
from .session import expand_kv, union_mask

Tensor = torch.Tensor

# =========================================================================== E3-0 Same-State


def cfg_combine(v_cond: Tensor, v_uncond: Tensor, scale: float) -> Tensor:
    """Classifier-free-guided output ``v_u + s (v_c - v_u)`` (primary E3-0 readout)."""
    return v_uncond + scale * (v_cond - v_uncond)


def same_state_effect(v_base: Tensor, v_int: Tensor, sigma: float, sigma_next: float,
                      channel_dim: int = 1) -> dict[str, Tensor]:
    """One-step effect at a fixed state: ``raw = ||Δv||_2`` over channels and ``weighted = |Δσ| raw``.

    This is a local derivative at ``x_σ``, not a causal effect on ``x_0`` (see :func:`rollout_bridge`).
    """
    raw = (v_int.float() - v_base.float()).norm(dim=channel_dim)
    return {"raw": raw, "weighted": abs(sigma_next - sigma) * raw}


def direct_effect_spec(keys: TokenSpan | Sequence[TokenSpan], *, blocks: Optional[Sequence[int]] = None,
                       window: StepWindow = StepWindow(), cfg_scope: CfgScope = "both") -> InterventionSpec:
    """``direct_effect``: knock out ``img → keys`` pre-softmax logits (direct attention path only)."""
    return InterventionSpec("logit_knockout", blocks=blocks, window=window, cfg_scope=cfg_scope,
                            params={"keys": keys, "query": "img"})


def check_total_effect_pair(base: TokenMap, neutral: TokenMap, target: TokenSpan | str) -> None:
    """``total_effect`` precondition: neutral prompt differs from base only inside ``target``."""
    base.assert_strict_token_alignment(neutral, editable=[target])


def random_token_control(token_map: TokenMap, n_tokens: int, *, exclude: Sequence[TokenSpan] = (),
                         span_name: str = "S", generator: Optional[torch.Generator] = None) -> list[int]:
    """Same-count random-token control row: ``n_tokens`` positions from ``span_name`` outside ``exclude``."""
    pool = [i for i in token_map.span(span_name).indices() if not any(i in e for e in exclude)]
    if len(pool) < n_tokens:
        raise ValueError(f"only {len(pool)} candidate tokens for {n_tokens}")
    perm = torch.randperm(len(pool), generator=generator)[:n_tokens]
    return sorted(pool[i] for i in perm.tolist())


def rollout_bridge(delta_v: Sequence[Tensor], sigmas: Sequence[float], delta_x0: Tensor,
                   channel_dim: int = 1) -> dict[str, float]:
    """Compare the first-order Euler prediction ``Σ_i Δσ_i Δv_i`` with the actual paired ``Δx_0``.

    ``delta_v[i]`` is the Same-State ``Δv`` at step ``i`` (``Δσ_i = σ_{i+1} - σ_i``).
    ``bridge_corr`` = spatial Pearson correlation of the channel-norm maps; ``cosine`` = full vector cosine.
    """
    if len(sigmas) < len(delta_v) + 1:
        raise ValueError("need sigma_{i+1} for every delta_v[i]")
    pred = sum((sigmas[i + 1] - sigmas[i]) * dv.float() for i, dv in enumerate(delta_v))
    a, b = pred.norm(dim=channel_dim).flatten(), delta_x0.float().norm(dim=channel_dim).flatten()
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    return {"bridge_corr": float(a @ b / denom) if denom > 0 else float("nan"),
            "cosine": float(F.cosine_similarity(pred.flatten(), delta_x0.float().flatten(), dim=0))}


# =========================================================================== E3-1 K/V patching


class KVCache:
    """Cache-and-inject K/V slices keyed by ``(step, block, branch, sample)``.

    Source pass: :meth:`store` (call at the hook point; pre-RoPE is mandatory on MOSS-Image).
    Target pass: :meth:`patch` returns patched copies; sources map branch-to-branch
    (source-cond → target-cond, source-uncond → target-uncond), so batched vs split CFG calls
    are equivalent. ``which="txt"`` = TextGen text K/V; ``"img"`` = StableFlow image K/V.
    K/V layout ``[B, Hkv, Nk, d]`` (pre GQA repeat): patching KV head ``g`` affects Q heads
    ``{g*G, ..., g*G + G - 1}`` only.
    """

    def __init__(self, which: Literal["txt", "img"] = "txt", to_cpu: bool = False) -> None:
        self.which, self.to_cpu = which, to_cpu
        self.store_: dict[tuple, tuple[Tensor, Tensor]] = {}

    def _slice(self, layout: SeqLayout) -> slice:
        s = layout.k_txt if self.which == "txt" else layout.k_img
        if s is None:
            raise ValueError("image K/V do not exist in cross-attention")
        return s

    def store(self, block: int, ctx: StepContext, k: Tensor, v: Tensor, layout: SeqLayout) -> None:
        s = self._slice(layout)
        for row, (branch, sample) in enumerate(ctx.row_map()):
            kk, vv = k[row, :, s].detach().clone(), v[row, :, s].detach().clone()
            if self.to_cpu:
                kk, vv = kk.cpu(), vv.cpu()
            self.store_[(ctx.step_index, block, branch, sample)] = (kk, vv)

    def patch(self, block: int, ctx: StepContext, k: Tensor, v: Tensor, layout: SeqLayout, *,
              kv_heads: Optional[Sequence[int]] = None, spans: Optional[Sequence[TokenSpan]] = None,
              cfg_scope: CfgScope = "both") -> tuple[Tensor, Tensor]:
        """Return ``(k', v')`` with cached slices injected for rows in ``cfg_scope``.

        ``spans`` (text-local, ``which="txt"`` only) restrict the patched token positions.
        """
        s = self._slice(layout)
        n = s.stop - s.start
        pos = slice(0, n) if spans is None else union_mask(n, [sp.as_slice() for sp in as_spans(spans)], k.device)
        heads = range(k.shape[1]) if kv_heads is None else list(kv_heads)
        k2, v2 = k.clone(), v.clone()
        rmap = ctx.row_map()
        for row in ctx.rows_for_scope(cfg_scope):
            key = (ctx.step_index, block, *rmap[row])
            if key not in self.store_:
                raise KeyError(f"no cached K/V for {key}")
            ck, cv = self.store_[key]
            if ck.shape[-2] != n:
                raise ValueError(f"cached length {ck.shape[-2]} != target {n}; use length-matched prompts")
            for dst, src in ((k2, ck), (v2, cv)):
                seg, src = dst[row, :, s], src.to(dst.device, dst.dtype)  # seg is a view into dst
                for h in heads:
                    seg[h, pos] = src[h, pos]
        return k2, v2


def kv_group_heads(kv_head: int, group: int) -> range:
    """Q heads that read KV head ``kv_head`` under ``repeat_interleave(group)``."""
    return range(kv_head * group, (kv_head + 1) * group)


# =========================================================================== E3-2 PCI

DenoiseFn = Callable[[Tensor, float, int, Any], Tensor]


def euler_step(x: Tensor, v: Tensor, sigma: float, sigma_next: float) -> Tensor:
    """Deterministic rectified-flow Euler update ``x + (σ_{i+1} - σ_i) v``."""
    return x + (sigma_next - sigma) * v


class PCIController:
    """Prompt-switch rollouts with a cached base trajectory (PCI, ``O(S)`` saving per switch).

    ``denoise(x, sigma, i, cond) -> v`` must be deterministic (``euler`` baseline). Switching
    at step ``k`` runs steps ``i < k`` under ``cond_base`` and ``i >= k`` under ``cond_concept``
    (step ``k`` itself uses the concept). ``k = 0`` is pure concept; ``k = S`` is pure base.
    """

    def __init__(self, denoise: DenoiseFn, sigmas: Sequence[float]) -> None:
        if len(sigmas) < 2:
            raise ValueError("need at least one step")
        self.denoise, self.sigmas = denoise, [float(s) for s in sigmas]
        self.base_states: Optional[list[Tensor]] = None
        self.calls = 0

    @property
    def num_steps(self) -> int:
        return len(self.sigmas) - 1

    def _run(self, x: Tensor, start: int, conds: Callable[[int], Any]) -> Tensor:
        for i in range(start, self.num_steps):
            self.calls += 1
            x = euler_step(x, self.denoise(x, self.sigmas[i], i, conds(i)), self.sigmas[i], self.sigmas[i + 1])
        return x

    def cache_base(self, x_T: Tensor, cond_base: Any) -> list[Tensor]:
        """Run and cache the base trajectory ``[x_0 (=x_T), x_1, ..., x_S]`` (state *before* step ``i``)."""
        states = [x_T]
        for i in range(self.num_steps):
            self.calls += 1
            v = self.denoise(states[-1], self.sigmas[i], i, cond_base)
            states.append(euler_step(states[-1], v, self.sigmas[i], self.sigmas[i + 1]))
        self.base_states = states
        return states

    def switch(self, k: int, cond_concept: Any) -> Tensor:
        """Final ``x`` for ``P_b --k--> P_c`` reusing the cached prefix ``x_k``."""
        if self.base_states is None:
            raise RuntimeError("call cache_base first")
        if not 0 <= k <= self.num_steps:
            raise ValueError(f"switch step {k} outside [0, {self.num_steps}]")
        return self._run(self.base_states[k], k, lambda _: cond_concept)

    def full_rollout(self, x_T: Tensor, k: int, cond_base: Any, cond_concept: Any) -> Tensor:
        """Uncached reference rollout (for the prefix-cache equivalence invariant)."""
        return self._run(x_T, 0, lambda i: cond_base if i < k else cond_concept)


def conditioning_sequence(k: int, num_steps: int) -> list[str]:
    """Expected per-step conditioning labels for a switch at ``k``: ``k × 'base' + (S-k) × 'concept'``."""
    return ["base" if i < k else "concept" for i in range(num_steps)]


def switch_points(num_steps: int, stride: int = 1) -> list[int]:
    """Switch steps ``0, stride, ..., S`` (always includes the ``k = S`` pure-base boundary)."""
    pts = list(range(0, num_steps + 1, stride))
    return pts if pts[-1] == num_steps else pts + [num_steps]


def tau_crossings(tau: Sequence[float], cis: Sequence[float], qs: Sequence[float] = (0.5, 0.7)) -> dict[str, Any]:
    """``τ_q = min{τ : CIS(τ) >= q}`` on ``τ := σ`` sorted ascending (0 = clean, 1 = noise).

    Non-crossing curves return ``NaN`` with status ``"never"`` (CIS < q everywhere) or
    ``"always"`` (CIS >= q already at the smallest τ, i.e. lock-in below the sampled range).
    ``W = τ_0.7 - τ_0.5`` (``NaN`` if either is non-crossing).
    """
    pairs = sorted(zip((float(t) for t in tau), (float(c) for c in cis)))
    out: dict[str, Any] = {}
    for q in qs:
        hits = [t for t, c in pairs if c >= q]
        tag = f"{round(q * 100)}"
        if not hits:
            out[f"tau_{tag}"], out[f"status_{tag}"] = float("nan"), "never"
        elif pairs[0][1] >= q:
            out[f"tau_{tag}"], out[f"status_{tag}"] = float("nan"), "always"
        else:
            out[f"tau_{tag}"], out[f"status_{tag}"] = min(hits), "ok"
    if "tau_50" in out and "tau_70" in out:
        out["W"] = out["tau_70"] - out["tau_50"]
    return out


def reference_normalized_score(s_x: Tensor | float, s_base: Tensor | float, s_concept: Tensor | float,
                               eps: float = 1e-8) -> Tensor:
    """``(s(x) - s(P_b)) / (s(P_c) - s(P_b))``; ``NaN`` where the reference gap is below ``eps``."""
    s_x, s_b, s_c = (torch.as_tensor(t, dtype=torch.float64) for t in (s_x, s_base, s_concept))
    gap = s_c - s_b
    return torch.where(gap.abs() > eps, (s_x - s_b) / gap, torch.full_like(gap, float("nan")))


# =========================================================================== E3-3 TACA


def taca_gamma(sigma: float, gamma0: float = 1.2, sigma_thresh: float = 0.97) -> float:
    """``γ(σ) = γ0`` if ``σ >= σ_thresh`` else ``1``."""
    return gamma0 if sigma >= sigma_thresh else 1.0


def taca_spec(*, keys: Optional[TokenSpan | Sequence[TokenSpan]] = None, gamma0: float = 1.2, sigma_thresh: float = 0.97,
              bias: float = 0.0, cfg_scope: CfgScope = "both", blocks: Optional[Sequence[int]] = None,
              first_n_steps: Optional[int] = None) -> InterventionSpec:
    """TACA as a ``logit_scale`` intervention on ``img → keys`` (default all text keys).

    Normative window ``σ >= sigma_thresh``; ``first_n_steps`` is the separate ``i < n`` ablation.
    ``bias`` adds the sign-independent log-bias ``s' = s + δ`` (Anima post-``k_norm`` steering).
    """
    window = StepWindow(end=first_n_steps) if first_n_steps is not None else StepWindow(sigma_min=sigma_thresh)
    return InterventionSpec("logit_scale", blocks=blocks, window=window, cfg_scope=cfg_scope,
                            params={"gamma": gamma0, "bias": bias, "keys": keys, "query": "img"})


def _sdpa_gqa(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    g = q.shape[1] // k.shape[1]
    return F.scaled_dot_product_attention(q, expand_kv(k, g), expand_kv(v, g))


def taca_listing2_attention(q: Tensor, k: Tensor, v: Tensor, layout: SeqLayout, gamma: float, *,
                            keys: Optional[TokenSpan | Sequence[TokenSpan]] = None,
                            attn_fn: Callable[[Tensor, Tensor, Tensor], Tensor] = _sdpa_gqa) -> Tensor:
    """TACA App. A Listing 2: scale selected ``K_txt`` by ``γ`` (post QK-norm/RoPE), run fused
    attention, then restore text-query rows from an unscaled pass.

    Image rows then equal the ``img × keys`` logit scaling exactly; text rows are unchanged.
    ``q [B,Hq,Nq,d]``, ``k/v [B,Hkv,Nk,d]``.
    """
    if gamma == 1.0:
        return attn_fn(q, k, v)
    parts = [layout.txt_keys(s) for s in as_spans(keys)] if keys is not None else [layout.k_txt]
    kmask = union_mask(layout.n_key, parts, k.device)
    k_scaled = torch.where(kmask[None, None, :, None], k * gamma, k)
    out = attn_fn(q, k_scaled, v)
    if layout.q_txt is not None:
        out = out.clone()
        out[:, :, layout.q_txt] = attn_fn(q[:, :, layout.q_txt], k, v)
    return out


# =========================================================================== E3-4 DiT-ST / SAP


def attention_change(a_prev: Tensor, a_cur: Tensor, theta: float = 1e-8) -> Tensor:
    """DiT-ST Eq. 5: ``||A_i - A_{i-1}||_F / (||A_{i-1}||_F + θ)`` over the last two axes, in fp32."""
    a_prev, a_cur = a_prev.float(), a_cur.float()
    return (a_cur - a_prev).flatten(-2).norm(dim=-1) / (a_prev.flatten(-2).norm(dim=-1) + theta)


class AttentionChangeTracker:
    """Streaming DiT-ST ``Δ̄_i``: keeps only the previous step's map per block (on CPU by default).

    ``add(block, step, A)`` with ``A [B, H, N, L]`` (cond rows, fp32). ``series()`` returns
    ``{i: Δ̄_i}`` averaged over blocks, heads and samples (Eq. 6).
    """

    def __init__(self, store_device: torch.device | str = "cpu", pin_memory: bool = False, theta: float = 1e-8) -> None:
        self.store_device, self.pin, self.theta = torch.device(store_device), pin_memory, theta
        self._prev: dict[int, tuple[int, Tensor]] = {}
        self._sums: dict[int, list[float]] = {}

    def add(self, block: int, step: int, attn: Tensor) -> None:
        cur = attn.detach().float()
        if block in self._prev:
            pstep, prev = self._prev[block]
            if step != pstep + 1:
                raise ValueError(f"block {block}: step {step} does not follow {pstep}")
            d = attention_change(prev.to(cur.device), cur, self.theta).mean()
            self._sums.setdefault(step, []).append(float(d))
        stored = cur.to(self.store_device)
        if self.pin and stored.device.type == "cpu" and torch.cuda.is_available():
            stored = stored.pin_memory()
        self._prev[block] = (step, stored)

    def series(self) -> dict[int, float]:
        return {i: sum(v) / len(v) for i, v in sorted(self._sums.items())}


def moving_average(series: Mapping[int, float], w: int = 3) -> dict[int, float]:
    """Trailing mean ``(1/w) Σ_{k<w} Δ̄_{i-k}`` over available entries."""
    keys = sorted(series)
    return {i: sum(series[j] for j in keys if i - w < j <= i) / sum(1 for j in keys if i - w < j <= i) for i in keys}


def normalize_by_dsigma(series: Mapping[int, float], sigmas: Sequence[float]) -> dict[int, float]:
    """``Δ̂_i / |σ_i - σ_{i-1}|`` (step-size-normalised change; zero steps skipped)."""
    return {i: v / abs(sigmas[i] - sigmas[i - 1]) for i, v in series.items() if i >= 1 and sigmas[i] != sigmas[i - 1]}


def attribute_boundary(delta_hat: Mapping[int, float], tau: float = 1e-4,
                       rel_drop: float = 0.1) -> tuple[Optional[int], str]:
    """``s_attr = min{i : Δ̂_i < τ}``; fallback ``Δ̂_i < rel_drop · max Δ̂`` (after the peak)."""
    keys = sorted(delta_hat)
    for i in keys:
        if delta_hat[i] < tau:
            return i, "tau"
    if keys:
        peak = max(keys, key=delta_hat.__getitem__)
        cap = rel_drop * delta_hat[peak]
        for i in keys:
            if i > peak and delta_hat[i] < cap:
                return i, "relative_drop"
    return None, "none"


def snr(sigmas: Sequence[float], eps_sigma: float = 1e-4) -> list[float]:
    """Rectified-flow ``snr_i = ((1 - σ_i) / σ_i)^2`` with ``σ_i >= eps_sigma``."""
    return [((1 - max(s, eps_sigma)) / max(s, eps_sigma)) ** 2 for s in sigmas]


def discrete_curvature(y: Sequence[float]) -> list[float]:
    """Eq. 8-9 ``κ_i = |Δ²y_i| / (1 + (Δy_i)^2)^{3/2}`` at interior points (unit grid; ends = 0)."""
    n = len(y)
    out = [0.0] * n
    for i in range(1, n - 1):
        d1 = (y[i + 1] - y[i - 1]) / 2
        d2 = y[i + 1] - 2 * y[i] + y[i - 1]
        out[i] = abs(d2) / (1 + d1 * d1) ** 1.5
    return out


def relation_boundary(y: Mapping[int, float], x: Sequence[float], s_attr: int,
                      grid: Literal["uniform", "index"] = "uniform") -> Optional[int]:
    """``s_rel = argmax_i κ_i`` on the prefix ``i < s_attr``.

    ``x[i]`` is the abscissa per step (e.g. :func:`snr` or ``σ``). ``uniform`` resamples ``y`` onto a
    uniformly spaced ``x`` grid (duplicates skipped) so linear ``y(x)`` gives ``κ = 0``; the argmax is
    mapped back to the nearest step.
    """
    steps = [i for i in sorted(y) if i < s_attr]
    pts: list[tuple[float, float, int]] = []
    for i in steps:
        if not pts or x[i] != pts[-1][0]:
            pts.append((float(x[i]), float(y[i]), i))
    if len(pts) < 3:
        return None
    if grid == "index":
        kappa = discrete_curvature([p[1] for p in pts])
        return pts[max(range(len(pts)), key=kappa.__getitem__)][2]
    xs = torch.tensor([p[0] for p in pts], dtype=torch.float64)
    ys = torch.tensor([p[1] for p in pts], dtype=torch.float64)
    if not bool((xs[1:] > xs[:-1]).all() or (xs[1:] < xs[:-1]).all()):
        raise ValueError("x must be strictly monotonic over the prefix")
    if xs[0] > xs[-1]:
        xs, ys = xs.flip(0), ys.flip(0)
        pts = pts[::-1]
    grid_x = torch.linspace(float(xs[0]), float(xs[-1]), len(pts), dtype=torch.float64)
    j = torch.searchsorted(xs, grid_x).clamp(1, len(pts) - 1)
    w = (grid_x - xs[j - 1]) / (xs[j] - xs[j - 1])
    grid_y = ys[j - 1] + w * (ys[j] - ys[j - 1])
    kappa = discrete_curvature(grid_y.tolist())
    gx = float(grid_x[max(range(len(kappa)), key=kappa.__getitem__)])
    return min(pts, key=lambda p: abs(p[0] - gx))[2]


@dataclass(frozen=True, eq=False)
class CondBundle:
    """Complete conditioning for one SAP stage: never mix embeddings with another prompt's length/RoPE."""

    text_emb: Tensor
    token_len: int
    rope_ids: Optional[Any] = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class StagePromptSchedule:
    """SAP deterministic stage schedule over the *cond* branch (``uncond`` stays fixed).

    Give either ``switch_steps`` (strictly increasing, in ``(0, S)``) or ``switch_sigmas``
    (strictly decreasing, in ``(0, 1)``). Stage ``s`` is active once ``i >= switch_steps[s-1]`` or
    ``σ_i <= switch_sigmas[s-1]``.
    """

    bundles: Sequence[CondBundle]
    switch_steps: Optional[Sequence[int]] = None
    switch_sigmas: Optional[Sequence[float]] = None

    def __post_init__(self) -> None:
        if not self.bundles:
            raise ValueError("empty prompt list")
        if self.switch_steps is not None and self.switch_sigmas is not None:
            raise ValueError("use switch_steps or switch_sigmas, not both")
        pts = self.switch_steps if self.switch_steps is not None else self.switch_sigmas
        pts = list(pts or [])
        if len(pts) != len(self.bundles) - 1:
            raise ValueError(f"{len(self.bundles)} stages need {len(self.bundles) - 1} switch points")
        if self.switch_steps is not None:
            if any(b <= a for a, b in zip(pts, pts[1:])) or any(p <= 0 for p in pts):
                raise ValueError("switch_steps must be positive and strictly increasing")
        elif any(b >= a for a, b in zip(pts, pts[1:])) or any(not 0 < p < 1 for p in pts):
            raise ValueError("switch_sigmas must lie in (0, 1) and strictly decrease")

    def validate_steps(self, num_steps: int) -> None:
        if self.switch_steps is not None and any(p >= num_steps for p in self.switch_steps):
            raise ValueError(f"switch step out of range for S={num_steps}")

    def stage(self, step_index: int, sigma: float) -> int:
        if self.switch_steps is not None:
            return sum(1 for p in self.switch_steps if step_index >= p)
        return sum(1 for p in (self.switch_sigmas or []) if sigma <= p)

    def bundle_at(self, ctx: StepContext) -> CondBundle:
        self.validate_steps(ctx.num_steps)
        return self.bundles[self.stage(ctx.step_index, ctx.sigma)]

    def sequence(self, sigmas: Sequence[float]) -> list[int]:
        """Stage index for each of the ``S = len(sigmas) - 1`` updates."""
        self.validate_steps(len(sigmas) - 1)
        return [self.stage(i, sigmas[i]) for i in range(len(sigmas) - 1)]


# =========================================================================== offline cond cache


class CondCache:
    """Hash-keyed on-disk cache of pre-encoded conditioning (BF16 text-encoder outputs).

    Key = ``sha256(json({encoder_id, prompt, **meta}))``. Floating tensors are stored in
    ``dtype`` on CPU; writes are atomic. ``manifest_hash`` gives ``te_cache_sha256``.
    """

    def __init__(self, root: str | os.PathLike, encoder_id: str, dtype: torch.dtype = torch.bfloat16) -> None:
        self.root, self.encoder_id, self.dtype = Path(root), encoder_id, dtype

    def key(self, prompt: str, **meta: Any) -> str:
        payload = json.dumps({"encoder_id": self.encoder_id, "prompt": prompt, **meta}, sort_keys=True,
                             ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def path(self, prompt: str, **meta: Any) -> Path:
        return self.root / f"{self.key(prompt, **meta)}.pt"

    def get_or_encode(self, prompt: str, encode: Callable[[str], Mapping[str, Any]], **meta: Any) -> dict[str, Any]:
        path = self.path(prompt, **meta)
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=True)
        out = {k: (v.detach().to("cpu", self.dtype if v.is_floating_point() else v.dtype)
                   if isinstance(v, Tensor) else v) for k, v in encode(prompt).items()}
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(out, tmp)
        os.replace(tmp, path)
        return out

    def manifest_hash(self, prompts: Sequence[str], **meta: Any) -> str:
        h = hashlib.sha256()
        for p in prompts:
            h.update(self.key(p, **meta).encode())
        return h.hexdigest()
