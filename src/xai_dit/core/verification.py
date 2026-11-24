"""Direction 04 verification & mechanistic interpretability (``docs/specs/04_verification_mi.md``).

* E4-1 AttentionSinksDiT: two-pass dynamic sink masking report with ``residual_ratio``
  re-read at the *pre-intervention* indices; static logit-bias / prob-scale / value edits;
  ``xlogy`` entropy.
* E4-2 TemplateRegisters: ``S/R/P`` incoming-mass decomposition, normalised I2T share,
  ``R → I`` / ``R → S`` pre-softmax masking specs + isolation report, pre-DiT span swap.
* E4-3 GatedAttention (Krea 2 G1 gate): ``gate_logits`` vs ``gate_value`` separation,
  post-sigmoid ``gate_override``, sparsity / M-Act / mean-gate, dose-response fit.
* E4-4 DiTRelationCircuits: Attention Map Synopsis (cond/uncond separate), token-specific
  head ablation spec, overall vs object-conditional relation accuracy.
* E4-5 DiffusionProbe / ProbeSelect: ``residual_out`` tap contract (explicit cond rows and
  grid metadata) and the text-only-softmax ``img_to_txt`` tap.

Pure PyTorch: no ``comfy.*`` imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping, Optional, Sequence

import torch

from ..adapters.base import (
    AdapterCapability,
    CfgScope,
    InterventionSpec,
    SeqLayout,
    StepContext,
    StepWindow,
    TokenMap,
    TokenSpan,
    as_spans,
)
from .layer_selection import spearman_rho
from .session import (
    SINK_MASK_VALUE,
    AttentionCall,
    IncomingMass,
    LogitEdit,
    ProbObserver,
    Region,
    RunningMap,
    SinkSelection,
    chunked_attention,
    dynamic_sink_edit,
    grouped_scores,
    index_mask,
    select_dynamic_sinks,
    union_mask,
)

Tensor = torch.Tensor


class StructurallyNotApplicable(ValueError):
    """The requested intervention cannot exist on this architecture (e.g. Anima ``R → I``)."""


class _Collect(ProbObserver):
    def __init__(self, shape: tuple[int, ...], device: torch.device) -> None:
        self.probs = torch.zeros(shape, dtype=torch.float32, device=device)

    def __call__(self, probs: Tensor, q_sl: slice, k_sl: slice) -> None:
        self.probs[:, :, q_sl, k_sl] = probs.float()


def materialize_probs(q: Tensor, k: Tensor, **kwargs) -> Tensor:
    """Full ``[B, Hq, Nq, Nk]`` probabilities through the chunked engine (tests / small inputs only)."""
    obs = _Collect((q.shape[0], q.shape[1], q.shape[2], k.shape[2]), q.device)
    chunked_attention(q, k, None, observers=[obs], **kwargs)
    return obs.probs


def entropy(p: Tensor, dim: int = -1) -> Tensor:
    """Shannon entropy with ``xlogy`` (exact 0 for one-hot rows; no clamp drift)."""
    return -torch.special.xlogy(p, p).sum(dim)


# =========================================================================== E4-1 dynamic sinks


def sink_masking_report(q: Tensor, k: Tensor, layout: SeqLayout, top_k: int, *,
                        selection: Literal["per_head", "union"] = "per_head",
                        groups: Sequence[TokenSpan] = (), random: bool = False,
                        generator: Optional[torch.Generator] = None, value: float = SINK_MASK_VALUE,
                        scale: Optional[float] = None, query_chunk: int = 1024,
                        key_chunk: Optional[int] = None) -> dict[str, Tensor | SinkSelection]:
    """Two-pass chunked dynamic sink masking with pre/post mass at the *pre-selected* indices.

    Returns ``mass_before`` / ``mass_after`` ``[B, Hq]`` (summed over ``S_k``), ``residual_ratio``
    (``after / before``; fixes the repo's ``top1_incoming_mass_after`` re-selection bug) and
    the :class:`SinkSelection`.
    """
    dev = q.device
    B, Hq, Nq = q.shape[:3]
    Nk = k.shape[2]
    q_img = index_mask(Nq, layout.q_img, dev)
    kw = dict(scale=scale, query_chunk=query_chunk, key_chunk=key_chunk)
    before = IncomingMass(q_img, (B, Hq, Nk), dev)
    chunked_attention(q, k, None, observers=[before], **kw)
    sel = select_dynamic_sinks(before.mass, top_k, selection=selection,
                               groups=[layout.txt_keys(s) for s in groups], random=random, generator=generator)
    after = IncomingMass(q_img, (B, Hq, Nk), dev)
    chunked_attention(q, k, None, logit_edits=[dynamic_sink_edit(sel, q_img, value)], observers=[after], **kw)
    mb = (before.mass * sel.mask).sum(-1)
    ma = (after.mass * sel.mask).sum(-1)
    return {"mass_before": mb, "mass_after": ma, "residual_ratio": ma / mb.clamp_min(torch.finfo(mb.dtype).tiny),
            "selection": sel}


def static_sink_logit_bias(sink_keys: Tensor, eta: float, query: Optional[Tensor] = None) -> LogitEdit:
    """Paper App. A.5.1: add ``log η`` to sink-key logits (``mode="logit_bias"``)."""
    if eta <= 0:
        raise ValueError("eta must be > 0 for logit_bias")
    return LogitEdit(Region(query=query, key=sink_keys), "add", math.log(eta))


def expected_logit_bias_mass(sink_mass: Tensor, eta: float) -> Tensor:
    """Analytical sink mass after ``log η`` bias: ``η S / (1 - (1 - η) S)``."""
    return eta * sink_mass / (1 - (1 - eta) * sink_mass)


def prob_scale(probs: Tensor, sink_keys: Tensor, eta: float) -> Tensor:
    """Repo ``_apply_prob_scale``: sink mass → ``η Σ A_s``, remaining keys renormalised to keep row sum 1.

    Requires strictly positive non-sink mass in every row.
    """
    s = (probs * sink_keys).sum(-1, keepdim=True)
    rest = 1 - s
    if bool((rest <= 0).any()):
        raise ValueError("prob_scale needs positive non-sink mass in every row")
    factor = (1 - eta * s) / rest
    return torch.where(sink_keys, probs * eta, probs * factor)


def replace_sink_values(v: Tensor, sink_keys: Tensor, mode: Literal["zero", "mean", "noise", "lerp"] = "zero", *,
                        alpha: float = 0.5, generator: Optional[torch.Generator] = None) -> Tensor:
    """Static value intervention on ``v [B, H, N, d]`` at ``sink_keys [N]`` (``mean`` over all joint tokens).

    Under GQA any V edit affects every Q head of the KV group.
    """
    out = v.clone()
    mean = v.mean(2, keepdim=True).expand_as(v)
    if mode == "zero":
        repl = torch.zeros_like(v)
    elif mode == "mean":
        repl = mean
    elif mode == "noise":
        repl = torch.randn(v.shape, generator=generator, dtype=v.dtype, device="cpu").to(v.device) * v.std()
    elif mode == "lerp":
        repl = torch.lerp(v, mean, alpha)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    out[:, :, sink_keys] = repl[:, :, sink_keys]
    return out


def sink_mask_spec(top_k: int, *, blocks: Sequence[int], selection: Literal["per_head", "union"] = "per_head",
                   groups: Sequence[TokenSpan] = (), random: bool = False, cfg_scope: CfgScope = "both",
                   window: StepWindow = StepWindow()) -> InterventionSpec:
    """E4-1 ``two_pass_sink_mask`` spec (Anima: pass ``groups=[P]`` so tied pad logits act as one unit)."""
    return InterventionSpec("two_pass_sink_mask", blocks=blocks, window=window, cfg_scope=cfg_scope,
                            params={"top_k": top_k, "selection": selection, "groups": tuple(groups),
                                    "random": random})


# =========================================================================== E4-2 template registers


def span_masses(mass: Tensor, layout: SeqLayout, token_map: TokenMap) -> dict[str, Tensor]:
    """Decompose incoming mass ``m_j`` (``[..., Nk]``, mean over image queries of the full softmax) into
    text spans: ``{span_name: Σ_{j∈span} m_j}`` plus ``txt_total``. Anima: ``S + R + P == 1``."""
    out = {name: mass[..., layout.txt_keys(span)].sum(-1) for name, span in token_map.spans.items()}
    out["txt_total"] = mass[..., layout.k_txt].sum(-1)
    return out


def normalized_register_share(masses: Mapping[str, Tensor],
                              register: Sequence[str] = ("R_prefix", "R"), semantic: str = "S") -> Tensor:
    """``m_R / (m_R + m_S)`` — the only share comparable across cross- vs joint-attention."""
    m_r = sum(masses[n] for n in register if n in masses)
    return m_r / (m_r + masses[semantic])


class SpanMassTracker:
    """Streaming per-block ``S/R/P`` incoming mass for one branch (``[B_branch, Hq]`` per span)."""

    def __init__(self, token_map: TokenMap, branch: Literal["cond", "uncond"] = "cond") -> None:
        self.token_map, self.branch = token_map, branch
        self.maps: dict[tuple[str, int], RunningMap] = {}

    def factory(self, call: AttentionCall) -> Optional[ProbObserver]:
        rows = call.ctx.rows(self.branch)
        if not rows:
            return None
        tracker, layout = self, call.layout
        B, Hq, Nq, Nk = call.q.shape[0], call.q.shape[1], call.q.shape[2], call.k.shape[2]

        class _Obs(IncomingMass):
            def finish(self) -> None:
                m = self.mass.index_select(0, torch.tensor(list(rows), device=self.mass.device))
                for name, val in span_masses(m, layout, tracker.token_map).items():
                    tracker.maps.setdefault((name, call.block), RunningMap("mean")).add(val)

        return _Obs(index_mask(Nq, layout.q_img, call.device), (B, Hq, Nk), call.device)


def register_mask_spec(capability: AdapterCapability, token_map: TokenMap,
                       key_target: Literal["img", "S"] = "img", *, blocks: Sequence[int],
                       register: Sequence[str] = ("R_prefix", "R"), cfg_scope: CfgScope = "both") -> InterventionSpec:
    """E4-2 Sec 4.4: mask register queries ``R`` → image keys (``R→I``) or semantic keys (``R→S``).

    Raises :class:`StructurallyNotApplicable` on cross-attention models (Anima ``ctx`` is never updated).
    """
    if capability.attention_kind == "cross_attn":
        raise StructurallyNotApplicable(f"{capability.name}: register queries do not exist (R→I is N/A)")
    q_spans = tuple(token_map.spans[n] for n in register if n in token_map.spans)
    target = "img" if key_target == "img" else token_map.spans["S"]
    return InterventionSpec("span_query_mask", blocks=blocks, cfg_scope=cfg_scope,
                            params={"query_span": q_spans, "key_target": target})


def masking_isolation_report(q: Tensor, k: Tensor, layout: SeqLayout, query_spans: Sequence[TokenSpan],
                             key_target: Literal["img"] | Sequence[TokenSpan] = "img",
                             scale: Optional[float] = None) -> dict[str, float]:
    """Check the E4-2 masking invariants on small inputs.

    Returns ``masked_prob_max`` (``< 1e-6``), ``logit_outside_max_abs_diff`` (``== 0``),
    ``register_row_sum_err`` (``≈ 0``), ``other_rows_max_abs_diff`` (``< 1e-6``).
    """
    dev = q.device
    Nq, Nk = q.shape[2], k.shape[2]
    qm = union_mask(Nq, [layout.txt_queries(s) for s in as_spans(query_spans)], dev)
    km = (index_mask(Nk, layout.k_img, dev) if key_target == "img"
          else union_mask(Nk, [layout.txt_keys(s) for s in as_spans(key_target)], dev))
    edit = LogitEdit(Region(query=qm, key=km), "set", float("-inf"))
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    s = grouped_scores(q.float(), k.float(), scale)
    s2 = edit(s, slice(0, Nq), slice(0, Nk))
    region = qm[:, None] & km[None, :]
    base = materialize_probs(q, k, scale=scale)
    masked = materialize_probs(q, k, scale=scale, logit_edits=[edit])
    return {
        "masked_prob_max": float(masked[:, :, region].max()) if bool(region.any()) else 0.0,
        "logit_outside_max_abs_diff": float((s2 - s)[:, :, ~region].abs().max()),
        "register_row_sum_err": float((masked[:, :, qm].sum(-1) - 1).abs().max()),
        "other_rows_max_abs_diff": float((masked[:, :, ~qm] - base[:, :, ~qm]).abs().max()),
    }


def pre_dit_span_swap(states: Tensor, span: TokenSpan, source: Tensor, mode: Literal["swap", "mean"] = "swap") -> Tensor:
    """Replace ``states[:, span]`` (``[B, T, D]``) by the donor's span (``swap``) or by the mean over donors
    (``mean``; ``source [P, T, D]``). Apply at ``te.out`` / ``refiner.out`` (MOSS) or ``text_in`` / ``txt_embed`` (Krea 2)."""
    out = states.clone()
    src = source[..., span.start:span.end, :]
    out[:, span.start:span.end] = (src.mean(0) if mode == "mean" else src).to(out.dtype)
    return out


# =========================================================================== E4-3 output gate


def gate_value(gate_logits: Tensor) -> Tensor:
    """``sigmoid(gate_logits)``; ``gate_logits`` is the raw pre-sigmoid ``attn.gate(x)`` output."""
    return torch.sigmoid(gate_logits)


def apply_gate_override(ctx: Tensor, gate_logits: Tensor, mode: Literal["none", "one", "mean", "const", "scale"],
                        value: Optional[float | Tensor] = None) -> Tensor:
    """Gated context ``ctx ⊙ g'`` where ``g'`` always acts on the post-sigmoid ``gate_value``.

    ``one``: ``g' = 1``; ``mean``: ``g' = value`` (calibrated mean gate, broadcastable);
    ``const``: ``g' = value``; ``scale`` (dose-response): ``g' = lerp(g, 1, value)`` with
    ``value ∈ [0, 1]`` (0 = normal gate, 1 = fully open); ``none``: normal gate.
    """
    g = gate_value(gate_logits.float())
    if mode == "none":
        g2 = g
    elif mode == "one":
        g2 = torch.ones_like(g)
    elif mode in ("mean", "const"):
        if value is None:
            raise ValueError(f"gate_override mode={mode!r} needs value")
        g2 = torch.as_tensor(value, dtype=g.dtype, device=g.device).expand_as(g)
    elif mode == "scale":
        g2 = torch.lerp(g, torch.ones_like(g), float(value if value is not None else 1.0))
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return (ctx.float() * g2).to(ctx.dtype)


def gate_stats(gate_values: Tensor, *, quantiles: Sequence[float] = (0.1, 0.5, 0.9),
               below: float = 0.1) -> dict[str, float]:
    """Mean / quantiles / fraction below ``below`` of ``gate_value`` (all in ``[0, 1]``)."""
    g = gate_values.float().flatten()
    if g.numel() > 16_000_000:
        g = g[torch.randperm(g.numel(), device=g.device)[:16_000_000]]
    qs = torch.quantile(g, torch.tensor(list(quantiles), device=g.device))
    out = {"mean": float(g.mean()), f"frac_below_{below:g}": float((g < below).float().mean()),
           "min": float(g.min()), "max": float(g.max())}
    out.update({f"q{round(q * 100)}": float(v) for q, v in zip(quantiles, qs)})
    return out


def massive_activation(hidden: Tensor) -> Tensor:
    """M-Act: max ``|h|`` per sample of a block output ``[B, T, D] -> [B]``."""
    return hidden.detach().abs().flatten(1).amax(1).float()


class MeanGate:
    """Running mean of ``gate_value`` over tokens/steps per block (channel-wise ``[Hq*d]``) for ``mode="mean"``."""

    def __init__(self) -> None:
        self.maps: dict[int, RunningMap] = {}

    def add(self, block: int, gate_logits: Tensor, token_slice: slice = slice(None)) -> None:
        g = gate_value(gate_logits.float())[:, token_slice].flatten(0, 1)
        self.maps.setdefault(block, RunningMap("mean")).add(g.mean(0), weight=float(g.shape[0]))

    def value(self, block: int) -> Tensor:
        return self.maps[block].value()


def dose_response_fit(doses: Sequence[float], responses: Sequence[float]) -> dict[str, float]:
    """Least-squares slope/intercept and Spearman ρ of response vs dose."""
    x = torch.tensor(doses, dtype=torch.float64)
    y = torch.tensor(responses, dtype=torch.float64)
    xc = x - x.mean()
    slope = float((xc * (y - y.mean())).sum() / (xc ** 2).sum())
    return {"slope": slope, "intercept": float(y.mean() - slope * x.mean()), "spearman": spearman_rho(x, y)}


# =========================================================================== E4-4 relation circuits


def attention_synopsis(img_to_txt: Tensor, region_masks: Tensor) -> Tensor:
    """Synopsis inner product ``Σ_n mask_r[n] A[h, n, l]``: ``[B, H, N, L] × [B, R, N] -> [B, H, R, L]``.

    Feed cond and uncond rows separately (never mixed) and sum over timesteps outside.
    """
    return torch.einsum("bhnl,brn->bhrl", img_to_txt.float(), region_masks.float())


def synopsis_by_branch(img_to_txt: Tensor, region_masks: Tensor, ctx: StepContext) -> dict[str, Tensor]:
    """Synopsis for ``cond`` and ``uncond`` rows of a CFG batch, kept separate."""
    out = {}
    for branch in ("cond", "uncond"):
        rows = ctx.rows(branch)
        if rows:
            idx = torch.tensor(rows, device=img_to_txt.device)
            out[branch] = attention_synopsis(img_to_txt.index_select(0, idx), region_masks.index_select(0, idx))
    return out


def token_head_ablation_spec(heads: Sequence[int], token: TokenSpan | Sequence[TokenSpan], *, blocks: Sequence[int],
                             cfg_scope: CfgScope = "both", window: StepWindow = StepWindow()) -> InterventionSpec:
    """Remove only ``A_{h,j} V_{h_kv,j}`` for target tokens inside the given Q heads (post-softmax)."""
    return InterventionSpec("per_head_token_ablation", blocks=blocks, window=window, cfg_scope=cfg_scope,
                            params={"heads": tuple(heads), "keys": token, "query": "img"})


def relation_accuracy(obj1_detected: Sequence[bool], obj2_detected: Sequence[bool],
                      relation_correct: Sequence[bool]) -> dict[str, float]:
    """Overall success (relation correct with both objects) and object-detected conditional accuracy."""
    rows = list(zip(obj1_detected, obj2_detected, relation_correct))
    if not rows:
        raise ValueError("no samples")
    both = [r for r in rows if r[0] and r[1]]
    n = len(rows)
    return {"overall": sum(1 for a, b, c in rows if a and b and c) / n,
            "conditional": (sum(1 for *_, c in both if c) / len(both)) if both else float("nan"),
            "object_accuracy": len(both) / n, "n": float(n), "n_both_detected": float(len(both))}


# =========================================================================== E4-5 taps


@dataclass(frozen=True)
class ResidualTapMeta:
    block: int
    step_index: int
    sigma: float
    branch: str
    sample_id: int
    img_token_range: tuple[int, int]
    grid_h: int
    grid_w: int


def early_step_index(num_steps: int, frac: float = 0.2) -> int:
    """``i = round(frac * S)`` counted from the noise end."""
    return round(frac * num_steps)


def residual_tap(hidden: Tensor, layout: SeqLayout, ctx: StepContext, block: int, grid: tuple[int, int],
                 branch: Literal["cond", "uncond"] = "cond") -> tuple[Tensor, list[ResidualTapMeta]]:
    """Read-only ``residual_out`` tap: image tokens of ``branch`` rows reshaped to ``[b, D, h, w]``.

    Rows come from ``cond_or_uncond`` metadata (never ``[0::2]``), with explicit per-row metadata.
    """
    h, w = grid
    if h * w != layout.n_img:
        raise ValueError(f"grid {grid} does not match n_img={layout.n_img}")
    rows = ctx.rows(branch)
    idx = torch.tensor(rows, device=hidden.device)
    feats = hidden.detach().index_select(0, idx)[:, layout.q_img].transpose(1, 2).reshape(len(rows), -1, h, w)
    rmap = ctx.row_map()
    metas = [ResidualTapMeta(block, ctx.step_index, ctx.sigma, branch, rmap[r][1],
                             (layout.q_img.start, layout.q_img.stop), h, w) for r in rows]
    return feats.clone(), metas


def img_to_txt_text_softmax(q: Tensor, k: Tensor, layout: SeqLayout, scale: Optional[float] = None) -> Tensor:
    """DiffusionProbe-style tap: softmax over *text keys only* for image queries (rows sum to 1).

    Distinct from the joint-softmax ``img_to_txt`` slice (rows sum to the text share < 1).
    """
    return materialize_probs(q[:, :, layout.q_img], k[:, :, layout.k_txt], scale=scale)
