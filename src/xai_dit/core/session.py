"""Streaming attention engine, running accumulators and the hook-policy session.

Evolves ``Anima_HEATMAP/anima_heatmap/compute.py`` (query-chunked softmax with OOM
key-chunked ``logsumexp`` fallback) and ``aggregate.py`` (``RunningMap``) into a
model-agnostic engine that also *produces the attention output* so interventions are
fed back into the model:

* GQA without materialising ``repeat_interleave``: Q head ``h`` reads KV head ``h // g``.
* Any sequence order (``[img|txt]``, ``[txt|img]``, cross) via :class:`SeqLayout` slices.
* Pre-softmax :class:`LogitEdit` (mask / add / scale), post-softmax :class:`ProbZero`
  (per-head token ablation ``A_hj V_{h_kv j}`` subtraction; never touches sibling Q heads),
  full-row :data:`ProbOverride` (e.g. :class:`TextAxisBlur`), and streaming
  :class:`ProbObserver` taps.
* Dynamic two-pass sink masking (pass 1 incoming mass over ``Q_img`` rows, pass 2 ``-1e4``).

:class:`XAISession` hands off to the model's ``native`` attention *unchanged* when no
capture, observer or intervention touches the current ``(block, step)`` (exact no-op).
Pure PyTorch: no ``comfy.*`` imports.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Literal, Mapping, Optional, Protocol, Sequence, Union

import torch
import torch.nn.functional as F

from ..adapters.base import (
    AdapterCapability,
    CapturedTensors,
    CaptureSpec,
    InterventionSpec,
    SeqLayout,
    StepContext,
    StepWindow,
    TokenSpan,
    as_spans,
)
from .profiling import measure, record_count

Tensor = torch.Tensor
Index = Union[slice, Sequence[int], Tensor, None]

_OOM = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)
NEG_INF = float("-inf")
SINK_MASK_VALUE = -1e4

# --------------------------------------------------------------------------- masks / edits


def index_mask(n: int, index: Index, device: Optional[torch.device] = None) -> Optional[Tensor]:
    """Boolean ``[n]`` mask from a slice / int list / bool tensor; ``None`` means "all"."""
    if index is None:
        return None
    if isinstance(index, Tensor) and index.dtype == torch.bool:
        if index.shape != (n,):
            raise ValueError(f"bool mask shape {tuple(index.shape)} != ({n},)")
        return index.to(device)
    m = torch.zeros(n, dtype=torch.bool, device=device)
    m[index.to(device) if isinstance(index, Tensor) else (index if isinstance(index, slice) else list(index))] = True
    return m


def slice_overlap(a: slice, b: slice) -> Optional[tuple[slice, slice]]:
    """Intersection of absolute slices ``a`` and ``b`` as ``(local-in-a, local-in-b)``; ``None`` if empty."""
    lo, hi = max(a.start, b.start), min(a.stop, b.stop)
    if lo >= hi:
        return None
    return slice(lo - a.start, hi - a.start), slice(lo - b.start, hi - b.start)


def union_mask(n: int, parts: Sequence[Index], device: Optional[torch.device] = None) -> Tensor:
    m = torch.zeros(n, dtype=torch.bool, device=device)
    for p in parts:
        pm = index_mask(n, p, device)
        m |= torch.ones_like(m) if pm is None else pm
    return m


@dataclass(frozen=True, eq=False)
class Region:
    """Boolean region of the ``[B, Hq, Nq, Nk]`` score tensor; ``None`` axes mean "all".

    ``key`` may be ``[Nk]`` or per-(sample, head) ``[B, Hq, Nk]`` (dynamic sinks).
    """

    batch: Optional[Tensor] = None
    heads: Optional[Tensor] = None
    query: Optional[Tensor] = None
    key: Optional[Tensor] = None

    def to(self, device: torch.device) -> Region:
        mv = lambda t: None if t is None else t.to(device)  # noqa: E731
        return Region(mv(self.batch), mv(self.heads), mv(self.query), mv(self.key))

    def chunk_mask(self, q_sl: slice, k_sl: slice) -> Optional[Tensor]:
        parts = []
        if self.batch is not None:
            parts.append(self.batch[:, None, None, None])
        if self.heads is not None:
            parts.append(self.heads[None, :, None, None])
        if self.query is not None:
            parts.append(self.query[q_sl][None, None, :, None])
        if self.key is not None:
            kk = self.key[..., k_sl]
            parts.append(kk[None, None, None, :] if kk.ndim == 1 else kk[:, :, None, :])
        if not parts:
            return None
        m = parts[0]
        for p in parts[1:]:
            m = m & p
        return m


@dataclass(frozen=True, eq=False)
class LogitEdit:
    """Pre-softmax edit on ``region``: ``set`` (e.g. ``-inf`` knockout), ``add`` (log-bias, sink ``-1e4``), ``scale`` (TACA γ)."""

    region: Region
    op: Literal["set", "add", "scale"]
    value: float

    def __call__(self, logits: Tensor, q_sl: slice, k_sl: slice) -> Tensor:
        m = self.region.chunk_mask(q_sl, k_sl)
        if self.op == "set":
            return torch.full_like(logits, self.value) if m is None else logits.masked_fill(m, self.value)
        new = logits + self.value if self.op == "add" else logits * self.value
        return new if m is None else torch.where(m, new, logits)


@dataclass(frozen=True, eq=False)
class ProbZero:
    """Post-softmax zeroing of ``A[b, h, i, j]`` in ``region`` (no renormalisation).

    Since the output is ``Σ_j A_hij V_{h_kv j}``, zeroing ``A_hij`` subtracts exactly the term
    ``A_hj V_{h_kv j}`` for Q head ``h`` only; GQA sibling heads keep their own ``A`` and are
    bit-identical (E4-4 per-head token ablation).
    """

    region: Region

    def __call__(self, probs: Tensor, q_sl: slice, k_sl: slice) -> Tensor:
        m = self.region.chunk_mask(q_sl, k_sl)
        return torch.zeros_like(probs) if m is None else probs.masked_fill(m, 0.0)


class ProbOverride(Protocol):
    """Full-key-row post-softmax replacement ``(probs[B,Hq,Nc,Nk], q_sl) -> probs``."""

    def __call__(self, probs: Tensor, q_sl: slice) -> Tensor: ...


class ProbObserver:
    """Streaming tap: receives post-edit probabilities chunk by chunk (each ``(q, k)`` cell once).

    ``finish`` is called once after the attention call completes.
    """

    def __call__(self, probs: Tensor, q_sl: slice, k_sl: slice) -> None:
        raise NotImplementedError

    def finish(self) -> None:
        pass


# --------------------------------------------------------------------------- GQA math


def expand_kv(x: Tensor, group: int) -> Tensor:
    """Reference GQA expansion ``[B, Hkv, N, d] -> [B, Hkv*g, N, d]`` (``h_kv = h_q // g``)."""
    return x.repeat_interleave(group, dim=1)


def grouped_scores(q: Tensor, k: Tensor, scale: float) -> Tensor:
    B, Hq, Nc, d = q.shape
    Hkv = k.shape[1]
    s = torch.matmul(q.reshape(B, Hkv, Hq // Hkv, Nc, d), k.unsqueeze(2).transpose(-1, -2))
    return s.reshape(B, Hq, Nc, k.shape[2]) * scale


def grouped_pv(p: Tensor, v: Tensor) -> Tensor:
    B, Hq, Nc, Kc = p.shape
    Hkv = v.shape[1]
    o = torch.matmul(p.reshape(B, Hkv, Hq // Hkv, Nc, Kc), v.unsqueeze(2))
    return o.reshape(B, Hq, Nc, v.shape[-1])


# --------------------------------------------------------------------------- engine


def chunked_attention(
    q: Tensor,
    k: Tensor,
    v: Optional[Tensor] = None,
    *,
    scale: Optional[float] = None,
    attn_mask: Optional[Tensor] = None,
    logit_edits: Sequence[Callable[[Tensor, slice, slice], Tensor]] = (),
    prob_edits: Sequence[Callable[[Tensor, slice, slice], Tensor]] = (),
    prob_override: Optional[ProbOverride] = None,
    observers: Sequence[ProbObserver] = (),
    query_chunk: int = 1024,
    key_chunk: Optional[int] = None,
    oom_key_chunk: int = 1024,
    compute_dtype: torch.dtype = torch.float32,
) -> Optional[Tensor]:
    """Manual SDPA over query chunks, keeping *all* keys in each softmax denominator.

    Shapes: ``q [B, Hq, Nq, d]``, ``k/v [B, Hkv, Nk, d]`` with ``Hq % Hkv == 0``.
    ``attn_mask`` follows SDPA semantics (bool ``True`` = keep, or additive), broadcastable
    to ``[B, Hq, Nq, Nk]``. Returns ``[B, Hq, Nq, dv]`` in ``v.dtype`` (``None`` if ``v`` is None).

    ``key_chunk=None`` computes full key rows per query chunk and, on OOM, switches to a
    two-pass key-chunked ``logsumexp`` (exactly equivalent). ``prob_override`` needs full
    rows, so under OOM it halves the query chunk instead.
    """
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q/k must be [batch, heads, tokens, head_dim]")
    B, Hq, Nq, d = q.shape
    Hkv, Nk = k.shape[1], k.shape[2]
    if k.shape[0] != B or k.shape[-1] != d or Hq % Hkv:
        raise ValueError(f"incompatible q {tuple(q.shape)} / k {tuple(k.shape)}")
    if v is not None and tuple(v.shape[:3]) != tuple(k.shape[:3]):
        raise ValueError(f"v {tuple(v.shape)} does not match k {tuple(k.shape)}")
    if prob_override is not None and key_chunk is not None:
        raise ValueError("prob_override requires full key rows (key_chunk=None)")
    if query_chunk <= 0 or (key_chunk is not None and key_chunk <= 0) or Nk == 0:
        raise ValueError("chunk sizes and key length must be positive")

    scale = d ** -0.5 if scale is None else scale
    kc_ = k.to(compute_dtype)
    vc_ = None if v is None else v.to(compute_dtype)
    mask = None if attn_mask is None else torch.broadcast_to(attn_mask.to(q.device), (B, Hq, Nq, Nk))
    out = None if v is None else torch.empty(B, Hq, Nq, v.shape[-1], dtype=v.dtype, device=q.device)
    full = slice(0, Nk)
    observed = False

    def logits_for(qc: Tensor, q_sl: slice, k_sl: slice) -> Tensor:
        s = grouped_scores(qc, kc_[:, :, k_sl], scale)
        if mask is not None:
            m = mask[:, :, q_sl, k_sl]
            s = s.masked_fill(~m, NEG_INF) if m.dtype == torch.bool else s + m.to(s.dtype)
        for e in logit_edits:
            s = e(s, q_sl, k_sl)
        return s

    def post(p: Tensor, q_sl: slice, k_sl: slice) -> Tensor:
        nonlocal observed
        for e in prob_edits:
            p = e(p, q_sl, k_sl)
        for ob in observers:
            observed = True
            ob(p, q_sl, k_sl)
        return p

    def full_rows(qc: Tensor, q_sl: slice) -> Optional[Tensor]:
        s = logits_for(qc, q_sl, full)
        blocked = torch.isneginf(s).all(-1, keepdim=True)
        p = torch.softmax(s, dim=-1).masked_fill(blocked, 0.0)
        del s
        if prob_override is not None:
            p = prob_override(p, q_sl)
        p = post(p, q_sl, full)
        return None if vc_ is None else grouped_pv(p, vc_)

    def key_chunked(qc: Tensor, q_sl: slice, kc: int) -> Optional[Tensor]:
        k_slices = [slice(o, min(o + kc, Nk)) for o in range(0, Nk, kc)]
        lse = torch.full((B, Hq, qc.shape[2]), NEG_INF, dtype=compute_dtype, device=q.device)
        for k_sl in k_slices:
            lse = torch.logaddexp(lse, torch.logsumexp(logits_for(qc, q_sl, k_sl), dim=-1))
        dead = torch.isneginf(lse)[..., None]
        acc = None if vc_ is None else torch.zeros(B, Hq, qc.shape[2], vc_.shape[-1], dtype=compute_dtype, device=q.device)
        for k_sl in k_slices:
            p = torch.exp(logits_for(qc, q_sl, k_sl) - lse[..., None]).masked_fill(dead, 0.0)
            p = post(p, q_sl, k_sl)
            if acc is not None:
                acc += grouped_pv(p, vc_[:, :, k_sl])
        return acc

    qn, kc, start = query_chunk, key_chunk, 0
    while start < Nq:
        q_sl = slice(start, min(start + qn, Nq))
        qc = q[:, :, q_sl].to(compute_dtype)
        observed = False
        try:
            res = full_rows(qc, q_sl) if kc is None else key_chunked(qc, q_sl, kc)
        except _OOM:
            if observed:  # observers already saw part of this chunk; retrying would double count
                raise
            if kc is None and prob_override is None:
                kc = oom_key_chunk
            elif qn > 1:
                qn = max(1, qn // 2)
            else:
                raise
            record_count("oom_fallback")
            if q.is_cuda:
                torch.cuda.empty_cache()
            continue
        if out is not None:
            out[:, :, q_sl] = res.to(out.dtype)
        start = q_sl.stop
    for ob in observers:
        ob.finish()
    return out


# --------------------------------------------------------------------------- 1D text-axis blur


def gaussian_kernel_1d(size: int, sigma: float, device: Optional[torch.device] = None,
                       dtype: torch.dtype = torch.float32) -> Tensor:
    """Normalised odd-length Gaussian; ``sigma <= 0`` gives the identity (delta) kernel."""
    if size < 1 or size % 2 == 0:
        raise ValueError("kernel size must be a positive odd integer")
    x = torch.arange(size, device=device, dtype=dtype) - size // 2
    k = (x == 0).to(dtype) if sigma <= 0 else torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def blur_last_dim(x: Tensor, kernel: Tensor) -> Tensor:
    """1D convolution along the last axis with reflect padding (replicate if too short)."""
    L, pad = x.shape[-1], kernel.numel() // 2
    flat = x.reshape(-1, 1, L)
    if pad:
        flat = F.pad(flat, (pad, pad), mode="reflect" if L > pad else "replicate")
    return F.conv1d(flat, kernel.view(1, 1, -1).to(flat.dtype)).reshape(x.shape)


@dataclass(eq=False)
class TextAxisBlur:
    """Seg4Diff App. A.3 causal perturbation: 1D Gaussian blur along the text-key axis.

    Restricted to ``key_span`` (prompt content only, never pad/template keys) and to the
    selected query rows / heads / batch rows. With ``preserve_mass`` each row's mass inside
    the span is rescaled back to its original value, so row sums stay exactly 1 and every
    column outside the span is untouched. ``strength`` lerps baseline → blurred (``0`` = no-op).
    """

    key_span: slice
    query: Optional[Tensor] = None
    heads: Optional[Tensor] = None
    batch: Optional[Tensor] = None
    kernel_size: int = 5
    sigma: float = 9.0
    strength: float = 1.0
    preserve_mass: bool = True

    @classmethod
    def for_layout(cls, layout: SeqLayout, span: TokenSpan, *, batch: Optional[Tensor] = None,
                   **kwargs) -> TextAxisBlur:
        """Blur image-query rows over the text span ``span`` (text-local coordinates)."""
        return cls(key_span=layout.txt_keys(span), query=index_mask(layout.n_query, layout.q_img),
                   batch=batch, **kwargs)

    def __call__(self, probs: Tensor, q_sl: slice) -> Tensor:
        seg = probs[..., self.key_span]
        if seg.shape[-1] == 0 or self.strength == 0:
            return probs
        blurred = blur_last_dim(seg, gaussian_kernel_1d(self.kernel_size, self.sigma, seg.device, seg.dtype))
        if self.preserve_mass:
            tiny = torch.finfo(seg.dtype).tiny
            blurred = blurred * (seg.sum(-1, keepdim=True) / blurred.sum(-1, keepdim=True).clamp_min(tiny))
        new = torch.lerp(seg, blurred, self.strength)
        m = Region(self.batch, self.heads, self.query).to(probs.device).chunk_mask(q_sl, slice(None))
        if m is not None:
            new = torch.where(m, new, seg)
        out = probs.clone()
        out[..., self.key_span] = new
        return out


# --------------------------------------------------------------------------- dynamic sinks


class IncomingMass(ProbObserver):
    """Per-(sample, head) incoming mass ``m_j = (1/|Q|) Σ_{i∈Q} A_ij`` over selected query rows."""

    def __init__(self, query: Tensor, shape: tuple[int, int, int], device: torch.device) -> None:
        self.query = query.to(device)
        self.n = int(self.query.sum())
        self.total = torch.zeros(shape, dtype=torch.float32, device=device)

    def __call__(self, probs: Tensor, q_sl: slice, k_sl: slice) -> None:
        rows = self.query[q_sl]
        if rows.any():
            self.total[..., k_sl] += probs[:, :, rows].sum(2).float()

    @property
    def mass(self) -> Tensor:
        return self.total / max(self.n, 1)


@dataclass(frozen=True, eq=False)
class SinkSelection:
    """Dynamic sink choice: ``unit_ids [B, Hq, k]``, column ``mask [B, Hq, Nk]``, pre-intervention ``mass_before [B, Hq, Nk]``."""

    unit_ids: Tensor
    mask: Tensor
    mass_before: Tensor


def select_dynamic_sinks(mass: Tensor, top_k: int, *, selection: Literal["per_head", "union"] = "per_head",
                         groups: Sequence[slice] = (), random: bool = False,
                         generator: Optional[torch.Generator] = None) -> SinkSelection:
    """Top-``k`` keys by incoming mass (AttentionSinksDiT Sec 3.3.1 / 3.5.3).

    ``groups`` (e.g. Anima's zero-pad ``P`` span, whose logits tie) are scored as one unit
    (summed mass) and masked together. ``union`` ranks the head-mean mass so all heads share
    one index set. ``random=True`` draws a Random-k control over the same units.
    """
    B, H, Nk = mass.shape
    dev = mass.device
    unit = torch.arange(Nk, device=dev)
    valid = torch.ones(Nk, dtype=torch.bool, device=dev)
    for g in groups:
        unit[g] = g.start
        valid[g.start + 1:g.stop] = False
    unit_idx = unit.expand(B, H, Nk)
    score = torch.zeros_like(mass, dtype=torch.float32).scatter_add_(-1, unit_idx, mass.float())
    if random:
        score = torch.rand((B, 1 if selection == "union" else H, Nk), generator=generator,
                           device="cpu" if generator is None else generator.device).to(dev).expand(B, H, Nk)
    elif selection == "union":
        score = score.mean(1, keepdim=True).expand(B, H, Nk)
    elif selection != "per_head":
        raise ValueError(f"unknown selection {selection!r}")
    score = score.masked_fill(~valid, NEG_INF)
    k = min(int(top_k), int(valid.sum()))
    ids = score.topk(k, dim=-1).indices
    chosen = torch.zeros(B, H, Nk, dtype=torch.bool, device=dev).scatter_(-1, ids, True)
    return SinkSelection(ids, chosen.gather(-1, unit_idx), mass)


def dynamic_sink_edit(sel: SinkSelection, query: Tensor, value: float = SINK_MASK_VALUE,
                      batch: Optional[Tensor] = None) -> LogitEdit:
    """Pass-2 edit: add ``value`` (``-1e4``) to logits at ``(Q_img rows, S_k)``; text rows untouched."""
    return LogitEdit(Region(batch=batch, query=query, key=sel.mask), "add", value)


# --------------------------------------------------------------------------- streaming accumulator


class RunningMap:
    """Streaming reducer that never stores per-call history.

    ``mean``: weighted running mean over all ``add`` calls.
    ``group_mean_sum``: mean within each contiguous ``group`` (e.g. one step), summed across
    groups (DAAM-style). Groups must arrive contiguously.
    """

    def __init__(self, mode: Literal["mean", "group_mean_sum"] = "mean", *,
                 device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        if mode not in ("mean", "group_mean_sum"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode, self.device, self.dtype = mode, torch.device(device), dtype
        self.total: Optional[Tensor] = None
        self.weight = 0.0
        self.count = 0
        self._pending: Optional[Tensor] = None
        self._pending_n = 0
        self._group: object = None
        self._seen: set = set()

    @property
    def shape(self) -> Optional[torch.Size]:
        ref = self.total if self.total is not None else self._pending
        return None if ref is None else ref.shape

    def add(self, values: Tensor, *, weight: float = 1.0, group: object = None) -> None:
        v = values.detach().to(self.device, self.dtype)
        if self.shape is not None and v.shape != self.shape:
            raise ValueError(f"map shape changed {tuple(self.shape)} -> {tuple(v.shape)}")
        if self.mode == "mean":
            self.total = v * weight if self.total is None else self.total.add_(v, alpha=weight)
            self.weight += weight
        else:
            if group is None:
                raise ValueError("group_mean_sum requires group=")
            if group != self._group:
                if group in self._seen:
                    raise ValueError(f"group {group!r} is not contiguous")
                self._flush()
                self._group = group
                self._seen.add(group)
            self._pending = v.clone() if self._pending is None else self._pending.add_(v)
            self._pending_n += 1
        self.count += 1

    def _flush(self) -> None:
        if self._pending is not None:
            g = self._pending / self._pending_n
            self.total = g if self.total is None else self.total.add_(g)
            self._pending, self._pending_n = None, 0

    def value(self) -> Tensor:
        """Current reduction (non-destructive)."""
        if self.count == 0:
            raise ValueError("RunningMap is empty")
        if self.mode == "mean":
            return self.total / self.weight
        tail = self._pending / self._pending_n if self._pending is not None else 0
        return (self.total + tail) if self.total is not None else tail


# --------------------------------------------------------------------------- session

@dataclass(frozen=True, eq=False)
class AttentionCall:
    """Everything an observer factory may need about the current attention call.

    ``extras`` carries adapter-provided tensors, e.g. ``gate_logits`` (Krea 2, ``[B, Nq, Hq*d]``)
    and ``g1`` (timestep modulation applied after ``Wo``).
    """

    block: int
    ctx: StepContext
    layout: SeqLayout
    q: Tensor
    k: Tensor
    v: Tensor
    extras: Mapping[str, Tensor]

    @property
    def device(self) -> torch.device:
        return self.q.device


ObserverFactory = Callable[[AttentionCall], Optional[ProbObserver]]
OverrideFactory = Callable[[SeqLayout, Optional[Tensor]], ProbOverride]

_ATTN_HOOKS = frozenset({"logit_knockout", "logit_scale", "attn_override", "two_pass_sink_mask",
                         "span_query_mask", "per_head_token_ablation"})


@dataclass
class SinkRecord:
    block: int
    step_index: int
    sigma: float
    selection: SinkSelection


class _ProbsRecorder(ProbObserver):
    def __init__(self, session: XAISession, block: int, ctx: StepContext, rows: tuple[int, ...],
                 shape: tuple[int, int, int], device: torch.device) -> None:
        self.session, self.block, self.ctx = session, block, ctx
        self.rows = torch.tensor(rows, device=device)
        dev = "cpu" if session.capture.to_cpu else device
        self.buf = torch.empty((len(rows), *shape), dtype=torch.float32, device=dev)

    def __call__(self, probs: Tensor, q_sl: slice, k_sl: slice) -> None:
        self.buf[:, :, q_sl, k_sl] = probs.index_select(0, self.rows).to(self.buf.device, torch.float32)

    def finish(self) -> None:
        self.session._store("probs", self.buf, self.block, self.ctx)


class XAISession:
    """Fixed-policy hook manager consulted by adapters on every block call.

    Adapters call :meth:`attention` instead of their attention kernel (passing ``native`` =
    the original kernel returning ``[B, Hq, Nq, dv]``), :meth:`record` for non-attention taps,
    and :meth:`interventions_for` for model-side hooks (``kv_patch``, ``gate_override``,
    ``pre_dit_span_swap``, ``concept_stream``, ``true_block_bypass``).
    """

    def __init__(self, capability: AdapterCapability, *, capture: Optional[CaptureSpec] = None,
                 interventions: Sequence[InterventionSpec] = (), query_chunk: int = 1024,
                 key_chunk: Optional[int] = None, oom_key_chunk: int = 1024,
                 enforce_fixed_batching: bool = True) -> None:
        self.capability = capability
        self.capture = capture
        self.interventions = tuple(interventions)
        self.query_chunk, self.key_chunk, self.oom_key_chunk = query_chunk, key_chunk, oom_key_chunk
        self.enforce_fixed_batching = enforce_fixed_batching
        self.enabled = True
        self.captures: dict[tuple, CapturedTensors] = {}
        self.sink_log: list[SinkRecord] = []
        self._observers: list[tuple[ObserverFactory, Optional[frozenset[int]], StepWindow]] = []
        self._batching: Optional[tuple[int, int]] = None

    # ---- configuration
    def add_observer(self, factory: ObserverFactory, *, blocks: Optional[Sequence[int]] = None,
                     window: StepWindow = StepWindow()) -> None:
        self._observers.append((factory, None if blocks is None else frozenset(blocks), window))

    @contextmanager
    def disabled(self) -> Iterator[XAISession]:
        prev, self.enabled = self.enabled, False
        try:
            yield self
        finally:
            self.enabled = prev

    # ---- queries
    def interventions_for(self, hook: str, block: int, ctx: StepContext) -> list[InterventionSpec]:
        if not self.enabled:
            return []
        return [s for s in self.interventions if s.hook == hook and s.active(block, ctx)]

    def bypass_mode(self, block: int, ctx: StepContext) -> Optional[str]:
        """``true_block_bypass`` mode for this block (applied identically to cond and uncond)."""
        specs = self.interventions_for("true_block_bypass", block, ctx)
        return str(specs[0].params.get("mode", "full_block")) if specs else None

    def wants(self, name: str, block: int, ctx: StepContext) -> bool:
        return self.enabled and self.capture is not None and self.capture.wants(name, block, ctx)

    def check_batching(self, ctx: StepContext) -> None:
        """Enforce one CFG batching policy per run (no memory-driven batched/split switching)."""
        sig = (len(ctx.cond_or_uncond), ctx.chunk_size)
        if self._batching is None:
            self._batching = sig
        elif self.enforce_fixed_batching and sig != self._batching:
            raise RuntimeError(f"CFG batching changed {self._batching} -> {sig}; fix the runner batching policy")

    # ---- capture
    def _store(self, name: str, tensor: Tensor, block: int, ctx: StepContext) -> None:
        rows = self.capture.rows(ctx)
        samples = tuple(ctx.row_map()[r] for r in rows)
        key = (block, ctx.step_index, samples)
        entry = self.captures.get(key)
        if entry is None:
            entry = self.captures[key] = CapturedTensors(block, ctx.step_index, ctx.sigma, rows)
        entry.set(name, tensor)

    def record(self, name: str, tensor: Tensor, *, block: int, ctx: StepContext, batch_dim: int = 0) -> None:
        """Store ``tensor`` rows selected by ``CaptureSpec.branches`` (no-op if not requested)."""
        if not self.wants(name, block, ctx):
            return
        rows = self.capture.rows(ctx)
        if not rows:
            return
        t = tensor.detach().index_select(batch_dim, torch.tensor(rows, device=tensor.device))
        self._store(name, t.cpu() if self.capture.to_cpu else t.clone(), block, ctx)

    # ---- attention
    def attention(self, q: Tensor, k: Tensor, v: Tensor, *, block: int, ctx: StepContext,
                  layout: SeqLayout, native: Callable[[Tensor, Tensor, Tensor], Tensor],
                  scale: Optional[float] = None, attn_mask: Optional[Tensor] = None,
                  extras: Optional[Mapping[str, Tensor]] = None) -> Tensor:
        """Run attention for one block call; ``q [B,Hq,Nq,d]``, ``k/v [B,Hkv,Nk,d]`` (pre GQA repeat)."""
        if not self.enabled:
            return native(q, k, v)
        self.check_batching(ctx)
        for name, t in (("q", q), ("k", k), ("v", v)):
            self.record(name, t, block=block, ctx=ctx)
        logit, prob, override = self._attention_edits(q, k, block, ctx, layout, scale, attn_mask)
        call = AttentionCall(block, ctx, layout, q, k, v, extras or {})
        observers = [ob for f, blocks, window in self._observers
                     if (blocks is None or block in blocks) and window.contains(ctx)
                     for ob in (f(call),) if ob is not None]
        if self.wants("probs", block, ctx) and self.capture.rows(ctx):
            observers.append(_ProbsRecorder(self, block, ctx, self.capture.rows(ctx),
                                            (q.shape[1], q.shape[2], k.shape[2]), q.device))
        if not (logit or prob or override or observers):
            return native(q, k, v)  # exact no-op handoff
        with measure("xai_attention"):
            return chunked_attention(q, k, v, scale=scale, attn_mask=attn_mask, logit_edits=logit,
                                     prob_edits=prob, prob_override=override, observers=observers,
                                     query_chunk=self.query_chunk,
                                     key_chunk=None if override is not None else self.key_chunk,
                                     oom_key_chunk=self.oom_key_chunk)

    @staticmethod
    def _query_mask(layout: SeqLayout, which, dev: torch.device) -> Optional[Tensor]:
        if which == "all":
            return None
        if which == "img":
            return index_mask(layout.n_query, layout.q_img, dev)
        if which == "txt":
            return index_mask(layout.n_query, layout.q_txt or slice(0, 0), dev)
        return union_mask(layout.n_query, [layout.txt_queries(s) for s in as_spans(which)], dev)

    @staticmethod
    def _key_mask(layout: SeqLayout, which, dev: torch.device) -> Optional[Tensor]:
        if which == "all":
            return None
        if which is None or which == "txt":
            return index_mask(layout.n_key, layout.k_txt, dev)
        if which == "img":
            if layout.k_img is None:
                raise ValueError("no image keys in cross-attention")
            return index_mask(layout.n_key, layout.k_img, dev)
        return union_mask(layout.n_key, [layout.txt_keys(s) for s in as_spans(which)], dev)

    def _attention_edits(self, q: Tensor, k: Tensor, block: int, ctx: StepContext, layout: SeqLayout,
                         scale: Optional[float], attn_mask: Optional[Tensor]):
        dev = q.device
        B, Hq, Nq = q.shape[:3]
        Nk = k.shape[2]
        if (Nq, Nk) != (layout.n_query, layout.n_key):
            raise ValueError(f"q/k lengths {(Nq, Nk)} do not match layout {(layout.n_query, layout.n_key)}")
        logit: list[LogitEdit] = []
        prob: list[ProbZero] = []
        override: Optional[ProbOverride] = None
        sinks: list[tuple[InterventionSpec, Optional[Tensor]]] = []
        for spec in self.interventions:
            if spec.hook not in _ATTN_HOOKS or not spec.active(block, ctx):
                continue
            p = spec.params
            batch = None if spec.cfg_scope == "both" else index_mask(B, list(spec.rows(ctx)), dev)
            if batch is not None and not bool(batch.any()):
                continue  # this call holds no cond rows
            heads = None if p.get("heads") is None else index_mask(Hq, list(p["heads"]), dev)
            if spec.hook == "logit_knockout":
                region = Region(batch, heads, self._query_mask(layout, p.get("query", "img"), dev),
                                self._key_mask(layout, p["keys"], dev))
                logit.append(LogitEdit(region, "set", NEG_INF))
            elif spec.hook == "logit_scale":
                region = Region(batch, heads, self._query_mask(layout, p.get("query", "img"), dev),
                                self._key_mask(layout, p.get("keys"), dev))
                gamma, bias = float(p.get("gamma", 1.0)), float(p.get("bias", 0.0))
                if gamma != 1.0:
                    logit.append(LogitEdit(region, "scale", gamma))
                if bias != 0.0:
                    logit.append(LogitEdit(region, "add", bias))
            elif spec.hook == "span_query_mask":
                region = Region(batch, heads, self._query_mask(layout, p["query_span"], dev),
                                self._key_mask(layout, p["key_target"], dev))
                logit.append(LogitEdit(region, "set", NEG_INF))
            elif spec.hook == "per_head_token_ablation":
                region = Region(batch, heads, self._query_mask(layout, p.get("query", "img"), dev),
                                self._key_mask(layout, p["keys"], dev))
                prob.append(ProbZero(region))
            elif spec.hook == "attn_override":
                if override is not None:
                    raise ValueError("only one attn_override may be active per attention call")
                override = p["override"](layout, batch)
            else:
                sinks.append((spec, batch))
        for spec, batch in sinks:
            p = spec.params
            q_img = index_mask(Nq, layout.q_img, dev)
            obs = IncomingMass(q_img, (B, Hq, Nk), dev)
            chunked_attention(q, k, None, scale=scale, attn_mask=attn_mask, logit_edits=list(logit),
                              observers=[obs], query_chunk=self.query_chunk, key_chunk=self.key_chunk,
                              oom_key_chunk=self.oom_key_chunk)
            groups = [layout.txt_keys(s) for s in as_spans(p.get("groups", ()))]
            sel = select_dynamic_sinks(obs.mass, int(p["top_k"]), selection=p.get("selection", "per_head"),
                                       groups=groups, random=bool(p.get("random", False)),
                                       generator=p.get("generator"))
            logit.append(dynamic_sink_edit(sel, q_img, float(p.get("value", SINK_MASK_VALUE)), batch))
            self.sink_log.append(SinkRecord(block, ctx.step_index, ctx.sigma, sel))
        return logit, prob, override
