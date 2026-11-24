"""Model-adapter contract shared by all target DiTs (``docs/architecture.md`` §3).

Pure metadata module: importing it does not import torch. Tensor-typed fields are
annotated via ``TYPE_CHECKING`` only, so the contract can be validated on a CPU box
without PyTorch.
"""

from __future__ import annotations

import abc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Mapping, Optional, Sequence

if TYPE_CHECKING:
    import torch

    from ..core.session import XAISession

AttentionKind = Literal["cross_attn", "joint_attn"]
SeqOrder = Literal["img_txt", "txt_img", "cross"]
Branch = Literal["cond", "uncond"]
CfgScope = Literal["both", "cond_only"]

# ComfyUI ``cond_or_uncond`` chunk labels (index into ``calc_cond_batch(model, [cond, uncond], ...)``).
COND, UNCOND = 0, 1
BRANCH_LABEL: Mapping[str, int] = {"cond": COND, "uncond": UNCOND}

CAPTURE_FIELDS: tuple[str, ...] = (
    "q", "k", "v", "probs",
    "attn_pre_proj",  # context before gate and Wo (MOSS ``attn.out``, Krea 2 ``ctx``)
    "attn_out",       # after gate (if any) and Wo, before timestep modulation g1
    "gate_logits",    # Krea 2 only: raw ``self.gate(x)`` (pre-sigmoid)
    "gate_value",     # Krea 2 only: ``sigmoid(gate_logits)`` multiplier
    "residual_out",   # block output residual tap (E4-5)
)

INTERVENTION_HOOKS: tuple[str, ...] = (
    "kv_patch", "logit_knockout", "logit_scale", "attn_override", "two_pass_sink_mask",
    "span_query_mask", "pre_dit_span_swap", "gate_override", "per_head_token_ablation",
    "concept_stream", "true_block_bypass",
)

# Minimal required ``InterventionSpec.params`` per hook. Optional keys are documented on the spec.
_REQUIRED_PARAMS: Mapping[str, tuple[str, ...]] = {
    "kv_patch": ("source",),
    "logit_knockout": ("keys",),
    "logit_scale": (),
    "attn_override": ("override",),
    "two_pass_sink_mask": ("top_k",),
    "span_query_mask": ("query_span", "key_target"),
    "pre_dit_span_swap": ("span", "source", "point"),
    "gate_override": ("mode",),
    "per_head_token_ablation": ("heads", "keys"),
    "concept_stream": ("concepts",),
    "true_block_bypass": (),
}


class TokenAlignmentError(AssertionError):
    """Raised when two token maps are not strictly aligned outside the editable spans."""


# --------------------------------------------------------------------------- capability


@dataclass(frozen=True)
class AdapterCapability:
    """Static per-model metadata (``architecture.md`` §3-1)."""

    name: str
    attention_kind: AttentionKind
    seq_order: SeqOrder
    num_blocks: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    text_updated_per_block: bool
    supports_concept_attention: bool
    has_output_gate: bool
    text_pad_to: Optional[int] = None  # Anima: 512 (zero-padded ctx); None = no padding

    def __post_init__(self) -> None:
        if (self.attention_kind == "cross_attn") != (self.seq_order == "cross"):
            raise ValueError("seq_order must be 'cross' iff attention_kind == 'cross_attn'")
        if min(self.num_blocks, self.num_q_heads, self.num_kv_heads, self.head_dim) <= 0:
            raise ValueError("block/head counts and head_dim must be positive")
        if self.num_q_heads % self.num_kv_heads:
            raise ValueError("num_q_heads must be a multiple of num_kv_heads (GQA)")

    @property
    def gqa_group(self) -> int:
        """Q heads per KV head ``g``; GQA uses ``repeat_interleave(g)`` so ``h_kv = h_q // g``."""
        return self.num_q_heads // self.num_kv_heads

    def kv_head(self, q_head: int) -> int:
        return q_head // self.gqa_group

    def q_heads_of(self, kv_head: int) -> range:
        g = self.gqa_group
        return range(kv_head * g, (kv_head + 1) * g)


# --------------------------------------------------------------------------- tokens / layout


@dataclass(frozen=True, order=True)
class TokenSpan:
    """Half-open text-local token interval ``[start, end)``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if not (0 <= self.start <= self.end):
            raise ValueError(f"invalid half-open span [{self.start}, {self.end})")

    def __len__(self) -> int:
        return self.end - self.start

    def __contains__(self, index: object) -> bool:
        return isinstance(index, int) and self.start <= index < self.end

    def as_slice(self, offset: int = 0) -> slice:
        return slice(self.start + offset, self.end + offset)

    def indices(self) -> range:
        return range(self.start, self.end)

    def overlaps(self, other: TokenSpan) -> bool:
        return self.start < other.end and other.start < self.end


def as_spans(spans: TokenSpan | Sequence[TokenSpan]) -> tuple[TokenSpan, ...]:
    return (spans,) if isinstance(spans, TokenSpan) else tuple(spans)


@dataclass(frozen=True)
class TokenMap:
    """Exact text-slot classification for one branch of one prompt.

    ``spans`` is a disjoint half-open partition of ``[0, len(token_ids))``. Canonical keys:
    ``R_prefix`` (leading chat template), ``S`` (prompt content), ``R`` (trailing
    template / ``</s>``; i.e. the suffix register), ``P`` (zero padding). Anima:
    ``S=[0, Lt-1)``, ``R=[Lt-1, Lt)``, ``P=[Lt, L)``.
    ``targets`` holds tag/concept spans (text-local) used by interventions.
    """

    token_ids: tuple[int, ...]
    spans: Mapping[str, TokenSpan]
    targets: Mapping[str, TokenSpan] = field(default_factory=dict)
    branch_text_len: Mapping[str, int] = field(default_factory=dict)
    occurrence_idx: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(int(t) for t in self.token_ids))
        if not self.occurrence_idx:
            seen: dict[int, int] = {}
            occ = []
            for t in self.token_ids:
                occ.append(seen.get(t, 0))
                seen[t] = occ[-1] + 1
            object.__setattr__(self, "occurrence_idx", tuple(occ))
        if len(self.occurrence_idx) != len(self.token_ids):
            raise ValueError("occurrence_idx must have one entry per token")
        self.validate_partition()
        for name, span in self.targets.items():
            if span.end > self.text_len:
                raise ValueError(f"target {name!r} {span} exceeds text length {self.text_len}")

    @classmethod
    def from_lengths(cls, token_ids: Sequence[int], *, prefix: int = 0, suffix: int = 0,
                     pad: int = 0, **kwargs: Any) -> TokenMap:
        """Build ``R_prefix | S | R | P`` from segment lengths (zero-length spans omitted)."""
        n = len(token_ids)
        content = n - prefix - suffix - pad
        if content < 0 or min(prefix, suffix, pad) < 0:
            raise ValueError("segment lengths exceed token count")
        bounds = [("R_prefix", prefix), ("S", content), ("R", suffix), ("P", pad)]
        spans, start = {}, 0
        for name, length in bounds:
            if length:
                spans[name] = TokenSpan(start, start + length)
            start += length
        return cls(tuple(token_ids), spans, **kwargs)

    @property
    def text_len(self) -> int:
        return len(self.token_ids)

    def validate_partition(self) -> None:
        """Spans must be disjoint, non-empty, and tile ``[0, L)`` exactly (half-open)."""
        ordered = sorted(self.spans.values())
        cursor = 0
        for span in ordered:
            if len(span) == 0 or span.start != cursor:
                raise ValueError(f"spans do not tile [0, {self.text_len}) at {cursor}: {dict(self.spans)}")
            cursor = span.end
        if cursor != self.text_len:
            raise ValueError(f"spans cover [0, {cursor}) but text length is {self.text_len}")

    def span(self, name: str) -> TokenSpan:
        return self.spans[name] if name in self.spans else self.targets[name]

    def role_of(self, index: int) -> str:
        for name, span in self.spans.items():
            if index in span:
                return name
        raise IndexError(index)

    def occurrence(self, token_id: int, k: int = 0) -> int:
        """Text-local position of the ``k``-th occurrence of ``token_id``."""
        for i, (t, o) in enumerate(zip(self.token_ids, self.occurrence_idx)):
            if t == token_id and o == k:
                return i
        raise KeyError(f"token {token_id} occurrence {k} not found")

    def assert_strict_token_alignment(self, other: TokenMap,
                                      editable: Sequence[TokenSpan | str] = ()) -> None:
        """Assert ``other`` is a length-matched substitute of ``self``.

        Requires equal length, identical span partition, and identical token IDs at every
        position outside ``editable`` (spans or target names). Used before neutral-prompt
        swaps / K-V patches on MOSS-Image and Krea 2 (no padding, RoPE depends on length).
        """
        if self.text_len != other.text_len:
            raise TokenAlignmentError(f"length mismatch {self.text_len} != {other.text_len}")
        if dict(self.spans) != dict(other.spans):
            raise TokenAlignmentError(f"span partition mismatch {dict(self.spans)} != {dict(other.spans)}")
        free = [self.span(e) if isinstance(e, str) else e for e in editable]
        bad = [i for i, (a, b) in enumerate(zip(self.token_ids, other.token_ids))
               if a != b and not any(i in s for s in free)]
        if bad:
            raise TokenAlignmentError(f"token IDs differ outside editable spans at positions {bad[:16]}")


@dataclass(frozen=True)
class SeqLayout:
    """DiT attention-sequence slot coordinates.

    ``cross``: queries are image tokens ``[0, n_img)``, keys are text slots ``[0, n_txt)``.
    ``img_txt`` (MOSS-Image): joint ``[img N | txt L]``. ``txt_img`` (Krea 2): ``[txt T | img N]``.
    """

    order: SeqOrder
    n_img: int
    n_txt: int

    @property
    def is_joint(self) -> bool:
        return self.order != "cross"

    @property
    def n_query(self) -> int:
        return self.n_img + self.n_txt if self.is_joint else self.n_img

    @property
    def n_key(self) -> int:
        return self.n_img + self.n_txt if self.is_joint else self.n_txt

    @property
    def _img_off(self) -> int:
        return self.n_txt if self.order == "txt_img" else 0

    @property
    def _txt_off(self) -> int:
        return self.n_img if self.order == "img_txt" else 0

    @property
    def q_img(self) -> slice:
        return slice(self._img_off, self._img_off + self.n_img)

    @property
    def q_txt(self) -> Optional[slice]:
        return slice(self._txt_off, self._txt_off + self.n_txt) if self.is_joint else None

    @property
    def k_img(self) -> Optional[slice]:
        return self.q_img if self.is_joint else None

    @property
    def k_txt(self) -> slice:
        return slice(self._txt_off, self._txt_off + self.n_txt)

    def txt_keys(self, span: TokenSpan) -> slice:
        """Map a text-local span to key-axis positions."""
        if span.end > self.n_txt:
            raise ValueError(f"{span} exceeds n_txt={self.n_txt}")
        return span.as_slice(self.k_txt.start)

    def txt_queries(self, span: TokenSpan) -> slice:
        if not self.is_joint:
            raise ValueError("text queries do not exist in cross-attention")
        if span.end > self.n_txt:
            raise ValueError(f"{span} exceeds n_txt={self.n_txt}")
        return span.as_slice(self._txt_off)


# --------------------------------------------------------------------------- step / CFG


@dataclass(frozen=True)
class StepContext:
    """Per-forward step and CFG metadata.

    ``step_index`` is derived from the current sigma (``transformer_options["sigmas"]``)
    located in the schedule (``transformer_options["sample_sigmas"]``), never from call
    counting. ``cond_or_uncond`` lists the label of each equally sized batch chunk.
    """

    sample_sigmas: tuple[float, ...]
    sigma: float
    cond_or_uncond: tuple[int, ...]
    batch_size: int
    atol: float = 1e-5
    step_index: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.sample_sigmas:
            raise ValueError("empty sigma schedule")
        if not self.cond_or_uncond or self.batch_size % len(self.cond_or_uncond):
            raise ValueError(f"batch {self.batch_size} not divisible into chunks {self.cond_or_uncond}")
        if any(c not in (COND, UNCOND) for c in self.cond_or_uncond):
            raise ValueError(f"unknown cond_or_uncond labels {self.cond_or_uncond}")
        diffs = [abs(s - self.sigma) for s in self.sample_sigmas]
        i = min(range(len(diffs)), key=diffs.__getitem__)
        if diffs[i] > self.atol * max(1.0, abs(self.sigma)):
            raise ValueError(f"sigma {self.sigma} not found in schedule")
        object.__setattr__(self, "step_index", i)

    @classmethod
    def from_transformer_options(cls, options: Mapping[str, Any], batch_size: int,
                                 atol: float = 1e-5) -> StepContext:
        def floats(x: Any) -> list[float]:  # tensor, sequence or scalar -> flat float list
            if hasattr(x, "flatten"):
                return [float(v) for v in x.flatten().tolist()]
            return [float(v) for v in x] if isinstance(x, (list, tuple)) else [float(x)]

        values = floats(options["sigmas"])
        if max(values) - min(values) > atol:
            raise ValueError(f"mixed sigmas in one forward: {values}")
        return cls(tuple(floats(options["sample_sigmas"])), values[0],
                   tuple(int(c) for c in options["cond_or_uncond"]), int(batch_size), atol)

    @property
    def num_steps(self) -> int:
        """Number of denoising updates ``S`` (schedule has ``S + 1`` sigmas)."""
        return len(self.sample_sigmas) - 1

    @property
    def step_fraction(self) -> float:
        return self.step_index / max(self.num_steps, 1)

    @property
    def chunk_size(self) -> int:
        return self.batch_size // len(self.cond_or_uncond)

    def row_map(self) -> list[tuple[str, int]]:
        """``(branch, sample_id)`` for every batch row. Invariant to chunk order and call splits."""
        names = {COND: "cond", UNCOND: "uncond"}
        counts = {COND: 0, UNCOND: 0}
        out = []
        for label in self.cond_or_uncond:
            base = counts[label]
            out.extend((names[label], base + s) for s in range(self.chunk_size))
            counts[label] += self.chunk_size
        return out

    def rows(self, branch: str) -> tuple[int, ...]:
        """Batch rows of ``branch`` ordered by sample id."""
        pairs = [(sid, r) for r, (b, sid) in enumerate(self.row_map()) if b == branch]
        return tuple(r for _, r in sorted(pairs))

    def rows_for_scope(self, scope: CfgScope) -> tuple[int, ...]:
        return tuple(range(self.batch_size)) if scope == "both" else self.rows("cond")


@dataclass(frozen=True)
class StepWindow:
    """Active step window: half-open step indices ``[start, end)`` AND optional sigma bounds (inclusive)."""

    start: int = 0
    end: Optional[int] = None
    sigma_min: Optional[float] = None
    sigma_max: Optional[float] = None

    def contains(self, ctx: StepContext) -> bool:
        i, s = ctx.step_index, ctx.sigma
        return (i >= self.start and (self.end is None or i < self.end)
                and (self.sigma_min is None or s >= self.sigma_min)
                and (self.sigma_max is None or s <= self.sigma_max))


def _blocks(blocks: Optional[Sequence[int]]) -> Optional[frozenset[int]]:
    return None if blocks is None else frozenset(int(b) for b in blocks)


# --------------------------------------------------------------------------- capture / intervene


@dataclass(frozen=True)
class CaptureSpec:
    """Which tensors to record, where, and for which CFG branches."""

    fields: frozenset[str]
    blocks: Optional[frozenset[int]] = None
    window: StepWindow = StepWindow()
    branches: frozenset[str] = frozenset({"cond"})
    to_cpu: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", frozenset(self.fields))
        object.__setattr__(self, "blocks", _blocks(self.blocks))
        object.__setattr__(self, "branches", frozenset(self.branches))
        unknown = self.fields - set(CAPTURE_FIELDS)
        if unknown:
            raise ValueError(f"unknown capture fields {sorted(unknown)}")
        if not self.branches <= {"cond", "uncond"}:
            raise ValueError(f"invalid branches {self.branches}")

    def wants(self, name: str, block: int, ctx: StepContext) -> bool:
        return (name in self.fields and (self.blocks is None or block in self.blocks)
                and self.window.contains(ctx))

    def rows(self, ctx: StepContext) -> tuple[int, ...]:
        return tuple(r for b in ("cond", "uncond") if b in self.branches for r in ctx.rows(b))


@dataclass
class CapturedTensors:
    """Recorded tensors for one ``(block, step)``; rows follow ``CaptureSpec.rows`` order."""

    block: int
    step_index: int
    sigma: float
    rows: tuple[int, ...]
    q: Optional["torch.Tensor"] = None
    k: Optional["torch.Tensor"] = None
    v: Optional["torch.Tensor"] = None
    probs: Optional["torch.Tensor"] = None
    attn_pre_proj: Optional["torch.Tensor"] = None
    attn_out: Optional["torch.Tensor"] = None
    gate_logits: Optional["torch.Tensor"] = None
    gate_value: Optional["torch.Tensor"] = None
    residual_out: Optional["torch.Tensor"] = None

    def set(self, name: str, tensor: "torch.Tensor") -> None:
        if name not in CAPTURE_FIELDS:
            raise ValueError(f"unknown capture field {name!r}")
        setattr(self, name, tensor)

    def present(self) -> tuple[str, ...]:
        return tuple(f for f in CAPTURE_FIELDS if getattr(self, f) is not None)


@dataclass(frozen=True)
class InterventionSpec:
    """One intervention hook, scoped by blocks, step window and CFG scope.

    Spans in ``params`` are text-local :class:`TokenSpan` (mapped through :class:`SeqLayout`).

    ===========================  ==========================================================
    hook                         params (required **bold**, optional with default)
    ===========================  ==========================================================
    kv_patch                     **source** (cache/provider), heads=None (KV heads), keys=None
    logit_knockout               **keys** (spans), query="img", heads=None
    logit_scale                  gamma=1.0, bias=0.0, keys=None (all text), query="img", heads=None
    attn_override                **override** (``(SeqLayout, batch_mask|None) -> ProbOverride``)
    two_pass_sink_mask           **top_k**, selection="per_head", value=-1e4, groups=()
    span_query_mask              **query_span**, **key_target** ("img" or spans)
    pre_dit_span_swap            **span**, **source**, **point** (e.g. "te.out", "txt_embed")
    gate_override                **mode** ("one" | "mean" | "scale"), value=None
    per_head_token_ablation      **heads** (Q heads), **keys** (spans), query="img"
    concept_stream               **concepts** (pooled concept states ``[R, D]``)
    true_block_bypass            mode="full_block" | "attn_only" | "mlp_only" | "cross_attn_only"
    ===========================  ==========================================================
    """

    hook: str
    blocks: Optional[frozenset[int]] = None
    window: StepWindow = StepWindow()
    cfg_scope: CfgScope = "both"
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.hook not in INTERVENTION_HOOKS:
            raise ValueError(f"unknown hook {self.hook!r}")
        if self.cfg_scope not in ("both", "cond_only"):
            raise ValueError(f"invalid cfg_scope {self.cfg_scope!r}")
        object.__setattr__(self, "blocks", _blocks(self.blocks))
        missing = [p for p in _REQUIRED_PARAMS[self.hook] if p not in self.params]
        if missing:
            raise ValueError(f"{self.hook} requires params {missing}")

    def active(self, block: int, ctx: StepContext) -> bool:
        return (self.blocks is None or block in self.blocks) and self.window.contains(ctx)

    def rows(self, ctx: StepContext) -> tuple[int, ...]:
        return ctx.rows_for_scope(self.cfg_scope)


# --------------------------------------------------------------------------- adapter ABC


class ModelAdapter(abc.ABC):
    """Bridge between a concrete DiT implementation and the pure-PyTorch core.

    Concrete adapters (``anima``, ``moss_image``, ``krea2``) route every attention call
    through :meth:`XAISession.attention` with a ``native`` fallback, and record
    non-attention taps via :meth:`XAISession.record`.
    """

    @property
    @abc.abstractmethod
    def capability(self) -> AdapterCapability: ...

    @abc.abstractmethod
    def build_token_map(self, token_ids: Sequence[int], *, branch: Branch = "cond") -> TokenMap:
        """Classify the DiT text slots of one encoded prompt into ``S/R/P`` spans."""

    @abc.abstractmethod
    def layout(self, n_img: int, token_map: TokenMap) -> SeqLayout:
        """Attention-sequence layout for ``n_img`` image tokens and this token map."""

    @abc.abstractmethod
    def output_projection(self, block: int) -> Callable[["torch.Tensor"], "torch.Tensor"]:
        """Block attention ``Wo`` as a callable module (``wo(x)``; quantization-safe)."""

    @abc.abstractmethod
    def attach(self, session: XAISession) -> None:
        """Install hooks that consult ``session``."""

    @abc.abstractmethod
    def detach(self) -> None:
        """Remove all hooks; the model must behave exactly as unhooked afterwards."""

    def step_context(self, transformer_options: Mapping[str, Any], batch_size: int) -> StepContext:
        return StepContext.from_transformer_options(transformer_options, batch_size)

    @contextmanager
    def attached(self, session: XAISession) -> Iterator[XAISession]:
        self.attach(session)
        try:
            yield session
        finally:
            self.detach()
