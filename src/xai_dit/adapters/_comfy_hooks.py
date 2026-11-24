"""Shared machinery for the headless-ComfyUI adapters (Anima, MOSS-Image, Krea 2).

Hooks are installed as *instance-level* ``forward`` overrides on the ComfyUI DiT modules
(``attach``/``detach``) or handed to a ``ModelPatcher`` as object patches
(:meth:`ComfyDiTAdapter.object_patches`, keys relative to ``diffusion_model``).

Exact no-op contract: every patched ``forward`` first checks whether the session touches the
current ``(block, step)``; if not, it calls the module's *original* class-level ``forward``
with the original arguments, so the output is bit-identical to the unhooked model.

No ``comfy.*`` import happens here: model-module helpers (``optimized_attention``,
``AttentionTensorContainer``, ``apply_rope``, ...) are looked up from the namespace of the
module that defines the hooked class at hook-install time.
"""

from __future__ import annotations

import abc
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from torch import nn

from ..core.session import XAISession
from .base import (
    InterventionSpec,
    ModelAdapter,
    SeqLayout,
    StepContext,
    TokenMap,
    TokenSpan,
    as_spans,
)

Tensor = torch.Tensor

#: Pseudo block index used to scope ``pre_dit_span_swap`` (leave ``InterventionSpec.blocks=None``).
PRE_DIT_BLOCK = -1

_STEP_KEYS = ("sigmas", "sample_sigmas", "cond_or_uncond")


# --------------------------------------------------------------------------- module helpers


def model_namespace(module: nn.Module) -> dict[str, Any]:
    """Globals of the Python module defining ``type(module)`` (ComfyUI helpers live there)."""
    return vars(sys.modules[type(module).__module__])


def original_forward(module: nn.Module) -> Callable[..., Any]:
    """Class-level ``forward`` bound to ``module`` (ignores any instance override)."""
    return type(module).forward.__get__(module, type(module))


def resolve_diffusion_model(obj: Any) -> nn.Module:
    """Accept a ``ModelPatcher``, a ``BaseModel`` or the DiT ``nn.Module`` itself."""
    inner = getattr(obj, "model", None)  # ModelPatcher -> BaseModel
    if inner is not None and hasattr(inner, "diffusion_model"):
        return inner.diffusion_model
    if hasattr(obj, "diffusion_model"):
        return obj.diffusion_model
    if isinstance(obj, nn.Module):
        return obj
    raise TypeError(f"cannot find a diffusion model in {type(obj).__name__}")


def _resolve_attr(root: nn.Module, path: str) -> tuple[Any, str]:
    *parents, leaf = path.split(".")
    owner: Any = root
    for name in parents:
        owner = getattr(owner, name)
    return owner, leaf


class PatchSet:
    """Install / restore instance-level attribute overrides (exactly reversible)."""

    def __init__(self) -> None:
        self._saved: list[tuple[Any, str, bool, Any]] = []

    @property
    def active(self) -> bool:
        return bool(self._saved)

    def apply(self, root: nn.Module, patches: Mapping[str, Callable[..., Any]]) -> None:
        if self._saved:
            raise RuntimeError("patches already applied")
        try:
            for path, fn in patches.items():
                owner, leaf = _resolve_attr(root, path)
                had = leaf in vars(owner)
                self._saved.append((owner, leaf, had, vars(owner).get(leaf)))
                setattr(owner, leaf, fn)
        except Exception:
            self.restore()
            raise

    def restore(self) -> None:
        while self._saved:
            owner, leaf, had, old = self._saved.pop()
            if had:
                setattr(owner, leaf, old)
            else:
                delattr(owner, leaf)


# --------------------------------------------------------------------------- step / session helpers


def maybe_step_context(options: Optional[Mapping[str, Any]], batch_size: int) -> Optional[StepContext]:
    """``StepContext`` from ``transformer_options`` or ``None`` outside a sampler call."""
    if not options or any(k not in options for k in _STEP_KEYS):
        return None
    return StepContext.from_transformer_options(options, batch_size)


def observers_active(session: XAISession, block: int, ctx: StepContext) -> bool:
    # ``_observers`` is the session's (blocks, window)-scoped observer registry.
    return any((blocks is None or block in blocks) and window.contains(ctx)
               for _, blocks, window in getattr(session, "_observers", ()))


def session_touches(session: Optional[XAISession], block: int, ctx: StepContext) -> bool:
    """Whether anything (capture, intervention, observer, concept stream) acts on ``(block, step)``.

    ``pre_dit_span_swap`` acts before the DiT and never forces the block slow path.
    ``concept_stream`` is active on *every* block inside its step window because the concept
    state must evolve through all blocks (``blocks`` only selects where it is read out).
    """
    if session is None or not session.enabled:
        return False
    cap = session.capture
    if cap is not None and cap.rows(ctx) and any(cap.wants(f, block, ctx) for f in cap.fields):
        return True
    for spec in session.interventions:
        if spec.hook == "pre_dit_span_swap":
            continue
        if spec.hook == "concept_stream":
            if spec.window.contains(ctx):
                return True
        elif spec.active(block, ctx) and spec.rows(ctx):
            return True
    return observers_active(session, block, ctx)


def active_specs(session: XAISession, hook: str, block: int, ctx: StepContext) -> list[InterventionSpec]:
    return [s for s in session.interventions_for(hook, block, ctx) if s.rows(ctx)]


def _row_source(src: Any, row: int, ctx: StepContext, batch: int) -> Tensor:
    """Pick the source row for batch row ``row``: tensor ``[1|B, ...]`` or ``{branch: [n, ...]}``."""
    if isinstance(src, Mapping):
        branch, sample = ctx.row_map()[row]
        t = src[branch]
        return t[sample if t.shape[0] > 1 else 0]
    if src.shape[0] == 1:
        return src[0]
    if src.shape[0] != batch:
        raise ValueError(f"source batch {src.shape[0]} matches neither 1 nor call batch {batch}")
    return src[row]


def span_swap(session: Optional[XAISession], point: str, tensor: Tensor,
              ctx: Optional[StepContext]) -> Tensor:
    """Apply every active ``pre_dit_span_swap`` at ``point`` to ``tensor [B, L, ...]`` (token axis 1).

    ``params["source"]`` is a tensor ``[1|B, L, ...]`` in the same coordinates, a mapping
    ``{"cond": [n, L, ...], "uncond": ...}`` keyed by branch (sample id indexes dim 0), or a
    callable ``(point=, ctx=, tensor=) -> tensor|mapping``. Sources must be length-matched
    (checked); token-ID alignment is the caller's job (``TokenMap.assert_strict_token_alignment``).
    """
    if session is None or ctx is None or not session.enabled:
        return tensor
    specs = [s for s in active_specs(session, "pre_dit_span_swap", PRE_DIT_BLOCK, ctx)
             if s.params["point"] == point]
    if not specs:
        return tensor
    out = tensor.clone()
    for spec in specs:
        src = spec.params["source"]
        if callable(src) and not isinstance(src, (Tensor, Mapping)):
            src = src(point=point, ctx=ctx, tensor=tensor)
        for row in spec.rows(ctx):
            s = _row_source(src, row, ctx, tensor.shape[0]).to(out.device, out.dtype)
            if s.shape != out.shape[1:]:
                raise ValueError(f"{point} swap source {tuple(s.shape)} != target {tuple(out.shape[1:])}; "
                                 "use a length-matched prompt")
            for sp in as_spans(spec.params["span"]):
                out[row, sp.start:sp.end] = s[sp.start:sp.end]
    return out


def kv_patch(session: XAISession, block: int, ctx: StepContext, k: Tensor, v: Tensor,
             layout: SeqLayout) -> tuple[Tensor, Tensor]:
    """Apply ``kv_patch`` specs to ``k/v [B, Hkv, Nk, d]`` (pre GQA repeat).

    ``params["source"]``: a :class:`~xai_dit.core.patching.KVCache` (``mode="patch"`` default,
    ``mode="store"`` records the source pass) or a callable
    ``(block=, ctx=, k=, v=, layout=, spec=) -> (k, v)``.
    """
    for spec in active_specs(session, "kv_patch", block, ctx):
        p = spec.params
        src = p["source"]
        mode = p.get("mode", "patch")
        if mode == "store":
            src.store(block, ctx, k, v, layout)
        elif mode != "patch":
            raise ValueError(f"unknown kv_patch mode {mode!r}")
        elif hasattr(src, "patch"):
            k, v = src.patch(block, ctx, k, v, layout, kv_heads=p.get("heads"), spans=p.get("keys"),
                             cfg_scope=spec.cfg_scope)
        elif callable(src):
            k, v = src(block=block, ctx=ctx, k=k, v=v, layout=layout, spec=spec)
        else:
            raise TypeError(f"kv_patch source must be a KVCache or callable, got {type(src).__name__}")
    return k, v


def token_slice(layout: SeqLayout, which: Any) -> slice:
    if which in (None, "all"):
        return slice(0, layout.n_query)
    if which == "img":
        return layout.q_img
    if which == "txt":
        if layout.q_txt is None:
            raise ValueError("no text queries in cross-attention")
        return layout.q_txt
    raise ValueError(f"unknown token selector {which!r} (use 'all' | 'img' | 'txt')")


def gate_override(gate_value: Tensor, specs: Sequence[InterventionSpec], ctx: StepContext,
                  layout: SeqLayout) -> Tensor:
    """E4-3: replace the post-sigmoid multiplier ``gate_value [B, N, C]`` on ``spec.rows``.

    ``mode="one"`` → ``1.0``; ``"mean"`` → ``value`` (broadcastable to ``[n, C]``) or, if
    ``None``, the per-sample token mean of the current ``gate_value``; ``"scale"`` →
    ``value * gate_value``. ``params["tokens"]`` restricts to ``"all"`` (default) | ``"img"`` | ``"txt"``.
    """
    if not specs:
        return gate_value
    out = gate_value.clone()
    for spec in specs:
        p = spec.params
        rows = torch.tensor(spec.rows(ctx), device=out.device)
        tok = token_slice(layout, p.get("tokens", "all"))
        cur = out.index_select(0, rows)
        seg = cur[:, tok]
        mode, value = p["mode"], p.get("value")
        if mode == "one":
            new = torch.ones_like(seg)
        elif mode == "mean":
            ref = seg.mean(dim=1, keepdim=True) if value is None else torch.as_tensor(value)
            new = ref.to(seg.device, seg.dtype).expand_as(seg)
        elif mode == "scale":
            if value is None:
                raise ValueError("gate_override mode='scale' needs params['value']")
            new = seg * torch.as_tensor(value).to(seg.device, seg.dtype)
        else:
            raise ValueError(f"unknown gate_override mode {mode!r}")
        cur[:, tok] = new
        out.index_copy_(0, rows, cur)
    return out


# --------------------------------------------------------------------------- forward-scoped state


@dataclass
class BlockState:
    block: int
    ctx: StepContext
    bypass: Optional[str]
    g1: Optional[Tensor] = None


@dataclass
class ForwardState:
    """State shared between hooks within one DiT forward (reset at the pre-DiT hook)."""

    text_len: Optional[int] = None
    options: Optional[Mapping[str, Any]] = None
    concepts: Optional[Tensor] = None
    block: Optional[BlockState] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def reset(self, options: Optional[Mapping[str, Any]] = None) -> None:
        self.text_len, self.options, self.concepts, self.block = None, options, None, None
        self.extra.clear()


# --------------------------------------------------------------------------- adapter base


def find_subsequence(token_ids: Sequence[int], sub: Sequence[int], *, occurrence: int = 0,
                     within: Optional[TokenSpan] = None) -> TokenSpan:
    """Half-open span of the ``occurrence``-th exact match of ``sub`` inside ``within``."""
    sub = [int(t) for t in sub]
    if not sub:
        raise ValueError("empty token subsequence")
    lo, hi = (0, len(token_ids)) if within is None else (within.start, within.end)
    seen = 0
    for i in range(lo, hi - len(sub) + 1):
        if [int(t) for t in token_ids[i:i + len(sub)]] == sub:
            if seen == occurrence:
                return TokenSpan(i, i + len(sub))
            seen += 1
    raise KeyError(f"token subsequence {sub} (occurrence {occurrence}) not found in [{lo}, {hi})")


class ComfyDiTAdapter(ModelAdapter):
    """Common attach/detach, object-patch export and session plumbing for ComfyUI DiTs."""

    #: Hooks this model cannot express (validated at attach).
    UNSUPPORTED_HOOKS: frozenset[str] = frozenset()
    #: Valid ``pre_dit_span_swap`` points.
    SWAP_POINTS: frozenset[str] = frozenset()
    #: Valid ``true_block_bypass`` modes.
    BYPASS_MODES: frozenset[str] = frozenset({"full_block", "attn_only", "mlp_only"})

    def __init__(self, diffusion_model: Any = None) -> None:
        self.model: Optional[nn.Module] = None
        self.session: Optional[XAISession] = None
        self._patches = PatchSet()
        self.fwd = ForwardState()
        self.block_calls: Counter[int] = Counter()   # block bodies executed (bypass invariant: 0)
        self.bypass_calls: Counter[int] = Counter()  # full-block bypasses taken
        self.concept_log: dict[tuple[int, int], dict[str, Tensor]] = {}
        if diffusion_model is not None:
            self.bind(diffusion_model)

    # ---- binding / patch export
    def bind(self, model: Any) -> "ComfyDiTAdapter":
        dm = resolve_diffusion_model(model)
        self._check_model(dm)
        self.model = dm
        return self

    @abc.abstractmethod
    def _check_model(self, dm: nn.Module) -> None:
        """Raise if ``dm`` is not this adapter's DiT class / block count."""

    @abc.abstractmethod
    def _build_patches(self, session: XAISession) -> dict[str, Callable[..., Any]]:
        """Instance ``forward`` overrides keyed by attribute path relative to the DiT."""

    def object_patches(self, session: XAISession, prefix: str = "diffusion_model.") -> dict[str, Callable[..., Any]]:
        """Patches for ``ModelPatcher.add_object_patch`` (``prefix`` relative to ``BaseModel``)."""
        self._require_model()
        self.validate_session(session)
        return {prefix + k: fn for k, fn in self._build_patches(session).items()}

    def validate_session(self, session: XAISession) -> None:
        if session.capability != self.capability:
            raise ValueError(f"session capability {session.capability.name!r} != adapter {self.capability.name!r}")
        for spec in session.interventions:
            if spec.hook in self.UNSUPPORTED_HOOKS:
                raise ValueError(f"{spec.hook} is structurally N/A for {self.capability.name}")
            if spec.hook == "pre_dit_span_swap" and spec.params["point"] not in self.SWAP_POINTS:
                raise ValueError(f"{self.capability.name} swap points are {sorted(self.SWAP_POINTS)}, "
                                 f"got {spec.params['point']!r}")
            if spec.hook == "true_block_bypass":
                mode = spec.params.get("mode", "full_block")
                if mode not in self.BYPASS_MODES:
                    raise ValueError(f"{self.capability.name} bypass modes are {sorted(self.BYPASS_MODES)}, got {mode!r}")
            if spec.hook == "concept_stream" and spec.params.get("sink") is not None and not callable(spec.params["sink"]):
                raise TypeError("concept_stream params['sink'] must be callable")

    # ---- ModelAdapter
    def attach(self, session: XAISession) -> None:
        if self.session is not None or self._patches.active:
            raise RuntimeError("adapter already attached; detach first")
        self._require_model()
        self.validate_session(session)
        self.fwd.reset()
        self._patches.apply(self.model, self._build_patches(session))
        self.session = session

    def detach(self) -> None:
        self._patches.restore()
        self.session = None
        self.fwd.reset()

    def _require_model(self) -> None:
        if self.model is None:
            raise RuntimeError("adapter is not bound to a diffusion model; call bind(model) first")

    # ---- token helpers
    def target_span(self, token_map: TokenMap, sub_ids: Sequence[int], *, occurrence: int = 0) -> TokenSpan:
        """Locate a tag's token IDs inside the prompt span ``S``."""
        return find_subsequence(token_map.token_ids, sub_ids, occurrence=occurrence, within=token_map.span("S"))

    def check_alignment(self, base: TokenMap, other: TokenMap,
                        editable: Sequence[TokenSpan | str] = ()) -> None:
        base.assert_strict_token_alignment(other, editable)

    # ---- shared hook steps
    def _record(self, session: XAISession, name: str, tensor: Tensor, block: int, ctx: StepContext) -> None:
        session.record(name, tensor, block=block, ctx=ctx)

    def _emit_concepts(self, session: XAISession, block: int, ctx: StepContext, o_c: Tensor,
                       o_x: Tensor, layout: SeqLayout, **extras: Tensor) -> None:
        """Hand concept-stream outputs to each spec's ``sink`` (or :attr:`concept_log`)."""
        for spec in session.interventions_for("concept_stream", block, ctx):
            rows = list(spec.rows(ctx))
            if not rows:
                continue
            idx = torch.tensor(rows, device=o_c.device)
            payload = {"o_c": o_c.index_select(0, idx), "o_x": o_x.index_select(0, idx)[:, layout.q_img]}
            payload.update({k: v.index_select(0, idx) for k, v in extras.items()})
            sink = spec.params.get("sink")
            if sink is not None:
                sink(block=block, ctx=ctx, rows=tuple(rows), **payload)
            else:  # keep only the small concept-side tensors
                self.concept_log[(ctx.step_index, block)] = {
                    k: v.detach().float().cpu() for k, v in payload.items() if k != "o_x"}

    def _concept_init(self, session: XAISession, ctx: StepContext, batch: int, dim: int,
                      like: Tensor) -> Optional[Tensor]:
        specs = [s for s in session.interventions if s.hook == "concept_stream" and s.window.contains(ctx)]
        if not specs or not session.enabled:
            return None
        if self.fwd.concepts is None:
            c = torch.as_tensor(specs[0].params["concepts"]).to(like.device, like.dtype)
            if c.ndim == 2:
                c = c.unsqueeze(0)
            if c.shape[-1] != dim:
                raise ValueError(f"concept states have dim {c.shape[-1]}, DiT width is {dim}")
            self.fwd.concepts = c.expand(batch, -1, -1).clone()
        return self.fwd.concepts


def cfg_batching_wrapper(policy: str) -> Callable[..., Any]:
    """``CALC_COND_BATCH`` wrapper enforcing one CFG batching policy for the whole run.

    ``split``: every branch is evaluated in its own forward while ``cond_or_uncond`` keeps the
    original branch labels (other branches are passed as ``None``). ``batched``: ComfyUI's
    default path; :meth:`XAISession.check_batching` then rejects any memory-driven split.
    """
    if policy not in ("split", "batched"):
        raise ValueError(f"unknown CFG batching policy {policy!r}")

    def wrapper(executor: Callable[..., Any], model: Any, conds: list, x_in: Tensor, timestep: Tensor,
                model_options: dict) -> list:
        if policy == "batched" or sum(c is not None for c in conds) <= 1:
            return executor(model, conds, x_in, timestep, model_options)
        outs: list = []
        for i, c in enumerate(conds):
            if c is None:
                outs.append(None)
                continue
            only = [c if j == i else None for j in range(len(conds))]
            outs.append(executor(model, only, x_in, timestep, model_options)[i])
        return outs

    return wrapper
