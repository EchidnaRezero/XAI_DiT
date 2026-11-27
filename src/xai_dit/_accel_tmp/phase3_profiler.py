"""Phase 3 reusable timing, CUDA kernel profiling, and Block-0 dtype audit helpers.

Pure-Python dataclasses, kernel categorization, and summary helpers have ZERO ``torch``
imports at module import time so CPU unit tests run without PyTorch.
"""

from __future__ import annotations

import contextlib
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence


# --------------------------------------------------------------------------- TimingBreakdown

@dataclass
class TimingBreakdown:
    """Decomposed wall-clock & VRAM profile for a single Phase 3 run."""

    t_te_encode_s: float = 0.0
    t_compile_warmup_s: float = 0.0
    t_denoise_pure_s: float = 0.0
    actual_forwards: int = 0
    nominal_forwards: int = 0
    s_per_fwd: float = 0.0
    skip_ratio: float = 0.0
    t_vae_decode_s: float = 0.0
    vram_peak_gb: float = 0.0
    vram_reserved_gb: float = 0.0

    def __post_init__(self) -> None:
        if self.s_per_fwd == 0.0 and self.actual_forwards > 0 and self.t_denoise_pure_s > 0.0:
            self.s_per_fwd = round(self.t_denoise_pure_s / self.actual_forwards, 4)
        if self.skip_ratio == 0.0 and self.nominal_forwards > 0 and self.actual_forwards >= 0:
            self.skip_ratio = round(max(0.0, 1.0 - (self.actual_forwards / self.nominal_forwards)), 4)

    @classmethod
    def from_measurements(
        cls,
        *,
        t_te_encode_s: float = 0.0,
        t_compile_warmup_s: float = 0.0,
        t_denoise_pure_s: float = 0.0,
        actual_forwards: int = 0,
        nominal_forwards: int = 0,
        t_vae_decode_s: float = 0.0,
        vram_peak_gb: float = 0.0,
        vram_reserved_gb: float = 0.0,
    ) -> "TimingBreakdown":
        s_per_fwd = round(t_denoise_pure_s / actual_forwards, 4) if actual_forwards > 0 else 0.0
        skip_ratio = (
            round(max(0.0, 1.0 - (actual_forwards / nominal_forwards)), 4)
            if nominal_forwards > 0
            else 0.0
        )
        return cls(
            t_te_encode_s=round(float(t_te_encode_s), 4),
            t_compile_warmup_s=round(float(t_compile_warmup_s), 4),
            t_denoise_pure_s=round(float(t_denoise_pure_s), 4),
            actual_forwards=int(actual_forwards),
            nominal_forwards=int(nominal_forwards),
            s_per_fwd=s_per_fwd,
            skip_ratio=skip_ratio,
            t_vae_decode_s=round(float(t_vae_decode_s), 4),
            vram_peak_gb=round(float(vram_peak_gb), 3),
            vram_reserved_gb=round(float(vram_reserved_gb), 3),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- CUDA Event Timer

@dataclass
class TimerResult:
    elapsed_s: float = 0.0
    used_cuda_events: bool = False


@contextlib.contextmanager
def cuda_timer(torch_mod: Optional[Any] = None) -> Iterator[TimerResult]:
    """Context manager measuring elapsed seconds via ``torch.cuda.Event(enable_timing=True)``.

    Falls back cleanly to ``time.perf_counter()`` if CUDA is unavailable.
    """
    res = TimerResult()
    if torch_mod is None:
        try:
            import torch as _torch  # type: ignore[import-not-found]
            torch_mod = _torch
        except ImportError:
            torch_mod = None

    use_cuda = bool(torch_mod is not None and torch_mod.cuda.is_available())
    if use_cuda:
        start_evt = torch_mod.cuda.Event(enable_timing=True)
        end_evt = torch_mod.cuda.Event(enable_timing=True)
        torch_mod.cuda.synchronize()
        start_evt.record()
        try:
            yield res
        finally:
            end_evt.record()
            torch_mod.cuda.synchronize()
            res.elapsed_s = float(start_evt.elapsed_time(end_evt)) / 1000.0
            res.used_cuda_events = True
    else:
        t0 = time.perf_counter()
        try:
            yield res
        finally:
            res.elapsed_s = time.perf_counter() - t0
            res.used_cuda_events = False


# --------------------------------------------------------------------------- Kernel Categorization (Pure Python)

_RE_DTYPE_CAST = re.compile(
    r"(aten::_to_copy|aten::copy_|_to_copy|copy_kernel|direct_copy_kernel|unrolled_elementwise.*copy|cast_)",
    re.IGNORECASE,
)
_RE_ATTN = re.compile(
    r"(flash_fwd|flash_bwd|fmha|sageattn|sage_attn|sdpa|scaled_dot_product|"
    r"efficient_attention|sol_attn|int8_attention|flex_attention|attention_)",
    re.IGNORECASE,
)
_RE_INT8_GEMM = re.compile(
    r"(_int8_mm|int8_mm|int8_gemm|i8_gemm|s8s8|i8i8|w8a8|w6a8|w4a8|w4a4|int4_mm|"
    r"cutlass.*(?:i8|s8|int8|s4|int4).*gemm|comfy_kitchen.*int[48]|cublas.*i8)",
    re.IGNORECASE,
)
_RE_FP8_GEMM = re.compile(
    r"(_scaled_mm|fp8_gemm|e4m3|e5m2|fp8.*gemm|nvfp4|mxfp8|f8f8|fp8_matmul|cublas.*fp8)",
    re.IGNORECASE,
)
_RE_BF16_GEMM = re.compile(
    r"(bf16.*gemm|bfloat16.*gemm|gemm.*bf16|ampere_bf16|ada_bf16|sm89_.*bf16|sm90_.*bf16|"
    r"cutlass.*bf16|nvjet_tst|cublas.*bf16|aten::mm|aten::addmm|aten::bmm|gemm_bf16|gemm)",
    re.IGNORECASE,
)


def categorize_kernel(name: str) -> str:
    """Classify a PyTorch/CUDA kernel name into one of the Phase 3 verification categories.

    Categories: ``int8_gemm``, ``fp8_gemm``, ``bf16_gemm``, ``attn``, ``dtype_cast``, ``other``.
    """
    if not name:
        return "other"
    if _RE_DTYPE_CAST.search(name):
        return "dtype_cast"
    if _RE_ATTN.search(name):
        return "attn"
    if _RE_INT8_GEMM.search(name):
        return "int8_gemm"
    if _RE_FP8_GEMM.search(name):
        return "fp8_gemm"
    if _RE_BF16_GEMM.search(name):
        return "bf16_gemm"
    return "other"


def summarize_kernel_events(
    events: Sequence[Mapping[str, Any]],
    top_k: int = 15,
) -> dict[str, Any]:
    """Aggregate raw kernel records ``[{name, cuda_ms, calls}]`` into a JSON-serializable summary."""
    normalized: list[dict[str, Any]] = []
    category_totals_ms: dict[str, float] = {
        "int8_gemm": 0.0,
        "fp8_gemm": 0.0,
        "bf16_gemm": 0.0,
        "attn": 0.0,
        "dtype_cast": 0.0,
        "other": 0.0,
    }
    category_calls: dict[str, int] = {k: 0 for k in category_totals_ms}

    for ev in events:
        name = str(ev.get("name", ""))
        cuda_ms = float(ev.get("cuda_ms", 0.0))
        calls = int(ev.get("calls", 0))
        cat = str(ev.get("category") or categorize_kernel(name))
        if cat not in category_totals_ms:
            category_totals_ms[cat] = 0.0
            category_calls[cat] = 0
        category_totals_ms[cat] += cuda_ms
        category_calls[cat] += calls
        normalized.append(
            {
                "name": name,
                "cuda_ms": round(cuda_ms, 4),
                "calls": calls,
                "category": cat,
            }
        )

    normalized.sort(key=lambda r: r["cuda_ms"], reverse=True)
    top_kernels = normalized[: max(1, int(top_k))]
    total_cuda_ms = sum(r["cuda_ms"] for r in normalized)

    return {
        "total_cuda_ms": round(total_cuda_ms, 4),
        "top_kernels": top_kernels,
        "category_totals_ms": {k: round(v, 4) for k, v in category_totals_ms.items()},
        "category_calls": category_calls,
        "dtype_cast_calls": category_calls["dtype_cast"],
        "dtype_cast_ms": round(category_totals_ms["dtype_cast"], 4),
        "has_int8_gemm": category_calls["int8_gemm"] > 0,
        "has_fp8_gemm": category_calls["fp8_gemm"] > 0,
        "has_bf16_gemm": category_calls["bf16_gemm"] > 0,
    }


def _extract_cuda_time_us(evt: Any) -> float:
    """Handle PyTorch 2.x profiler attribute rename (``device_time_total`` vs ``cuda_time_total``)."""
    for attr in ("self_device_time_total", "device_time_total", "self_cuda_time_total", "cuda_time_total"):
        val = getattr(evt, attr, None)
        if val is not None and float(val) > 0.0:
            return float(val)
    return 0.0


def profile_single_step_kernels(
    fn: Callable[[], Any],
    *,
    top_k: int = 15,
    torch_mod: Optional[Any] = None,
) -> tuple[Any, dict[str, Any]]:
    """Run ``fn()`` under ``torch.profiler.profile`` and return ``(result, kernel_summary)``.

    Captures CUDA kernels (plus CPU dispatch names so ``aten::_to_copy`` / ``aten::copy_``
    appear alongside device kernel times) and returns a structured JSON-serializable summary.
    """
    if torch_mod is None:
        import torch as _torch  # type: ignore[import-not-found]
        torch_mod = _torch

    activities = [torch_mod.profiler.ProfilerActivity.CPU]
    if torch_mod.cuda.is_available():
        activities.append(torch_mod.profiler.ProfilerActivity.CUDA)
        torch_mod.cuda.synchronize()

    with torch_mod.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        result = fn()
        if torch_mod.cuda.is_available():
            torch_mod.cuda.synchronize()

    raw_events: list[dict[str, Any]] = []
    for evt in prof.key_averages():
        cuda_us = _extract_cuda_time_us(evt)
        name = str(getattr(evt, "key", ""))
        calls = int(getattr(evt, "count", 0))
        cat = categorize_kernel(name)
        # Keep events that either spent GPU time or are explicit dtype casts (aten::_to_copy)
        if cuda_us > 0.0 or cat == "dtype_cast":
            raw_events.append(
                {
                    "name": name,
                    "cuda_ms": cuda_us / 1000.0,
                    "calls": calls,
                    "category": cat,
                }
            )

    summary = summarize_kernel_events(raw_events, top_k=top_k)
    return result, summary


# --------------------------------------------------------------------------- Block-0 Dtype & Backend Audit

def _locate_block0(diffusion_model: Any) -> tuple[str, Optional[Any]]:
    for attr in ("blocks", "layers", "double_blocks", "single_blocks"):
        cand = getattr(diffusion_model, attr, None)
        if cand is not None and hasattr(cand, "__len__") and len(cand) > 0:
            return f"{attr}[0]", cand[0]
    return "diffusion_model", diffusion_model


def _detect_comfy_kitchen_cuda() -> dict[str, Any]:
    info: dict[str, Any] = {"installed": False, "cuda_backend_active": False, "backends": []}
    try:
        import comfy_kitchen as ck  # type: ignore[import-not-found]

        info["installed"] = True
        info["version"] = str(getattr(ck, "__version__", "unknown"))
        if hasattr(ck, "list_backends"):
            backends = ck.list_backends()
            if isinstance(backends, dict):
                info["backends"] = {
                    k: bool(v.get("available", False)) if isinstance(v, dict) else bool(v)
                    for k, v in backends.items()
                }
                cuda_entry = backends.get("cuda")
                info["cuda_backend_active"] = bool(
                    cuda_entry.get("available", False) if isinstance(cuda_entry, dict) else cuda_entry
                )
            else:
                info["backends"] = list(backends)
                info["cuda_backend_active"] = "cuda" in [str(b).lower() for b in backends]
        elif hasattr(ck, "registry") and hasattr(ck.registry, "list_backends"):
            backends = list(ck.registry.list_backends())
            info["backends"] = backends
            info["cuda_backend_active"] = any("cuda" in str(b).lower() for b in backends)
    except ImportError:
        pass
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def audit_block0_dtypes(
    patcher: Any,
    *,
    torch_mod: Optional[Any] = None,
    mm_mod: Optional[Any] = None,
) -> dict[str, Any]:
    """Audit Block-0 weight/compute dtypes, ``PRIORITIZE_FP16``, ``allow_fp16_accumulation``, and ``comfy_kitchen``."""
    if torch_mod is None:
        try:
            import torch as _torch  # type: ignore[import-not-found]
            torch_mod = _torch
        except ImportError:
            torch_mod = None
    if mm_mod is None:
        try:
            import comfy.model_management as _mm  # type: ignore[import-not-found]
            mm_mod = _mm
        except ImportError:
            mm_mod = None

    dm = patcher.model.diffusion_model
    block_label, block0 = _locate_block0(dm)

    first_linear_name: Optional[str] = None
    weight_dtype: Optional[str] = None
    weight_cls: Optional[str] = None
    quant_layout: Optional[str] = None
    if block0 is not None and hasattr(block0, "named_modules"):
        for mod_name, mod in block0.named_modules():
            w = getattr(mod, "weight", None)
            if w is not None and getattr(w, "ndim", 0) >= 2:
                first_linear_name = f"{block_label}.{mod_name}" if mod_name else block_label
                weight_dtype = str(getattr(w, "dtype", "unknown"))
                weight_cls = type(w).__name__
                quant_layout = str(getattr(w, "_layout_type", getattr(w, "layout_type", None)))
                break

    model_dtype = None
    if hasattr(patcher.model, "get_dtype"):
        try:
            model_dtype = str(patcher.model.get_dtype())
        except Exception:
            model_dtype = None
    manual_cast_dtype = str(getattr(patcher.model, "manual_cast_dtype", None))

    prioritize_fp16 = bool(getattr(mm_mod, "PRIORITIZE_FP16", False)) if mm_mod is not None else None
    allow_fp16_acc = None
    if torch_mod is not None and hasattr(torch_mod, "backends") and hasattr(torch_mod.backends, "cuda"):
        allow_fp16_acc = bool(getattr(torch_mod.backends.cuda.matmul, "allow_fp16_accumulation", False))

    return {
        "block0_label": block_label,
        "first_linear": first_linear_name,
        "weight_dtype": weight_dtype,
        "weight_class": weight_cls,
        "quant_layout": quant_layout if quant_layout != "None" else None,
        "model_dtype": model_dtype,
        "compute_dtype": manual_cast_dtype if manual_cast_dtype != "None" else model_dtype,
        "PRIORITIZE_FP16": prioritize_fp16,
        "allow_fp16_accumulation": allow_fp16_acc,
        "comfy_kitchen": _detect_comfy_kitchen_cuda(),
    }


def attach_block0_runtime_dtype_hook(patcher: Any) -> tuple[dict[str, Any], Callable[[], None]]:
    """Attach a one-shot forward hook on Block 0 to capture actual step-0 ``x.dtype`` and ``out.dtype``."""
    dm = patcher.model.diffusion_model
    block_label, block0 = _locate_block0(dm)
    captured: dict[str, Any] = {"block0_label": block_label, "x_dtype": None, "out_dtype": None, "captured": False}

    if block0 is None or not hasattr(block0, "register_forward_hook"):
        return captured, (lambda: None)

    def _hook(_mod: Any, inp: Any, out: Any) -> None:
        if captured["captured"]:
            return
        if isinstance(inp, (tuple, list)) and len(inp) > 0 and hasattr(inp[0], "dtype"):
            captured["x_dtype"] = str(inp[0].dtype)
            captured["x_shape"] = list(inp[0].shape)
        t_out = out[0] if isinstance(out, (tuple, list)) else out
        if hasattr(t_out, "dtype"):
            captured["out_dtype"] = str(t_out.dtype)
        captured["captured"] = True

    handle = block0.register_forward_hook(_hook)
    return captured, handle.remove
