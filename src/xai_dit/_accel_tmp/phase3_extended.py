"""Phase 3 Extended Acceleration Sweep Runner (Tier 1 Colab L4 & Tier 2 Blackwell).

Supports:
  - Tier-filtered presets (``--tier {l4,blackwell}``) across all 4 models
    (``anima``, ``moss_image``, ``krea2``, ``krea2_anime``) and 5 tracks
    (``quant``, ``cache``, ``cfg_attn``, ``distill``, ``compile``, ``all``).
  - Built-in **Scheduled CFG** wrapper (drops CFG to ``1.0`` after ``cfg_drop_percent``
    via ``WrappersMP.PREDICT_NOISE``, skipping ``uncond`` cleanly while preserving ``cond``).
  - Built-in **BlockSparseAttention** (``sol-attn``, ``tau=1.3``) via ComfyUI's
    ``comfy_extras/nodes_sparse_attention.py``.
  - Quantization weight overrides (Anima INT8 ConvRot/Rowwise, MOSS HMS W6A8+W4A8,
    Krea 2 Raw INT8 ConvRot, Krea2-Anime v3.0 FP8/INT8/W4A4, Blackwell NVFP4/MXFP8).
  - Distillation LoRAs, ``manual_sigmas`` (MOSS 4-step eDMD), and ``Raw -> Turbo`` 2-stage handoff.
  - L4-safe ``torch.compile(mode="max-autotune-no-cudagraphs", dynamic=False)`` & Regional/Block
    compile, plus Blackwell ``max-autotune+cudagraphs``, ``FlexAttention``, and ``SageAttention 3``.
  - Graceful optional detection for community custom nodes (skips cleanly with ``SKIPPED`` status
    if not installed in ``ComfyUI/custom_nodes/`` or if optional weights are absent).
  - Full integration with ``metrics_and_exif.py`` (3-chunk PNG EXIF, ``summary.json``,
    ``summary.md``, ``contact_sheet.png``) and ``phase3_profiler.py`` (``TimingBreakdown``).

Pure-Python preset catalog, Scheduled CFG math, and tier/track filtering have ZERO ``torch``
imports at module import time so CPU unit tests run without PyTorch.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..runners.cli import (
    MODEL_DEFAULTS,
    REPO_ROOT,
    comfyui_commit,
    empty_latent,
    load_local_env,
    log,
    resolve_weight,
    unload_all,
    with_shift,
    write_json,
)
from .metrics_and_exif import (
    build_comfy_prompt_graph,
    compute_image_metrics,
    load_png_float01,
    make_comparison_sheet,
    save_png_with_exif,
)
from .phase3_profiler import (
    TimingBreakdown,
    audit_block0_dtypes,
    cuda_timer,
    profile_single_step_kernels,
)
from .presets import SAFE_FAST_FEATURES, get_prompt
from .run_t2i import (
    apply_easycache,
    apply_lora_model_only,
    bootstrap_with_accel,
    ensure_conds,
    zero_out_conditioning,
)


# --------------------------------------------------------------------------- Pure-Python Scheduled CFG Math

def compute_scheduled_cfg_value(
    step_idx: int,
    total_steps: int,
    base_cfg: float,
    cfg_drop_percent: float,
    dropped_cfg: float = 1.0,
) -> float:
    """Return effective CFG at 0-based ``step_idx`` out of ``total_steps``.

    Once ``step_idx / max(total_steps, 1) >= cfg_drop_percent``, CFG drops to ``dropped_cfg`` (1.0)
    so ComfyUI's ``sampling_function`` skips the ``uncond`` forward pass cleanly.
    """
    if total_steps <= 0 or base_cfg <= 1.0:
        return float(base_cfg)
    progress = step_idx / float(total_steps)
    if progress >= cfg_drop_percent:
        return float(dropped_cfg)
    return float(base_cfg)


def compute_cfg_schedule_plan(
    total_steps: int,
    base_cfg: float,
    cfg_drop_percent: float,
    dropped_cfg: float = 1.0,
) -> dict[str, Any]:
    """Compute per-step CFG values and expected forward-pass savings (pure Python)."""
    schedule = [
        compute_scheduled_cfg_value(i, total_steps, base_cfg, cfg_drop_percent, dropped_cfg)
        for i in range(max(0, total_steps))
    ]
    nominal_fwds = max(0, total_steps) * (2 if base_cfg > 1.0 else 1)
    expected_fwds = sum(2 if c > 1.0 else 1 for c in schedule)
    skip_ratio = round(max(0.0, 1.0 - (expected_fwds / nominal_fwds)), 4) if nominal_fwds > 0 else 0.0
    return {
        "total_steps": total_steps,
        "base_cfg": base_cfg,
        "cfg_drop_percent": cfg_drop_percent,
        "dropped_cfg": dropped_cfg,
        "per_step_cfg": schedule,
        "nominal_forwards": nominal_fwds,
        "expected_forwards": expected_fwds,
        "expected_skip_ratio": skip_ratio,
    }


# --------------------------------------------------------------------------- Optional Custom Node Detection

KNOWN_CUSTOM_NODES: dict[str, tuple[str, ...]] = {
    "spectrum_ksampler": ("ComfyUI-Spectrum-KSampler", "ComfyUI_Spectrum_KSampler"),
    "spectrum_moss": ("ComfyUI-Spectrum-MOSS", "ComfyUI_Spectrum_MOSS"),
    "anima_teacache": ("ComfyUI-Anima-TeaCache", "ComfyUI_Anima_TeaCache"),
    "moss_accel": ("comfyui-moss-accel", "ComfyUI-MOSS-Accel"),
    "kjnodes": ("ComfyUI-KJNodes", "ComfyUI_KJNodes"),
    "krea2_accel": ("ComfyUI-Krea2-Accel", "comfyui-krea2-accel"),
}


def detect_custom_nodes(comfyui_dir: Optional[str | Path]) -> dict[str, Optional[Path]]:
    """Scan ``<comfyui_dir>/custom_nodes`` for optional Phase 3 community nodes."""
    found: dict[str, Optional[Path]] = {k: None for k in KNOWN_CUSTOM_NODES}
    if not comfyui_dir:
        return found
    cn_root = Path(comfyui_dir).expanduser().resolve() / "custom_nodes"
    if not cn_root.is_dir():
        return found
    existing = {p.name.lower(): p for p in cn_root.iterdir() if p.is_dir()}
    for key, candidates in KNOWN_CUSTOM_NODES.items():
        for cand in candidates:
            hit = existing.get(cand.lower())
            if hit is not None:
                found[key] = hit
                break
    return found


# --------------------------------------------------------------------------- Extended Preset Catalog (Pure Python)

@dataclass(frozen=True)
class ExtendedPreset:
    name: str
    model: str  # "anima" | "moss_image" | "krea2" | "krea2_anime"
    tier: str  # "l4" | "blackwell"
    track: str  # "baseline" | "quant" | "cache" | "cfg_attn" | "distill" | "compile"
    attention: str = "pytorch"  # "pytorch" | "ck" | "flash" | "sage" | "sage3" | "flex"
    fast_features: tuple[str, ...] = ()
    dit_override: Optional[str] = None
    lora: Optional[str] = None
    lora_strength: float = 1.0
    steps: Optional[int] = None
    cfg: Optional[float] = None
    shift: Optional[float] = None
    sampler: Optional[str] = None
    scheduler: Optional[str] = None
    zero_negative: bool = False
    manual_sigmas: Optional[tuple[float, ...]] = None
    # Two-stage Raw -> Turbo handoff
    two_stage_dit: Optional[str] = None
    two_stage_split_step: Optional[int] = None
    two_stage_cfg: float = 1.0
    # Built-in Phase 3 accelerations
    easycache: bool = False
    easycache_threshold: float = 0.20
    easycache_start: float = 0.15
    easycache_end: float = 0.95
    cfg_drop_percent: Optional[float] = None
    block_sparse_tau: Optional[float] = None
    block_sparse_start: float = 0.20
    block_sparse_end: float = 1.00
    block_sparse_min_tokens: int = 1024
    # Compiler options
    compile_mode: Optional[str] = None  # "max-autotune-no-cudagraphs" | "max-autotune" | "regional_blocks"
    compile_dynamic: bool = False
    # Blackwell NVFP4 metadata check
    nvfp4_full_precision_mm: Optional[bool] = None
    # Optional community custom node requirement
    requires_custom_node: Optional[str] = None
    custom_node_params: tuple[tuple[str, Any], ...] = ()
    notes: str = ""

    def process_flag_key(self) -> tuple[Any, ...]:
        return (self.attention, tuple(sorted(self.fast_features)))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fast_features"] = list(self.fast_features)
        d["manual_sigmas"] = list(self.manual_sigmas) if self.manual_sigmas else None
        d["custom_node_params"] = dict(self.custom_node_params)
        return d


# Krea2-Anime v3.0 shares Krea 2's TE/VAE/clip_type with Turbo 8-step defaults
KREA2_ANIME_DEFAULTS = SimpleNamespace(
    name="krea2_anime",
    dit="krea2_anime_v3_fp8_e4m3.safetensors",
    te=MODEL_DEFAULTS["krea2"].te,
    vae=MODEL_DEFAULTS["krea2"].vae,
    precision="fp8_e4m3",
    clip_type=MODEL_DEFAULTS["krea2"].clip_type,
    steps=8,
    cfg=1.0,
    shift=1.15,
    sampler="euler",
    scheduler="simple",
    negative="",
)


def get_model_defaults(model: str) -> Any:
    if model == "krea2_anime":
        return KREA2_ANIME_DEFAULTS
    return MODEL_DEFAULTS[model]


def build_extended_presets(
    model: str,
    tier: str = "l4",
    track: str = "all",
) -> list[ExtendedPreset]:
    """Return ordered Phase 3 extended presets filtered by ``model``, ``tier``, and ``track``."""
    all_presets: list[ExtendedPreset] = []

    # Always define an L4/tier baseline per model
    if model == "anima":
        all_presets.extend([
            ExtendedPreset("baseline", "anima", "l4", "baseline", notes="Anima BF16 SDPA baseline"),
            # Quantization (T1-1 & T2-3)
            ExtendedPreset("quant_int8_convrot", "anima", "l4", "quant", dit_override="anima-base-v1.0-int8-convrot.safetensors", notes="Anima INT8 ConvRot (ruwwww/Bedovyy)"),
            ExtendedPreset("quant_int8_rowwise", "anima", "l4", "quant", dit_override="anima-base-v1.0-int8-rowwise.safetensors", notes="Anima INT8 Rowwise (ComfyUI-INT8-Fast)"),
            ExtendedPreset("quant_mxfp8", "anima", "blackwell", "quant", dit_override="anima-base-v1.0-mxfp8.safetensors", notes="Blackwell MXFP8 vs INT8 ConvRot"),
            # Caching & Forecasting (T1-4)
            ExtendedPreset("cache_easycache_20", "anima", "l4", "cache", easycache=True, easycache_threshold=0.20, easycache_start=0.15, notes="EasyCache (t=0.20, s=0.15)"),
            ExtendedPreset("cache_spectrum_ks", "anima", "l4", "cache", requires_custom_node="spectrum_ksampler", custom_node_params=(("degree", 3), ("reg", 0.5), ("warmup", 8), ("stop", 0.8)), notes="Spectrum-KSampler (d=3, reg=0.5, w=8, stop=0.8)"),
            ExtendedPreset("cache_anima_teacache_0075", "anima", "l4", "cache", requires_custom_node="anima_teacache", custom_node_params=(("rel_l1_thresh", 0.075),), notes="Anima-TeaCache (thresh=0.075)"),
            # CFG & Attention (T1-7 & T2-1/T2-4)
            ExtendedPreset("cfg_scheduled_60", "anima", "l4", "cfg_attn", cfg_drop_percent=0.60, notes="Scheduled CFG: drop CFG->1.0 at 60%"),
            ExtendedPreset("cfg_scheduled_65", "anima", "l4", "cfg_attn", cfg_drop_percent=0.65, notes="Scheduled CFG: drop CFG->1.0 at 65%"),
            ExtendedPreset("cfg_sched60_ec20", "anima", "l4", "cfg_attn", cfg_drop_percent=0.60, easycache=True, easycache_threshold=0.20, easycache_start=0.15, notes="Scheduled CFG (60%) + EasyCache (t=0.20)"),
            ExtendedPreset("attn_block_sparse_sol13", "anima", "l4", "cfg_attn", block_sparse_tau=1.3, notes="ComfyUI BlockSparseAttention (sol-attn, tau=1.3)"),
            ExtendedPreset("attn_sage2_kj", "anima", "l4", "cfg_attn", attention="sage", requires_custom_node="kjnodes", notes="SageAttention 2 (KJNodes)"),
            ExtendedPreset("attn_sage3_fp4", "anima", "blackwell", "cfg_attn", attention="sage3", notes="Blackwell SageAttention 3 (FP4)"),
            ExtendedPreset("attn_flex", "anima", "blackwell", "cfg_attn", attention="flex", notes="Blackwell PyTorch FlexAttention"),
            # Distillation (T1-8)
            ExtendedPreset("distill_acc_lora_16s", "anima", "l4", "distill", lora="anima_16step_acc_lora.safetensors", steps=16, cfg=3.5, notes="Anima 16-step Acc LoRA"),
            ExtendedPreset("distill_turbo_v11_8s", "anima", "l4", "distill", lora="anima_turbo_v1.1.safetensors", steps=8, cfg=1.0, zero_negative=True, notes="Anima Turbo v1.1 (8 steps, CFG 1.0)"),
            ExtendedPreset("distill_turbo_4step", "anima", "l4", "distill", lora="anima_turbo_4step_v4.safetensors", steps=4, cfg=1.0, sampler="er_sde", zero_negative=True, notes="Anima Turbo 4-step (er_sde, CFG 1.0)"),
            # Compiler (T1-9 & T2-4)
            ExtendedPreset("compile_max_autotune_no_cg", "anima", "l4", "compile", compile_mode="max-autotune-no-cudagraphs", compile_dynamic=False, notes="L4-safe torch.compile(max-autotune-no-cudagraphs, dynamic=False)"),
            ExtendedPreset("compile_regional_blocks", "anima", "l4", "compile", compile_mode="regional_blocks", compile_dynamic=False, notes="L4 Regional/Block torch.compile"),
            ExtendedPreset("compile_max_autotune_cg", "anima", "blackwell", "compile", compile_mode="max-autotune", compile_dynamic=False, notes="Blackwell torch.compile(max-autotune + cudagraphs)"),
        ])
    elif model == "moss_image":
        all_presets.extend([
            ExtendedPreset("baseline", "moss_image", "l4", "baseline", notes="MOSS-Image SFT INT8 ConvRot baseline"),
            # Quantization (T1-2)
            ExtendedPreset("quant_hms_w6a8_w4a8", "moss_image", "l4", "quant", dit_override="moss_image_hms_v1.safetensors", notes="MOSS-Image HMS v1 (W6A8 Attn/FFN30-41 + W4A8 FFN0-29)"),
            ExtendedPreset("quant_moss_heal", "moss_image", "l4", "quant", dit_override="moss_image_heal.safetensors", notes="MOSS-Image-heal checkpoint"),
            # Caching & Forecasting (T1-5)
            ExtendedPreset("cache_easycache_20", "moss_image", "l4", "cache", easycache=True, easycache_threshold=0.20, easycache_start=0.15, notes="EasyCache (t=0.20, s=0.15)"),
            ExtendedPreset("cache_moss_accel_fast", "moss_image", "l4", "cache", requires_custom_node="moss_accel", custom_node_params=(("mode", "fast"),), notes="comfyui-moss-accel (fast CSF)"),
            ExtendedPreset("cache_spectrum_moss", "moss_image", "l4", "cache", requires_custom_node="spectrum_moss", custom_node_params=(("adaptive", True),), notes="Spectrum-MOSS (Adaptive)"),
            ExtendedPreset("cache_spectrum_ks", "moss_image", "l4", "cache", requires_custom_node="spectrum_ksampler", custom_node_params=(("degree", 3), ("reg", 0.5), ("warmup", 6), ("stop", 0.8)), notes="Spectrum-KSampler v2.10.1"),
            # CFG & Attention (T1-7 & T2-1/T2-4)
            ExtendedPreset("cfg_scheduled_60", "moss_image", "l4", "cfg_attn", cfg_drop_percent=0.60, notes="Scheduled CFG: drop CFG->1.0 at 60%"),
            ExtendedPreset("cfg_scheduled_65", "moss_image", "l4", "cfg_attn", cfg_drop_percent=0.65, notes="Scheduled CFG: drop CFG->1.0 at 65%"),
            ExtendedPreset("attn_block_sparse_sol13", "moss_image", "l4", "cfg_attn", block_sparse_tau=1.3, notes="ComfyUI BlockSparseAttention (sol-attn, tau=1.3)"),
            ExtendedPreset("attn_sage2_kj", "moss_image", "l4", "cfg_attn", attention="sage", requires_custom_node="kjnodes", notes="SageAttention 2 (KJNodes)"),
            ExtendedPreset("attn_sage3_fp4", "moss_image", "blackwell", "cfg_attn", attention="sage3", notes="Blackwell SageAttention 3 (FP4)"),
            ExtendedPreset("attn_flex", "moss_image", "blackwell", "cfg_attn", attention="flex", notes="Blackwell PyTorch FlexAttention"),
            # Distillation (T1-8)
            ExtendedPreset("distill_fast12_opd_12s", "moss_image", "l4", "distill", lora="moss_fast12_opd.safetensors", steps=12, cfg=1.0, sampler="euler", scheduler="normal", zero_negative=True, notes="MOSS Fast-12 OPD (12s, euler/normal)"),
            ExtendedPreset("distill_dmd_8s", "moss_image", "l4", "distill", lora="moss_lodestones_dmd_8s.safetensors", steps=8, cfg=1.0, sampler="euler", scheduler="simple", zero_negative=True, notes="MOSS lodestones 8s DMD"),
            ExtendedPreset("distill_edmd_4s_manual_sigmas", "moss_image", "l4", "distill", lora="moss_lvladikov_edmd_4s.safetensors", steps=4, cfg=1.0, sampler="euler", zero_negative=True, manual_sigmas=(1.0, 0.78, 0.50, 0.22, 0.0), notes="MOSS lvladikov 4s eDMD (manual_sigmas)"),
            # Compiler (T1-9 & T2-4)
            ExtendedPreset("compile_max_autotune_no_cg", "moss_image", "l4", "compile", compile_mode="max-autotune-no-cudagraphs", compile_dynamic=False, notes="L4-safe torch.compile(max-autotune-no-cudagraphs)"),
            ExtendedPreset("compile_regional_blocks", "moss_image", "l4", "compile", compile_mode="regional_blocks", compile_dynamic=False, notes="L4 Regional/Block torch.compile"),
            ExtendedPreset("compile_max_autotune_cg", "moss_image", "blackwell", "compile", compile_mode="max-autotune", compile_dynamic=False, notes="Blackwell torch.compile(max-autotune + cudagraphs)"),
        ])
    elif model == "krea2":
        all_presets.extend([
            ExtendedPreset("baseline", "krea2", "l4", "baseline", notes="Krea 2 Raw FP8 scaled SDPA baseline"),
            # Quantization (T1-3 & T2-2/T2-3)
            ExtendedPreset("quant_raw_int8_convrot", "krea2", "l4", "quant", dit_override="krea2_raw_int8_convrot.safetensors", notes="Krea 2 Raw INT8 ConvRot (Comfy-Org)"),
            ExtendedPreset("quant_turbo_nvfp4_stock", "krea2", "blackwell", "quant", dit_override="krea2_turbo_nvfp4.safetensors", steps=8, cfg=1.0, zero_negative=True, nvfp4_full_precision_mm=False, notes="Blackwell NVFP4 stock metadata"),
            ExtendedPreset("quant_turbo_nvfp4_fp_mm", "krea2", "blackwell", "quant", dit_override="krea2_turbo_nvfp4.safetensors", steps=8, cfg=1.0, zero_negative=True, nvfp4_full_precision_mm=True, notes="Blackwell NVFP4 + full_precision_matrix_mult=True (#25)"),
            ExtendedPreset("quant_turbo_mxfp8", "krea2", "blackwell", "quant", dit_override="krea2_turbo_mxfp8.safetensors", steps=8, cfg=1.0, zero_negative=True, notes="Blackwell Krea 2 Turbo MXFP8"),
            # Caching & Forecasting (T1-6)
            ExtendedPreset("cache_easycache_20", "krea2", "l4", "cache", easycache=True, easycache_threshold=0.20, easycache_start=0.25, notes="EasyCache (t=0.20, s=0.25)"),
            ExtendedPreset("cache_spectrum_ks_w20", "krea2", "l4", "cache", requires_custom_node="spectrum_ksampler", custom_node_params=(("degree", 3), ("reg", 0.5), ("warmup", 7), ("stop", 0.85)), notes="Spectrum-KSampler (>=20% warmup)"),
            ExtendedPreset("cache_krea2_accel", "krea2", "l4", "cache", requires_custom_node="krea2_accel", notes="ComfyUI-Krea2-Accel"),
            # CFG & Attention (T1-7 & T2-1/T2-4)
            ExtendedPreset("cfg_scheduled_60", "krea2", "l4", "cfg_attn", cfg_drop_percent=0.60, notes="Scheduled CFG: drop CFG->1.0 at 60%"),
            ExtendedPreset("cfg_scheduled_65", "krea2", "l4", "cfg_attn", cfg_drop_percent=0.65, notes="Scheduled CFG: drop CFG->1.0 at 65%"),
            ExtendedPreset("attn_block_sparse_sol13", "krea2", "l4", "cfg_attn", block_sparse_tau=1.3, notes="ComfyUI BlockSparseAttention (sol-attn, tau=1.3)"),
            ExtendedPreset("attn_sage3_fp4", "krea2", "blackwell", "cfg_attn", attention="sage3", notes="Blackwell SageAttention 3 (FP4, selective bypass)"),
            ExtendedPreset("attn_flex", "krea2", "blackwell", "cfg_attn", attention="flex", notes="Blackwell PyTorch FlexAttention"),
            # Distillation & 2-stage (T1-8)
            ExtendedPreset("distill_raw_to_turbo_2stage", "krea2", "l4", "distill", steps=8, cfg=4.0, two_stage_dit="krea2_turbo_fp8_scaled.safetensors", two_stage_split_step=3, two_stage_cfg=1.0, notes="2-stage handoff: Raw (steps 0..3, CFG 4) -> Turbo (steps 3..8, CFG 1)"),
            ExtendedPreset("distill_svd_8s_lora", "krea2", "l4", "distill", lora="krea2_divergent_8step_svd.safetensors", steps=8, cfg=1.0, zero_negative=True, notes="TheDivergentAI 8-step SVD LoRA on Raw"),
            ExtendedPreset("distill_turbo_4s_lora", "krea2", "l4", "distill", dit_override="krea2_turbo_fp8_scaled.safetensors", lora="krea2_lvladikov_4step_turbo.safetensors", steps=4, cfg=1.0, zero_negative=True, notes="lvladikov 4-step LoRA on Krea 2 Turbo"),
            # Compiler (T1-9 & T2-4)
            ExtendedPreset("compile_max_autotune_no_cg", "krea2", "l4", "compile", compile_mode="max-autotune-no-cudagraphs", compile_dynamic=False, notes="L4-safe torch.compile(max-autotune-no-cudagraphs)"),
            ExtendedPreset("compile_max_autotune_cg", "krea2", "blackwell", "compile", compile_mode="max-autotune", compile_dynamic=False, notes="Blackwell torch.compile(max-autotune + cudagraphs)"),
        ])
    elif model == "krea2_anime":
        all_presets.extend([
            ExtendedPreset("baseline", "krea2_anime", "l4", "baseline", dit_override="krea2_anime_v3_fp8_e4m3.safetensors", steps=8, cfg=1.0, zero_negative=True, notes="Krea2-Anime v3.0 FP8 E4M3 baseline (8s, CFG 1.0)"),
            ExtendedPreset("quant_krea2_anime_int8_convrot", "krea2_anime", "l4", "quant", dit_override="krea2_anime_v3_int8_convrot.safetensors", steps=8, cfg=1.0, zero_negative=True, notes="Krea2-Anime v3.0 INT8 ConvRot (12.57 GB)"),
            ExtendedPreset("quant_krea2_anime_int4_w4a4", "krea2_anime", "l4", "quant", dit_override="krea2_anime_v3_int4_w4a4.safetensors", steps=8, cfg=1.0, zero_negative=True, notes="Krea2-Anime v3.0 INT4 W4A4 (6.91 GB, comfy-kitchen)"),
            ExtendedPreset("cache_easycache_20", "krea2_anime", "l4", "cache", dit_override="krea2_anime_v3_fp8_e4m3.safetensors", steps=8, cfg=1.0, zero_negative=True, easycache=True, easycache_threshold=0.20, easycache_start=0.25, notes="Krea2-Anime v3.0 + EasyCache"),
            ExtendedPreset("attn_block_sparse_sol13", "krea2_anime", "l4", "cfg_attn", dit_override="krea2_anime_v3_fp8_e4m3.safetensors", steps=8, cfg=1.0, zero_negative=True, block_sparse_tau=1.3, notes="Krea2-Anime v3.0 + BlockSparseAttention (sol-attn)"),
            ExtendedPreset("distill_krea2_anime_turbo_4s_lora", "krea2_anime", "l4", "distill", dit_override="krea2_anime_v3_fp8_e4m3.safetensors", lora="krea2_lvladikov_4step_turbo.safetensors", steps=4, cfg=1.0, zero_negative=True, notes="Krea2-Anime v3.0 + lvladikov 4-step Turbo LoRA"),
            ExtendedPreset("compile_max_autotune_no_cg", "krea2_anime", "l4", "compile", dit_override="krea2_anime_v3_fp8_e4m3.safetensors", steps=8, cfg=1.0, zero_negative=True, compile_mode="max-autotune-no-cudagraphs", compile_dynamic=False, notes="Krea2-Anime v3.0 L4-safe torch.compile"),
            ExtendedPreset("compile_max_autotune_cg", "krea2_anime", "blackwell", "compile", dit_override="krea2_anime_v3_fp8_e4m3.safetensors", steps=8, cfg=1.0, zero_negative=True, compile_mode="max-autotune", compile_dynamic=False, notes="Krea2-Anime v3.0 Blackwell torch.compile + cudagraphs"),
        ])
    else:
        raise ValueError(f"Unsupported model: {model!r}")

    # Filter by tier
    if tier == "l4":
        tier_filtered = [p for p in all_presets if p.tier == "l4"]
    elif tier == "blackwell":
        # Blackwell tier includes baseline (for reference) + blackwell-specific presets
        tier_filtered = [p for p in all_presets if p.tier == "blackwell" or p.track == "baseline"]
    else:
        raise ValueError(f"Unsupported tier: {tier!r}; choices: 'l4', 'blackwell'")

    # Filter by track
    if track == "all":
        return tier_filtered
    wanted = {t.strip() for t in track.split(",") if t.strip()}
    selected = [p for p in tier_filtered if p.track in wanted]
    # Always prepend baseline so PSNR/HF Ratio are computed against reference
    if "baseline" not in wanted:
        base = next((p for p in tier_filtered if p.track == "baseline"), None)
        if base is not None and base not in selected:
            selected.insert(0, base)
    return selected


# --------------------------------------------------------------------------- GPU Patch Helpers

def apply_scheduled_cfg(
    c: SimpleNamespace,
    patcher: Any,
    base_cfg: float,
    cfg_drop_percent: float,
    dropped_cfg: float = 1.0,
) -> Any:
    """Attach a ``WrappersMP.PREDICT_NOISE`` wrapper that drops ``guider.cfg`` to ``1.0``
    after ``cfg_drop_percent`` of the sigma schedule, so ComfyUI skips ``uncond`` cleanly.
    """
    m = patcher.clone()
    model_sampling = m.get_model_object("model_sampling")
    drop_sigma = float(model_sampling.percent_to_sigma(cfg_drop_percent))

    def _scheduled_cfg_wrapper(executor: Any, x: Any, timestep: Any, model_options: Optional[dict[str, Any]] = None, seed: Any = None) -> Any:
        guider = executor.class_obj
        orig_cfg = guider.cfg
        sigma_val = float(timestep[0].item()) if hasattr(timestep, "__getitem__") else float(timestep)
        if sigma_val <= drop_sigma and orig_cfg > dropped_cfg:
            guider.cfg = dropped_cfg
        else:
            guider.cfg = base_cfg
        try:
            return executor(x, timestep, model_options or {}, seed)
        finally:
            guider.cfg = orig_cfg

    m.add_wrapper_with_key(c.pe.WrappersMP.PREDICT_NOISE, "phase3_scheduled_cfg", _scheduled_cfg_wrapper)
    return m


def apply_block_sparse_sol_attn(
    patcher: Any,
    *,
    tau: float = 1.3,
    start_percent: float = 0.20,
    end_percent: float = 1.00,
    min_tokens: int = 1024,
) -> Any:
    """Apply ComfyUI's built-in ``BlockSparseAttention`` (``sol-attn``)."""
    from comfy_extras.nodes_sparse_attention import apply_block_sparse_attention

    return apply_block_sparse_attention(
        patcher,
        tau=tau,
        topk_ratio=0.0,
        vsa=False,
        start_percent=start_percent,
        end_percent=end_percent,
        min_tokens=min_tokens,
        dense_blocks=set(),
        sink_conditioning="exact_kv_and_rows",
        extra_tokens=256,
        verbose=False,
    )


def apply_phase3_compile(
    patcher: Any,
    *,
    compile_mode: str,
    dynamic: bool = False,
) -> Any:
    """Apply L4-safe ``max-autotune-no-cudagraphs``, Regional/Block compile, or Blackwell ``max-autotune``."""
    from comfy_api.torch_helpers import set_torch_compile_wrapper
    from comfy_extras.nodes_torch_compile import skip_torch_compile_dict

    m = patcher.clone(disable_dynamic=True)
    if compile_mode == "regional_blocks":
        dm = m.get_model_object("diffusion_model")
        block_keys: list[str] = []
        for attr in ("blocks", "layers", "double_blocks", "single_blocks"):
            cand = getattr(dm, attr, None)
            if cand is not None and hasattr(cand, "__len__") and len(cand) > 0:
                block_keys = [f"diffusion_model.{attr}.{i}" for i in range(len(cand))]
                break
        set_torch_compile_wrapper(
            model=m,
            backend="inductor",
            mode="max-autotune-no-cudagraphs",
            dynamic=dynamic,
            options={"guard_filter_fn": skip_torch_compile_dict},
            keys=block_keys or ["diffusion_model"],
        )
    else:
        set_torch_compile_wrapper(
            model=m,
            backend="inductor",
            mode=compile_mode,
            dynamic=dynamic,
            options={"guard_filter_fn": skip_torch_compile_dict},
        )
    return m


# --------------------------------------------------------------------------- Prompt file loader (pure Python)

def load_prompt_spec(model: str, prompt_id: Optional[str], prompt_file: Optional[Path]) -> tuple[str, str, str]:
    """Return ``(prompt_id, prompt_text, negative_text)`` from ``--prompt-file`` JSON or built-in pool."""
    if prompt_file is not None:
        data = json.loads(prompt_file.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("models"), dict) and model in data["models"]:
            items = data["models"][model]
            entry = next(
                (e for e in items if prompt_id is None or e.get("id") == prompt_id or e.get("prompt_id") == prompt_id),
                items[0],
            )
        elif isinstance(data, list):
            entry = next(
                (e for e in data if prompt_id is None or e.get("id") == prompt_id or e.get("prompt_id") == prompt_id),
                data[0],
            )
        elif isinstance(data, dict) and model in data:
            entry = data[model]
        else:
            entry = data
        return (
            str(entry.get("id", entry.get("prompt_id", prompt_id or "custom"))),
            str(entry["prompt"]),
            str(entry.get("negative_prompt", entry.get("negative", "")) or ""),
        )
    base_model = "krea2" if model == "krea2_anime" else model
    pspec = get_prompt(base_model, prompt_id)
    return pspec.prompt_id, pspec.prompt, pspec.negative


# --------------------------------------------------------------------------- Single preset GPU worker

def _try_resolve_weight(kind: str, name: str, models_dir: Optional[Path]) -> Optional[Path]:
    if Path(name).is_file():
        return Path(name)
    try:
        return resolve_weight(kind, name, models_dir)
    except SystemExit:
        return None


def _run_extended_group_worker(args: argparse.Namespace) -> int:
    catalog = {p.name: p for p in build_extended_presets(args.model, args.tier, "all")}
    selected = [catalog[n.strip()] for n in args.presets.split(",") if n.strip()]
    if not selected:
        return 0

    custom_nodes = detect_custom_nodes(args.comfyui_dir)
    d = get_model_defaults(args.model)
    prompt_id, prompt_text, negative_text = load_prompt_spec(args.model, args.prompt_id, args.prompt_file)

    # Filter out presets whose required custom node or DiT/LoRA weight is unavailable BEFORE heavy init
    runnable: list[ExtendedPreset] = []
    for preset in selected:
        if preset.requires_custom_node and custom_nodes.get(preset.requires_custom_node) is None:
            skip_res = {
                "preset": preset.name,
                "model": args.model,
                "tier": preset.tier,
                "track": preset.track,
                "status": "SKIPPED",
                "skip_reason": f"custom node {preset.requires_custom_node!r} not installed in ComfyUI/custom_nodes",
                "notes": preset.notes,
            }
            write_json(args.out_dir / f"{preset.name}_result.json", skip_res)
            log(f"[{args.model}/{preset.name}] SKIPPED ({skip_res['skip_reason']})")
            continue
        dit_name = preset.dit_override or d.dit
        if _try_resolve_weight("dit", dit_name, args.models_dir) is None:
            skip_res = {
                "preset": preset.name,
                "model": args.model,
                "tier": preset.tier,
                "track": preset.track,
                "status": "SKIPPED",
                "skip_reason": f"DiT weight {dit_name!r} not found in models_dir",
                "notes": preset.notes,
            }
            write_json(args.out_dir / f"{preset.name}_result.json", skip_res)
            log(f"[{args.model}/{preset.name}] SKIPPED ({skip_res['skip_reason']})")
            continue
        if preset.lora and _try_resolve_weight("dit", preset.lora, args.models_dir) is None:
            # Also check loras/ subdir
            lora_alt = (args.models_dir / "loras" / preset.lora) if args.models_dir else None
            if lora_alt is None or not lora_alt.is_file():
                skip_res = {
                    "preset": preset.name,
                    "model": args.model,
                    "tier": preset.tier,
                    "track": preset.track,
                    "status": "SKIPPED",
                    "skip_reason": f"LoRA weight {preset.lora!r} not found",
                    "notes": preset.notes,
                }
                write_json(args.out_dir / f"{preset.name}_result.json", skip_res)
                log(f"[{args.model}/{preset.name}] SKIPPED ({skip_res['skip_reason']})")
                continue
        runnable.append(preset)

    if not runnable:
        return 0

    first = runnable[0]
    attn_backend = "pytorch" if first.attention in ("sage3", "flex") else first.attention
    c = bootstrap_with_accel(
        comfyui_dir=args.comfyui_dir,
        attention=attn_backend,
        fast_features=first.fast_features,
        force_upcast_attention=False,
        dont_upcast_attention=False,
        no_highvram=args.no_highvram,
    )

    te_path = resolve_weight("te", d.te, args.models_dir)
    vae_path = resolve_weight("vae", d.vae, args.models_dir)
    with cuda_timer(c.torch) as te_timer:
        conds = ensure_conds(
            c,
            model_name=args.model,
            te_path=te_path,
            clip_type=d.clip_type,
            texts=[prompt_text, negative_text],
            cache_dir=args.cond_cache_dir,
        )
    vae = c.sd.VAE(sd=c.utils.load_torch_file(str(vae_path)))

    loaded_dit_name: Optional[str] = None
    base_patcher: Any = None
    baseline_png = args.out_dir / "baseline.png"

    for preset in runnable:
        dit_name = preset.dit_override or d.dit
        if dit_name != loaded_dit_name:
            if base_patcher is not None:
                del base_patcher
                unload_all(c)
                gc.collect()
            dit_path = resolve_weight("dit", dit_name, args.models_dir)
            base_patcher = c.sd.load_diffusion_model(str(dit_path))
            loaded_dit_name = dit_name

        steps = preset.steps if preset.steps is not None else d.steps
        cfg = preset.cfg if preset.cfg is not None else d.cfg
        shift = preset.shift if preset.shift is not None else d.shift
        sampler = preset.sampler or d.sampler
        scheduler = preset.scheduler or d.scheduler

        patcher = with_shift(c, base_patcher, shift)

        if preset.lora:
            lora_p = _try_resolve_weight("dit", preset.lora, args.models_dir)
            if lora_p is None and args.models_dir is not None:
                lora_p = args.models_dir / "loras" / preset.lora
            assert lora_p is not None
            patcher = apply_lora_model_only(c, patcher, lora_p, preset.lora_strength)

        if preset.block_sparse_tau is not None:
            patcher = apply_block_sparse_sol_attn(
                patcher,
                tau=preset.block_sparse_tau,
                start_percent=preset.block_sparse_start,
                end_percent=preset.block_sparse_end,
                min_tokens=preset.block_sparse_min_tokens,
            )

        if preset.cfg_drop_percent is not None:
            patcher = apply_scheduled_cfg(c, patcher, cfg, preset.cfg_drop_percent)

        ec_holder = None
        if preset.easycache:
            patcher, ec_holder = apply_easycache(
                patcher, preset.easycache_threshold, preset.easycache_start, preset.easycache_end
            )

        if preset.compile_mode:
            patcher = apply_phase3_compile(
                patcher, compile_mode=preset.compile_mode, dynamic=preset.compile_dynamic
            )

        actual_fwds = 0
        kernel_profile: Optional[dict[str, Any]] = None

        def _count_fwd(executor: Any, *a: Any, **kw: Any) -> Any:
            nonlocal actual_fwds, kernel_profile
            actual_fwds += 1
            if args.profile_kernels and kernel_profile is None and actual_fwds == 2:
                res, kernel_profile = profile_single_step_kernels(
                    lambda: executor(*a, **kw), top_k=15, torch_mod=c.torch
                )
                return res
            return executor(*a, **kw)

        patcher.add_wrapper_with_key(c.pe.WrappersMP.DIFFUSION_MODEL, "phase3_ext_counter", _count_fwd)

        pos_cond = conds[prompt_text]
        neg_cond = zero_out_conditioning(c, pos_cond) if preset.zero_negative else conds[negative_text]

        repeat = 2 if preset.compile_mode else 1
        rep_denoise: list[float] = []
        t_vae_s = 0.0
        vram_peak_gb = 0.0
        vram_reserved_gb = 0.0
        last_img_f32: Optional[np.ndarray] = None

        for _rep in range(repeat):
            actual_fwds = 0
            if ec_holder is not None and hasattr(ec_holder, "reset"):
                ec_holder.reset()
            if c.torch.cuda.is_available():
                c.torch.cuda.reset_peak_memory_stats()
                c.torch.cuda.synchronize()

            latent = empty_latent(c, patcher, width=args.width, height=args.height)
            noise = c.sample.prepare_noise(latent, args.seed, None)

            with c.torch.inference_mode():
                with cuda_timer(c.torch) as denoise_timer:
                    if preset.manual_sigmas is not None:
                        sigmas_t = c.torch.FloatTensor(list(preset.manual_sigmas))
                        ksampler = c.samplers.ksampler(sampler)
                        samples = c.sample.sample_custom(
                            patcher, noise, cfg, ksampler, sigmas_t,
                            pos_cond, neg_cond, latent,
                            noise_mask=None, callback=None, disable_pbar=True, seed=args.seed,
                        )
                    elif preset.two_stage_dit is not None and preset.two_stage_split_step is not None:
                        # Stage 1: Raw high-sigma steps
                        mid_latent = c.sample.sample(
                            patcher, noise, steps, cfg, sampler, scheduler,
                            pos_cond, neg_cond, latent,
                            denoise=1.0, disable_noise=False,
                            start_step=0, last_step=preset.two_stage_split_step,
                            force_full_denoise=False, disable_pbar=True, seed=args.seed,
                        )
                        # Stage 2: Turbo remaining sigmas
                        s2_path = resolve_weight("dit", preset.two_stage_dit, args.models_dir)
                        s2_patcher = with_shift(c, c.sd.load_diffusion_model(str(s2_path)), shift)
                        s2_patcher.add_wrapper_with_key(c.pe.WrappersMP.DIFFUSION_MODEL, "phase3_ext_counter_s2", _count_fwd)
                        samples = c.sample.sample(
                            s2_patcher, noise, steps, preset.two_stage_cfg, sampler, scheduler,
                            pos_cond, zero_out_conditioning(c, pos_cond), mid_latent,
                            denoise=1.0, disable_noise=True,
                            start_step=preset.two_stage_split_step, last_step=steps,
                            force_full_denoise=True, disable_pbar=True, seed=args.seed,
                        )
                        del s2_patcher
                    else:
                        samples = c.sample.sample(
                            patcher, noise, steps, cfg, sampler, scheduler,
                            pos_cond, neg_cond, latent,
                            denoise=1.0, disable_noise=False,
                            start_step=None, last_step=None,
                            force_full_denoise=True, disable_pbar=True, seed=args.seed,
                        )
                with cuda_timer(c.torch) as vae_timer:
                    decoded = vae.decode(samples)

            if c.torch.cuda.is_available():
                vram_peak_gb = max(vram_peak_gb, c.torch.cuda.max_memory_allocated() / (1024 ** 3))
                vram_reserved_gb = max(vram_reserved_gb, c.torch.cuda.max_memory_reserved() / (1024 ** 3))
            rep_denoise.append(round(denoise_timer.elapsed_s, 4))
            t_vae_s = vae_timer.elapsed_s
            img_t = decoded
            while img_t.ndim > 3:
                img_t = img_t[0]
            last_img_f32 = img_t.float().cpu().numpy()

        nominal_fwds = steps * (2 if cfg > 1.0 else 1)
        t_warmup = max(0.0, rep_denoise[0] - rep_denoise[-1]) if len(rep_denoise) > 1 else 0.0
        timing = TimingBreakdown.from_measurements(
            t_te_encode_s=te_timer.elapsed_s,
            t_compile_warmup_s=t_warmup,
            t_denoise_pure_s=rep_denoise[-1],
            actual_forwards=actual_fwds,
            nominal_forwards=nominal_fwds,
            t_vae_decode_s=t_vae_s,
            vram_peak_gb=vram_peak_gb,
            vram_reserved_gb=vram_reserved_gb,
        )

        assert last_img_f32 is not None
        ref_f32 = load_png_float01(baseline_png) if (baseline_png.is_file() and preset.name != "baseline") else None
        img_metrics = compute_image_metrics(last_img_f32, ref_f32)
        rgb_u8 = (np.clip(np.nan_to_num(last_img_f32), 0.0, 1.0) * 255.0).round().astype(np.uint8)

        weights_meta = {"dit": Path(dit_name).name, "te": te_path.name, "vae": vae_path.name}
        accel_meta = {
            "tier": preset.tier,
            "track": preset.track,
            "attention": preset.attention,
            "fast_features": list(preset.fast_features),
            "easycache": preset.easycache,
            "easycache_threshold": preset.easycache_threshold,
            "easycache_start": preset.easycache_start,
            "easycache_end": preset.easycache_end,
            "cfg_drop_percent": preset.cfg_drop_percent,
            "block_sparse_tau": preset.block_sparse_tau,
            "compile_mode": preset.compile_mode,
            "lora": preset.lora,
            "two_stage_dit": preset.two_stage_dit,
            "manual_sigmas": list(preset.manual_sigmas) if preset.manual_sigmas else None,
        }
        result: dict[str, Any] = {
            "preset": preset.name,
            "model": args.model,
            "tier": preset.tier,
            "track": preset.track,
            "status": "COLLAPSED" if img_metrics["is_collapsed"] else "OK",
            "notes": preset.notes,
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "negative_prompt": negative_text if not preset.zero_negative else "<ConditioningZeroOut>",
            "steps": steps,
            "cfg": cfg,
            "shift": shift,
            "sampler": sampler,
            "scheduler": scheduler,
            "seed": args.seed,
            "width": args.width,
            "height": args.height,
            "weights": weights_meta,
            "acceleration": accel_meta,
            "wall_clock_sec": round(timing.t_denoise_pure_s + timing.t_vae_decode_s, 3),
            "timing": timing.to_dict(),
            "dtype_audit": audit_block0_dtypes(patcher, torch_mod=c.torch, mm_mod=c.mm),
            "kernel_profile": kernel_profile,
            "metrics": img_metrics,
            "comfyui_commit": comfyui_commit(c.pkg_dir),
        }

        comfy_graph = build_comfy_prompt_graph(
            model_name="krea2" if args.model == "krea2_anime" else args.model,
            dit_name=weights_meta["dit"],
            te_name=weights_meta["te"],
            vae_name=weights_meta["vae"],
            clip_type=d.clip_type,
            prompt=prompt_text,
            negative=negative_text,
            steps=steps,
            cfg=cfg,
            shift=shift,
            sampler=sampler,
            scheduler=scheduler,
            seed=args.seed,
            width=args.width,
            height=args.height,
            accel=accel_meta,
            filename_prefix=f"p3_{args.model}_{preset.name}",
            zero_negative=preset.zero_negative,
        )
        png_path = args.out_dir / f"{preset.name}.png"
        save_png_with_exif(rgb_u8, png_path, result, comfy_graph)
        result["png_file"] = png_path.name
        write_json(args.out_dir / f"{preset.name}_result.json", result)
        log(
            f"[{args.model}/{preset.name}] {result['status']} | "
            f"denoise={timing.t_denoise_pure_s:.2f}s ({timing.s_per_fwd:.3f}s/fwd, skip={timing.skip_ratio:.1%})"
        )

    return 0


# --------------------------------------------------------------------------- Summary Rendering & CLI

def render_phase3_markdown(model: str, tier: str, results: Sequence[Mapping[str, Any]]) -> str:
    base_denoise = next(
        (r.get("timing", {}).get("t_denoise_pure_s") for r in results if r.get("preset") == "baseline" and r.get("status") == "OK"),
        None,
    )
    lines = [
        f"# Phase 3 Extended Acceleration Summary — `{model}` (tier: `{tier}`)",
        "",
        "| Preset | Track | Status | Denoise (s) | Speedup | s/fwd | Skip % | VRAM (GB) | PSNR (dB) | HF Ratio | Notes |",
        "|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        status = r.get("status", "ERROR")
        if status in ("SKIPPED", "ERROR"):
            reason = r.get("skip_reason") or r.get("error") or ""
            lines.append(f"| `{r['preset']}` | {r.get('track', '-')} | **{status}** | - | - | - | - | - | - | - | {reason} |")
            continue
        t = r.get("timing", {})
        m = r.get("metrics", {})
        den = float(t.get("t_denoise_pure_s", 0.0))
        spd = f"{base_denoise / den:.2f}x" if (base_denoise and den > 0) else "-"
        psnr = f"{m['psnr_db']:.2f}" if "psnr_db" in m else "-"
        hfr = f"{m['hf_ratio_vs_base']:.3f}" if "hf_ratio_vs_base" in m else "-"
        lines.append(
            f"| `{r['preset']}` | {r.get('track', '-')} | **{status}** | {den:.2f} | {spd} | "
            f"{t.get('s_per_fwd', 0.0):.3f} | {t.get('skip_ratio', 0.0) * 100:.1f}% | "
            f"{t.get('vram_peak_gb', 0.0):.2f} | {psnr} | {hfr} | {r.get('notes', '')} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 3 extended acceleration sweep runner (Tier 1 L4 & Tier 2 Blackwell).")
    p.add_argument("--tier", default="l4", choices=["l4", "blackwell"])
    p.add_argument("--model", required=True, choices=["anima", "moss_image", "krea2", "krea2_anime"])
    p.add_argument("--track", default="all", choices=["quant", "cache", "cfg_attn", "distill", "compile", "all"])
    p.add_argument("--presets", default=None, help="Explicit comma-separated preset names (overrides --track).")
    p.add_argument("--prompt-file", type=Path, default=None, help="Optional JSON prompt file.")
    p.add_argument("--prompt-id", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--comfyui-dir", default=None)
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--cond-cache-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/phase3_extended"))
    p.add_argument("--profile-kernels", action="store_true", help="Run 1-step torch.profiler CUDA kernel summary per preset.")
    p.add_argument("--no-highvram", action="store_true")
    p.add_argument("--list-presets", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_local_env(REPO_ROOT / "local/.env")
    args = build_parser().parse_args(argv)

    if args._worker:
        return _run_extended_group_worker(args)

    if args.presets:
        catalog = {p.name: p for p in build_extended_presets(args.model, args.tier, "all")}
        selected = [catalog[n.strip()] for n in args.presets.split(",") if n.strip()]
    else:
        selected = build_extended_presets(args.model, args.tier, args.track)

    if args.list_presets:
        for p in selected:
            print(f"{p.name:34s} [{p.tier:9s}/{p.track:8s}] attn={p.attention:7s} | {p.notes}")
        return 0

    groups: OrderedDict[tuple[Any, ...], list[ExtendedPreset]] = OrderedDict()
    for p in selected:
        groups.setdefault(p.process_flag_key(), []).append(p)

    log(f"phase3_extended: tier={args.tier} model={args.model} track={args.track} presets={len(selected)} groups={len(groups)}")
    for idx, (fkey, plist) in enumerate(groups.items(), 1):
        log(f"  group {idx}/{len(groups)} flags={fkey} -> {[p.name for p in plist]}")

    if args.dry_run:
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cond_cache = args.cond_cache_dir or (args.out_dir / "_cond_cache")

    for fkey, plist in groups.items():
        cmd = [
            sys.executable,
            "-m",
            "xai_dit._accel_tmp.phase3_extended",
            "--_worker",
            "--tier", args.tier,
            "--model", args.model,
            "--presets", ",".join(p.name for p in plist),
            "--seed", str(args.seed),
            "--width", str(args.width),
            "--height", str(args.height),
            "--out-dir", str(args.out_dir),
            "--cond-cache-dir", str(cond_cache),
        ]
        if args.prompt_file:
            cmd += ["--prompt-file", str(args.prompt_file)]
        if args.prompt_id:
            cmd += ["--prompt-id", args.prompt_id]
        if args.comfyui_dir:
            cmd += ["--comfyui-dir", str(args.comfyui_dir)]
        if args.models_dir:
            cmd += ["--models-dir", str(args.models_dir)]
        if args.profile_kernels:
            cmd.append("--profile-kernels")
        if args.no_highvram:
            cmd.append("--no-highvram")
        subprocess.run(cmd, check=False)

    results: list[dict[str, Any]] = []
    sheet_items: list[tuple[str, Path]] = []
    for p in selected:
        rp = args.out_dir / f"{p.name}_result.json"
        if rp.is_file():
            r = json.loads(rp.read_text(encoding="utf-8"))
            results.append(r)
            if r.get("png_file"):
                t = r.get("timing", {})
                sheet_items.append((f"{p.name} ({t.get('t_denoise_pure_s', 0):.1f}s)", args.out_dir / r["png_file"]))
        else:
            results.append({"preset": p.name, "model": args.model, "tier": p.tier, "track": p.track, "status": "ERROR", "error": "missing result json"})

    write_json(args.out_dir / "summary.json", {"model": args.model, "tier": args.tier, "track": args.track, "results": results})
    (args.out_dir / "summary.md").write_text(render_phase3_markdown(args.model, args.tier, results), encoding="utf-8")
    if sheet_items:
        make_comparison_sheet(sheet_items, args.out_dir / "contact_sheet.png", cols=min(4, len(sheet_items)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
