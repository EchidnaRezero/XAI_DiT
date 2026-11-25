"""Unified single-pass E1-1 + E1-3 + E4-2 + E4-3 profiler & visualizer for Colab L4.

Captures in ONE hooked generation pass per model (zero GPU attention matrix retention):
- E1-1: Streamed img_to_txt spatial [64, 64] heatmaps for all 4 manifest targets (all steps + 6 rep steps)
- E1-3: Per-block V-norm ratio ||V_prompt||_2 / ||V_img||_2 (pre-repeat 12 KV heads on Krea 2; proxy on Anima)
- E4-2: Exact per-block/step S / R_prefix / R_suffix / P mass & normalized I2T template share
- E4-3: Krea 2 per-block gate_value sparsity/mean/std + 3-model M-Act (max |attn_out|)

Full protocol: every sampling step of every block is observed (no step skipping); target spans
are resolved with the tokenizer only (zero TE weights); all per-block/step accumulators stay on
the GPU (no ``.item()`` / host syncs inside the attention chunk loop; one transfer at the end);
progress is logged per step (``flush=True``).

Saves standard outputs under <out_root>/e1_e4_profile_<model>_<precision>_<run_id>/
and cross-model charts under <out_root>/cross_model/{e4_2,e4_3}/. Models not run in this
invocation (e.g. an Anima profile finished earlier) are picked up from their ``metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

from .cli import (
    MODEL_DEFAULTS,
    REPO_ROOT,
    SCHEMA_VERSION,
    bootstrap,
    empty_latent,
    ensure_cond_cache,
    install_runtime_wrappers,
    load_manifest,
    log,
    pick_prompt,
    resolve_weight,
    save_png,
    to_conditioning,
    token_map_json,
    unload_all,
    weight_precision,
    with_shift,
    write_json,
)

CROSS_MODEL_ORDER = ("anima", "moss_image", "krea2")
MODEL_COLORS = {"anima": "#1f77b4", "moss_image": "#ff7f0e", "krea2": "#9467bd"}

# Heatmap aggregation / display options (defaults match reference_repo/Anima_HEATMAP v1 maps.py:
# span ``.mean(axis=-1)`` L117, ``render_map(colormap="turbo", normalization="relative")`` L145-170).
SPAN_AGG_CHOICES = ("mean", "sum")
HEATMAP_NORM_CHOICES = ("per_tag", "shared")
HEATMAP_CMAP_CHOICES = ("turbo", "inferno")
DEFAULT_SPAN_AGG = "mean"
DEFAULT_HEATMAP_NORM = "per_tag"
DEFAULT_HEATMAP_CMAP = "turbo"
SHARED_CLIP_PERCENTILE = 99.5  # legacy ``shared`` scale: 0 .. 99.5th percentile
# Display-only post-processing of the 64x64 E1-1 maps (raw npz / metrics stay unsmoothed):
# Gaussian sigma in latent-patch units (0 disables), and the ``per_tag`` upper percentile so a
# single corner / outlier patch (observed up to 4.65x the rest) cannot crush the map (100 = exact max).
DEFAULT_HEATMAP_SMOOTH_SIGMA = 1.5
DEFAULT_HEATMAP_CLIP_PCT = 99.0
GAUSSIAN_TRUNCATE = 4.0  # kernel radius = round(truncate * sigma), as scipy.ndimage.gaussian_filter


def span_agg_divisor(span_len: int, span_agg: str) -> float:
    """Divisor applied to the per-query attention summed over a target span's tokens.

    ``mean`` divides by the full span length (so key-chunked partial sums still add up to the
    exact mean); ``sum`` keeps the legacy token-count-scaled total.
    """
    if span_agg == "mean":
        return float(max(int(span_len), 1))
    if span_agg == "sum":
        return 1.0
    raise ValueError(f"span_agg must be one of {SPAN_AGG_CHOICES}, got {span_agg!r}")


def _check_clip_pct(clip_pct: float) -> float:
    clip_pct = float(clip_pct)
    if not 0.0 < clip_pct <= 100.0:
        raise ValueError(f"heatmap clip percentile must be in (0, 100], got {clip_pct}")
    return clip_pct


def heatmap_range(maps: Sequence[Any], norm: str, clip_pct: float = 100.0) -> tuple[float, float]:
    """Color range ``(lo, hi)`` for the given map(s).

    ``per_tag``: relative ``min .. P(clip_pct)`` (``clip_pct=100`` = exact ``min .. max``,
    Anima_HEATMAP v1 ``normalization="relative"``), pass one map.
    ``shared``: legacy ``0 .. 99.5th percentile`` over all given maps (one scale; ``clip_pct`` ignored).
    """
    import numpy as np

    flat = np.concatenate([np.asarray(m, dtype=np.float64).ravel() for m in maps])
    if norm == "per_tag":
        clip_pct = _check_clip_pct(clip_pct)
        hi = float(flat.max()) if clip_pct >= 100.0 else float(np.percentile(flat, clip_pct))
        return float(flat.min()), hi
    if norm == "shared":
        return 0.0, float(np.percentile(flat, SHARED_CLIP_PERCENTILE))
    raise ValueError(f"heatmap norm must be one of {HEATMAP_NORM_CHOICES}, got {norm!r}")


def gaussian_kernel_1d(sigma: float, truncate: float = GAUSSIAN_TRUNCATE) -> Any:
    """Normalized 1D Gaussian taps (radius ``round(truncate * sigma)``, at least 1)."""
    import numpy as np

    radius = max(int(truncate * float(sigma) + 0.5), 1)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    return k / k.sum()


def smooth_heatmap(m: Any, sigma: float = DEFAULT_HEATMAP_SMOOTH_SIGMA) -> Any:
    """Pure-NumPy separable 2D Gaussian blur of one ``[H, W]`` map (``sigma`` in pixels/patches).

    ``sigma <= 0`` returns the map unchanged (as float64). Borders use ``symmetric`` padding
    (= scipy.ndimage ``mode="reflect"``), so mass is preserved and corners are not darkened.
    """
    import numpy as np

    a = np.asarray(m, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"smooth_heatmap expects a 2D map, got shape {a.shape}")
    if float(sigma) <= 0.0:
        return a.copy()
    k = gaussian_kernel_1d(sigma)
    r = (k.size - 1) // 2
    # rows (axis 1) then columns (axis 0): two 1D passes == full 2D Gaussian
    p = np.pad(a, ((0, 0), (r, r)), mode="symmetric")
    a = sum(k[i] * p[:, i:i + a.shape[1]] for i in range(k.size))
    p = np.pad(a, ((r, r), (0, 0)), mode="symmetric")
    return sum(k[i] * p[i:i + a.shape[0], :] for i in range(k.size))


def prepare_heatmap(
    m: Any,
    norm: str = DEFAULT_HEATMAP_NORM,
    smooth_sigma: float = 0.0,
    clip_pct: float = 100.0,
    rng: Optional[tuple[float, float]] = None,
    resize: Optional[Any] = None,
) -> Any:
    """Shared display pipeline: smooth (``smooth_sigma``, on the native grid) -> optional ``resize``
    (e.g. bilinear to image size) -> range (``rng`` or ``heatmap_range``) -> clipped [0, 1].

    Defaults (no smoothing, exact min..max) reproduce ``normalize_heatmap(m, *heatmap_range([m], norm))``.
    Pass ``rng`` for a group scale that was computed on maps smoothed the same way.
    """
    s = smooth_heatmap(m, smooth_sigma)
    if resize is not None:
        s = resize(s)
    lo, hi = rng if rng is not None else heatmap_range([s], norm, clip_pct)
    return normalize_heatmap(s, lo, hi)


def heatmap_scale_label(norm: str, smooth_sigma: float, clip_pct: float) -> str:
    """Short plot-title text describing the display scale."""
    smooth = f"gauss σ={float(smooth_sigma):g}px" if float(smooth_sigma) > 0 else "no smoothing"
    if norm == "shared":
        return f"shared 0–{SHARED_CLIP_PERCENTILE:g}% scale, {smooth}"
    top = "max" if float(clip_pct) >= 100.0 else f"p{float(clip_pct):g}"
    return f"per-panel min–{top} scale, {smooth}"


def _nonneg_float(s: str) -> float:
    v = float(s)
    if v < 0.0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {v}")
    return v


def _pct_float(s: str) -> float:
    try:
        return _check_clip_pct(float(s))
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def add_heatmap_display_args(p: argparse.ArgumentParser) -> None:
    """``--heatmap-norm`` / ``--heatmap-cmap`` / ``--heatmap-smooth-sigma`` / ``--heatmap-clip-pct``
    shared by the E1/E4 profile and E3-0 runners."""
    p.add_argument("--heatmap-norm", choices=HEATMAP_NORM_CHOICES, default=DEFAULT_HEATMAP_NORM,
                   help="heatmap color scale: per_tag = relative min..p<clip-pct> per map (default, Anima_HEATMAP v1); "
                        "shared = legacy one 0..99.5%% scale across tags")
    p.add_argument("--heatmap-cmap", choices=HEATMAP_CMAP_CHOICES, default=DEFAULT_HEATMAP_CMAP,
                   help="heatmap colormap (default: turbo, Anima_HEATMAP v1; inferno = legacy)")
    p.add_argument("--heatmap-smooth-sigma", type=_nonneg_float, default=DEFAULT_HEATMAP_SMOOTH_SIGMA,
                   help="display-only Gaussian blur sigma on the 64x64 map, applied before clipping/normalization "
                        "(default: %(default)s; 0 disables)")
    p.add_argument("--heatmap-clip-pct", type=_pct_float, default=DEFAULT_HEATMAP_CLIP_PCT,
                   help="per_tag upper percentile of the color scale (default: %(default)s; 100 = exact min..max; "
                        "ignored by shared)")


def normalize_heatmap(m: Any, lo: float, hi: float) -> Any:
    """Map values to [0, 1] with ``(m - lo) / (hi - lo)``, clipped (``lo=0`` reproduces ``m / vmax``)."""
    import numpy as np

    # float64 like ``heatmap_range`` so ``m == lo`` maps to exactly 0 (no float32 round-off / 1e-12 blow-up)
    out = (np.asarray(m, dtype=np.float64) - lo) / max(hi - lo, 1e-12)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def profile_dir(out_root: Path, model_name: str, run_id: str) -> Path:
    return out_root / f"e1_e4_profile_{model_name}_{MODEL_DEFAULTS[model_name].precision}_{run_id}"


def find_profile_dir(out_root: Path, model_name: str, run_id: str) -> Optional[Path]:
    """Finished E1/E4 profile (``metrics.json`` present) for ``run_id``, else the newest one at the same precision."""
    exact = profile_dir(out_root, model_name, run_id)
    if (exact / "metrics.json").is_file():
        return exact
    prefix = f"e1_e4_profile_{model_name}_{MODEL_DEFAULTS[model_name].precision}_"
    done = [p for p in out_root.glob(prefix + "*") if (p / "metrics.json").is_file()]
    return max(done, key=lambda p: (p / "metrics.json").stat().st_mtime) if done else None


def _load_lightweight_tokenizer(model_name: str) -> Any:
    """HF tokenizer behind the DiT text slots, built from ComfyUI's bundled vocab files (zero TE weights).

    Anima's DiT slots are T5 IDs; MOSS-Image / Krea 2 are Qwen byte-level BPE. ``SD1Tokenizer``
    subclasses keep the inner ``SDTokenizer`` under ``self.<clip>`` (no ``.tokenizer`` attribute).
    """
    if model_name == "anima":
        from comfy.text_encoders.anima import T5XXLTokenizer
        return T5XXLTokenizer().tokenizer
    if model_name == "moss_image":
        from comfy.text_encoders.moss_image import MOSSImageTokenizer
        wrapper = MOSSImageTokenizer()
    elif model_name == "krea2":
        from comfy.text_encoders.krea2 import Krea2Tokenizer
        wrapper = Krea2Tokenizer()
    else:
        raise ValueError(f"no tokenizer for model {model_name!r}")
    return getattr(wrapper, wrapper.clip).tokenizer


def _token_piece(tok: Any, tid: int) -> str:
    """Surface text of one token; SentencePiece (T5) word starts ``▁`` become a space (``decode`` drops it)."""
    piece = tok.convert_ids_to_tokens(tid)
    if isinstance(piece, str) and piece.startswith("▁"):
        return " " + piece[1:]
    return tok.decode([tid], skip_special_tokens=False)


def _match_target_spans_in_S(
    tok: Any,
    full_ids: Sequence[int],
    span_S: Any,
    targets: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, tuple[int, int]], dict[str, str]]:
    """Resolve each target's text-local half-open span [start, end) inside S (tokenizer only).

    Returns ``(spans, method)``; ``method`` is ``char_align`` | ``subsequence`` | ``word_fallback``
    | ``unresolved`` (the last two are approximate and logged as warnings).
    """
    s_start, s_end = span_S.start, span_S.end
    s_ids = [int(t) for t in full_ids[s_start:s_end]]
    pieces = [_token_piece(tok, tid) for tid in s_ids]
    joined_low = "".join(pieces).lower()
    char_to_tok = [idx for idx, p in enumerate(pieces) for _ in p]

    def unescape(s: str) -> str:
        return s.replace("\\(", "(").replace("\\)", ")")

    def find_nth(hay: str, needle: str, occ: int) -> int:
        pos = -1
        for _ in range(occ + 1):
            pos = hay.find(needle, pos + 1)
            if pos < 0:
                return -1
        return pos

    resolved: dict[str, tuple[int, int]] = {}
    method: dict[str, str] = {}
    for t in targets:
        raw_text = str(t["text"])
        occ = int(t.get("occurrence_idx", 0))
        needles = list(dict.fromkeys([unescape(raw_text).strip().lower(),
                                      re.sub(r"\s+", " ", unescape(raw_text)).strip().lower()]))
        span_found: Optional[tuple[int, int]] = None

        for needle in needles:
            pos = find_nth(joined_low, needle, occ) if needle else -1
            if pos >= 0:
                end_pos = min(pos + len(needle) - 1, len(char_to_tok) - 1)
                span_found = (s_start + char_to_tok[pos], s_start + char_to_tok[end_pos] + 1)
                method[raw_text] = "char_align"
                break

        cased = unescape(raw_text).strip()  # token IDs are case-sensitive
        for text in ((cased, " " + cased) if span_found is None else ()):  # BPE: mid-sentence words carry a space
            sub_ids = [int(i) for i in tok.encode(text, add_special_tokens=False)]
            hits = [i for i in range(len(s_ids) - len(sub_ids) + 1) if sub_ids and s_ids[i:i + len(sub_ids)] == sub_ids]
            if len(hits) > occ:
                i = hits[occ]
                span_found = (s_start + i, s_start + i + len(sub_ids))
                method[raw_text] = "subsequence"
                break

        if span_found is None:
            words = [w for w in re.split(r"[\s,()\"]+", needles[-1]) if len(w) >= 3]
            hits = [s_start + idx for idx, p in enumerate(pieces) if any(w in p.lower() for w in words)]
            if hits:
                span_found, method[raw_text] = (min(hits), max(hits) + 1), "word_fallback"
            else:
                span_found, method[raw_text] = (s_start, min(s_start + 1, s_end)), "unresolved"
            log(f"WARNING target {raw_text!r} resolved approximately ({method[raw_text]}) -> {span_found}")
        resolved[raw_text] = span_found
    return resolved, method


def _render_heatmap_visuals(
    base_img: Any,
    target_names: Sequence[str],
    maps_all: Mapping[str, Any],
    maps_by_step: Mapping[tuple[str, int], Any],
    rep_steps: Sequence[int],
    sigma_table: Sequence[float],
    out_visuals: Path,
    model_title: str,
    heatmap_norm: str = DEFAULT_HEATMAP_NORM,
    heatmap_cmap: str = DEFAULT_HEATMAP_CMAP,
    smooth_sigma: float = DEFAULT_HEATMAP_SMOOTH_SIGMA,
    clip_pct: float = DEFAULT_HEATMAP_CLIP_PCT,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    img_np = base_img.float().clamp(0, 1).cpu().numpy()
    H_img, W_img = img_np.shape[:2]

    def resize_map(m2d: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(m2d.astype(np.float32), mode="F").resize((W_img, H_img), Image.BILINEAR)
        return np.asarray(pil, dtype=np.float32)

    def smooth_resized(m2d: np.ndarray) -> np.ndarray:  # blur on the 64x64 patch grid, then upsample
        return resize_map(smooth_heatmap(m2d, smooth_sigma))

    cmap = plt.get_cmap(heatmap_cmap)
    scale_lbl = heatmap_scale_label(heatmap_norm, smooth_sigma, clip_pct)

    # 1. Per-target heatmap & overlay (smooth -> per_tag: min..p<clip>; shared: legacy 0..99.5% of this map)
    for tag in target_names:
        if tag not in maps_all:
            continue
        mn = prepare_heatmap(maps_all[tag], heatmap_norm, smooth_sigma, clip_pct, resize=resize_map)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_")[:48]

        heat_rgb = (cmap(mn)[..., :3] * 255.0).round().astype(np.uint8)
        Image.fromarray(heat_rgb).save(out_visuals / f"heatmap_{safe}.png")

        ov = np.clip(0.55 * img_np + 0.45 * cmap(mn)[..., :3], 0.0, 1.0)
        Image.fromarray((ov * 255.0).round().astype(np.uint8)).save(out_visuals / f"overlay_{safe}.png")

    # 2. [4 tags x 6 rep steps] overlay grid (smoothed; per_tag: each panel min..p<clip>; shared: one 0..99.5%)
    all_keys = [(t, s) for t in target_names for s in rep_steps if (t, s) in maps_by_step]
    if all_keys:
        shared_range = (heatmap_range([smooth_resized(maps_by_step[k]) for k in all_keys], "shared")
                        if heatmap_norm == "shared" else None)
        nrows, ncols = len(target_names), len(rep_steps)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.1 * nrows), dpi=130)
        if nrows == 1:
            axes = np.expand_dims(axes, 0)
        if ncols == 1:
            axes = np.expand_dims(axes, 1)
        for r, tag in enumerate(target_names):
            for c_idx, step_i in enumerate(rep_steps):
                ax = axes[r, c_idx]
                m_raw = maps_by_step.get((tag, step_i))
                if m_raw is None:
                    ax.axis("off")
                    continue
                mn = prepare_heatmap(m_raw, "per_tag", smooth_sigma, clip_pct,
                                     rng=shared_range, resize=resize_map)
                ov = np.clip(0.52 * img_np + 0.48 * cmap(mn)[..., :3], 0.0, 1.0)
                ax.imshow(ov)
                sig = sigma_table[step_i] if step_i < len(sigma_table) else float("nan")
                ax.set_title(f"{tag[:18]}\ni={step_i} (σ={sig:.2f}) μ={float(m_raw.mean()):.2e}", fontsize=7.5)
                ax.axis("off")
        scale_txt = (f"{scale_lbl}, 99.5%={shared_range[1]:.2e}" if shared_range
                     else f"{scale_lbl}; compare raw μ across panels")
        fig.suptitle(
            f"{model_title} — E1-1 img→txt Attention [tags × steps] ({scale_txt}, {heatmap_cmap}, bilinear)",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_visuals / "grid_attn_steps.png")
        plt.close(fig)


def _render_profile_plots(
    model_name: str,
    num_blocks: int,
    vnorm_blocks: Sequence[float],
    span_blocks: Mapping[str, Sequence[float]],
    mact_blocks: Sequence[float],
    gate_stats: Optional[Mapping[str, Sequence[float]]],
    out_visuals: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    xs = np.arange(num_blocks)

    # E1-3 V-norm profile
    fig, ax = plt.subplots(figsize=(8.5, 3.6), dpi=140)
    ax.plot(xs, vnorm_blocks, marker="o", linewidth=1.8, markersize=3.5, color="#1f77b4")
    ax.set_title(f"{model_name} — E1-3 Seg4Diff V-Norm Profile (||V_prompt||_2 / ||V_img||_2 or proxy)")
    ax.set_xlabel("Block Index")
    ax.set_ylabel("V-Norm Ratio")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_visuals / "e1_3_vnorm_profile.png")
    plt.close(fig)

    # E4-2 Template share per block
    fig, ax = plt.subplots(figsize=(9.0, 4.0), dpi=140)
    for k, color in (("S", "#2ca02c"), ("R_prefix", "#ff7f0e"), ("R", "#d62728"), ("P", "#7f7f7f")):
        if k in span_blocks and any(v > 0 for v in span_blocks[k]):
            ax.plot(xs, span_blocks[k], marker="o", markersize=3, linewidth=1.6, label=k, color=color)
    ax.set_title(f"{model_name} — E4-2 Attention Mass Decomposition by Block (cond, mean over steps)")
    ax.set_xlabel("Block Index")
    ax.set_ylabel("Mean I2T Attention Mass")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_visuals / "e4_2_template_blocks.png")
    plt.close(fig)

    # E4-3 Gate (Krea 2) + M-Act
    if gate_stats is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0), dpi=140)
        ax1.plot(xs, gate_stats["mean"], marker="o", markersize=3.5, label="mean(gate)", color="#1f77b4")
        ax1.plot(xs, gate_stats["frac_lt_01"], marker="s", markersize=3.5, label="frac(gate < 0.1)", color="#d62728")
        ax1.plot(xs, gate_stats["frac_lt_05"], marker="^", markersize=3.0, label="frac(gate < 0.5)", color="#ff7f0e")
        ax1.set_title(f"{model_name} — E4-3 G1 Output Gate Sparsity (sigmoid(gate_logits))")
        ax1.set_xlabel("Block Index")
        ax1.set_ylabel("Value / Fraction")
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(xs, mact_blocks, marker="o", markersize=3.5, color="#9467bd")
        ax2.set_title(f"{model_name} — E4-3 Massive Activation M-Act (max |attn_out|)")
        ax2.set_xlabel("Block Index")
        ax2.set_ylabel("max |attn_out|")
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_visuals / "e4_3_krea2_gate.png")
        plt.close(fig)


def run_model_profile(
    c: SimpleNamespace,
    model_name: str,
    prompt_id: str,
    comfy_dir: Path,
    models_dir: Path,
    out_root: Path,
    cache_root: Path,
    run_id: str,
    span_agg: str = DEFAULT_SPAN_AGG,
    heatmap_norm: str = DEFAULT_HEATMAP_NORM,
    heatmap_cmap: str = DEFAULT_HEATMAP_CMAP,
    heatmap_smooth_sigma: float = DEFAULT_HEATMAP_SMOOTH_SIGMA,
    heatmap_clip_pct: float = DEFAULT_HEATMAP_CLIP_PCT,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from ..adapters import get_adapter
    from ..adapters.base import CaptureSpec
    from ..core.session import XAISession

    d = MODEL_DEFAULTS[model_name]
    manifest_path = REPO_ROOT / f"experiments/design/manifest_{model_name}.json"
    manifest, manifest_sha = load_manifest(manifest_path)
    entry = pick_prompt(manifest, prompt_id)
    prompt = entry["prompt"]
    negative = entry.get("negative_prompt", manifest.get("default_negative_prompt", d.negative))
    seed = 42
    targets = entry.get("targets", [])[:4]
    target_names = [t["text"] for t in targets]

    args = SimpleNamespace(
        model=model_name,
        comfyui_dir=str(comfy_dir),
        models_dir=models_dir,
        te=None,
        cache_root=str(cache_root),
        no_highvram=False,
        enable_dynamic_vram=False,
    )

    adapter = get_adapter(model_name)
    texts = [prompt, negative]
    with torch.inference_mode():  # TE pass only runs on a cache miss (TE unloaded right after)
        cache, conds = ensure_cond_cache(c, args, d, adapter, texts, manifest_sha)
    tm_cond = adapter.build_token_map(conds[prompt]["token_ids"], branch="cond")

    # Resolve target spans with the tokenizer only (zero TE weight loading)
    tok = _load_lightweight_tokenizer(model_name)
    resolved_spans, span_method = _match_target_spans_in_S(tok, tm_cond.token_ids, tm_cond.spans["S"], targets)
    del tok
    log(f"[{model_name}] resolved target spans in S={tm_cond.spans['S']}: {resolved_spans} ({span_method})")
    # Target-span token aggregation: mean (default, v1) or legacy sum; applied per key chunk so split spans stay exact
    span_div = {t: span_agg_divisor(resolved_spans[t][1] - resolved_spans[t][0], span_agg) for t in target_names}
    log(f"[{model_name}] E1-1 heatmap: span_agg={span_agg}, norm={heatmap_norm}, cmap={heatmap_cmap}, "
        f"smooth_sigma={heatmap_smooth_sigma}, clip_pct={heatmap_clip_pct}")

    # Load DiT + VAE (quantized int8_convrot / fp8_scaled weights load natively; no dtype forced)
    dit_path = resolve_weight("dit", d.dit, models_dir)
    vae_path = resolve_weight("vae", d.vae, models_dir)
    log(f"[{model_name}] loading DiT {dit_path.name} ({d.precision}) + VAE {vae_path.name}")
    patcher = c.sd.load_diffusion_model(str(dit_path))
    patcher = with_shift(c, patcher, d.shift)
    install_runtime_wrappers(c, patcher, d.batching)
    adapter.bind(patcher)
    vae = c.sd.VAE(sd=c.utils.load_torch_file(str(vae_path)))

    ms = patcher.get_model_object("model_sampling")
    sigmas = c.samplers.calculate_sigmas(ms, d.scheduler, d.steps).cpu()
    sigma_table = [float(s) for s in sigmas]
    rep_set = set(d.rep_steps)
    latent = empty_latent(c, patcher)

    positive = to_conditioning(conds[prompt])
    negative_c = to_conditioning(conds[negative])

    num_blocks = adapter.capability.num_blocks
    dev_gpu = c.mm.get_torch_device()
    # Keep ALL per-block/per-step accumulators on GPU (zero .item() CPU syncs inside loop!)
    side = 64
    n_tags = len(target_names)
    maps_all_gpu = torch.zeros((n_tags, side * side), dtype=torch.float32, device=dev_gpu)
    maps_all_cnt = 0
    rep_list = sorted(rep_set)
    rep_to_col = {s: idx for idx, s in enumerate(rep_list)}
    maps_step_gpu = torch.zeros((n_tags, len(rep_list), side * side), dtype=torch.float32, device=dev_gpu)
    maps_step_cnt = [0] * len(rep_list)

    vnorm_gpu = torch.zeros(num_blocks, dtype=torch.float64, device=dev_gpu)
    vnorm_cnt = np.zeros(num_blocks, dtype=np.int64)

    span_names = list(tm_cond.spans.keys())
    span_gpu = torch.zeros((len(span_names), num_blocks), dtype=torch.float64, device=dev_gpu)
    span_cnt = np.zeros(num_blocks, dtype=np.int64)

    mact_gpu = torch.zeros(num_blocks, dtype=torch.float32, device=dev_gpu)
    gate_mean_gpu = torch.zeros(num_blocks, dtype=torch.float64, device=dev_gpu)
    gate_lt01_gpu = torch.zeros(num_blocks, dtype=torch.float64, device=dev_gpu)
    gate_lt05_gpu = torch.zeros(num_blocks, dtype=torch.float64, device=dev_gpu)
    gate_cnt = np.zeros(num_blocks, dtype=np.int64)

    cap_fields = {"attn_out"}
    if adapter.capability.has_output_gate:
        cap_fields.add("gate_value")

    class _StreamingSession(XAISession):
        def _store(self, field_name: str, tensor: Any, block: int, ctx: Any) -> None:
            if not ctx.rows("cond"):
                return
            t = tensor.detach()
            if field_name == "attn_out":
                val = t.float().abs().max()
                mact_gpu[block] = torch.maximum(mact_gpu[block], val)
            elif field_name == "gate_value":
                gf = t.float()
                gate_mean_gpu[block] += gf.mean().double()
                gate_lt01_gpu[block] += (gf < 0.1).float().mean().double()
                gate_lt05_gpu[block] += (gf < 0.5).float().mean().double()
                gate_cnt[block] += 1

    session = _StreamingSession(
        adapter.capability,
        capture=CaptureSpec(fields=frozenset(cap_fields), branches=("cond",), to_cpu=False),
        query_chunk=1024,
    )

    last_logged_step = [-1]
    t_steps = [time.monotonic()]

    def observer_factory(call: Any):
        cond_rows = call.ctx.rows("cond")
        if not cond_rows:
            return None

        b = call.block
        step_i = call.ctx.step_index
        if b == 0 and step_i != last_logged_step[0]:  # host-side clock only, no CUDA sync
            now = time.monotonic()
            last_logged_step[0] = step_i
            log(f"[{model_name}] step {step_i + 1}/{d.steps} (σ={call.ctx.sigma:.3f}) "
                f"+{now - t_steps[0]:.1f}s{' [rep]' if step_i in rep_set else ''}")
            t_steps[0] = now

        layout = call.layout
        k_txt = layout.k_txt
        q_img = layout.q_img
        n_img = layout.n_img
        txt_offset = k_txt.start

        # E1-3 V-norm ratio on GPU (no .item() sync)
        v_cond = call.v[cond_rows[0]].float()
        s_span = tm_cond.spans["S"]
        v_prompt = v_cond[:, txt_offset + s_span.start : txt_offset + s_span.end, :]
        vp_norm = v_prompt.norm(dim=-1).mean().double()
        if layout.order == "cross":
            vnorm_gpu[b] += vp_norm
        else:
            v_img = v_cond[:, layout.k_img, :]
            vi_norm = v_img.norm(dim=-1).mean().double().clamp_min(1e-12)
            vnorm_gpu[b] += vp_norm / vi_norm
        vnorm_cnt[b] += 1

        tag_buf = torch.zeros((n_tags, n_img), dtype=torch.float32, device=call.device)
        span_buf = torch.zeros(len(span_names), dtype=torch.float64, device=call.device)
        q_img_count = 0

        def obs(probs: Any, q_sl: slice, k_sl: slice) -> None:
            nonlocal q_img_count
            qi0 = max(q_sl.start, q_img.start)
            qi1 = min(q_sl.stop, q_img.stop)
            if qi0 >= qi1:
                return
            local_q = slice(qi0 - q_sl.start, qi1 - q_sl.start)
            img_slice = slice(qi0 - q_img.start, qi1 - q_img.start)
            p_h = probs[cond_rows[0], :, local_q, :].float().mean(dim=0)
            if k_sl.start == 0:  # count each query row once (OOM key-chunked fallback revisits it per key chunk)
                q_img_count += (qi1 - qi0)

            ks0, ks1 = k_sl.start, k_sl.stop
            for s_idx, sname in enumerate(span_names):
                sp = tm_cond.spans[sname]
                a0 = max(ks0, txt_offset + sp.start)
                a1 = min(ks1, txt_offset + sp.end)
                if a0 < a1:
                    span_buf[s_idx] += p_h[:, a0 - ks0 : a1 - ks0].sum(dtype=torch.float64)

            for t_idx, tname in enumerate(target_names):
                ts0, ts1 = resolved_spans[tname]
                a0 = max(ks0, txt_offset + ts0)
                a1 = min(ks1, txt_offset + ts1)
                if a0 < a1:
                    tag_buf[t_idx, img_slice] += p_h[:, a0 - ks0 : a1 - ks0].sum(dim=-1) / span_div[tname]

        def finish() -> None:
            nonlocal maps_all_cnt
            if q_img_count > 0:
                span_gpu[:, b] += span_buf / float(q_img_count)
                span_cnt[b] += 1
            if n_img == side * side:
                maps_all_gpu.add_(tag_buf)
                maps_all_cnt += 1
                col = rep_to_col.get(step_i)
                if col is not None:
                    maps_step_gpu[:, col, :].add_(tag_buf)
                    maps_step_cnt[col] += 1

        obs.finish = finish  # type: ignore[attr-defined]
        return obs

    session.add_observer(observer_factory)

    t_gen = time.monotonic()
    with torch.inference_mode():
        noise = c.sample.prepare_noise(latent, seed)
        sampler_obj = c.samplers.sampler_object(d.sampler)
        with adapter.attached(session):
            samples = c.sample.sample_custom(
                patcher, noise, d.cfg, sampler_obj, sigmas, positive, negative_c,
                latent, disable_pbar=True, seed=seed,
            )
        img = vae.decode(samples)
        base_img = img.reshape(-1, *img.shape[-3:])[0].detach().cpu()
    elapsed = time.monotonic() - t_gen
    log(f"[{model_name}] single-pass E1/E4 profile done in {elapsed:.1f}s "
        f"({d.steps} steps, latent {list(latent.shape)}, heatmap calls={maps_all_cnt})")
    if maps_all_cnt == 0:
        log(f"WARNING [{model_name}] no E1-1 heatmap accumulated: image tokens != {side}x{side}")

    # Single GPU->CPU transfer after generation completes
    maps_all_np = (maps_all_gpu / max(maps_all_cnt, 1)).reshape(n_tags, side, side).cpu().numpy()
    maps_step_np = maps_step_gpu.reshape(n_tags, len(rep_list), side, side).cpu().numpy()
    maps_all = {t: maps_all_np[i] for i, t in enumerate(target_names)}
    maps_by_step = {
        (t, s): maps_step_np[i, col] / max(maps_step_cnt[col], 1)
        for i, t in enumerate(target_names)
        for col, s in enumerate(rep_list)
        if maps_step_cnt[col] > 0
    }
    vnorm_blocks = (vnorm_gpu.cpu().numpy() / np.maximum(vnorm_cnt, 1)).tolist()
    span_np = span_gpu.cpu().numpy()
    span_blocks = {sname: (span_np[i] / np.maximum(span_cnt, 1)).tolist() for i, sname in enumerate(span_names)}
    mact_max = mact_gpu.cpu().numpy().astype(np.float64)
    gate_mean_sum = gate_mean_gpu.cpu().numpy()
    gate_lt01_sum = gate_lt01_gpu.cpu().numpy()
    gate_lt05_sum = gate_lt05_gpu.cpu().numpy()

    # Free GPU before next model
    del patcher, vae, samples, maps_all_gpu, maps_step_gpu, vnorm_gpu, span_gpu, mact_gpu
    unload_all(c)

    # Normalized I2T template share: m_R_total / (m_R_total + m_S)
    m_S_mean = float(np.mean(span_blocks.get("S", [0.0])))
    m_R_pre_mean = float(np.mean(span_blocks.get("R_prefix", [0.0])))
    m_R_suf_mean = float(np.mean(span_blocks.get("R", [0.0])))
    m_P_mean = float(np.mean(span_blocks.get("P", [0.0])))
    m_R_total = m_R_pre_mean + m_R_suf_mean
    norm_template_share = m_R_total / max(m_R_total + m_S_mean, 1e-12)

    gate_stats = None
    if adapter.capability.has_output_gate:
        denom = np.maximum(gate_cnt, 1)
        gate_stats = {
            "mean": (gate_mean_sum / denom).tolist(),
            "frac_lt_01": (gate_lt01_sum / denom).tolist(),
            "frac_lt_05": (gate_lt05_sum / denom).tolist(),
        }

    # Write outputs
    out_dir = profile_dir(out_root, model_name, run_id)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "visuals").mkdir(parents=True, exist_ok=True)

    save_png(base_img, out_dir / "visuals" / "generated.png")
    _render_heatmap_visuals(
        base_img, target_names, maps_all, maps_by_step, d.rep_steps, sigma_table,
        out_dir / "visuals", f"{model_name} ({prompt_id})",
        heatmap_norm=heatmap_norm, heatmap_cmap=heatmap_cmap,
        smooth_sigma=heatmap_smooth_sigma, clip_pct=heatmap_clip_pct,
    )
    _render_profile_plots(
        model_name, num_blocks, vnorm_blocks, span_blocks, mact_max.tolist(), gate_stats,
        out_dir / "visuals",
    )

    np.savez_compressed(
        out_dir / "raw" / "profiles.npz",
        vnorm_blocks=np.array(vnorm_blocks),
        mact_blocks=mact_max,
        **{f"span_{k}": np.array(v) for k, v in span_blocks.items()},
        **({f"gate_{k}": np.array(v) for k, v in gate_stats.items()} if gate_stats else {}),
        **{f"map_all_{i}": maps_all[t] for i, t in enumerate(target_names) if t in maps_all},
    )

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model": model_name,
        "prompt_id": prompt_id,
        "elapsed_sec": elapsed,
        "resolved_target_spans": {k: list(v) for k, v in resolved_spans.items()},
        "target_span_method": span_method,
        "e1_1_heatmap_settings": {"span_agg": span_agg, "norm": heatmap_norm, "cmap": heatmap_cmap,
                                  "smooth_sigma": heatmap_smooth_sigma, "clip_pct": heatmap_clip_pct,
                                  "display_only": True},
        "e1_3_vnorm": {
            "blocks": vnorm_blocks,
            "peak_block": int(np.argmax(vnorm_blocks)),
            "peak_value": float(np.max(vnorm_blocks)),
        },
        "e4_2_template_registers": {
            "m_S_mean": m_S_mean,
            "m_R_prefix_mean": m_R_pre_mean,
            "m_R_suffix_mean": m_R_suf_mean,
            "m_P_mean": m_P_mean,
            "normalized_template_share": norm_template_share,
            "per_block": span_blocks,
        },
        "e4_3_gate_and_mact": {
            "mact_max_overall": float(np.max(mact_max)),
            "mact_peak_block": int(np.argmax(mact_max)),
            "mact_per_block": mact_max.tolist(),
            "gate_stats": gate_stats,
        },
    }
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "config.json", {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "exp_id": "e1_e4_profile",
        "model": {"name": model_name, "precision": d.precision, "dit": dit_path.name, "te": d.te,
                  "te_precision": weight_precision(d.te), "vae": vae_path.name,
                  "te_cache_sha256": cache.manifest_hash(texts)},
        "runtime": {"sampler": d.sampler, "scheduler": d.scheduler, "steps": d.steps, "cfg": d.cfg, "shift": d.shift,
                    "cfg_batching_policy": d.batching, "seed": seed, "latent_shape": list(latent.shape),
                    "sigma_table": sigma_table},
        "visualization": {"span_agg": span_agg, "heatmap_norm": heatmap_norm, "heatmap_cmap": heatmap_cmap,
                          "heatmap_smooth_sigma": heatmap_smooth_sigma, "heatmap_clip_pct": heatmap_clip_pct},
        "prompt": {"prompt_id": prompt_id, "prompt": prompt, "negative_prompt": negative},
        "token_map": token_map_json(tm_cond),
    })
    return metrics


def _render_cross_model_charts(results: Mapping[str, dict[str, Any]], out_root: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = [m for m in CROSS_MODEL_ORDER if m in results]
    if not models:
        return
    bar_colors = [MODEL_COLORS[m] for m in models]

    # Cross-model E4-2 Template Share
    e4_2_dir = out_root / "cross_model" / "e4_2"
    e4_2_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=140)
    x = np.arange(len(models))
    s_vals = [results[m]["e4_2_template_registers"]["m_S_mean"] for m in models]
    rp_vals = [results[m]["e4_2_template_registers"]["m_R_prefix_mean"] for m in models]
    rs_vals = [results[m]["e4_2_template_registers"]["m_R_suffix_mean"] for m in models]
    p_vals = [results[m]["e4_2_template_registers"]["m_P_mean"] for m in models]
    norm_r = [results[m]["e4_2_template_registers"]["normalized_template_share"] * 100.0 for m in models]

    ax1.bar(x, s_vals, label="S (Prompt)", color="#2ca02c")
    ax1.bar(x, rp_vals, bottom=s_vals, label="R_prefix", color="#ff7f0e")
    b2 = [a + b for a, b in zip(s_vals, rp_vals)]
    ax1.bar(x, rs_vals, bottom=b2, label="R_suffix / </s>", color="#d62728")
    b3 = [a + b for a, b in zip(b2, rs_vals)]
    ax1.bar(x, p_vals, bottom=b3, label="P (0-pad)", color="#7f7f7f")
    ax1.set_xticks(x, models)
    ax1.set_ylabel("Raw Mean I2T Attention Mass")
    ax1.set_title("E4-2 Raw Span Mass (Cross-Attn vs Joint-Attn)")
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    bars = ax2.bar(x, norm_r, color=bar_colors)
    for bar, v in zip(bars, norm_r):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(x, models)
    ax2.set_ylabel("Normalized Template Share R / (R + S) (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("E4-2 Normalized I2T Template Share (Comparable Across All 3)")
    ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(e4_2_dir / "e4_2_template_share.png")
    plt.close(fig)

    # Cross-model E4-3 M-Act comparison (normalized block depth 0..1)
    e4_3_dir = out_root / "cross_model" / "e4_3"
    e4_3_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.0), dpi=140)
    for m in models:
        vals = results[m]["e4_3_gate_and_mact"]["mact_per_block"]
        depth = np.linspace(0.0, 1.0, len(vals))
        ax.plot(depth, vals, marker="o", markersize=3, linewidth=1.8, label=m, color=MODEL_COLORS[m])
    ax.set_xlabel("Relative Block Depth (0 = first, 1 = last)")
    ax.set_ylabel("Peak max |attn_out| (M-Act)")
    ax.set_title("E4-3 Cross-Model Massive Activation (M-Act) Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(e4_3_dir / "e4_3_mact_compare.png")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run single-pass E1-1 + E1-3 + E4-2 + E4-3 profile across models")
    p.add_argument("--models", nargs="+", default=["moss_image", "krea2", "anima"])
    p.add_argument("--comfyui-dir", required=True)
    p.add_argument("--models-dir", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, default=REPO_ROOT / "cache" / "cond")
    p.add_argument("--run-id", default="l4_01")
    p.add_argument("--skip-existing", action="store_true",
                   help="reuse a finished profile (metrics.json) of a requested model instead of re-running it")
    add_heatmap_display_args(p)
    p.add_argument("--span-agg", choices=SPAN_AGG_CHOICES, default=DEFAULT_SPAN_AGG,
                   help="E1-1 target-span token aggregation (default: mean, Anima_HEATMAP v1; sum = legacy)")
    args = p.parse_args(argv)

    prompt_map = {"anima": "P-A1", "moss_image": "P-M1", "krea2": "P-K1"}
    unknown = [m for m in args.models if m not in prompt_map]
    if unknown:
        p.error(f"unknown models {unknown}; choose from {list(prompt_map)}")

    def reuse(m: str, why: str) -> bool:
        found = find_profile_dir(args.out_root, m, args.run_id)
        if found is None:
            return False
        results[m] = json.loads((found / "metrics.json").read_text(encoding="utf-8"))
        log(f"[{m}] reusing finished profile {found.name} ({why})")
        return True

    results: dict[str, dict[str, Any]] = {}
    to_run = [m for m in args.models if not (args.skip_existing and reuse(m, "--skip-existing"))]
    if to_run:
        c = bootstrap(SimpleNamespace(comfyui_dir=args.comfyui_dir, enable_dynamic_vram=False, no_highvram=False))
        for n, m in enumerate(to_run, 1):
            log(f"=== profile {n}/{len(to_run)}: {m} ({MODEL_DEFAULTS[m].precision}) ===")
            results[m] = run_model_profile(
                c, m, prompt_map[m], Path(args.comfyui_dir), args.models_dir,
                args.out_root, args.cache_root, args.run_id,
                span_agg=args.span_agg, heatmap_norm=args.heatmap_norm, heatmap_cmap=args.heatmap_cmap,
                heatmap_smooth_sigma=args.heatmap_smooth_sigma, heatmap_clip_pct=args.heatmap_clip_pct,
            )
    for m in CROSS_MODEL_ORDER:  # e.g. an Anima profile finished in an earlier session
        if m not in results:
            reuse(m, "cross-model charts")
    log(f"cross-model charts over {[m for m in CROSS_MODEL_ORDER if m in results]}")
    _render_cross_model_charts(results, args.out_root)
    (args.out_root / "cross_model").mkdir(parents=True, exist_ok=True)
    write_json(args.out_root / "cross_model" / f"summary_e1_e4_{args.run_id}.json", results)
    log("ALL E1/E4 PROFILES + CROSS-MODEL CHARTS DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
