"""Step 2 Anima Unified 4-Axis & Tier-3 suite runner (Colab L4; ``plan/handoff_heatmap_and_showcase.md`` Step 2).

Every stage reads the frozen prompt set ``experiments/design/unified_anima_set.json`` (same prompts,
same seed, same base images across all axes) and runs on the pure XAI L4 baseline: deterministic
``euler`` / ``simple``, 30 steps, cfg 4.0, shift 3.0, 1024x1024, ``--disable-dynamic-vram``
``--highvram``; NO ``--fast``, NO ``--use-ck-attention``, NO SageAttention, NO EasyCache.

Stages (``--stages``; ``all`` = every stage in this order)::

    multiprompt_profile   E1-1 heatmaps (span mean, per-tag min..max, turbo) + E4-2 S/R/P mass
                          + E4-3 M-Act, one hooked pass per prompt, 5-prompt mean +/- std charts
    block_third_knockout  E3-0 per-category (style/character/outfit/prop + random_ctrl) Early/Mid/Late
                          Logit Knockout (cond, image queries) vs the path-matched reference
    e2_1_stableflow       E2-1 true single-block bypass 0..27, latent vitality 1 - cos + pixel delta,
                          Top-3 vital vs 3 safe-to-skip blocks
    e3_2_pci              E3-2 prefix-cached prompt switch (P_b -> P_c at step k) filmstrip + lock-in curve
    e3_4_sap              E3-4 DiT-ST attention-change curve (transition step) + SAP 1-stage vs 2-stage
    e4_1_sinks            E4-1 two-pass dynamic sink masking (Anima P span grouped) vs Random-k control
    e4_4_relations        E4-4 left/right synopsis heads + token-specific relation-head ablation (A & swapped B)

Outputs (incremental: images are written as soon as they exist, ``progress.json`` after every unit,
``summary.json`` after every stage; ``--resume`` skips finished units)::

    <out_dir>/run_config.json  summary.json  _shared/{base,ref}_<prompt>_s<seed>.png (+ latent .npy)
    <out_dir>/<stage>/{metrics.json, progress.json, *.png, <prompt>/...}

Torch / ComfyUI are imported lazily: ``--dry-run`` and ``--help`` work on a CPU box without torch.
Image-space scores (pixel delta, latent cosine, colour-layout proxy, CIS proxy) are training-free
proxies; DINOv2 / CLIP / VQA evaluators run in a separate offline pass.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional, Sequence

from .cli import (
    MODEL_DEFAULTS,
    REPO_ROOT,
    SCHEMA_VERSION,
    _IdentityObserver,
    empty_latent,
    ensure_cond_cache,
    install_runtime_wrappers,
    load_manifest,
    log,
    resolve_weight,
    to_conditioning,
    token_map_json,
    unload_all,
    weight_precision,
    with_shift,
    write_json,
)
from .e1_e4_profile import (
    DEFAULT_HEATMAP_CMAP,
    DEFAULT_HEATMAP_NORM,
    DEFAULT_SPAN_AGG,
    SPAN_AGG_CHOICES,
    _load_lightweight_tokenizer,
    _match_target_spans_in_S,
    _render_heatmap_visuals,
    add_heatmap_display_args,
    span_agg_divisor,
)
from .e3_same_state import _pick_random_ctrl_span

MODEL = "anima"
SET_PATH = REPO_ROOT / "experiments" / "design" / "unified_anima_set.json"
DEFAULT_OUT = REPO_ROOT / "experiments" / "step2_anima_l4_01"
STAGES = ("multiprompt_profile", "block_third_knockout", "e2_1_stableflow", "e3_2_pci",
          "e3_4_sap", "e4_1_sinks", "e4_4_relations")

ROLE_CATEGORY = {"style": "style", "character": "character", "left_char": "character",
                 "right_char": "character", "outfit": "outfit", "prop": "prop", "left_prop": "prop",
                 "right_prop": "prop"}
CATEGORY_ORDER = ("style", "character", "outfit", "prop", "random_ctrl")
THIRD_ORDER = ("early", "mid", "late")
HEATMAP_SIDE = 64  # 1024x1024 -> 64x64 Anima image tokens
# L4 30-35 min budget (~100 generations): representative prompt subsets, overridable on the CLI.
DEFAULT_KNOCKOUT_PROMPTS = ("P-A1", "P-A2")
DEFAULT_STABLEFLOW_PROMPTS = ("P-A1",)
PROMPT_OVERRIDE_FLAGS = {"block_third_knockout": "knockout_prompts", "e2_1_stableflow": "stableflow_prompts"}
LOG_EVERY = 5

# Validated reference categorical palette (dataviz references/palette.md, light mode), fixed slot order.
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
C_NEUTRAL, C_INK, C_INK2, C_SURFACE = "#c3c2b7", "#0b0b0b", "#52514e", "#fcfcfb"


# =========================================================================== torch-free helpers


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")[:48] or "x"


def _np() -> Any:
    import numpy as np
    return np


def save_img(img: Any, path: Path) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def load_img(path: Path) -> Any:
    from PIL import Image
    return _np().asarray(Image.open(path).convert("RGB"), dtype=_np().uint8)


def f32(img: Any) -> Any:
    return img.astype(_np().float32) / 255.0


def small(img: Any, side: int = 128) -> Any:
    """Bilinear downsample to ``side`` x ``side`` float [0, 1] (noise-robust image distances)."""
    from PIL import Image
    return f32(_np().asarray(Image.fromarray(img).resize((side, side), Image.BILINEAR)))


def thumb(img: Any, side: int = 384) -> Any:
    from PIL import Image
    return _np().asarray(Image.fromarray(img).resize((side, side), Image.BILINEAR))


def img_delta(a: Any, b: Any) -> dict[str, float]:
    d = _np().abs(f32(a) - f32(b))
    h, w = d.shape[:2]
    return {"mean_abs": float(d.mean()), "max_abs": float(d.max()),
            "left_mean_abs": float(d[:, : w // 2].mean()), "right_mean_abs": float(d[:, w // 2:].mean())}


def delta_map(a: Any, b: Any) -> Any:
    return _np().abs(f32(a) - f32(b)).mean(-1)


def latent_cos_vitality(full: Any, other: Any) -> float:
    """StableFlow-style ``1 - cos(M_full, M_-l)`` on flattened final latents (DINOv2 CLS offline)."""
    np = _np()
    a, b = np.asarray(full, dtype=np.float64).ravel(), np.asarray(other, dtype=np.float64).ravel()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(1.0 - (a @ b) / den) if den > 0 else float("nan")


def color_layout(img: Any) -> dict[str, float]:
    """Left/right share of saturated red and blue pixels (training-free layout proxy for P-A3:
    red umbrella with the left character, blue sketchbook / blue hair with the right character)."""
    np = _np()
    x = f32(img)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    red = (r > 0.45) & (r - g > 0.2) & (r - b > 0.2)
    blue = (b > 0.45) & (b - r > 0.15) & (b - g > 0.05)
    w = x.shape[1] // 2
    out = {}
    for name, m in (("red", red), ("blue", blue)):
        lft, rgt = float(m[:, :w].sum()), float(m[:, w:].sum())
        out[f"{name}_left_share"] = lft / (lft + rgt) if lft + rgt > 0 else float("nan")
        out[f"{name}_pixels"] = lft + rgt
    return out


def _plt() -> Any:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.edgecolor": C_INK2, "axes.labelcolor": C_INK, "xtick.color": C_INK2,
                         "ytick.color": C_INK2, "figure.facecolor": "white", "axes.facecolor": C_SURFACE})
    return plt


def _style(ax: Any, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, alpha=0.25)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    _plt().close(fig)


def _panels(panels: Sequence[tuple[Any, str]], title: str, path: Path, ncols: Optional[int] = None) -> None:
    """Image panels (uint8 RGB or [H, W] float maps rendered with turbo) in a grid."""
    plt = _plt()
    n = len(panels)
    ncols = ncols or n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.7 * nrows), dpi=130, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (img, label) in zip(axes.ravel(), panels):
        if img.ndim == 2:
            ax.imshow(img, cmap=DEFAULT_HEATMAP_CMAP)
        else:
            ax.imshow(thumb(img))
        ax.set_title(label, fontsize=8.5, color=C_INK2)
    fig.suptitle(title, fontsize=11, color=C_INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, path)


class StageStore:
    """Per-stage output folder with crash-safe incremental ``progress.json`` (one entry per unit)."""

    def __init__(self, root: Path, stage: str, resume: bool) -> None:
        self.stage, self.dir = stage, root / stage
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "progress.json"
        self.units: dict[str, Any] = {}
        if resume and self.path.is_file():
            self.units = json.loads(self.path.read_text(encoding="utf-8")).get("units", {})
            log(f"[{stage}] resume: {len(self.units)} finished unit(s) in progress.json")

    def done(self, key: str, *files: str) -> bool:
        return key in self.units and all((self.dir / f).is_file() for f in files)

    def put(self, key: str, value: Any) -> None:
        self.units[key] = value
        write_json(self.path, {"schema_version": SCHEMA_VERSION, "stage": self.stage,
                               "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "units": self.units})

    def finish(self, metrics: Mapping[str, Any]) -> None:
        write_json(self.dir / "metrics.json", {"schema_version": SCHEMA_VERSION, "stage": self.stage, **metrics})


# =========================================================================== plan (torch-free)


def load_unified_set(path: Path) -> tuple[dict[str, Any], str]:
    uset, sha = load_manifest(path)
    if uset.get("model") != MODEL:
        raise SystemExit(f"{path.name} is not an Anima set (model={uset.get('model')!r})")
    return uset, sha


def check_runtime(uset: Mapping[str, Any]) -> dict[str, Any]:
    """The frozen set's runtime defaults must equal the L4 baseline in ``cli.MODEL_DEFAULTS`` (never altered here)."""
    d, rt = MODEL_DEFAULTS[MODEL], uset["runtime_defaults"]
    pairs = {"steps": (rt["steps"], d.steps), "cfg": (rt["cfg"], d.cfg), "shift": (rt["shift"], d.shift),
             "sampler": (rt["sampler"], d.sampler), "scheduler": (rt["scheduler"], d.scheduler),
             "same_state_steps": (list(rt["same_state_steps"]), list(d.rep_steps)),
             "checkpoint": (rt["checkpoint"], d.dit), "text_encoder": (rt["text_encoder"], d.te),
             "resolution": (list(rt["resolution"]), [1024, 1024])}
    bad = {k: v for k, v in pairs.items() if v[0] != v[1]}
    if bad:
        raise SystemExit(f"unified set runtime != L4 baseline (agreed spec, not auto-fixed): {bad}")
    thirds = rt["block_thirds"]
    blocks = sorted(b for t in THIRD_ORDER for b in thirds[t])
    if blocks != list(range(rt["num_blocks"])):
        raise SystemExit(f"block_thirds do not tile 0..{rt['num_blocks'] - 1}: {thirds}")
    return {k: v[0] for k, v in pairs.items()}


def stage_prompts(uset: Mapping[str, Any], stage: str, include_optional: bool,
                  override: Optional[Sequence[str]] = None) -> list[str]:
    proto = uset["axis_protocols"]
    key = {"multiprompt_profile": "E4-2_srp_length_consistency", "block_third_knockout": "E3-0_block_third_knockout",
           "e2_1_stableflow": "E2-1_stableflow", "e4_1_sinks": "E4-1_attention_sinks"}.get(stage)
    if key is None:
        return []
    pids = list(proto[key]["prompts"])
    if override:  # L4 budget subset (--knockout-prompts / --stableflow-prompts)
        known = {p["prompt_id"] for p in uset["prompts"]}
        bad = [p for p in override if p not in known]
        if bad:
            raise SystemExit(f"[{stage}] unknown prompt id(s) {bad}; known: {sorted(known)}")
        pids = list(dict.fromkeys(override))
    if stage == "multiprompt_profile":  # E1-1 heatmap prompts are the same 5, keep the length order
        pids += [p for p in proto["E1-1_heatmap"]["prompts"] if p not in pids]
    if include_optional:
        pids += [p for p in proto[key].get("optional_prompts", []) if p not in pids]
    return pids


def run_prompts(uset: Mapping[str, Any], stage: str, args: argparse.Namespace) -> list[str]:
    """``stage_prompts`` with the CLI prompt override for the stages that have one."""
    flag = PROMPT_OVERRIDE_FLAGS.get(stage)
    return stage_prompts(uset, stage, args.include_optional, getattr(args, flag, None) if flag else None)


def build_plan(uset: Mapping[str, Any], stages: Sequence[str], args: argparse.Namespace) -> dict[str, Any]:
    """Stage -> prompts / units / generation count, plus every text needing conditioning (one TE pass)."""
    rt = uset["runtime_defaults"]
    P = {p["prompt_id"]: p for p in uset["prompts"]}
    proto = uset["axis_protocols"]
    S = int(rt["steps"])
    nb = int(rt["num_blocks"])
    texts: dict[str, None] = {rt["default_negative_prompt"]: None}
    plan: dict[str, Any] = {}
    shared_base: set[str] = set()
    shared_ref: set[str] = set()

    def need(*ts: str) -> None:
        for t in ts:
            texts[t] = None

    for st in stages:
        pids = run_prompts(uset, st, args)
        for pid in pids:
            need(P[pid]["prompt"], P[pid].get("negative_prompt", rt["default_negative_prompt"]))
        if st == "multiprompt_profile":
            shared_base.update(pids)
            plan[st] = {"prompts": pids, "generations": len(pids), "note": "1 hooked pass / prompt + shared base"}
        elif st == "block_third_knockout":
            shared_ref.update(pids)
            rows = sum(len(P[p]["targets"]) + (0 if args.no_random_ctrl else 1) for p in pids)
            plan[st] = {"prompts": pids, "thirds": {t: rt["block_thirds"][t] for t in THIRD_ORDER},
                        "generations": rows * 3, "note": "logit_knockout cond img->target span, path-matched"}
        elif st == "e2_1_stableflow":
            shared_base.update(pids)
            blocks = list(proto["E2-1_stableflow"]["blocks"])
            plan[st] = {"prompts": pids, "blocks": f"{blocks[0]}..{blocks[-1]}", "generations": len(pids) * len(blocks)}
        elif st == "e3_2_pci":
            pr = proto["E3-2_pci"]
            pairs = [p["pair_id"] for p in pr["pairs"]]
            for p in pr["pairs"]:
                need(p["prompt_off_Pb"], p["prompt_on_Pc"])
            ks = [k for k in pr["injection_steps"] if k < S]
            plan[st] = {"pairs": pairs, "injection_steps": ks + [S],
                        "generations": len(pairs) * (2 + len(ks)), "note": "P_b rollout + prefix-cached switches + 1 identity check"}
        elif st == "e3_4_sap":
            bundles = proto["E3-4_sap_2stage"]["bundles"]
            for bd in bundles:
                need(bd["stage1_layout_proxy"], bd["stage2_detailed_target"])
                if bd["stage2_detailed_target"] == P[bd["base_prompt_id"]]["prompt"]:
                    shared_base.add(bd["base_prompt_id"])
            plan[st] = {"bundles": [b["bundle_id"] for b in bundles], "generations": 3 * len(bundles),
                        "switch_step": args.sap_switch_step if args.sap_switch_step is not None else "auto (DiT-ST s_rel)"}
        elif st == "e4_1_sinks":
            shared_ref.update(pids)
            plan[st] = {"prompts": pids, "blocks": args.sink_blocks or "relative depth 25/50/75% -> [7, 14, 21]",
                        "top_k": args.sink_top_k, "generations": 2 * len(pids), "note": "sink top-k vs Random-k, P span grouped"}
        elif st == "e4_4_relations":
            pairs = proto["E4-4_spatial_relation_circuits"]["pairs"]
            for pr in pairs:
                need(pr["prompt_a_left_emilia_right_arona"], pr["prompt_b_left_arona_right_emilia"])
            plan[st] = {"pairs": [p["pair_id"] for p in pairs], "top_heads": args.relation_top_heads,
                        "generations": 10 * len(pairs), "note": "2 synopsis + 2 refs + 3 ablation arms x 2 prompts"}
    plan["_shared"] = {"base_prompts": sorted(shared_base), "identity_ref_prompts": sorted(shared_ref),
                       "generations": len(shared_base) + len(shared_ref)}
    plan["_total_generations"] = sum(v["generations"] for v in plan.values() if isinstance(v, dict) and "generations" in v)
    plan["_texts"] = list(texts)
    return plan


# =========================================================================== runtime (torch / ComfyUI)


def bootstrap_baseline(args: argparse.Namespace) -> tuple[SimpleNamespace, dict[str, Any]]:
    """``cli.bootstrap`` with every accelerator forced OFF before ``model_management`` is imported."""
    from .cli import bootstrap

    comfy_dir = args.comfyui_dir or os.environ.get("COMFYUI_DIR")
    if comfy_dir:
        sys.path.insert(0, str(Path(comfy_dir).expanduser().resolve()))
    try:
        import comfy.options
    except ImportError as e:
        raise SystemExit("ComfyUI not importable: pass --comfyui-dir or set COMFYUI_DIR") from e
    comfy.options.enable_args_parsing(False)
    import comfy.cli_args

    a = comfy.cli_args.args
    a.fast = set()  # no fp16_accumulation / fp8_matrix_mult / cublas_ops / autotune
    for name in ("use_ck_attention", "use_sage_attention"):
        if hasattr(a, name):
            setattr(a, name, False)
    c = bootstrap(SimpleNamespace(comfyui_dir=comfy_dir, enable_dynamic_vram=False, no_highvram=False))
    flags = {"fast": sorted(str(f) for f in (a.fast or ())), "use_ck_attention": getattr(a, "use_ck_attention", None),
             "use_sage_attention": getattr(a, "use_sage_attention", None),
             "disable_dynamic_vram": bool(a.disable_dynamic_vram), "highvram": bool(a.highvram),
             "easycache": False, "torch_compile": False}
    if flags["fast"] or flags["use_ck_attention"] or flags["use_sage_attention"] or not flags["disable_dynamic_vram"] \
            or not flags["highvram"]:
        raise SystemExit(f"runtime flags deviate from the pure XAI baseline: {flags}")
    log(f"pure XAI baseline flags: {flags}")
    return c, flags


class AnimaSuite:
    """Loaded Anima DiT + VAE + conditioning; shared base / reference images; generation helpers."""

    def __init__(self, c: SimpleNamespace, args: argparse.Namespace, uset: Mapping[str, Any], set_sha: str,
                 texts: Sequence[str]) -> None:
        import numpy as np
        import torch
        from ..adapters import get_adapter

        self.c, self.args, self.uset, self.torch, self.np = c, args, uset, torch, np
        self.d = d = MODEL_DEFAULTS[MODEL]
        self.rt = uset["runtime_defaults"]
        self.P = {p["prompt_id"]: p for p in uset["prompts"]}
        self.proto = uset["axis_protocols"]
        self.seed = int(args.seed)
        self.negative = self.rt["default_negative_prompt"]
        self.adapter = get_adapter(MODEL)
        self.cap = self.adapter.capability
        self.out = Path(args.out_dir)
        self.shared = self.out / "_shared"
        self.shared.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, tuple[Any, Any]] = {}
        self._tm: dict[str, Any] = {}
        self.n_generations = 0
        t0 = time.monotonic()

        log(f"conditioning cache for {len(texts)} texts (TE loaded only on misses, then unloaded)")
        cc = SimpleNamespace(te=args.te, models_dir=args.models_dir, cache_root=str(args.cache_root))
        self.cache, self.conds = ensure_cond_cache(c, cc, d, self.adapter, list(texts), set_sha)
        self.texts = list(texts)
        self.tok = _load_lightweight_tokenizer(MODEL)  # span resolution only (zero TE weights)

        self.dit_path = resolve_weight("dit", d.dit, args.models_dir)
        self.vae_path = resolve_weight("vae", d.vae, args.models_dir)
        log(f"loading DiT {self.dit_path.name} ({d.precision}) + VAE {self.vae_path.name}")
        patcher = c.sd.load_diffusion_model(str(self.dit_path))
        self.patcher = with_shift(c, patcher, d.shift)
        install_runtime_wrappers(c, self.patcher, d.batching)
        if "easycache" in self.patcher.model_options.get("transformer_options", {}):
            raise SystemExit("EasyCache is attached; the pure XAI baseline forbids it")
        self.adapter.bind(self.patcher)
        self.vae = c.sd.VAE(sd=c.utils.load_torch_file(str(self.vae_path)))
        self.sigmas = c.samplers.calculate_sigmas(self.patcher.get_model_object("model_sampling"),
                                                  d.scheduler, d.steps).cpu()
        self.sigma_table = [float(s) for s in self.sigmas]
        self.S = len(self.sigma_table) - 1
        self.rep_steps = [i for i in self.rt["same_state_steps"] if i < self.S]
        self.latent = empty_latent(c, self.patcher)
        self.device = c.mm.get_torch_device()
        log(f"ready in {time.monotonic() - t0:.1f}s (steps={self.S}, blocks={self.cap.num_blocks}, "
            f"latent={list(self.latent.shape)}, seed={self.seed})")

    # ------------------------------------------------------------------ tokens
    def token_map(self, text: str) -> Any:
        if text not in self._tm:
            self._tm[text] = self.adapter.build_token_map(self.conds[text]["token_ids"], branch="cond")
        return self._tm[text]

    def resolve(self, text: str, targets: Sequence[Mapping[str, Any]]) -> tuple[dict[str, tuple[int, int]], dict[str, str]]:
        tm = self.token_map(text)
        return _match_target_spans_in_S(self.tok, tm.token_ids, tm.spans["S"], targets)

    # ------------------------------------------------------------------ sessions
    def session(self, specs: Sequence[Any] = (), *, identity: bool = True, cls: Any = None, **kw: Any) -> Any:
        """``identity=True`` forces the manual chunked SDPA on every attention call (cond and uncond),
        so interventions are compared against a path-matched reference (``ref``)."""
        from ..core.session import XAISession
        sess = (cls or XAISession)(self.cap, interventions=list(specs), query_chunk=self.args.query_chunk, **kw)
        if identity:
            sess.add_observer(lambda call: _IdentityObserver())
        return sess

    def continue_sampler(self, x_start: Any, k: int) -> Any:
        """Deterministic Euler (k-diffusion ``sample_euler``, ``s_churn=0``) restarted from the cached
        state ``x_k`` (prefix cache); conditioning comes from the ``sample_custom`` call."""
        def fn(model: Any, x: Any, sigmas: Any, extra_args: Optional[dict] = None, callback: Any = None,
               disable: Any = None) -> Any:
            extra_args = extra_args or {}
            s_in = x.new_ones([x.shape[0]])
            x = x_start.to(device=x.device, dtype=x.dtype)
            for i in range(k, len(sigmas) - 1):
                denoised = model(x, sigmas[i] * s_in, **extra_args)
                d = (x - denoised) / sigmas[i]
                if callback is not None:
                    callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
                x = x + d * (sigmas[i + 1] - sigmas[i])
            return x

        return self.c.samplers.KSAMPLER(fn)

    # ------------------------------------------------------------------ generation
    def generate(self, tag: str, text: str, *, session: Any = None, sampler: Any = None,
                 callback: Optional[Callable[..., None]] = None) -> tuple[Any, Any]:
        """One full sampling run -> (uint8 HxWx3 image, final latent on CPU). Logs every few steps."""
        c, d, torch = self.c, self.d, self.torch
        pos, neg = to_conditioning(self.conds[text]), to_conditioning(self.conds[self.negative])
        t0 = last = time.monotonic()
        self.n_generations += 1
        log(f"  >>> gen #{self.n_generations} '{tag}' ({'hooked' if session is not None else 'plain'}"
            f"{', custom sampler' if sampler is not None else ''})")

        def cb(step: int, x0: Any, x: Any, total: int) -> None:  # host clock only, no CUDA sync
            nonlocal last
            if callback is not None:
                callback(step, x0, x, total)
            if (step + 1) % LOG_EVERY == 0 or step + 1 == total:
                now = time.monotonic()
                log(f"      {tag} step {step + 1}/{total} (σ={self.sigma_table[step]:.3f}) +{now - last:.1f}s")
                last = now

        noise = c.sample.prepare_noise(self.latent, self.seed)
        smp = sampler or c.samplers.sampler_object(d.sampler)
        run = lambda: c.sample.sample_custom(self.patcher, noise, d.cfg, smp, self.sigmas, pos, neg,  # noqa: E731
                                             self.latent, callback=cb, disable_pbar=True, seed=self.seed)
        if session is None:
            samples = run()
        else:
            with self.adapter.attached(session):
                samples = run()
        img = self.vae.decode(samples)
        img = img.reshape(-1, *img.shape[-3:])[0].detach().float().clamp(0, 1).cpu().numpy()
        log(f"  <<< '{tag}' done in {time.monotonic() - t0:.1f}s")
        return (img * 255.0).round().astype(self.np.uint8), samples.detach().float().cpu()

    def _cached(self, name: str, text: str, identity: bool) -> tuple[Any, Any]:
        """Shared image + latent (``_shared/<name>_s<seed>``), generated once per run and reused by every stage."""
        if name in self._mem:
            return self._mem[name]
        png, lat = self.shared / f"{name}_s{self.seed}.png", self.shared / f"{name}_s{self.seed}_latent.npy"
        if self.args.resume and png.is_file() and lat.is_file():
            out = (load_img(png), self.np.load(lat))
            log(f"  resume: shared {png.name}")
        else:
            img, latent = self.generate(name, text, session=self.session() if identity else None)
            save_img(img, png)
            self.np.save(lat, latent.numpy().astype(self.np.float32))
            out = (img, latent.numpy())
        self._mem[name] = out
        return out

    def base(self, pid: str) -> tuple[Any, Any]:
        """Plain fused-SDPA base image (the 'same image' shown across all axes)."""
        return self._cached(f"base_{pid}", self.P[pid]["prompt"], identity=False)

    def ref(self, key: str, text: Optional[str] = None) -> tuple[Any, Any]:
        """Path-matched reference (manual chunked SDPA everywhere, nothing edited) for attention interventions.
        A text identical to a unified prompt reuses that prompt's reference (same image across stages)."""
        if text is not None:
            key = next((pid for pid, e in self.P.items() if e["prompt"] == text), key)
        return self._cached(f"ref_{key}", text or self.P[key]["prompt"], identity=True)

    def config(self) -> dict[str, Any]:
        d = self.d
        return {"model": {"name": MODEL, "precision": d.precision, "dit": self.dit_path.name, "te": Path(self.args.te or d.te).name,
                          "te_precision": weight_precision(self.args.te or d.te), "vae": self.vae_path.name,
                          "te_cache_sha256": self.cache.manifest_hash(self.texts)},
                "runtime": {"sampler": d.sampler, "scheduler": d.scheduler, "steps": d.steps, "cfg": d.cfg, "shift": d.shift,
                            "cfg_batching_policy": d.batching, "seed": self.seed, "query_chunk": self.args.query_chunk,
                            "latent_shape": list(self.latent.shape), "sigma_table": self.sigma_table}}

    def close(self) -> None:
        self.patcher = self.vae = None
        self._mem.clear()
        unload_all(self.c)


# =========================================================================== stage: multiprompt_profile


def _profile_pass(s: AnimaSuite, pid: str, out: Path) -> dict[str, Any]:
    """One hooked pass: E1-1 tag heatmaps (all steps + rep steps), E4-2 S/R/P mass per block, E4-3 M-Act."""
    torch, np = s.torch, s.np
    from ..adapters.base import CaptureSpec, TokenSpan
    from ..core.session import XAISession, slice_overlap

    entry = s.P[pid]
    text = entry["prompt"]
    targets = list(entry.get("targets", []))[:4]
    names = [t["text"] for t in targets]
    tm = s.token_map(text)
    spans, method = s.resolve(text, targets)
    div = {t: span_agg_divisor(spans[t][1] - spans[t][0], s.args.span_agg) for t in names}
    L, dev, side = s.cap.num_blocks, s.device, HEATMAP_SIDE
    rep_col = {st: i for i, st in enumerate(s.rep_steps)}
    n_tags = len(names)
    maps_all = torch.zeros((n_tags, side * side), dtype=torch.float32, device=dev)
    maps_step = torch.zeros((n_tags, len(s.rep_steps), side * side), dtype=torch.float32, device=dev)
    cnt_all, cnt_step = [0], [0] * len(s.rep_steps)
    span_names = list(tm.spans)
    span_acc = torch.zeros((len(span_names), L), dtype=torch.float64, device=dev)
    span_cnt = [0] * L
    mact = torch.zeros(L, dtype=torch.float32, device=dev)
    log(f"[profile] {pid}: spans={spans} ({method}), S/R/P={ {k: [v.start, v.end] for k, v in tm.spans.items()} }")

    class _Sess(XAISession):
        def _store(self, name: str, tensor: Any, block: int, ctx: Any) -> None:  # GPU-only M-Act
            if name == "attn_out" and ctx.rows("cond"):
                mact[block] = torch.maximum(mact[block], tensor.detach().float().abs().max())

    sess = s.session(identity=False, cls=_Sess,
                     capture=CaptureSpec(fields=frozenset({"attn_out"}), branches=("cond",), to_cpu=False))

    def factory(call: Any) -> Any:
        rows = call.ctx.rows("cond")
        if not rows:
            return None
        row, b, step, lay = rows[0], call.block, call.ctx.step_index, call.layout
        tag_buf = torch.zeros((n_tags, lay.n_img), dtype=torch.float32, device=call.device)
        span_buf = torch.zeros(len(span_names), dtype=torch.float64, device=call.device)
        nq = [0]
        span_keys = [lay.txt_keys(tm.spans[n]) for n in span_names]
        tag_keys = [lay.txt_keys(TokenSpan(*spans[n])) for n in names]

        def obs(probs: Any, q_sl: slice, k_sl: slice) -> None:
            ov = slice_overlap(q_sl, lay.q_img)
            if ov is None:
                return
            p_h = probs[row, :, ov[0], :].float().mean(dim=0)  # head mean [qc, kc]
            if k_sl.start == 0:  # count each query row once (key-chunked OOM fallback revisits it)
                nq[0] += ov[0].stop - ov[0].start
            for si, ks in enumerate(span_keys):
                ko = slice_overlap(k_sl, ks)
                if ko is not None:
                    span_buf[si] += p_h[:, ko[0]].sum(dtype=torch.float64)
            for ti, ks in enumerate(tag_keys):
                ko = slice_overlap(k_sl, ks)
                if ko is not None:
                    tag_buf[ti, ov[1]] += p_h[:, ko[0]].sum(dim=-1) / div[names[ti]]

        def finish() -> None:
            if nq[0]:
                span_acc[:, b] += span_buf / float(nq[0])
                span_cnt[b] += 1
            if lay.n_img == side * side:
                maps_all.add_(tag_buf)
                cnt_all[0] += 1
                col = rep_col.get(step)
                if col is not None:
                    maps_step[:, col].add_(tag_buf)
                    cnt_step[col] += 1

        obs.finish = finish  # type: ignore[attr-defined]
        return obs

    sess.add_observer(factory)
    base_img, _ = s.base(pid)
    hooked, _ = s.generate(f"{pid}_profile", text, session=sess)
    save_img(hooked, out / "generated_hooked.png")

    m_all = (maps_all / max(cnt_all[0], 1)).reshape(n_tags, side, side).cpu().numpy()
    m_step = maps_step.reshape(n_tags, len(s.rep_steps), side, side).cpu().numpy()
    maps_all_d = {t: m_all[i] for i, t in enumerate(names)}
    maps_by_step = {(t, st): m_step[i, col] / max(cnt_step[col], 1) for i, t in enumerate(names)
                    for col, st in enumerate(s.rep_steps) if cnt_step[col] > 0}
    sp = span_acc.cpu().numpy() / np.maximum(np.array(span_cnt), 1)
    span_blocks = {n: sp[i].tolist() for i, n in enumerate(span_names)}
    mact_np = mact.cpu().numpy().astype(np.float64)
    if cnt_all[0] == 0:
        log(f"WARNING [profile] {pid}: no heatmap accumulated (image tokens != {side}x{side})")

    _render_heatmap_visuals(torch.from_numpy(f32(base_img)), names, maps_all_d, maps_by_step, s.rep_steps,
                            s.sigma_table, out, f"anima ({pid})", heatmap_norm=s.args.heatmap_norm,
                            heatmap_cmap=s.args.heatmap_cmap)
    _render_srp_one(pid, span_blocks, mact_np, out / "e4_2_srp_blocks.png")
    np.savez_compressed(out / "profiles.npz", mact=mact_np, **{f"span_{k}": np.array(v) for k, v in span_blocks.items()},
                        **{f"map_all_{i}": m_all[i] for i in range(n_tags)})

    m = {k: float(np.mean(v)) for k, v in span_blocks.items()}
    r = m.get("R_prefix", 0.0) + m.get("R", 0.0)
    return {"prompt_id": pid, "word_count": entry.get("word_count"), "length_bucket": entry.get("length_bucket"),
            "text_len_Lt": tm.spans["R"].end, "targets": names,
            "resolved_target_spans": {k: list(v) for k, v in spans.items()}, "target_span_method": method,
            "srp_mean": m, "normalized_template_share": r / max(r + m.get("S", 0.0), 1e-12),
            "srp_per_block": span_blocks, "mact_per_block": mact_np.tolist(),
            "mact_peak_block": int(np.argmax(mact_np)), "mact_max": float(np.max(mact_np)),
            "heatmap_mean_per_tag": {t: float(maps_all_d[t].mean()) for t in names},
            "path_check_hooked_vs_base": img_delta(hooked, base_img)}


def _render_srp_one(pid: str, span_blocks: Mapping[str, Sequence[float]], mact: Sequence[float], path: Path) -> None:
    plt = _plt()
    import numpy as np
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8), dpi=130)
    colors = {"S": SLOTS[0], "R": SLOTS[1], "P": SLOTS[2], "R_prefix": SLOTS[3]}
    for k, v in span_blocks.items():
        a1.plot(np.arange(len(v)), v, color=colors.get(k, C_NEUTRAL), lw=2, marker="o", ms=3.5, label=k)
    a1.set_xlabel("Block index")
    a1.set_ylabel("mean img→txt mass (cond, mean over steps)")
    a1.set_title(f"{pid} — E4-2 S / R (</s>) / P (0-pad) mass", fontsize=10)
    a1.legend(frameon=False, fontsize=8)
    _style(a1, "both")
    a2.plot(np.arange(len(mact)), mact, color=SLOTS[0], lw=2, marker="o", ms=3.5)
    a2.set_xlabel("Block index")
    a2.set_ylabel("max |attn_out| over steps")
    a2.set_title(f"{pid} — E4-3 M-Act (cross-attn output)", fontsize=10)
    _style(a2, "both")
    fig.tight_layout()
    _save(fig, path)


def _render_multiprompt(units: Mapping[str, Mapping[str, Any]], order: Sequence[str], out: Path) -> dict[str, Any]:
    plt = _plt()
    import numpy as np
    pids = [p for p in order if p in units]
    if not pids:
        return {}
    keys = [k for k in ("S", "R", "P") if all(k in units[p]["srp_per_block"] for p in pids)]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.0), dpi=130, squeeze=False)
    agg: dict[str, Any] = {}
    for ax, k in zip(axes[0], keys):
        arr = np.array([units[p]["srp_per_block"][k] for p in pids])
        xs = np.arange(arr.shape[1])
        mu, sd = arr.mean(0), arr.std(0)
        ax.fill_between(xs, mu - sd, mu + sd, color=C_NEUTRAL, alpha=0.45, lw=0, label="mean ± std")
        for i, p in enumerate(pids):
            ax.plot(xs, arr[i], color=SLOTS[i % len(SLOTS)], lw=1.4, alpha=0.9,
                    label=f"{p} ({units[p]['word_count']}w)")
        ax.plot(xs, mu, color=C_INK, lw=2.2, label="mean")
        ax.set_title(f"E4-2 {k} mass by block", fontsize=10)
        ax.set_xlabel("Block index")
        ax.set_ylabel("mean img→txt mass")
        _style(ax, "both")
        agg[k] = {"mean_per_block": mu.tolist(), "std_per_block": sd.tolist(),
                  "mean_over_blocks_mean": float(arr.mean()), "cv_across_prompts": float(arr.mean(1).std() / max(arr.mean(), 1e-12))}
    axes[0][-1].legend(frameon=False, fontsize=7.5, loc="best")
    fig.suptitle("Anima unified set — S/R/P attention mass per block, short → long prompts (cond, all steps)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, out / "srp_blocks_multiprompt.png")

    share = [units[p]["normalized_template_share"] * 100 for p in pids]
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=130)
    bars = ax.bar(np.arange(len(pids)), share, color=SLOTS[0], width=0.6, edgecolor="white", linewidth=2)
    for bar, v in zip(bars, share):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5, color=C_INK)
    ax.set_xticks(np.arange(len(pids)), [f"{p}\n{units[p]['word_count']}w / Lt={units[p]['text_len_Lt']}" for p in pids], fontsize=8)
    ax.set_ylabel("R / (R + S)  (%)")
    ax.set_title("E4-2 normalized template share by prompt length", fontsize=10)
    _style(ax)
    fig.tight_layout()
    _save(fig, out / "template_share_by_length.png")

    arr = np.array([units[p]["mact_per_block"] for p in pids])
    xs = np.arange(arr.shape[1])
    fig, ax = plt.subplots(figsize=(8.5, 3.9), dpi=130)
    ax.fill_between(xs, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0), color=C_NEUTRAL, alpha=0.45, lw=0, label="mean ± std")
    for i, p in enumerate(pids):
        ax.plot(xs, arr[i], color=SLOTS[i % len(SLOTS)], lw=1.4, label=p)
    ax.plot(xs, arr.mean(0), color=C_INK, lw=2.2, label="mean")
    ax.set_xlabel("Block index")
    ax.set_ylabel("max |attn_out|")
    ax.set_title("E4-3 M-Act per block across the unified prompt set", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    _style(ax, "both")
    fig.tight_layout()
    _save(fig, out / "mact_multiprompt.png")
    agg["mact"] = {"peak_blocks": {p: units[p]["mact_peak_block"] for p in pids},
                   "mean_per_block": arr.mean(0).tolist(), "std_per_block": arr.std(0).tolist()}
    agg["normalized_template_share_pct"] = dict(zip(pids, share))
    return agg


def stage_multiprompt_profile(s: AnimaSuite, st: StageStore) -> dict[str, Any]:
    pids = stage_prompts(s.uset, "multiprompt_profile", s.args.include_optional)
    for n, pid in enumerate(pids, 1):
        if st.done(pid, f"{pid}/grid_attn_steps.png"):
            continue
        log(f"[multiprompt_profile] prompt {n}/{len(pids)}: {pid}")
        st.put(pid, _profile_pass(s, pid, st.dir / pid))
    agg = _render_multiprompt(st.units, pids, st.dir)
    pair = s.proto["E4-2_srp_length_consistency"].get("controlled_length_pair", [])
    ctrl = None
    if len(pair) == 2 and all(p in st.units for p in pair):
        a, b = (st.units[p] for p in pair)
        ctrl = {"pair": pair, "normalized_template_share": [a["normalized_template_share"], b["normalized_template_share"]],
                "srp_mean": [a["srp_mean"], b["srp_mean"]], "mact_peak_block": [a["mact_peak_block"], b["mact_peak_block"]]}
    return {"status": "ok", "prompts": pids, "per_prompt": st.units, "aggregate": agg, "controlled_length_pair": ctrl,
            "settings": {"span_agg": s.args.span_agg, "heatmap_norm": s.args.heatmap_norm, "heatmap_cmap": s.args.heatmap_cmap,
                         "rep_steps": s.rep_steps, "branch": "cond", "srp": "head-mean img→txt mass, mean over img queries, mean over steps",
                         "mact": "max |attn_out| (after output_proj, before gate_cross_attn) over steps"}}


# =========================================================================== stage: block_third_knockout


def stage_block_third_knockout(s: AnimaSuite, st: StageStore) -> dict[str, Any]:
    from ..adapters.base import InterventionSpec, TokenSpan

    thirds = {t: list(s.rt["block_thirds"][t]) for t in THIRD_ORDER}
    pids = run_prompts(s.uset, "block_third_knockout", s.args)
    for n, pid in enumerate(pids, 1):
        entry, text = s.P[pid], s.P[pid]["prompt"]
        tm = s.token_map(text)
        spans, method = s.resolve(text, entry["targets"])
        rows = [(t["text"], ROLE_CATEGORY.get(t["role"], t["role"]), spans[t["text"]]) for t in entry["targets"]]
        if not s.args.no_random_ctrl:
            rows.append(("random_ctrl", "random_ctrl", tuple(_pick_random_ctrl_span(tm.spans["S"], spans, seed=s.seed))))
        log(f"[block_third_knockout] prompt {n}/{len(pids)}: {pid} rows={[(r[0], r[2]) for r in rows]}")
        ref, _ = s.ref(pid)
        for name, cat, span in rows:
            for third, blocks in thirds.items():
                key, rel = f"{pid}|{name}|{third}", f"{pid}/{_safe(name)}_{third}.png"
                if st.done(key, rel):
                    continue
                spec = InterventionSpec("logit_knockout", blocks=blocks, cfg_scope="cond_only",
                                        params={"keys": (TokenSpan(*span),), "query": "img"})
                img, _ = s.generate(f"{pid}_ko_{_safe(name)}_{third}", text, session=s.session([spec]))
                save_img(img, st.dir / rel)
                st.put(key, {"prompt_id": pid, "target": name, "category": cat, "third": third,
                             "blocks": [blocks[0], blocks[-1]], "span": list(span),
                             "span_method": method.get(name, "random_ctrl"), "image": rel,
                             "delta_vs_ref": img_delta(img, ref)})
        _render_knockout_grid(s, st, pid, rows, ref)
    return {"status": "ok", "prompts": pids, "thirds": thirds, "units": st.units,
            "aggregate": _render_knockout_summary(st), "reference": "path-matched identity (manual chunked SDPA, cond+uncond)",
            "intervention": {"hook": "logit_knockout", "cfg_scope": "cond_only", "query": "img", "steps": "all"}}


def _render_knockout_grid(s: AnimaSuite, st: StageStore, pid: str, rows: Sequence[tuple], ref: Any) -> None:
    plt = _plt()
    import numpy as np
    cells = {(r[0], t): load_img(st.dir / st.units[f"{pid}|{r[0]}|{t}"]["image"])
             for r in rows for t in THIRD_ORDER if f"{pid}|{r[0]}|{t}" in st.units}
    dmaps = {k: delta_map(thumb(v, 256), thumb(ref, 256)) for k, v in cells.items()}
    vmax = float(np.percentile(np.stack(list(dmaps.values())), 99.5)) if dmaps else 1.0
    cmap = plt.get_cmap(s.args.heatmap_cmap)
    ref_t = f32(thumb(ref, 256))
    ncols = 1 + 2 * len(THIRD_ORDER)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(2.3 * ncols, 2.5 * len(rows)), dpi=120, squeeze=False)
    for r, (name, cat, span) in enumerate(rows):
        axes[r, 0].imshow(ref_t)
        axes[r, 0].set_title(f"{name[:20]}\n[{cat}] ref", fontsize=7.5, color=C_INK2)
        for j, t in enumerate(THIRD_ORDER):
            a_img, a_d = axes[r, 1 + j], axes[r, 1 + len(THIRD_ORDER) + j]
            if (name, t) not in cells:
                continue
            a_img.imshow(thumb(cells[(name, t)], 256))
            dm = dmaps[(name, t)]
            a_d.imshow(np.clip(0.5 * ref_t + 0.5 * cmap(np.clip(dm / max(vmax, 1e-12), 0, 1))[..., :3], 0, 1))
            mu = st.units[f"{pid}|{name}|{t}"]["delta_vs_ref"]["mean_abs"]
            a_img.set_title(f"KO {t}", fontsize=7.5, color=C_INK2)
            a_d.set_title(f"|Δ| {t}  μ={mu:.4f}", fontsize=7.5, color=C_INK2)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"anima ({pid}) — E3-0 Early/Mid/Late Logit Knockout per tag (cond img→tag; |Δ| shared 99.5%={vmax:.3f})",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, st.dir / f"grid_knockout_{pid}.png")


def _render_knockout_summary(st: StageStore) -> dict[str, Any]:
    plt = _plt()
    import numpy as np
    vals: dict[tuple[str, str], list[float]] = {}
    for u in st.units.values():
        vals.setdefault((u["category"], u["third"]), []).append(u["delta_vs_ref"]["mean_abs"])
    cats = [c for c in CATEGORY_ORDER if any((c, t) in vals for t in THIRD_ORDER)]
    if not cats:
        return {}
    mat = np.array([[np.mean(vals.get((c, t), [np.nan])) for t in THIRD_ORDER] for c in cats])
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=130)
    w = 0.26
    for j, t in enumerate(THIRD_ORDER):
        xs = np.arange(len(cats)) + (j - 1) * w
        ax.bar(xs, mat[:, j], width=w, color=SLOTS[j], edgecolor="white", linewidth=2, label=t)
        for x, c in zip(xs, cats):
            pts = vals.get((c, t), [])
            ax.scatter([x] * len(pts), pts, s=12, color=C_INK2, zorder=3, linewidths=0)
    ax.set_xticks(np.arange(len(cats)), cats)
    ax.set_ylabel("mean |Δ pixel| vs path-matched ref")
    ax.set_title("E3-0 which depth controls which tag category (bars = mean, dots = prompt×tag)", fontsize=10)
    ax.legend(title="block third", frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    _save(fig, st.dir / "category_third_bars.png")

    share = mat / np.nansum(mat, axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.8, 0.75 * len(cats) + 1.6), dpi=130)
    im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(cats)):
        for j in range(len(THIRD_ORDER)):
            ax.text(j, i, f"{share[i, j]:.0%}\n{mat[i, j]:.4f}", ha="center", va="center", fontsize=8,
                    color="white" if share[i, j] > 0.55 else C_INK)
    ax.set_xticks(range(len(THIRD_ORDER)), THIRD_ORDER)
    ax.set_yticks(range(len(cats)), cats)
    ax.set_title("Share of knockout effect per category (row-normalized)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, st.dir / "category_third_matrix.png")
    return {c: {t: {"mean": float(mat[i, j]), "share": float(share[i, j]), "n": len(vals.get((c, t), []))}
                for j, t in enumerate(THIRD_ORDER)} | {"dominant_third": THIRD_ORDER[int(np.nanargmax(mat[i]))]}
            for i, c in enumerate(cats)}


# =========================================================================== stage: e2_1_stableflow


def stage_e2_1_stableflow(s: AnimaSuite, st: StageStore) -> dict[str, Any]:
    import numpy as np
    from ..adapters.base import InterventionSpec

    blocks = list(s.proto["E2-1_stableflow"]["blocks"])
    pids = run_prompts(s.uset, "e2_1_stableflow", s.args)
    per_prompt: dict[str, Any] = {}
    for n, pid in enumerate(pids, 1):
        log(f"[e2_1_stableflow] prompt {n}/{len(pids)}: {pid} ({len(blocks)} single-block bypasses)")
        base, base_lat = s.base(pid)
        text = s.P[pid]["prompt"]
        for b in blocks:
            key, rel = f"{pid}|{b}", f"{pid}/bypass_{b:02d}.png"
            if st.done(key, rel):
                continue
            spec = InterventionSpec("true_block_bypass", blocks=[b], params={"mode": "full_block"})
            calls0, bypass0 = s.adapter.block_calls[b], s.adapter.bypass_calls[b]
            img, lat = s.generate(f"{pid}_bypass_{b:02d}", text, session=s.session([spec], identity=False))
            save_img(img, st.dir / rel)
            st.put(key, {"prompt_id": pid, "block": b, "image": rel,
                         "latent_vitality": latent_cos_vitality(base_lat, lat.numpy()),
                         "delta_vs_base": img_delta(img, base),
                         "bypass_calls": s.adapter.bypass_calls[b] - bypass0,
                         "block_body_calls": s.adapter.block_calls[b] - calls0})
        per_prompt[pid] = _stableflow_prompt(s, st, pid, blocks, base)
    agg = _render_stableflow_mean(st, pids, blocks, per_prompt)
    return {"status": "ok", "prompts": pids, "blocks": blocks, "per_prompt": per_prompt, "aggregate": agg,
            "units": st.units, "vitality": "1 - cos(final latent full, final latent with block l bypassed); DINOv2 CLS offline",
            "bypass": "true_block_bypass full_block (block returns its input residual; body never runs), cond+uncond, all steps"}


def _stableflow_prompt(s: AnimaSuite, st: StageStore, pid: str, blocks: Sequence[int], base: Any) -> dict[str, Any]:
    plt = _plt()
    import numpy as np
    import torch
    from matplotlib.patches import Patch
    from ..core.layer_selection import select_vital_layers

    vit = np.array([st.units[f"{pid}|{b}"]["latent_vitality"] for b in blocks])
    pix = np.array([st.units[f"{pid}|{b}"]["delta_vs_base"]["mean_abs"] for b in blocks])
    order = np.argsort(-vit, kind="stable")
    top3, safe3 = [blocks[i] for i in order[:3]], [blocks[i] for i in order[::-1][:3]]
    vital, rule = select_vital_layers(torch.from_numpy(vit))
    fig = plt.figure(figsize=(15, 9.5), dpi=120)
    gs = fig.add_gridspec(2, 7, height_ratios=[1.0, 1.15])
    ax = fig.add_subplot(gs[0, :])
    colors = [SLOTS[0] if b in top3 else SLOTS[1] if b in safe3 else C_NEUTRAL for b in blocks]
    ax.bar(blocks, vit, color=colors, width=0.8, edgecolor="white", linewidth=1.5)
    ax.set_xticks(blocks)
    ax.set_xlabel("Bypassed block")
    ax.set_ylabel("vitality = 1 − cos(latent)")
    ax.set_title(f"anima ({pid}) — E2-1 StableFlow single-block bypass vitality (pixel Δ in metrics; rule {rule}: {vital})",
                 fontsize=10)
    ax.legend(handles=[Patch(color=SLOTS[0], label="Top-3 vital"), Patch(color=SLOTS[1], label="3 safe-to-skip"),
                       Patch(color=C_NEUTRAL, label="other")], frameon=False, fontsize=8)
    _style(ax)
    panels = [(base, "Base (all blocks)")] + \
             [(load_img(st.dir / f"{pid}/bypass_{b:02d}.png"), f"vital: −block {b}\nvit={vit[blocks.index(b)]:.4f}") for b in top3] + \
             [(load_img(st.dir / f"{pid}/bypass_{b:02d}.png"), f"safe: −block {b}\nvit={vit[blocks.index(b)]:.4f}") for b in safe3]
    for j, (img, label) in enumerate(panels):
        a = fig.add_subplot(gs[1, j])
        a.imshow(thumb(img, 320))
        a.set_title(label, fontsize=8.5, color=C_INK2)
        a.axis("off")
    fig.tight_layout()
    _save(fig, st.dir / f"stableflow_{pid}.png")
    return {"latent_vitality": vit.tolist(), "pixel_mean_abs": pix.tolist(), "top3_vital": top3, "safe3": safe3,
            "selected_vital": vital, "selection_rule": rule,
            "bypass_invariant_ok": all(st.units[f"{pid}|{b}"]["block_body_calls"] == 0
                                       and st.units[f"{pid}|{b}"]["bypass_calls"] > 0 for b in blocks
                                       if "bypass_calls" in st.units[f"{pid}|{b}"])}


def _render_stableflow_mean(st: StageStore, pids: Sequence[str], blocks: Sequence[int],
                            per_prompt: Mapping[str, Any]) -> dict[str, Any]:
    plt = _plt()
    import numpy as np
    import torch
    from ..core.layer_selection import spearman_rho

    arr = np.array([per_prompt[p]["latent_vitality"] for p in pids])
    fig, ax = plt.subplots(figsize=(11, 4), dpi=130)
    ax.bar(blocks, arr.mean(0), color=SLOTS[0], width=0.75, edgecolor="white", linewidth=1.5, label="mean")
    for i, p in enumerate(pids):
        ax.scatter(blocks, arr[i], s=14, color=SLOTS[(i + 1) % len(SLOTS)], zorder=3, label=p, linewidths=0)
    ax.set_xticks(blocks)
    ax.set_xlabel("Bypassed block")
    ax.set_ylabel("vitality = 1 − cos(latent)")
    ax.set_title("E2-1 block vitality across the unified prompts (bar = mean, dots = prompt)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    _save(fig, st.dir / "vitality_mean.png")
    rho = {f"{a}~{b}": spearman_rho(torch.from_numpy(arr[i]), torch.from_numpy(arr[j]))
           for i, a in enumerate(pids) for j, b in enumerate(pids) if i < j}
    order = np.argsort(-arr.mean(0), kind="stable")
    return {"mean_vitality": arr.mean(0).tolist(), "top3_mean": [blocks[i] for i in order[:3]],
            "safe3_mean": [blocks[i] for i in order[::-1][:3]], "spearman_between_prompts": rho}


# =========================================================================== prompt-switch helper (PCI / SAP)


def _switch_rollouts(s: AnimaSuite, tag: str, off_text: str, on_text: str, ks: Sequence[int], *,
                     check_k: Optional[int] = None) -> tuple[Any, Any, dict[int, Any], Optional[float]]:
    """Run ``off_text`` once caching ``x_k`` (prefix cache), then finish every switch ``k`` with ``on_text``.

    Returns (off image, off latent, {k: image}, prefix-cache identity error). ``check_k`` re-finishes the
    cached prefix with ``off_text`` and must reproduce the off latent (max |Δ| ~ 0)."""
    need = {k for k in ks if k < s.S} | ({check_k} if check_k is not None else set())
    states: dict[int, Any] = {}

    def grab(step: int, x0: Any, x: Any, total: int) -> None:
        if step in need:
            states[step] = x.detach().clone()

    off_img, off_lat = s.generate(f"{tag}_off", off_text, callback=grab)
    missing = sorted(need - set(states))
    if missing:
        raise RuntimeError(f"base rollout did not reach steps {missing}")
    out: dict[int, Any] = {}
    for k in sorted(ks):
        out[k] = off_img if k >= s.S else s.generate(f"{tag}_switch@{k}", on_text,
                                                       sampler=s.continue_sampler(states[k], k))[0]
    err = None
    if check_k is not None:
        _, lat = s.generate(f"{tag}_prefix_identity@{check_k}", off_text, sampler=s.continue_sampler(states[check_k], check_k))
        err = float((lat - off_lat).abs().max())
        log(f"  prefix-cache identity @k={check_k}: max|Δlatent|={err:.3g}")
    return off_img, off_lat, out, err


# =========================================================================== stage: e3_2_pci


def stage_e3_2_pci(s: AnimaSuite, st: StageStore) -> dict[str, Any]:
    from ..core.patching import tau_crossings

    pr = s.proto["E3-2_pci"]
    ks = sorted({k for k in pr["injection_steps"] if k < s.S} | {s.S})
    results: dict[str, Any] = {}
    for n, pair in enumerate(pr["pairs"], 1):
        pid = pair["pair_id"]
        files = [f"{pid}/switch_{k:02d}.png" for k in ks]
        if st.done(pid, *files):
            results[pid] = st.units[pid]
            continue
        log(f"[e3_2_pci] pair {n}/{len(pr['pairs'])}: {pid} ({pair['concept']!r}, switches {ks})")
        check_k = ks[len(ks) // 2] if ks[len(ks) // 2] < s.S else None
        _, _, imgs, err = _switch_rollouts(s, pid, pair["prompt_off_Pb"], pair["prompt_on_Pc"], ks, check_k=check_k)
        for k, f in zip(ks, files):
            save_img(imgs[k], st.dir / f)
        pc, pb = small(imgs[0]) if 0 in imgs else None, small(imgs[s.S])
        d_bc = float(abs(pb - pc).mean()) if pc is not None else float("nan")
        cis = {k: (1.0 - float(abs(small(imgs[k]) - pc).mean()) / d_bc) if pc is not None and d_bc > 0 else float("nan")
               for k in ks}
        sig = {k: s.sigma_table[k] for k in ks}
        tau = tau_crossings([sig[k] for k in ks], [cis[k] for k in ks])
        res = {"pair_id": pid, "concept": pair["concept"], "concept_type": pair["concept_type"],
               "localizable": pair["localizable"], "switch_steps": ks, "sigma_at_switch": sig,
               "cis_proxy": cis, "tau": tau, "d_pixel_Pb_Pc": d_bc, "prefix_cache_identity_max_abs": err,
               "images": dict(zip(map(str, ks), files)),
               "cis_definition": "1 - d(x_k, P_c) / d(P_b, P_c), d = mean |Δ| on 128x128 (pixel proxy; VQA/CLIP offline)"}
        st.put(pid, res)
        results[pid] = res
        _panels([(load_img(st.dir / f), f"{'P_c (k=0)' if k == 0 else 'P_b (k=S)' if k == s.S else f'switch@{k}'}\n"
                  f"σ={sig[k]:.3f}  CIS={cis[k]:.2f}") for k, f in zip(ks, files)],
                f"anima — E3-2 PCI {pid}: P_b → P_c ('{pair['concept']}' added at step k)", st.dir / f"filmstrip_{pid}.png")
    _render_pci_curves(results, st.dir / "lockin_curves.png")
    return {"status": "ok", "pairs": results, "injection_steps": ks}


def _render_pci_curves(results: Mapping[str, Any], path: Path) -> None:
    plt = _plt()
    if not results:
        return
    fig, ax = plt.subplots(figsize=(8, 4), dpi=130)
    for i, (pid, r) in enumerate(results.items()):
        ks = r["switch_steps"]
        ax.plot(ks, [r["cis_proxy"][k] if k in r["cis_proxy"] else r["cis_proxy"][str(k)] for k in ks],
                color=SLOTS[i], lw=2, marker="o", ms=6, label=f"{pid} ({r['concept']})")
    ax.axhline(0.5, color=C_NEUTRAL, lw=1, ls="--")
    ax.axhline(0.7, color=C_NEUTRAL, lw=1, ls=":")
    ax.set_xlabel("switch step k (P_b for i < k, P_c for i ≥ k)")
    ax.set_ylabel("CIS proxy (1 = like P_c, 0 = like P_b)")
    ax.set_title("E3-2 concept lock-in: how late can the concept still be injected?", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "both")
    fig.tight_layout()
    _save(fig, path)


# =========================================================================== stage: e3_4_sap


def _attention_change_pass(s: AnimaSuite, tag: str, text: str) -> tuple[Any, dict[int, float]]:
    """DiT-ST Eq. 5-6: per-step Δ̄_i = mean_{block, head} ||A_i - A_{i-1}||_F / ||A_{i-1}||_F on cond img→txt
    maps restricted to the live text slots [0, Lt) (S + </s>; zero pad excluded), streamed on the GPU."""
    torch = s.torch
    from ..core.patching import attention_change
    from ..core.session import slice_overlap

    tm = s.token_map(text)
    lt = tm.spans["R"].end
    prev: dict[int, tuple[int, Any]] = {}
    step_sum = torch.zeros(s.S, dtype=torch.float64, device=s.device)
    step_cnt = [0] * s.S

    def factory(call: Any) -> Any:
        rows = call.ctx.rows("cond")
        if not rows:
            return None
        row, b, i, lay = rows[0], call.block, call.ctx.step_index, call.layout
        keys = slice(lay.k_txt.start, lay.k_txt.start + lt)
        buf = torch.zeros((call.q.shape[1], lay.n_img, lt), dtype=torch.float32, device=call.device)

        class _Obs:
            def __call__(self, probs: Any, q_sl: slice, k_sl: slice) -> None:
                qo, ko = slice_overlap(q_sl, lay.q_img), slice_overlap(k_sl, keys)
                if qo is not None and ko is not None:
                    buf[:, qo[1], ko[1]] = probs[row, :, qo[0], ko[0]].float()

            def finish(self) -> None:
                p = prev.get(b)
                if p is not None and p[0] == i - 1:
                    step_sum[i] += attention_change(p[1], buf).mean().double()
                    step_cnt[i] += 1
                prev[b] = (i, buf)

        return _Obs()

    sess = s.session(identity=False)
    sess.add_observer(factory)
    img, _ = s.generate(tag, text, session=sess)
    vals = step_sum.cpu().tolist()
    prev.clear()
    return img, {i: vals[i] / step_cnt[i] for i in range(s.S) if step_cnt[i]}


def stage_e3_4_sap(s: AnimaSuite, st: StageStore) -> dict[str, Any]:
    from ..core.patching import attribute_boundary, moving_average, normalize_by_dsigma, relation_boundary, snr

    results: dict[str, Any] = {}
    for bd in s.proto["E3-4_sap_2stage"]["bundles"]:
        bid = bd["bundle_id"]
        files = [f"{bid}/{n}.png" for n in ("stage1_only", "one_stage", "two_stage", "curve_pass")]
        if st.done(bid, *files):
            results[bid] = st.units[bid]
            continue
        s1, s2 = bd["stage1_layout_proxy"], bd["stage2_detailed_target"]
        log(f"[e3_4_sap] {bid}: attention-change curve on the detailed prompt")
        curve_img, raw = _attention_change_pass(s, f"{bid}_curve", s2)
        save_img(curve_img, st.dir / files[3])
        ma = moving_average(raw, 3)
        dhat = normalize_by_dsigma(ma, s.sigma_table)
        s_attr, attr_rule = attribute_boundary(dhat)
        try:
            s_rel = relation_boundary(ma, snr(s.sigma_table), s_attr if s_attr is not None else s.S)
        except ValueError as e:
            log(f"  relation_boundary failed ({e})")
            s_rel = None
        if s.args.sap_switch_step is not None:
            k, rule = int(s.args.sap_switch_step), "--sap-switch-step"
        elif s_rel is not None and 0 < s_rel < s.S:
            k, rule = int(s_rel), "DiT-ST s_rel (max curvature of Δ̄ over SNR before s_attr)"
        else:
            k, rule = max(1, round(0.2 * s.S)), "fallback round(0.2 S) (no s_rel)"
        log(f"  s_rel={s_rel}, s_attr={s_attr} ({attr_rule}) -> SAP switch k={k} [{rule}]")
        if s2 == s.P[bd["base_prompt_id"]]["prompt"]:
            one = s.base(bd["base_prompt_id"])[0]
        else:
            one = s.generate(f"{bid}_one_stage", s2)[0]
        stage1, _, two, _ = _switch_rollouts(s, bid, s1, s2, [k])
        for f, img in zip(files[:3], (stage1, one, two[k])):
            save_img(img, st.dir / f)
        res = {"bundle_id": bid, "attention_change": raw, "moving_average": ma, "dsigma_normalized": dhat,
               "s_rel": s_rel, "s_attr": s_attr, "attr_rule": attr_rule, "switch_step": k, "switch_rule": rule,
               "switch_sigma": s.sigma_table[k], "delta_two_vs_one": img_delta(two[k], one),
               "delta_two_vs_stage1": img_delta(two[k], stage1), "path_check_curve_vs_one": img_delta(curve_img, one),
               "images": dict(zip(("stage1_only", "one_stage", "two_stage", "curve_pass"), files))}
        st.put(bid, res)
        results[bid] = res
        _render_sap(s, st, bid, res, stage1, one, two[k])
    return {"status": "ok", "bundles": results,
            "notes": "Δ̄ restricted to live text keys [0, Lt) of the cond branch, per-head then averaged; uncond fixed (negative)."}


def _render_sap(s: AnimaSuite, st: StageStore, bid: str, res: Mapping[str, Any], stage1: Any, one: Any, two: Any) -> None:
    plt = _plt()
    raw, ma = res["attention_change"], res["moving_average"]
    fig, ax = plt.subplots(figsize=(9, 4), dpi=130)
    xs = sorted(int(i) for i in raw)
    ax.plot(xs, [raw[i] for i in xs], color=C_NEUTRAL, lw=1.5, marker="o", ms=3.5, label="Δ̄_i (raw)")
    ax.plot(xs, [ma[i] for i in xs], color=SLOTS[0], lw=2.2, label="moving average (w=3)")
    for val, lab, col, ls in ((res["s_rel"], "s_rel", SLOTS[1], "--"), (res["s_attr"], "s_attr", SLOTS[2], ":"),
                              (res["switch_step"], "SAP switch", C_INK, "-")):
        if val is not None:
            ax.axvline(val, color=col, ls=ls, lw=1.4, label=f"{lab} = {val}")
    ax.set_xlabel("sampling step i")
    ax.set_ylabel("relative img→txt attention change")
    ax.set_title(f"anima — E3-4 DiT-ST attention-shift curve ({bid}, detailed prompt, cond)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "both")
    fig.tight_layout()
    _save(fig, st.dir / f"attention_shift_{bid}.png")
    k = res["switch_step"]
    _panels([(stage1, "stage-1 layout proxy only"), (one, "1-stage (detailed P-A3)"),
             (two, f"2-stage (proxy → detailed @ step {k}, σ={res['switch_sigma']:.3f})"),
             (delta_map(thumb(two, 512), thumb(one, 512)), f"|Δ| 2-stage vs 1-stage  μ={res['delta_two_vs_one']['mean_abs']:.4f}")],
            f"anima — E3-4 SAP 1-stage vs 2-stage prompting ({bid})", st.dir / f"sap_before_after_{bid}.png")


# =========================================================================== stage: e4_1_sinks


def _sink_tracker(s: AnimaSuite, sess: Any, blocks: Sequence[int], tm: Any) -> tuple[Callable[[Any], Any], dict[str, Any]]:
    """Per sink block (cond): pre/post mass at the PRE-selected sink indices (fixes the repo's re-selection bug)
    and which span (S / R=</s> / P=0-pad group) the selected sinks belong to."""
    torch = s.torch
    from ..core.session import slice_overlap

    bi = {b: i for i, b in enumerate(blocks)}
    nb = len(blocks)
    acc = {k: torch.zeros(nb, dtype=torch.float64, device=s.device) for k in ("before", "after", "sel_S", "sel_R", "sel_P")}
    acc["n"] = [0] * nb
    sS, sR, sP = tm.spans["S"], tm.spans["R"], tm.spans["P"]

    def factory(call: Any) -> Any:
        rows = call.ctx.rows("cond")
        if not rows or call.block not in bi or not sess.sink_log:
            return None
        rec = sess.sink_log[-1]
        if rec.block != call.block or rec.step_index != call.ctx.step_index:
            return None
        sess.sink_log.pop()  # bounded memory
        row, lay, j = rows[0], call.layout, bi[call.block]
        mask = rec.selection.mask[row]  # [H, Nk] (key axis)
        before = (rec.selection.mass_before[row] * mask).sum(-1)  # [H]
        total = torch.zeros(mask.shape, dtype=torch.float32, device=call.device)
        k0 = lay.k_txt.start

        class _Obs:
            def __call__(self, probs: Any, q_sl: slice, k_sl: slice) -> None:
                qo = slice_overlap(q_sl, lay.q_img)
                if qo is not None:
                    total[:, k_sl] += probs[row, :, qo[0], :].float().sum(1)

            def finish(self) -> None:
                after = (total / max(lay.n_img, 1) * mask).sum(-1)
                acc["before"][j] += before.mean().double()
                acc["after"][j] += after.mean().double()
                acc["sel_S"][j] += mask[:, k0 + sS.start:k0 + sS.end].any(-1).float().mean().double()
                acc["sel_R"][j] += mask[:, k0 + sR.start].float().mean().double()
                acc["sel_P"][j] += mask[:, k0 + sP.start].float().mean().double()
                acc["n"][j] += 1

        return _Obs()

    return factory, acc


def stage_e4_1_sinks(s: AnimaSuite, st: StageStore) -> dict[str, Any]:
    import numpy as np
    torch = s.torch
    from ..adapters.base import InterventionSpec
    from ..core.layer_selection import relative_depth_blocks

    blocks = list(s.args.sink_blocks or relative_depth_blocks(s.cap.num_blocks))
    pids = stage_prompts(s.uset, "e4_1_sinks", s.args.include_optional)
    arms = ("sink_topk", "random_k")
    for n, pid in enumerate(pids, 1):
        text = s.P[pid]["prompt"]
        tm = s.token_map(text)
        ref, _ = s.ref(pid)
        log(f"[e4_1_sinks] prompt {n}/{len(pids)}: {pid} blocks={blocks} top_k={s.args.sink_top_k} P={tm.spans['P']}")
        for arm in arms:
            key, rel = f"{pid}|{arm}", f"{pid}/{arm}.png"
            if st.done(key, rel):
                continue
            params = {"top_k": s.args.sink_top_k, "selection": "per_head", "groups": (tm.spans["P"],),
                      "random": arm == "random_k"}
            if arm == "random_k":
                params["generator"] = torch.Generator().manual_seed(s.seed)
            spec = InterventionSpec("two_pass_sink_mask", blocks=blocks, cfg_scope="cond_only", params=params)
            sess = s.session([spec])
            factory, acc = _sink_tracker(s, sess, blocks, tm)
            sess.add_observer(factory, blocks=blocks)
            img, _ = s.generate(f"{pid}_{arm}", text, session=sess)
            save_img(img, st.dir / rel)
            cnt = np.maximum(np.array(acc["n"]), 1)
            v = {k: (acc[k].cpu().numpy() / cnt).tolist() for k in ("before", "after", "sel_S", "sel_R", "sel_P")}
            st.put(key, {"prompt_id": pid, "arm": arm, "blocks": blocks, "image": rel, "delta_vs_ref": img_delta(img, ref),
                         "sink_mass_before": v["before"], "sink_mass_after": v["after"],
                         "residual_ratio": [a / b if b > 0 else float("nan") for a, b in zip(v["after"], v["before"])],
                         "selected_span_share": {"S": v["sel_S"], "R(</s>)": v["sel_R"], "P(pad group)": v["sel_P"]},
                         "calls_per_block": acc["n"]})
        u = {a: st.units[f"{pid}|{a}"] for a in arms}
        _panels([(ref, "reference (path-matched)"),
                 (load_img(st.dir / u["sink_topk"]["image"]), f"sink top-{s.args.sink_top_k} masked\nμ|Δ|={u['sink_topk']['delta_vs_ref']['mean_abs']:.4f}"),
                 (load_img(st.dir / u["random_k"]["image"]), f"random-{s.args.sink_top_k} masked\nμ|Δ|={u['random_k']['delta_vs_ref']['mean_abs']:.4f}"),
                 (delta_map(thumb(load_img(st.dir / u["sink_topk"]["image"]), 512), thumb(ref, 512)), "|Δ| sink vs ref")],
                f"anima ({pid}) — E4-1 dynamic sink masking, blocks {blocks} (cond img queries, −1e4)",
                st.dir / f"sink_before_after_{pid}.png")
    agg = _render_sinks(st, pids, blocks, arms)
    return {"status": "ok", "prompts": pids, "blocks": blocks, "top_k": s.args.sink_top_k, "units": st.units, "aggregate": agg,
            "method": "two-pass dynamic sink mask (AttentionSinksDiT Sec 3.3.1): pass-1 incoming mass over image queries, "
                      "per-head top-k units (Anima zero-pad P span = one unit), pass-2 −1e4 at (Q_img, S_k)",
            "adaptation": "cfg_scope=cond_only (P-span groups follow the cond token map); reference = path-matched identity"}


def _render_sinks(st: StageStore, pids: Sequence[str], blocks: Sequence[int], arms: Sequence[str]) -> dict[str, Any]:
    plt = _plt()
    import numpy as np
    pids = [p for p in pids if all(f"{p}|{a}" in st.units for a in arms)]
    if not pids:
        return {}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2), dpi=130)
    before = np.array([st.units[f"{p}|sink_topk"]["sink_mass_before"] for p in pids]).mean(0)
    after = np.array([st.units[f"{p}|sink_topk"]["sink_mass_after"] for p in pids]).mean(0)
    xs, w = np.arange(len(blocks)), 0.36
    a1.bar(xs - w / 2, before, width=w, color=SLOTS[0], edgecolor="white", linewidth=2, label="before (pass 1)")
    a1.bar(xs + w / 2, np.maximum(after, 1e-30), width=w, color=SLOTS[1], edgecolor="white", linewidth=2, label="after (at same S_k)")
    a1.set_yscale("log")
    a1.set_xticks(xs, [f"block {b}" for b in blocks])
    a1.set_ylabel("sink incoming mass (head mean, log)")
    a1.set_title("E4-1 sink mass before vs after masking (mean over prompts)", fontsize=10)
    a1.legend(frameon=False, fontsize=8)
    _style(a1)
    xs = np.arange(len(pids))
    for j, a in enumerate(arms):
        a2.bar(xs + (j - 0.5) * w, [st.units[f"{p}|{a}"]["delta_vs_ref"]["mean_abs"] for p in pids], width=w,
               color=SLOTS[j], edgecolor="white", linewidth=2, label=a)
    a2.set_xticks(xs, pids)
    a2.set_ylabel("mean |Δ pixel| vs path-matched ref")
    a2.set_title("Image effect: true sinks vs Random-k control", fontsize=10)
    a2.legend(frameon=False, fontsize=8)
    _style(a2)
    fig.tight_layout()
    _save(fig, st.dir / "sink_summary.png")
    comp = {k: np.array([st.units[f"{p}|sink_topk"]["selected_span_share"][k] for p in pids]).mean(0).tolist()
            for k in ("S", "R(</s>)", "P(pad group)")}
    return {"mass_before_mean": before.tolist(), "mass_after_mean": after.tolist(),
            "reduction_factor": [b / a if a > 0 else float("inf") for a, b in zip(after, before)],
            "selected_span_share_mean": comp}


# =========================================================================== stage: e4_4_relations


def _synopsis_pass(s: AnimaSuite, tag: str, text: str, groups: Mapping[str, Sequence[Any]]) -> tuple[Any, Any]:
    """Attention Map Synopsis (cond only; the negative prompt has no relation/entity tokens) with left/right
    image-half region masks: returns [L, H, G, 2] mean img→group mass for left / right query halves."""
    torch, np = s.torch, s.np
    from ..core.session import slice_overlap

    gnames = list(groups)
    L, H = s.cap.num_blocks, s.cap.num_q_heads
    acc = torch.zeros((L, H, len(gnames), 2), dtype=torch.float64, device=s.device)
    cnt = [0] * L
    lmask: dict[int, Any] = {}

    def factory(call: Any) -> Any:
        rows = call.ctx.rows("cond")
        if not rows:
            return None
        row, b, lay = rows[0], call.block, call.layout
        side = int(round(lay.n_img ** 0.5))
        if lay.n_img not in lmask:
            lmask[lay.n_img] = ((torch.arange(lay.n_img, device=call.device) % side) < side // 2).float()
        lm = lmask[lay.n_img]
        gkeys = [[lay.txt_keys(sp) for sp in groups[g]] for g in gnames]

        class _Obs:
            def __call__(self, probs: Any, q_sl: slice, k_sl: slice) -> None:
                qo = slice_overlap(q_sl, lay.q_img)
                if qo is None:
                    return
                p = probs[row, :, qo[0], :].float()  # [H, qc, kc]
                lmc = lm[qo[1]]
                for gi, keys in enumerate(gkeys):
                    for ks in keys:
                        ko = slice_overlap(k_sl, ks)
                        if ko is None:
                            continue
                        m = p[:, :, ko[0]].sum(-1)  # [H, qc]
                        acc[b, :, gi, 0] += (m * lmc).sum(-1).double()
                        acc[b, :, gi, 1] += (m * (1 - lmc)).sum(-1).double()

            def finish(self) -> None:
                cnt[b] += 1

        return _Obs()

    sess = s.session(identity=False)
    sess.add_observer(factory)
    img, _ = s.generate(tag, text, session=sess)
    n_half = max(next(iter(lmask)) // 2, 1) if lmask else 1
    out = acc.cpu().numpy() / (np.maximum(np.array(cnt), 1)[:, None, None, None] * n_half)
    return img, out


def stage_e4_4_relations(s: AnimaSuite, st: StageStore) -> dict[str, Any]:
    import numpy as np
    from ..adapters.base import InterventionSpec, TokenSpan

    K = s.args.relation_top_heads
    results: dict[str, Any] = {}
    for pr in s.proto["E4-4_spatial_relation_circuits"]["pairs"]:
        rid = pr["pair_id"]
        prompts = {"A": (pr["prompt_a_left_emilia_right_arona"], pr["targets_prompt_a"]),
                   "B": (pr["prompt_b_left_arona_right_emilia"], pr["targets_prompt_b"])}
        spans: dict[str, dict[str, Any]] = {}
        for x, (text, ent) in prompts.items():
            tgt = [{"text": "left"}, {"text": "right"}, {"text": ent["left_entity"]}, {"text": ent["right_entity"]}]
            sp, method = s.resolve(text, tgt)
            tm = s.token_map(text)
            spans[x] = {"rel_left": [TokenSpan(*sp["left"])], "rel_right": [TokenSpan(*sp["right"])],
                        "ent_left": [TokenSpan(*sp[ent["left_entity"]])], "ent_right": [TokenSpan(*sp[ent["right_entity"]])],
                        "eos": [tm.spans["R"]]}
            log(f"[e4_4_relations] {rid}/{x} spans={ {k: [(v.start, v.end) for v in vs] for k, vs in spans[x].items()} } ({method})")

        # 1. synopsis on A and B -> relation-head ranking (consistent across the swap)
        syn: dict[str, Any] = {}
        for x, (text, _) in prompts.items():
            key = f"{rid}|synopsis_{x}"
            if st.done(key, f"{rid}/synopsis_{x}.png") and (st.dir / rid / f"synopsis_{x}.npy").is_file():
                syn[x] = np.load(st.dir / rid / f"synopsis_{x}.npy")
                continue
            img, arr = _synopsis_pass(s, f"{rid}_synopsis_{x}", text, spans[x])
            save_img(img, st.dir / rid / f"synopsis_{x}.png")
            np.save(st.dir / rid / f"synopsis_{x}.npy", arr)
            syn[x] = arr
            st.put(key, {"groups": list(spans[x]), "shape": list(arr.shape)})
        g = list(spans["A"])

        def selectivity(arr: Any, a: str, b: str) -> Any:  # token a should read the left half, token b the right
            return (arr[:, :, g.index(a), 0] - arr[:, :, g.index(a), 1]) + (arr[:, :, g.index(b), 1] - arr[:, :, g.index(b), 0])

        rel = {x: selectivity(syn[x], "rel_left", "rel_right") for x in syn}
        ent = {x: selectivity(syn[x], "ent_left", "ent_right") for x in syn}
        rel_mean = (rel["A"] + rel["B"]) / 2
        flat = np.argsort(-rel_mean.ravel(), kind="stable")
        top = [divmod(int(i), rel_mean.shape[1]) for i in flat[:K]]
        rng = np.random.RandomState(s.seed)
        pool = [divmod(int(i), rel_mean.shape[1]) for i in flat[K:]]
        rand = [pool[i] for i in sorted(rng.choice(len(pool), size=K, replace=False))]

        def specs(heads: Sequence[tuple[int, int]], keys: Sequence[Any]) -> list[Any]:
            by_block: dict[int, list[int]] = {}
            for b, h in heads:
                by_block.setdefault(b, []).append(h)
            return [InterventionSpec("per_head_token_ablation", blocks=[b], cfg_scope="cond_only",
                                     params={"heads": tuple(sorted(hs)), "keys": tuple(keys), "query": "img"})
                    for b, hs in sorted(by_block.items())]

        # 2. token-specific head ablation on A and B vs path-matched references
        arms = {"rel@top_heads": (top, ("rel_left", "rel_right")), "rel@random_heads": (rand, ("rel_left", "rel_right")),
                "eos@top_heads": (top, ("eos",))}
        refs = {x: s.ref(f"{rid}_{x}", text)[0] for x, (text, _) in prompts.items()}
        for x, (text, _) in prompts.items():
            for arm, (heads, gkeys) in arms.items():
                key, rel_f = f"{rid}|{x}|{arm}", f"{rid}/{x}_{_safe(arm)}.png"
                if st.done(key, rel_f):
                    continue
                keys = [sp for gk in gkeys for sp in spans[x][gk]]
                img, _ = s.generate(f"{rid}_{x}_{arm}", text, session=s.session(specs(heads, keys)))
                save_img(img, st.dir / rel_f)
                st.put(key, {"prompt": x, "arm": arm, "heads": heads, "image": rel_f,
                             "delta_vs_ref": img_delta(img, refs[x]), "color_layout": color_layout(img),
                             "delta_vs_swapped_ref": img_delta(img, refs["B" if x == "A" else "A"])})
        res = {"pair_id": rid, "top_relation_heads": top, "random_heads": rand,
               "relation_selectivity_mean": rel_mean.tolist(), "entity_selectivity_mean": ((ent["A"] + ent["B"]) / 2).tolist(),
               "relation_head_consistency_A_vs_B": float(np.corrcoef(rel["A"].ravel(), rel["B"].ravel())[0, 1]),
               "ref_color_layout": {x: color_layout(refs[x]) for x in refs},
               "ref_A_vs_ref_B": img_delta(refs["A"], refs["B"]),
               "arms": {k: v for k, v in st.units.items() if k.startswith(f"{rid}|") and "|synopsis" not in k}}
        results[rid] = res
        _render_relations(s, st, rid, res, rel_mean, ent, refs, list(arms))
    return {"status": "ok", "pairs": results, "top_heads_k": K,
            "method": "synopsis = img→token mass split by left/right image half (cond, all steps); relation selectivity = "
                      "(left-token L−R) + (right-token R−L), averaged over A and swapped B; ablation = per_head_token_ablation "
                      "(post-softmax A_hj zeroed for the relation tokens inside the selected heads only), cond_only",
            "layout_proxy": "red/blue pixel left-share (red umbrella with emilia, blue sketchbook/hair with arona); detector-based accuracy offline"}


def _render_relations(s: AnimaSuite, st: StageStore, rid: str, res: Mapping[str, Any], rel_mean: Any,
                      ent: Mapping[str, Any], refs: Mapping[str, Any], arms: Sequence[str]) -> None:
    plt = _plt()
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), dpi=120)
    for ax, (mat, title) in zip(axes, ((rel_mean, "relation tokens ('left'/'right')"),
                                       ((ent["A"] + ent["B"]) / 2, "entity tokens (left / right character)"))):
        lim = float(np.abs(mat).max()) or 1.0
        im = ax.imshow(mat, cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -lim, lim), aspect="auto")
        ax.set_xlabel("head")
        ax.set_ylabel("block")
        ax.set_title(f"spatial selectivity: {title}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for b, h in res["top_relation_heads"]:
        axes[0].add_patch(Rectangle((h - 0.5, b - 0.5), 1, 1, fill=False, ec=C_INK, lw=2))
    fig.suptitle(f"anima — E4-4 Attention Map Synopsis ({rid}, mean of A and swapped B; boxes = top relation heads)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, st.dir / f"synopsis_heads_{rid}.png")

    panels = []
    for x in ("A", "B"):
        cl = color_layout(refs[x])
        panels.append((refs[x], f"{x}: reference\nred left-share={cl['red_left_share']:.2f}"))
        for arm in arms:
            u = st.units.get(f"{rid}|{x}|{arm}")
            if u is None:
                continue
            panels.append((load_img(st.dir / u["image"]),
                           f"{x}: ablate {arm}\nμ|Δ|={u['delta_vs_ref']['mean_abs']:.4f}  red L={u['color_layout']['red_left_share']:.2f}"))
    _panels(panels, f"anima — E4-4 relation-head token ablation ({rid}; top-{s.args.relation_top_heads} heads)",
            st.dir / f"relation_before_after_{rid}.png", ncols=1 + len(arms))


# =========================================================================== driver

STAGE_FUNCS: dict[str, Callable[[AnimaSuite, StageStore], dict[str, Any]]] = {
    "multiprompt_profile": stage_multiprompt_profile, "block_third_knockout": stage_block_third_knockout,
    "e2_1_stableflow": stage_e2_1_stableflow, "e3_2_pci": stage_e3_2_pci, "e3_4_sap": stage_e3_4_sap,
    "e4_1_sinks": stage_e4_1_sinks, "e4_4_relations": stage_e4_4_relations}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 2 Anima unified 4-axis & Tier-3 suite (pure XAI L4 baseline)")
    p.add_argument("--stages", nargs="+", default=["all"], choices=(*STAGES, "all"),
                   help="stages to run in canonical order (default: all)")
    p.add_argument("--models-dir", type=Path, default=os.environ.get("XAI_DIT_MODELS_DIR"),
                   help="root with diffusion_models/ text_encoders/ vae/ (default: $XAI_DIT_MODELS_DIR)")
    p.add_argument("--comfyui-dir", default=os.environ.get("COMFYUI_DIR"), help="ComfyUI checkout (default: $COMFYUI_DIR)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true", help="print the plan without importing torch / ComfyUI")
    p.add_argument("--resume", action="store_true", help="skip units already in <stage>/progress.json (images on disk)")
    p.add_argument("--set", type=Path, default=SET_PATH, help="frozen unified prompt set JSON")
    p.add_argument("--cache-root", type=Path, default=REPO_ROOT / "cache" / "cond")
    p.add_argument("--te", help="override TE file (only used on conditioning-cache misses)")
    p.add_argument("--query-chunk", type=int, default=1024)
    p.add_argument("--include-optional", action="store_true", help="also run optional_prompts (E2-1, E4-1)")
    p.add_argument("--knockout-prompts", nargs="+", default=list(DEFAULT_KNOCKOUT_PROMPTS), metavar="PID",
                   help="block_third_knockout prompt ids (default: P-A1 P-A2; set's full list: P-A1 P-A2 P-A3 P-A4 P-A-SHORT)")
    p.add_argument("--stableflow-prompts", nargs="+", default=list(DEFAULT_STABLEFLOW_PROMPTS), metavar="PID",
                   help="e2_1_stableflow prompt ids (default: P-A1; set's full list: P-A1 P-A2 P-A3)")
    p.add_argument("--no-random-ctrl", action="store_true", help="block_third_knockout without the random_ctrl row")
    p.add_argument("--sink-top-k", type=int, default=1, help="E4-1 dynamic sink units per head")
    p.add_argument("--sink-blocks", type=int, nargs="+", help="E4-1 blocks (default: 25/50/75%% depth = 7 14 21)")
    p.add_argument("--sap-switch-step", type=int, help="E3-4 override of the SAP switch step (default: DiT-ST s_rel)")
    p.add_argument("--relation-top-heads", type=int, default=4, help="E4-4 number of (block, head) relation heads")
    p.add_argument("--span-agg", choices=SPAN_AGG_CHOICES, default=DEFAULT_SPAN_AGG, help="E1-1 span aggregation (default mean)")
    add_heatmap_display_args(p)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    stages = list(STAGES) if "all" in args.stages else [st for st in STAGES if st in args.stages]
    uset, set_sha = load_unified_set(args.set)
    runtime = check_runtime(uset)
    plan = build_plan(uset, stages, args)
    log(f"unified set {uset['set_id']} (sha256 {set_sha[:16]}), seed={args.seed}, runtime={runtime}")
    log("runtime flags: --disable-dynamic-vram --highvram; fast=OFF ck_attention=OFF sage=OFF EasyCache=OFF")
    for st in stages:
        log(f"plan [{st}] {json.dumps(plan[st], ensure_ascii=False)} -> {Path(args.out_dir).name}/{st}/")
    log(f"plan [_shared] {json.dumps(plan['_shared'])}; texts to condition: {len(plan['_texts'])}; "
        f"~{plan['_total_generations']} generations total")
    if args.dry_run:
        return 0
    if args.models_dir is None:
        p.error("--models-dir (or $XAI_DIT_MODELS_DIR) is required unless --dry-run")
    args.models_dir = Path(args.models_dir).expanduser()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    c, flags = bootstrap_baseline(args)
    import torch

    summary: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "set_id": uset["set_id"], "seed": args.seed,
                               "stages": {}, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    t_all = time.monotonic()
    suite: Optional[AnimaSuite] = None
    try:
        with torch.inference_mode():
            suite = AnimaSuite(c, args, uset, set_sha, plan["_texts"])
            write_json(out / "run_config.json", {
                "schema_version": SCHEMA_VERSION, "exp_id": "step2_anima_suite", "set_id": uset["set_id"],
                "set_sha256": set_sha, "stages": stages, "plan": {k: v for k, v in plan.items() if k != "_texts"},
                "comfyui_flags": flags, **suite.config(),
                "visualization": {"span_agg": args.span_agg, "heatmap_norm": args.heatmap_norm, "heatmap_cmap": args.heatmap_cmap},
                "token_maps": {pid: token_map_json(suite.token_map(e["prompt"])) for pid, e in suite.P.items()
                               if e["prompt"] in suite.conds},
                "options": {"sink_top_k": args.sink_top_k, "sink_blocks": args.sink_blocks, "sap_switch_step": args.sap_switch_step,
                            "relation_top_heads": args.relation_top_heads, "include_optional": args.include_optional,
                            "knockout_prompts": args.knockout_prompts, "stableflow_prompts": args.stableflow_prompts,
                            "random_ctrl": not args.no_random_ctrl, "query_chunk": args.query_chunk, "resume": args.resume}})
            for n, st in enumerate(stages, 1):
                log(f"=== stage {n}/{len(stages)}: {st} ===")
                t0 = time.monotonic()
                store = StageStore(out, st, args.resume)
                try:
                    res = STAGE_FUNCS[st](suite, store)
                    store.finish({**res, "elapsed_sec": time.monotonic() - t0})
                    summary["stages"][st] = {"status": res.get("status", "ok"), "wall_sec": time.monotonic() - t0}
                except Exception as ex:  # one failing stage never aborts the rest; partial units stay on disk
                    log(f"ERROR [{st}] {type(ex).__name__}: {ex}")
                    print(traceback.format_exc(), flush=True)
                    summary["stages"][st] = {"status": "error", "error": f"{type(ex).__name__}: {ex}",
                                             "units_done": len(store.units), "wall_sec": time.monotonic() - t0}
                    c.mm.soft_empty_cache()
                summary["generations"] = suite.n_generations
                summary["wall_clock_sec"] = time.monotonic() - t_all
                write_json(out / "summary.json", summary)
    finally:
        if suite is not None:
            suite.close()
        else:
            unload_all(c)
    statuses = {k: v["status"] for k, v in summary["stages"].items()}
    log(f"STEP 2 ANIMA SUITE DONE in {time.monotonic() - t_all:.1f}s -> {out.name}: {statuses}")
    return 0 if all(v == "ok" for v in statuses.values()) and len(statuses) == len(stages) else 1


if __name__ == "__main__":
    sys.exit(main())
