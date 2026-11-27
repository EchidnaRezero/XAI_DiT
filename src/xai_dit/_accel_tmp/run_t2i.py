"""Single-process Headless ComfyUI T2I acceleration runner + optional internal diagnostic probe.

Sets ComfyUI process-level CLI flags (attention backend, individual ``--fast`` features,
upcast flags) BEFORE importing ``comfy.model_management`` so kernels initialize cleanly.
Supports running one or more presets sharing the same process-level flags without reloading
the DiT, caching text conditioning on disk, and saving ComfyUI 3-chunk PNG EXIF metadata.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import numpy as np

from ..runners.cli import (
    MODEL_DEFAULTS,
    REPO_ROOT,
    comfyui_commit,
    empty_latent,
    load_local_env,
    load_text_encoder,
    log,
    resolve_weight,
    unload_all,
    weight_precision,
    with_shift,
    write_json,
)
from .metrics_and_exif import (
    build_comfy_prompt_graph,
    compute_image_metrics,
    load_png_float01,
    save_png_with_exif,
)
from .presets import AccelPreset, get_prompt, presets_by_name


# --------------------------------------------------------------------------- ComfyUI bootstrap

def bootstrap_with_accel(
    *,
    comfyui_dir: Optional[str],
    attention: str,
    fast_features: Sequence[str],
    force_upcast_attention: bool,
    dont_upcast_attention: bool,
    no_highvram: bool = False,
) -> SimpleNamespace:
    """Import torch + ComfyUI with exact process-level acceleration flags applied first."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cdir = comfyui_dir or os.environ.get("COMFYUI_DIR")
    if cdir:
        sys.path.insert(0, str(Path(cdir).expanduser().resolve()))
    try:
        import comfy.options
    except ImportError as exc:
        raise SystemExit("ComfyUI not importable: pass --comfyui-dir or set COMFYUI_DIR") from exc

    comfy.options.enable_args_parsing(False)
    import comfy.cli_args

    comfy.cli_args.args.disable_dynamic_vram = True
    comfy.cli_args.args.highvram = not no_highvram

    # Attention backend (mutually exclusive)
    comfy.cli_args.args.use_pytorch_cross_attention = False
    comfy.cli_args.args.use_ck_attention = False
    comfy.cli_args.args.use_flash_attention = False
    comfy.cli_args.args.use_sage_attention = False
    if attention == "pytorch":
        comfy.cli_args.args.use_pytorch_cross_attention = True
    elif attention == "ck":
        comfy.cli_args.args.use_ck_attention = True
    elif attention == "flash":
        comfy.cli_args.args.use_flash_attention = True
    elif attention == "sage":
        comfy.cli_args.args.use_sage_attention = True
    else:
        raise ValueError(f"Unsupported attention backend: {attention!r}")

    # Individual --fast PerformanceFeature selection
    if fast_features:
        by_val = {pf.value: pf for pf in comfy.cli_args.PerformanceFeature}
        selected = set()
        for feat in fast_features:
            if feat not in by_val:
                raise ValueError(f"Unknown --fast feature {feat!r}; choices: {sorted(by_val)}")
            selected.add(by_val[feat])
        comfy.cli_args.args.fast = selected
    else:
        comfy.cli_args.args.fast = set()

    comfy.cli_args.args.force_upcast_attention = bool(force_upcast_attention)
    comfy.cli_args.args.dont_upcast_attention = bool(dont_upcast_attention)

    import comfy.model_management
    import comfy.patcher_extension
    import comfy.sample
    import comfy.samplers
    import comfy.sd
    import comfy.utils
    import torch

    return SimpleNamespace(
        torch=torch,
        comfy=comfy,
        mm=comfy.model_management,
        sd=comfy.sd,
        sample=comfy.sample,
        samplers=comfy.samplers,
        utils=comfy.utils,
        pe=comfy.patcher_extension,
        pkg_dir=Path(comfy.sd.__file__).resolve().parent,
    )


# --------------------------------------------------------------------------- Disk conditioning cache

def _cond_cache_path(cache_dir: Path, model_name: str, te_name: str, text: str) -> Path:
    key = hashlib.sha256(f"{model_name}|{Path(te_name).name}|{text}".encode("utf-8")).hexdigest()[:24]
    return cache_dir / f"{model_name}_{key}.pt"


def ensure_conds(
    c: SimpleNamespace,
    *,
    model_name: str,
    te_path: Path,
    clip_type: str,
    texts: Sequence[str],
    cache_dir: Optional[Path],
) -> dict[str, list]:
    """Load cached conditioning tensors or encode missing ones once and unload the TE."""
    unique = list(dict.fromkeys(texts))
    conds: dict[str, list] = {}
    missing: list[str] = []

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for text in unique:
            p = _cond_cache_path(cache_dir, model_name, te_path.name, text)
            if p.is_file():
                conds[text] = c.torch.load(p, map_location="cpu", weights_only=False)
            else:
                missing.append(text)
    else:
        missing = unique

    if missing:
        t0 = time.perf_counter()
        log(f"encoding {len(missing)} text(s) with {te_path.name} ({weight_precision(te_path.name)})")
        clip = load_text_encoder(c, te_path, clip_type)
        for text in missing:
            tokens = clip.tokenize(text)
            out = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            cond = out.pop("cond").cpu()
            extras = {k: v.cpu() if isinstance(v, c.torch.Tensor) else v for k, v in out.items()}
            entry = [[cond, extras]]
            conds[text] = entry
            if cache_dir is not None:
                c.torch.save(entry, _cond_cache_path(cache_dir, model_name, te_path.name, text))
        del clip
        unload_all(c)
        gc.collect()
        if c.torch.cuda.is_available():
            c.torch.cuda.empty_cache()
        log(f"TE encode done in {time.perf_counter() - t0:.1f}s (TE unloaded)")

    return conds


def zero_out_conditioning(c: SimpleNamespace, cond_list: list) -> list:
    """ComfyUI ConditioningZeroOut equivalent (used by Krea 2 Turbo)."""
    out = []
    for cond, meta in cond_list:
        new_meta = dict(meta)
        if "pooled_output" in new_meta and isinstance(new_meta["pooled_output"], c.torch.Tensor):
            new_meta["pooled_output"] = c.torch.zeros_like(new_meta["pooled_output"])
        out.append([c.torch.zeros_like(cond), new_meta])
    return out


# --------------------------------------------------------------------------- Node-level accelerations

def apply_easycache(
    patcher: Any,
    reuse_threshold: float,
    start_percent: float,
    end_percent: float,
) -> tuple[Any, Optional[Any]]:
    from comfy_extras.nodes_easycache import EasyCacheNode

    out = EasyCacheNode.execute(patcher, reuse_threshold, start_percent, end_percent, verbose=False)
    patched = out.result[0] if hasattr(out, "result") else out[0]
    holder = patched.model_options.get("transformer_options", {}).get("easycache")
    return patched, holder


def apply_torch_compile(patcher: Any, backend: str) -> Any:
    from comfy_extras.nodes_torch_compile import TorchCompileModel

    out = TorchCompileModel.execute(patcher, backend)
    return out.result[0] if hasattr(out, "result") else out[0]


def apply_lora_model_only(c: SimpleNamespace, patcher: Any, lora_path: Path, strength: float) -> Any:
    lora = c.utils.load_torch_file(str(lora_path), safe_load=True)
    patched_model, _ = c.sd.load_lora_for_models(patcher, None, lora, strength, 0.0)
    return patched_model


# --------------------------------------------------------------------------- Internal diagnostic probe

class InternalDiagProbe:
    """Lightweight forward hooks on selected DiT blocks to detect activation spikes and NaN/Inf onset."""

    def __init__(self, torch_mod: Any) -> None:
        self.torch = torch_mod
        self.handles: list[Any] = []
        self.records: dict[str, dict[str, Any]] = {}
        self.forward_calls = 0

    def attach(self, patcher: Any) -> int:
        dm = patcher.model.diffusion_model
        blocks = None
        for attr in ("blocks", "layers", "double_blocks", "single_blocks"):
            cand = getattr(dm, attr, None)
            if cand is not None and hasattr(cand, "__len__") and len(cand) > 0:
                blocks = cand
                break
        if blocks is None:
            return 0
        n = len(blocks)
        indices = sorted({0, 1, n // 4, n // 2, (3 * n) // 4, n - 1})
        for idx in indices:
            name = f"block_{idx}"
            self.records[name] = {"max_abs": 0.0, "nan_hits": 0, "inf_hits": 0, "calls": 0}
            self.handles.append(blocks[idx].register_forward_hook(self._make_hook(name)))
        return len(indices)

    def _make_hook(self, name: str):
        torch = self.torch

        def hook(_mod: Any, _inp: Any, out: Any) -> None:
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not isinstance(t, torch.Tensor):
                return
            rec = self.records[name]
            rec["calls"] += 1
            with torch.no_grad():
                nan_c = int(torch.isnan(t).sum().item())
                inf_c = int(torch.isinf(t).sum().item())
                rec["nan_hits"] += nan_c
                rec["inf_hits"] += inf_c
                if nan_c == 0 and inf_c == 0:
                    m = float(t.detach().abs().max().item())
                    if m > rec["max_abs"]:
                        rec["max_abs"] = round(m, 3)

        return hook

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def summary(self) -> dict[str, Any]:
        return {
            "blocks": self.records,
            "any_nan": any(v["nan_hits"] > 0 for v in self.records.values()),
            "any_inf": any(v["inf_hits"] > 0 for v in self.records.values()),
            "peak_abs": max((v["max_abs"] for v in self.records.values()), default=0.0),
        }


# --------------------------------------------------------------------------- Single preset execution

def run_single_preset(
    c: SimpleNamespace,
    *,
    model_name: str,
    preset: AccelPreset,
    base_patcher: Any,
    vae: Any,
    conds: dict[str, list],
    prompt_text: str,
    negative_text: str,
    weights_meta: dict[str, str],
    clip_type: str,
    seed: int,
    width: int,
    height: int,
    out_dir: Path,
    models_dir: Optional[Path],
    baseline_png: Optional[Path],
    diag_probe: bool,
    repeat: int,
    no_highvram: bool,
) -> dict[str, Any]:
    d = MODEL_DEFAULTS[model_name]
    steps = preset.steps if preset.steps is not None else d.steps
    cfg = preset.cfg if preset.cfg is not None else d.cfg
    shift = preset.shift if preset.shift is not None else d.shift
    sampler = preset.sampler or d.sampler
    scheduler = preset.scheduler or d.scheduler

    patcher = with_shift(c, base_patcher, shift)

    if preset.lora:
        lora_path = resolve_weight("dit", preset.lora, models_dir) if not Path(preset.lora).is_file() else Path(preset.lora)
        patcher = apply_lora_model_only(c, patcher, lora_path, preset.lora_strength)

    ec_holder = None
    if preset.easycache:
        patcher, ec_holder = apply_easycache(
            patcher,
            preset.easycache_threshold,
            preset.easycache_start,
            preset.easycache_end,
        )

    if preset.torch_compile:
        patcher = apply_torch_compile(patcher, preset.torch_compile)

    # Count actual DiT forwards (excludes EasyCache-skipped forwards when wrapped outside,
    # and measures true compute time).
    fwd_times: list[float] = []

    def _time_fwd(executor: Any, *a: Any, **kw: Any) -> Any:
        if c.torch.cuda.is_available():
            c.torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = executor(*a, **kw)
        if c.torch.cuda.is_available():
            c.torch.cuda.synchronize()
        fwd_times.append(time.perf_counter() - t0)
        return res

    patcher.add_wrapper_with_key(c.pe.WrappersMP.DIFFUSION_MODEL, "accel_tmp_timer", _time_fwd)

    probe = InternalDiagProbe(c.torch) if diag_probe else None
    if probe is not None:
        probe.attach(patcher)

    pos_cond = conds[prompt_text]
    neg_cond = zero_out_conditioning(c, pos_cond) if preset.zero_negative else conds[negative_text]

    run_timings: list[float] = []
    last_img_f32: Optional[np.ndarray] = None
    peak_vram_gb = 0.0
    ec_skipped = 0

    try:
        for rep_idx in range(max(1, repeat)):
            fwd_times.clear()
            if ec_holder is not None and hasattr(ec_holder, "reset"):
                ec_holder.reset()
            if c.torch.cuda.is_available():
                c.torch.cuda.reset_peak_memory_stats()
                c.torch.cuda.synchronize()

            latent = empty_latent(c, patcher, width=width, height=height)
            noise = c.sample.prepare_noise(latent, seed, None)
            t0 = time.perf_counter()
            with c.torch.inference_mode():
                samples = c.sample.sample(
                    patcher,
                    noise,
                    steps,
                    cfg,
                    sampler,
                    scheduler,
                    pos_cond,
                    neg_cond,
                    latent,
                    denoise=1.0,
                    disable_noise=False,
                    start_step=None,
                    last_step=None,
                    force_full_denoise=True,
                    noise_mask=None,
                    callback=None,
                    disable_pbar=True,
                    seed=seed,
                )
                decoded = vae.decode(samples)
            if c.torch.cuda.is_available():
                c.torch.cuda.synchronize()
                peak_vram_gb = max(peak_vram_gb, c.torch.cuda.max_memory_allocated() / (1024 ** 3))
            elapsed = time.perf_counter() - t0
            run_timings.append(round(elapsed, 3))
            if ec_holder is not None:
                ec_skipped = int(getattr(ec_holder, "steps_skipped", 0))
            img_t = decoded
            while img_t.ndim > 3:
                img_t = img_t[0]
            last_img_f32 = img_t.float().cpu().numpy()
            if repeat > 1:
                log(f"  [{preset.name}] rep {rep_idx + 1}/{repeat}: {elapsed:.2f}s (fwds={len(fwd_times)})")
    finally:
        if probe is not None:
            probe.remove()

    assert last_img_f32 is not None
    ref_f32 = load_png_float01(baseline_png) if (baseline_png and baseline_png.is_file()) else None
    img_metrics = compute_image_metrics(last_img_f32, ref_f32)
    rgb_u8 = (np.clip(np.nan_to_num(last_img_f32), 0.0, 1.0) * 255.0).round().astype(np.uint8)

    accel_meta: dict[str, Any] = {
        "disable_dynamic_vram": True,
        "highvram": not no_highvram,
        "attention": preset.attention,
        "fast_features": list(preset.fast_features),
        "force_upcast_attention": preset.force_upcast_attention,
        "dont_upcast_attention": preset.dont_upcast_attention,
        "easycache": preset.easycache,
        "easycache_threshold": preset.easycache_threshold if preset.easycache else None,
        "easycache_start": preset.easycache_start if preset.easycache else None,
        "easycache_end": preset.easycache_end if preset.easycache else None,
        "easycache_steps_skipped": ec_skipped,
        "torch_compile": preset.torch_compile,
        "lora": preset.lora,
        "lora_strength": preset.lora_strength if preset.lora else None,
    }

    sec_per_fwd = round(sum(fwd_times) / max(len(fwd_times), 1), 4)
    result: dict[str, Any] = {
        "preset": preset.name,
        "stage": preset.stage,
        "notes": preset.notes,
        "model": model_name,
        "prompt": prompt_text,
        "negative_prompt": negative_text if not preset.zero_negative else "<ConditioningZeroOut>",
        "steps": steps,
        "cfg": cfg,
        "shift": shift,
        "sampler": sampler,
        "scheduler": scheduler,
        "seed": seed,
        "width": width,
        "height": height,
        "weights": weights_meta,
        "acceleration": accel_meta,
        "wall_clock_sec": run_timings[-1],
        "all_run_secs": run_timings,
        "forward_calls": len(fwd_times),
        "sec_per_forward": sec_per_fwd,
        "peak_vram_gb": round(peak_vram_gb, 3),
        "metrics": img_metrics,
        "diag_probe": probe.summary() if probe is not None else None,
        "comfyui_commit": comfyui_commit(c.pkg_dir),
    }

    comfy_graph = build_comfy_prompt_graph(
        model_name=model_name,
        dit_name=weights_meta["dit"],
        te_name=weights_meta["te"],
        vae_name=weights_meta["vae"],
        clip_type=clip_type,
        prompt=prompt_text,
        negative=negative_text,
        steps=steps,
        cfg=cfg,
        shift=shift,
        sampler=sampler,
        scheduler=scheduler,
        seed=seed,
        width=width,
        height=height,
        accel=accel_meta,
        filename_prefix=f"{model_name}_{preset.name}",
        zero_negative=preset.zero_negative,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{preset.name}.png"
    save_png_with_exif(rgb_u8, png_path, result, comfy_graph)
    result["png_file"] = png_path.name
    write_json(out_dir / f"{preset.name}_result.json", result)

    status = "COLLAPSED" if img_metrics["is_collapsed"] else "OK"
    psnr_str = f" PSNR={img_metrics['psnr_db']}dB HF_ratio={img_metrics['hf_ratio_vs_base']}" if "psnr_db" in img_metrics else ""
    log(
        f"[{model_name}/{preset.name}] {status} | {result['wall_clock_sec']}s "
        f"({sec_per_fwd:.3f}s/fwd, fwds={len(fwd_times)}, vram={peak_vram_gb:.2f}GB){psnr_str}"
    )
    return result


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Single-process Phase 1 T2I acceleration runner.")
    p.add_argument("--model", required=True, choices=["anima", "moss_image", "krea2"])
    p.add_argument("--presets", required=True, help="Comma-separated preset names sharing process-level flags.")
    p.add_argument("--prompt-id", default=None, help="Prompt ID from presets.PROMPTS_BY_MODEL.")
    p.add_argument("--prompt", default=None, help="Custom positive prompt override.")
    p.add_argument("--negative", default=None, help="Custom negative prompt override.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--cfg", type=float, default=None)
    p.add_argument("--shift", type=float, default=None)
    p.add_argument("--sampler", default=None)
    p.add_argument("--scheduler", default=None)
    p.add_argument("--dit", default=None)
    p.add_argument("--te", default=None)
    p.add_argument("--vae", default=None)
    p.add_argument("--comfyui-dir", default=None)
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--cond-cache-dir", type=Path, default=None)
    p.add_argument("--baseline-png", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--diag-probe", action="store_true", help="Attach block-level NaN/M-Act diagnostic hooks.")
    p.add_argument("--repeat", type=int, default=1, help="Repeat runs (use 2 for TorchCompile warmup vs steady-state).")
    p.add_argument("--no-highvram", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_local_env(REPO_ROOT / "local/.env")
    args = build_parser().parse_args(argv)
    d = MODEL_DEFAULTS[args.model]
    catalog = presets_by_name(args.model)

    selected: list[AccelPreset] = []
    for raw_name in args.presets.split(","):
        name = raw_name.strip()
        if not name:
            continue
        if name not in catalog:
            raise SystemExit(f"Unknown preset {name!r} for {args.model}; valid: {sorted(catalog)}")
        selected.append(catalog[name])
    if not selected:
        raise SystemExit("No presets selected.")

    flag_keys = {p.process_flag_key() for p in selected}
    if len(flag_keys) > 1:
        raise SystemExit(
            f"All presets in a single run_t2i invocation must share process-level flags; got {flag_keys}. "
            "Use sweep.py to run mixed flag groups."
        )

    pspec = get_prompt(args.model, args.prompt_id)
    prompt_text = args.prompt if args.prompt is not None else pspec.prompt
    negative_text = args.negative if args.negative is not None else pspec.negative

    if args.dry_run:
        log(
            f"dry-run: model={args.model} presets={[p.name for p in selected]} "
            f"flags={selected[0].process_flag_key()} prompt_id={pspec.prompt_id}"
        )
        return 0

    first = selected[0]
    c = bootstrap_with_accel(
        comfyui_dir=args.comfyui_dir,
        attention=first.attention,
        fast_features=first.fast_features,
        force_upcast_attention=first.force_upcast_attention,
        dont_upcast_attention=first.dont_upcast_attention,
        no_highvram=args.no_highvram,
    )

    te_name = args.te or d.te
    vae_name = args.vae or d.vae
    te_path = resolve_weight("te", te_name, args.models_dir)
    vae_path = resolve_weight("vae", vae_name, args.models_dir)

    conds = ensure_conds(
        c,
        model_name=args.model,
        te_path=te_path,
        clip_type=d.clip_type,
        texts=[prompt_text, negative_text],
        cache_dir=args.cond_cache_dir,
    )

    vae = c.sd.VAE(sd=c.utils.load_torch_file(str(vae_path)))

    # Group by DiT weight file so we only reload DiT if a preset overrides it (e.g., Krea 2 Turbo)
    loaded_dit_name: Optional[str] = None
    base_patcher: Any = None

    for preset in selected:
        # Apply CLI overrides if given
        eff_preset = AccelPreset(
            **{
                **preset.to_dict(),
                "fast_features": preset.fast_features,
                "steps": args.steps if args.steps is not None else preset.steps,
                "cfg": args.cfg if args.cfg is not None else preset.cfg,
                "shift": args.shift if args.shift is not None else preset.shift,
                "sampler": args.sampler if args.sampler is not None else preset.sampler,
                "scheduler": args.scheduler if args.scheduler is not None else preset.scheduler,
            }
        )
        dit_name = args.dit or eff_preset.dit_override or d.dit
        if dit_name != loaded_dit_name:
            if base_patcher is not None:
                del base_patcher
                unload_all(c)
                gc.collect()
            dit_path = resolve_weight("dit", dit_name, args.models_dir)
            log(f"loading DiT {dit_path.name} ({weight_precision(dit_path.name)})")
            base_patcher = c.sd.load_diffusion_model(str(dit_path))
            loaded_dit_name = dit_name

        weights_meta = {"dit": Path(dit_name).name, "te": te_path.name, "vae": vae_path.name}
        rep = max(args.repeat, 2 if eff_preset.torch_compile else 1)
        run_single_preset(
            c,
            model_name=args.model,
            preset=eff_preset,
            base_patcher=base_patcher,
            vae=vae,
            conds=conds,
            prompt_text=prompt_text,
            negative_text=negative_text,
            weights_meta=weights_meta,
            clip_type=d.clip_type,
            seed=args.seed,
            width=args.width,
            height=args.height,
            out_dir=args.out_dir,
            models_dir=args.models_dir,
            baseline_png=args.baseline_png or (args.out_dir / "baseline.png"),
            diag_probe=args.diag_probe,
            repeat=rep,
            no_highvram=args.no_highvram,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
