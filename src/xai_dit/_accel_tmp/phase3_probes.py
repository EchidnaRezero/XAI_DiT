"""Phase 3 surgical code-level root-cause verification probes (P3-A, P3-B, P3-C, P3-D).

Empirically verifies the 4 static-audit hypotheses from issues/004–007 on Colab L4:
  - P3-A: ``PRIORITIZE_FP16`` (Anima BF16 & Krea 2 FP8: stock ``--fast fp16_accumulation`` vs
          ``PRIORITIZE_FP16=False`` with ``allow_fp16_accumulation=True``) + Block-0 dtype audit
          + 1-step ``torch.profiler`` kernel summary quantifying ``aten::_to_copy`` casts.
  - P3-B: ``TorchCompile + EasyCache`` on Anima (stock vs patched ``.item()`` float accumulation
          on ``cumulative_change_rate``), logging ``torch._dynamo.utils.counters["recompiles"]``
          and Rep 1 / Rep 2 wall times.
  - P3-C: ``Krea 2 + SageAttention`` NaN isolation in 3 modes: (1) stock global, (2) bypass
          ``TextFusionTransformer`` (``low_precision_attention=False``), (3) bypass + ``smooth_k=True``,
          with lightweight NaN/Inf forward hooks pinpointing exact first-NaN block & step.
  - P3-D: ``EasyCache`` ``subsample_factor=8`` vs ``1`` on Anima (``end=0.95, thresh=0.20``),
          logging per-step ``approx_output_change_rate`` and late-step HF Laplacian ratio.

Each probe variant runs in an isolated subprocess when invoked from the top-level CLI so
ComfyUI process-level flags (``--fast``, ``--use-sage-attention``) never leak across modes.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence

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
    weight_precision,
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
    attach_block0_runtime_dtype_hook,
    audit_block0_dtypes,
    cuda_timer,
    profile_single_step_kernels,
)
from .presets import get_prompt
from .run_t2i import (
    apply_easycache,
    apply_torch_compile,
    bootstrap_with_accel,
    ensure_conds,
)


# --------------------------------------------------------------------------- Probe spec registry (pure Python)

PROBE_IDS: tuple[str, ...] = ("P3-A", "P3-B", "P3-C", "P3-D")

PROBE_SUBJOBS: dict[str, list[dict[str, Any]]] = {
    "P3-A": [
        {"probe": "P3-A", "model": "anima", "mode": "baseline", "attention": "pytorch", "fast": ()},
        {"probe": "P3-A", "model": "anima", "mode": "stock_fp16acc", "attention": "pytorch", "fast": ("fp16_accumulation",)},
        {"probe": "P3-A", "model": "anima", "mode": "patch_no_prioritize_fp16", "attention": "pytorch", "fast": ("fp16_accumulation",)},
        {"probe": "P3-A", "model": "krea2", "mode": "baseline", "attention": "pytorch", "fast": ()},
        {"probe": "P3-A", "model": "krea2", "mode": "stock_fp16acc", "attention": "pytorch", "fast": ("fp16_accumulation",)},
        {"probe": "P3-A", "model": "krea2", "mode": "patch_no_prioritize_fp16", "attention": "pytorch", "fast": ("fp16_accumulation",)},
    ],
    "P3-B": [
        {"probe": "P3-B", "model": "anima", "mode": "stock_compile_ec20", "attention": "pytorch", "fast": ()},
        {"probe": "P3-B", "model": "anima", "mode": "patched_item_compile_ec20", "attention": "pytorch", "fast": ()},
    ],
    "P3-C": [
        {"probe": "P3-C", "model": "krea2", "mode": "baseline_sdpa", "attention": "pytorch", "fast": ()},
        {"probe": "P3-C", "model": "krea2", "mode": "sage_stock_global", "attention": "sage", "fast": ()},
        {"probe": "P3-C", "model": "krea2", "mode": "sage_bypass_text_fusion", "attention": "sage", "fast": ()},
        {"probe": "P3-C", "model": "krea2", "mode": "sage_bypass_and_smooth_k", "attention": "sage", "fast": ()},
    ],
    "P3-D": [
        {"probe": "P3-D", "model": "anima", "mode": "baseline", "attention": "pytorch", "fast": ()},
        {"probe": "P3-D", "model": "anima", "mode": "ec20_subsample_8", "attention": "pytorch", "fast": ()},
        {"probe": "P3-D", "model": "anima", "mode": "ec20_subsample_1", "attention": "pytorch", "fast": ()},
    ],
}


# --------------------------------------------------------------------------- Probe-specific patches & hooks

class FirstNaNTracker:
    """Forward hooks across TextFusionTransformer + all DiT blocks to pinpoint exact first-NaN origin."""

    def __init__(self, torch_mod: Any) -> None:
        self.torch = torch_mod
        self.handles: list[Any] = []
        self.step_counter = 0
        self.first_nan: Optional[dict[str, Any]] = None
        self.first_inf: Optional[dict[str, Any]] = None
        self.max_abs_by_module: dict[str, float] = {}

    def attach(self, patcher: Any) -> int:
        dm = patcher.model.diffusion_model
        targets: list[tuple[str, Any]] = []

        # Krea 2 TextFusionTransformer (layerwise_blocks + refiner_blocks)
        for attr in ("text_fusion", "txt_fusion", "text_encoder_fusion"):
            tf = getattr(dm, attr, None)
            if tf is not None:
                for sub in ("layerwise_blocks", "refiner_blocks"):
                    sub_blocks = getattr(tf, sub, None)
                    if sub_blocks is not None:
                        for i, blk in enumerate(sub_blocks):
                            targets.append((f"{attr}.{sub}[{i}]", blk))
                targets.append((attr, tf))

        # Generic check for TextFusionTransformer by class name if attribute name differs
        if not targets:
            for mod_name, mod in dm.named_modules():
                if type(mod).__name__ in ("TextFusionTransformer", "TextFusionBlock"):
                    targets.append((mod_name, mod))

        # All DiT blocks
        for attr in ("blocks", "layers", "double_blocks", "single_blocks"):
            cand = getattr(dm, attr, None)
            if cand is not None and hasattr(cand, "__len__") and len(cand) > 0:
                for i, blk in enumerate(cand):
                    targets.append((f"{attr}[{i}]", blk))
                break

        for name, mod in targets:
            self.handles.append(mod.register_forward_hook(self._make_hook(name)))
        return len(targets)

    def _make_hook(self, name: str) -> Callable[..., None]:
        torch = self.torch

        def _hook(_mod: Any, _inp: Any, out: Any) -> None:
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not isinstance(t, torch.Tensor):
                return
            with torch.no_grad():
                nan_c = int(torch.isnan(t).sum().item())
                inf_c = int(torch.isinf(t).sum().item())
                if nan_c > 0 and self.first_nan is None:
                    self.first_nan = {
                        "module": name,
                        "step": self.step_counter,
                        "nan_count": nan_c,
                        "shape": list(t.shape),
                        "dtype": str(t.dtype),
                    }
                if inf_c > 0 and self.first_inf is None:
                    self.first_inf = {
                        "module": name,
                        "step": self.step_counter,
                        "inf_count": inf_c,
                        "shape": list(t.shape),
                        "dtype": str(t.dtype),
                    }
                if nan_c == 0 and inf_c == 0:
                    m = float(t.detach().abs().max().item())
                    prev = self.max_abs_by_module.get(name, 0.0)
                    if m > prev:
                        self.max_abs_by_module[name] = round(m, 3)

        return _hook

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def summary(self) -> dict[str, Any]:
        top_peaks = sorted(self.max_abs_by_module.items(), key=lambda kv: kv[1], reverse=True)[:10]
        return {
            "first_nan": self.first_nan,
            "first_inf": self.first_inf,
            "any_nan": self.first_nan is not None,
            "any_inf": self.first_inf is not None,
            "top_max_abs": dict(top_peaks),
        }


def _patch_easycache_item_fix() -> Callable[[], None]:
    """Monkey-patch ``nodes_easycache.easycache_forward_wrapper`` so ``cumulative_change_rate``
    accumulates a Python ``float`` (via ``.item()``) instead of a 0-D ``torch.Tensor``.
    """
    import comfy_extras.nodes_easycache as nec

    orig_wrapper = nec.easycache_forward_wrapper

    def _patched_wrapper(executor: Any, *args: Any, **kwargs: Any) -> Any:
        transformer_options = args[-1]
        if not isinstance(transformer_options, dict):
            transformer_options = kwargs.get("transformer_options") or args[-2]
        easycache = transformer_options["easycache"]
        x, ax = nec._extract_tensor(args[0], easycache.output_channels)
        sigmas = transformer_options["sigmas"]
        uuids = transformer_options["uuids"]
        if sigmas is not None and easycache.is_past_end_timestep(sigmas):
            return executor(*args, **kwargs)
        has_first_cond_uuid = easycache.has_first_cond_uuid(uuids)
        next_x_prev = x
        input_change = None
        do_easycache = easycache.should_do_easycache(sigmas)
        if do_easycache:
            easycache.check_metadata(x)
            can_apply_cache_diff = easycache.can_apply_cache_diff(uuids)
            if easycache.skip_current_step and can_apply_cache_diff:
                result = easycache.apply_cache_diff(x, uuids)
                if ax is not None:
                    return [result, easycache.apply_cache_diff(ax, uuids, is_audio=True)]
                return result
            if easycache.initial_step:
                easycache.first_cond_uuid = uuids[0]
                has_first_cond_uuid = easycache.has_first_cond_uuid(uuids)
                easycache.initial_step = False
            if has_first_cond_uuid:
                if easycache.has_x_prev_subsampled():
                    input_change = (
                        easycache.subsample(x, uuids, clone=False) - easycache.x_prev_subsampled
                    ).flatten().abs().mean()
                if easycache.has_output_prev_norm() and easycache.has_relative_transformation_rate():
                    approx_output_change_rate = (
                        easycache.relative_transformation_rate * input_change
                    ) / easycache.output_prev_norm
                    # FIX: convert 0-D Tensor to Python float so TorchDynamo type guards stay stable
                    val = (
                        float(approx_output_change_rate.item())
                        if hasattr(approx_output_change_rate, "item")
                        else float(approx_output_change_rate)
                    )
                    easycache.cumulative_change_rate = float(easycache.cumulative_change_rate) + val
                    if easycache.cumulative_change_rate < easycache.reuse_threshold and can_apply_cache_diff:
                        easycache.skip_current_step = True
                        result = easycache.apply_cache_diff(x, uuids)
                        if ax is not None:
                            return [result, easycache.apply_cache_diff(ax, uuids, is_audio=True)]
                        return result
                    else:
                        easycache.cumulative_change_rate = 0.0

        full_output = executor(*args, **kwargs)
        output, audio_output = nec._extract_tensor(full_output, easycache.output_channels)
        if has_first_cond_uuid and easycache.has_output_prev_norm():
            output_change = (
                easycache.subsample(output, uuids, clone=False) - easycache.output_prev_subsampled
            ).flatten().abs().mean()
            if easycache.verbose:
                output_change_rate = output_change / easycache.output_prev_norm
                easycache.output_change_rates.append(float(output_change_rate.item()))
            if easycache.has_relative_transformation_rate():
                approx_output_change_rate = (
                    easycache.relative_transformation_rate * input_change
                ) / easycache.output_prev_norm
                easycache.approx_output_change_rates.append(float(approx_output_change_rate.item()))
            if input_change is not None:
                easycache.relative_transformation_rate = float((output_change / input_change).item())
        easycache.update_cache_diff(output, next_x_prev, uuids)
        if audio_output is not None:
            easycache.update_cache_diff(audio_output, ax, uuids, is_audio=True)
        if has_first_cond_uuid:
            easycache.x_prev_subsampled = easycache.subsample(next_x_prev, uuids)
            easycache.output_prev_subsampled = easycache.subsample(output, uuids)
            easycache.output_prev_norm = float(output.flatten().abs().mean().item())
        return full_output

    nec.easycache_forward_wrapper = _patched_wrapper

    def _restore() -> None:
        nec.easycache_forward_wrapper = orig_wrapper

    return _restore


def _patch_krea2_text_fusion_sdpa(patcher: Any) -> Callable[[], None]:
    """Force ``low_precision_attention=False`` inside Krea 2's ``TextFusionTransformer``."""
    import comfy.ldm.modules.attention as attn_mod

    dm = patcher.model.diffusion_model
    restores: list[Callable[[], None]] = []

    for mod_name, mod in dm.named_modules():
        if type(mod).__name__ == "TextFusionTransformer":
            orig_fwd = mod.forward

            def _make_wrapped(fn: Callable[..., Any]) -> Callable[..., Any]:
                def _wrapped(x: Any, mask: Any = None, transformer_options: Optional[dict[str, Any]] = None) -> Any:
                    orig_opt_attn = attn_mod.optimized_attention_masked

                    def _sdpa_only(*a: Any, **kw: Any) -> Any:
                        kw["low_precision_attention"] = False
                        return orig_opt_attn(*a, **kw)

                    # Patch inside krea2.model module namespace as well
                    import comfy.ldm.krea2.model as krea2_mod

                    orig_krea_opt = krea2_mod.optimized_attention_masked
                    krea2_mod.optimized_attention_masked = _sdpa_only
                    try:
                        return fn(x, mask=mask, transformer_options=transformer_options or {})
                    finally:
                        krea2_mod.optimized_attention_masked = orig_krea_opt

                return _wrapped

            mod.forward = _make_wrapped(orig_fwd)
            restores.append(lambda m=mod, f=orig_fwd: setattr(m, "forward", f))

    def _restore_all() -> None:
        for r in restores:
            r()

    return _restore_all


def _patch_sage_smooth_k(enable: bool = True) -> Callable[[], None]:
    """Monkey-patch ComfyUI's ``attention_sage`` to pass ``smooth_k=True`` to ``sageattn``."""
    import comfy.ldm.modules.attention as attn_mod

    orig_sageattn = getattr(attn_mod, "sageattn", None)
    if orig_sageattn is None:
        return lambda: None

    def _wrapped_sageattn(*args: Any, **kwargs: Any) -> Any:
        kwargs["smooth_k"] = enable
        return orig_sageattn(*args, **kwargs)

    attn_mod.sageattn = _wrapped_sageattn

    def _restore() -> None:
        attn_mod.sageattn = orig_sageattn

    return _restore


# --------------------------------------------------------------------------- Single subprobe execution (GPU)

def _run_subprobe(args: argparse.Namespace) -> dict[str, Any]:
    d = MODEL_DEFAULTS[args.model]
    pspec = get_prompt(args.model, args.prompt_id)
    prompt_text = pspec.prompt
    negative_text = pspec.negative
    fast_features = tuple(f for f in (args.fast.split(",") if args.fast else []) if f)

    c = bootstrap_with_accel(
        comfyui_dir=args.comfyui_dir,
        attention=args.attention,
        fast_features=fast_features,
        force_upcast_attention=False,
        dont_upcast_attention=False,
        no_highvram=args.no_highvram,
    )

    # Probe P3-A: monkey-patch PRIORITIZE_FP16 = False while keeping allow_fp16_accumulation = True
    if args.probe == "P3-A" and args.mode == "patch_no_prioritize_fp16":
        c.mm.PRIORITIZE_FP16 = False
        c.torch.backends.cuda.matmul.allow_fp16_accumulation = True

    te_path = resolve_weight("te", d.te, args.models_dir)
    vae_path = resolve_weight("vae", d.vae, args.models_dir)
    dit_path = resolve_weight("dit", d.dit, args.models_dir)

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
    base_patcher = c.sd.load_diffusion_model(str(dit_path))
    patcher = with_shift(c, base_patcher, d.shift)

    restores: list[Callable[[], None]] = []
    ec_holder = None
    extra_probe_data: dict[str, Any] = {}

    if args.probe == "P3-B":
        if args.mode == "patched_item_compile_ec20":
            restores.append(_patch_easycache_item_fix())
        patcher, ec_holder = apply_easycache(patcher, 0.20, 0.15, 0.95)
        patcher = apply_torch_compile(patcher, "inductor")
    elif args.probe == "P3-C":
        if args.mode in ("sage_bypass_text_fusion", "sage_bypass_and_smooth_k"):
            restores.append(_patch_krea2_text_fusion_sdpa(patcher))
        if args.mode == "sage_bypass_and_smooth_k":
            restores.append(_patch_sage_smooth_k(True))
    elif args.probe == "P3-D":
        if args.mode in ("ec20_subsample_8", "ec20_subsample_1"):
            from comfy_extras.nodes_easycache import EasyCacheNode

            out = EasyCacheNode.execute(patcher, 0.20, 0.15, 0.95, verbose=True)
            patcher = out.result[0] if hasattr(out, "result") else out[0]
            ec_holder = patcher.model_options.get("transformer_options", {}).get("easycache")
            if ec_holder is not None:
                ec_holder.subsample_factor = 8 if args.mode == "ec20_subsample_8" else 1

    # Block-0 dtype audit + runtime step-0 capture
    dtype_audit = audit_block0_dtypes(patcher, torch_mod=c.torch, mm_mod=c.mm)
    runtime_dtype, rm_dtype_hook = attach_block0_runtime_dtype_hook(patcher)
    restores.append(rm_dtype_hook)

    # First-NaN tracker for P3-C
    nan_tracker = FirstNaNTracker(c.torch) if args.probe == "P3-C" else None
    if nan_tracker is not None:
        nan_tracker.attach(patcher)
        restores.append(nan_tracker.remove)

    # Forward counter
    actual_fwds = 0
    step_kernel_profile: Optional[dict[str, Any]] = None
    profile_first_fwd = args.probe == "P3-A"

    def _count_and_profile_fwd(executor: Any, *a: Any, **kw: Any) -> Any:
        nonlocal actual_fwds, step_kernel_profile
        actual_fwds += 1
        if nan_tracker is not None:
            nan_tracker.step_counter = actual_fwds
        if profile_first_fwd and step_kernel_profile is None and actual_fwds == 2:
            # Profile 2nd forward (steady-state after any 1st-step lazy init)
            res, step_kernel_profile = profile_single_step_kernels(
                lambda: executor(*a, **kw),
                top_k=15,
                torch_mod=c.torch,
            )
            return res
        return executor(*a, **kw)

    patcher.add_wrapper_with_key(c.pe.WrappersMP.DIFFUSION_MODEL, "phase3_probe_counter", _count_and_profile_fwd)

    repeat = 2 if args.probe == "P3-B" else 1
    rep_denoise_secs: list[float] = []
    t_vae_s = 0.0
    vram_peak_gb = 0.0
    vram_reserved_gb = 0.0
    last_img_f32: Optional[np.ndarray] = None

    try:
        for rep_idx in range(repeat):
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
                    samples = c.sample.sample(
                        patcher,
                        noise,
                        d.steps,
                        d.cfg,
                        d.sampler,
                        d.scheduler,
                        conds[prompt_text],
                        conds[negative_text],
                        latent,
                        denoise=1.0,
                        disable_noise=False,
                        start_step=None,
                        last_step=None,
                        force_full_denoise=True,
                        noise_mask=None,
                        callback=None,
                        disable_pbar=True,
                        seed=args.seed,
                    )
                with cuda_timer(c.torch) as vae_timer:
                    decoded = vae.decode(samples)

            if c.torch.cuda.is_available():
                vram_peak_gb = max(vram_peak_gb, c.torch.cuda.max_memory_allocated() / (1024 ** 3))
                vram_reserved_gb = max(vram_reserved_gb, c.torch.cuda.max_memory_reserved() / (1024 ** 3))
            rep_denoise_secs.append(round(denoise_timer.elapsed_s, 4))
            t_vae_s = vae_timer.elapsed_s

            img_t = decoded
            while img_t.ndim > 3:
                img_t = img_t[0]
            last_img_f32 = img_t.float().cpu().numpy()
            log(f"[{args.probe}/{args.model}/{args.mode}] rep {rep_idx + 1}/{repeat}: denoise={denoise_timer.elapsed_s:.2f}s fwds={actual_fwds}")
    finally:
        for r in reversed(restores):
            try:
                r()
            except Exception:
                pass

    # Collect probe-specific diagnostics
    if args.probe == "P3-B":
        try:
            import torch._dynamo.utils as dynamo_utils

            recompiles = dict(dynamo_utils.counters.get("recompiles", {}))
            extra_probe_data["dynamo_recompiles"] = recompiles
            extra_probe_data["total_recompiles"] = int(sum(recompiles.values()))
        except Exception as exc:
            extra_probe_data["dynamo_recompiles_error"] = str(exc)
        extra_probe_data["rep_denoise_secs"] = rep_denoise_secs
    elif args.probe == "P3-C" and nan_tracker is not None:
        extra_probe_data["nan_tracker"] = nan_tracker.summary()
    elif args.probe == "P3-D" and ec_holder is not None:
        approx_rates = [round(float(v), 5) for v in getattr(ec_holder, "approx_output_change_rates", [])]
        actual_rates = [round(float(v), 5) for v in getattr(ec_holder, "output_change_rates", [])]
        extra_probe_data["subsample_factor"] = int(getattr(ec_holder, "subsample_factor", -1))
        extra_probe_data["total_steps_skipped"] = int(getattr(ec_holder, "total_steps_skipped", 0))
        extra_probe_data["approx_output_change_rates"] = approx_rates
        extra_probe_data["output_change_rates"] = actual_rates

    nominal_fwds = d.steps * (2 if d.cfg > 1.0 else 1)
    t_compile_warmup_s = max(0.0, rep_denoise_secs[0] - rep_denoise_secs[-1]) if len(rep_denoise_secs) > 1 else 0.0
    timing = TimingBreakdown.from_measurements(
        t_te_encode_s=te_timer.elapsed_s,
        t_compile_warmup_s=t_compile_warmup_s,
        t_denoise_pure_s=rep_denoise_secs[-1],
        actual_forwards=actual_fwds,
        nominal_forwards=nominal_fwds,
        t_vae_decode_s=t_vae_s,
        vram_peak_gb=vram_peak_gb,
        vram_reserved_gb=vram_reserved_gb,
    )

    assert last_img_f32 is not None
    out_dir = args.out_dir / args.probe / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_png = out_dir / ("baseline_sdpa.png" if args.probe == "P3-C" else "baseline.png")
    ref_f32 = load_png_float01(baseline_png) if (baseline_png.is_file() and args.mode not in ("baseline", "baseline_sdpa")) else None
    img_metrics = compute_image_metrics(last_img_f32, ref_f32)
    rgb_u8 = (np.clip(np.nan_to_num(last_img_f32), 0.0, 1.0) * 255.0).round().astype(np.uint8)

    weights_meta = {"dit": dit_path.name, "te": te_path.name, "vae": vae_path.name}
    accel_meta = {
        "disable_dynamic_vram": True,
        "highvram": not args.no_highvram,
        "attention": args.attention,
        "fast_features": list(fast_features),
        "probe": args.probe,
        "mode": args.mode,
    }
    result: dict[str, Any] = {
        "probe": args.probe,
        "model": args.model,
        "mode": args.mode,
        "preset": f"{args.probe}_{args.model}_{args.mode}",
        "prompt": prompt_text,
        "negative_prompt": negative_text,
        "steps": d.steps,
        "cfg": d.cfg,
        "shift": d.shift,
        "sampler": d.sampler,
        "scheduler": d.scheduler,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "weights": weights_meta,
        "acceleration": accel_meta,
        "wall_clock_sec": round(timing.t_denoise_pure_s + timing.t_vae_decode_s, 3),
        "timing": timing.to_dict(),
        "dtype_audit": {**dtype_audit, "runtime_step0": runtime_dtype},
        "kernel_profile": step_kernel_profile,
        "probe_data": extra_probe_data,
        "metrics": img_metrics,
        "comfyui_commit": comfyui_commit(c.pkg_dir),
    }

    comfy_graph = build_comfy_prompt_graph(
        model_name=args.model,
        dit_name=weights_meta["dit"],
        te_name=weights_meta["te"],
        vae_name=weights_meta["vae"],
        clip_type=d.clip_type,
        prompt=prompt_text,
        negative=negative_text,
        steps=d.steps,
        cfg=d.cfg,
        shift=d.shift,
        sampler=d.sampler,
        scheduler=d.scheduler,
        seed=args.seed,
        width=args.width,
        height=args.height,
        accel=accel_meta,
        filename_prefix=f"{args.probe}_{args.model}_{args.mode}",
    )

    png_path = out_dir / f"{args.mode}.png"
    save_png_with_exif(rgb_u8, png_path, result, comfy_graph)
    result["png_file"] = str(png_path.relative_to(args.out_dir))
    write_json(out_dir / f"{args.mode}_result.json", result)
    unload_all(c)
    gc.collect()
    return result


# --------------------------------------------------------------------------- Orchestrator & CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 3 surgical root-cause verification probes (P3-A..P3-D).")
    p.add_argument("--probe", required=True, choices=["P3-A", "P3-B", "P3-C", "P3-D", "all"])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--comfyui-dir", default=None)
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--cond-cache-dir", type=Path, default=None)
    p.add_argument("--prompt-id", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--no-highvram", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    # Internal single-subprobe worker flags
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--model", default=None, help=argparse.SUPPRESS)
    p.add_argument("--mode", default=None, help=argparse.SUPPRESS)
    p.add_argument("--attention", default="pytorch", help=argparse.SUPPRESS)
    p.add_argument("--fast", default="", help=argparse.SUPPRESS)
    return p


def select_probe_jobs(probe: str) -> list[dict[str, Any]]:
    if probe == "all":
        jobs: list[dict[str, Any]] = []
        for pid in PROBE_IDS:
            jobs.extend(PROBE_SUBJOBS[pid])
        return jobs
    return list(PROBE_SUBJOBS[probe])


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_local_env(REPO_ROOT / "local/.env")
    args = build_parser().parse_args(argv)

    if args._worker:
        _run_subprobe(args)
        return 0

    jobs = select_probe_jobs(args.probe)
    log(f"phase3_probes: probe={args.probe} subjobs={len(jobs)}")
    for j in jobs:
        log(f"  - {j['probe']} | {j['model']:10s} | {j['mode']:28s} | attn={j['attention']} fast={list(j['fast'])}")

    if args.dry_run:
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cond_cache = args.cond_cache_dir or (args.out_dir / "_cond_cache")
    all_results: list[dict[str, Any]] = []

    for j in jobs:
        cmd = [
            sys.executable,
            "-m",
            "xai_dit._accel_tmp.phase3_probes",
            "--_worker",
            "--probe", j["probe"],
            "--model", j["model"],
            "--mode", j["mode"],
            "--attention", j["attention"],
            "--fast", ",".join(j["fast"]),
            "--out-dir", str(args.out_dir),
            "--cond-cache-dir", str(cond_cache),
            "--seed", str(args.seed),
            "--width", str(args.width),
            "--height", str(args.height),
        ]
        if args.comfyui_dir:
            cmd += ["--comfyui-dir", str(args.comfyui_dir)]
        if args.models_dir:
            cmd += ["--models-dir", str(args.models_dir)]
        if args.prompt_id:
            cmd += ["--prompt-id", args.prompt_id]
        if args.no_highvram:
            cmd.append("--no-highvram")

        log(f"launching subprocess: {j['probe']} / {j['model']} / {j['mode']}")
        proc = subprocess.run(cmd, check=False)
        res_path = args.out_dir / j["probe"] / j["model"] / f"{j['mode']}_result.json"
        if proc.returncode == 0 and res_path.is_file():
            all_results.append(json.loads(res_path.read_text(encoding="utf-8")))
        else:
            all_results.append(
                {
                    "probe": j["probe"],
                    "model": j["model"],
                    "mode": j["mode"],
                    "error": f"subprocess exit code {proc.returncode}",
                }
            )

    write_json(args.out_dir / "probes_summary.json", {"probe": args.probe, "results": all_results})

    # Contact sheet per probe/model
    by_group: dict[tuple[str, str], list[tuple[str, Path]]] = {}
    for r in all_results:
        if r.get("error"):
            continue
        key = (r["probe"], r["model"])
        t = r.get("timing", {})
        m = r.get("metrics", {})
        label = f"{r['mode']} ({t.get('t_denoise_pure_s', 0):.1f}s, HF={m.get('hf_ratio_vs_base', 1.0):.3f})"
        by_group.setdefault(key, []).append((label, args.out_dir / r["png_file"]))
    for (pid, model), items in by_group.items():
        make_comparison_sheet(items, args.out_dir / pid / model / "contact_sheet.png", cols=min(4, len(items)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
