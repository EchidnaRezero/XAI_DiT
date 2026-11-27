"""Phase 2: Accelerated XAI Heatmap (E1-1) Speed & Fidelity Evaluation + Multi-Seed/Prompt Robustness.

Compares unaccelerated full-block E1-1 heatmaps against each model's Phase 1 validated
Best Preset:
  - ``unaccel_full``:      PyTorch SDPA, no EasyCache, all blocks observed (ground-truth reference)
  - ``best_ec_full``:      Phase 1 Best Preset (EasyCache + verified flags), all blocks observed
  - ``best_ec_selective``: Phase 1 Best Preset, selective mid/semantic blocks only (unhooked blocks
                           stay 100% on fast kernel; cond-only)
  - ``turbo_8step_*``:     (Krea 2) 8-step Krea 2 Turbo FP8 heatmap vs 32-step Raw teacher

Logs Pearson r, Spearman rho, Top-15% IoU, wall-clock speedup, saves ComfyUI 3-chunk PNG EXIF,
and renders side-by-side heatmap overlay comparison grids.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..runners.cli import (
    MODEL_DEFAULTS,
    REPO_ROOT,
    empty_latent,
    install_runtime_wrappers,
    load_local_env,
    load_text_encoder,
    log,
    resolve_weight,
    unload_all,
    with_shift,
    write_json,
)
from ..runners.e1_e4_profile import (
    DEFAULT_HEATMAP_CLIP_PCT,
    DEFAULT_HEATMAP_CMAP,
    DEFAULT_HEATMAP_NORM,
    DEFAULT_HEATMAP_SMOOTH_SIGMA,
    _load_lightweight_tokenizer,
    _match_target_spans_in_S,
    prepare_heatmap,
    smooth_heatmap,
    span_agg_divisor,
)
from .metrics_and_exif import build_comfy_prompt_graph, save_png_with_exif
from .presets import SAFE_FAST_FEATURES, get_prompt
from .run_t2i import apply_easycache, bootstrap_with_accel, zero_out_conditioning


# Target tags per prompt_id (3-4 semantic tags each)
TARGETS_BY_PROMPT: dict[str, tuple[str, ...]] = {
    "anima_kanao": ("@ogipote", "tsuyuri kanao", "demon slayer uniform", "sketchbook"),
    "anima_emilia": ("emilia \\(re:zero\\)", "soft window sunlight", "anime coloring"),
    "moss_emilia": ("emilia \\(re:zero\\)", "soft window sunlight", "anime coloring"),
    "moss_idol_photo": ("Korean woman", "cream knit cardigan", "silver necklace", "iced latte"),
    "krea2_idol_cafe": ("Korean woman", "ivory ribbed knit cardigan", "golden-hour sunlight", "silver earrings"),
    "krea2_idol_neon": ("Korean woman", "navy trench coat", "red cashmere scarf", "neon reflections"),
}

# Selective semantic/mid-block subsets (keeps ~75-85% of blocks on native fast kernel)
SELECTIVE_BLOCKS: dict[str, tuple[int, ...]] = {
    "anima": (9, 11, 13, 15, 17, 19),          # 6 / 28 blocks (21%)
    "moss_image": (12, 16, 20, 24, 28, 32),    # 6 / 42 blocks (14%)
    "krea2": (8, 12, 16, 20, 24),              # 5 / 28 blocks (18%)
}


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    return ranks


def heatmap_fidelity(ref_map: np.ndarray, cand_map: np.ndarray, top_frac: float = 0.15) -> dict[str, float]:
    """Spatial Pearson r, Spearman rho, and Top-k% IoU on smoothed 64x64 heatmaps."""
    r = smooth_heatmap(ref_map, DEFAULT_HEATMAP_SMOOTH_SIGMA).ravel()
    c = smooth_heatmap(cand_map, DEFAULT_HEATMAP_SMOOTH_SIGMA).ravel()
    r0 = r - r.mean()
    c0 = c - c.mean()
    denom = float(np.sqrt((r0 * r0).sum() * (c0 * c0).sum()))
    pearson = float((r0 * c0).sum() / denom) if denom > 1e-12 else 0.0

    rr = _rankdata(r)
    cr = _rankdata(c)
    rr0 = rr - rr.mean()
    cr0 = cr - cr.mean()
    rdenom = float(np.sqrt((rr0 * rr0).sum() * (cr0 * cr0).sum()))
    spearman = float((rr0 * cr0).sum() / rdenom) if rdenom > 1e-12 else 0.0

    r_thr = float(np.quantile(r, 1.0 - top_frac))
    c_thr = float(np.quantile(c, 1.0 - top_frac))
    rm = r >= r_thr
    cm = c >= c_thr
    inter = int(np.logical_and(rm, cm).sum())
    union = int(np.logical_or(rm, cm).sum())
    iou = float(inter / max(union, 1))
    return {"pearson": round(pearson, 4), "spearman": round(spearman, 4), "top15_iou": round(iou, 4)}


def encode_with_ids(c: Any, adapter: Any, te_path: Path, clip_type: str, texts: Sequence[str], cache_dir: Path) -> dict[str, dict[str, Any]]:
    import hashlib

    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for t in dict.fromkeys(texts):
        key = hashlib.sha256(f"{te_path.name}|{t}".encode("utf-8")).hexdigest()[:24]
        p = cache_dir / f"p2_{key}.pt"
        if p.is_file():
            out[t] = c.torch.load(p, map_location="cpu", weights_only=False)
        else:
            missing.append(t)
    if missing:
        clip = load_text_encoder(c, te_path, clip_type)
        for t in missing:
            tokens = clip.tokenize(t)
            enc = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
            payload: dict[str, Any] = {"cond": enc.pop("cond").cpu(), "token_ids": adapter.dit_token_ids(tokens)}
            for k, v in enc.items():
                if isinstance(v, c.torch.Tensor):
                    payload[f"extra.{k}"] = v.cpu()
            out[t] = payload
            key = hashlib.sha256(f"{te_path.name}|{t}".encode("utf-8")).hexdigest()[:24]
            c.torch.save(payload, cache_dir / f"p2_{key}.pt")
        del clip
        unload_all(c)
        gc.collect()
    return out


def to_cond(entry: Mapping[str, Any]) -> list:
    extras = {k[len("extra."):]: v for k, v in entry.items() if k.startswith("extra.")}
    return [[entry["cond"], extras]]


def run_heatmap_mode(
    c: Any,
    *,
    model_name: str,
    mode_name: str,
    base_patcher: Any,
    vae: Any,
    adapter: Any,
    conds: Mapping[str, dict[str, Any]],
    prompt_text: str,
    negative_text: str,
    target_names: Sequence[str],
    resolved_spans: Mapping[str, tuple[int, int]],
    dit_name: str,
    te_name: str,
    vae_name: str,
    seed: int,
    steps: int,
    cfg: float,
    shift: float,
    sampler: str,
    scheduler: str,
    attention: str,
    fast_features: Sequence[str],
    easycache: bool,
    ec_thresh: float,
    ec_start: float,
    ec_end: float,
    block_subset: Optional[Sequence[int]],
    zero_negative: bool,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    from ..core.session import XAISession

    d = MODEL_DEFAULTS[model_name]
    torch = c.torch
    patcher = with_shift(c, base_patcher, shift)
    # CRITICAL: Attach EasyCache BEFORE install_runtime_wrappers("split")!
    # ComfyUI executes CALC_COND_BATCH wrappers in registration order (first = outermost).
    # If cfg_batching_wrapper("split") wraps outside easycache_calc_cond_batch_wrapper,
    # the uncond branch resets easycache.skip_current_step=False and computes uncond FRESH
    # while cond uses the CACHED output -> asymmetric CFG speckled noise collapse!
    ec_holder = None
    if easycache:
        patcher, ec_holder = apply_easycache(patcher, ec_thresh, ec_start, ec_end)

    install_runtime_wrappers(c, patcher, d.batching)

    adapter.bind(patcher)

    ms = patcher.get_model_object("model_sampling")
    sigmas = c.samplers.calculate_sigmas(ms, scheduler, steps).cpu()
    latent = empty_latent(c, patcher, width=1024, height=1024)

    positive = to_cond(conds[prompt_text])
    negative_c = zero_out_conditioning(c, positive) if zero_negative else to_cond(conds[negative_text])

    dev_gpu = c.mm.get_torch_device()
    side = 64
    n_tags = len(target_names)
    maps_gpu = torch.zeros((n_tags, side * side), dtype=torch.float32, device=dev_gpu)
    maps_cnt = 0
    span_div = {t: span_agg_divisor(resolved_spans[t][1] - resolved_spans[t][0], "mean") for t in target_names}
    allowed_blocks = set(block_subset) if block_subset is not None else None

    session = XAISession(
        adapter.capability,
        query_chunk=1024,
    )

    def observer_factory(call: Any) -> Any:
        if allowed_blocks is not None and call.block not in allowed_blocks:
            return None
        cond_rows = call.ctx.rows("cond")
        if not cond_rows:
            return None
        layout = call.layout
        q_img = layout.q_img
        n_img = layout.n_img
        if n_img != side * side:
            return None
        txt_offset = layout.k_txt.start
        tag_buf = torch.zeros((n_tags, n_img), dtype=torch.float32, device=call.device)

        def obs(probs: Any, q_sl: slice, k_sl: slice) -> None:
            qi0 = max(q_sl.start, q_img.start)
            qi1 = min(q_sl.stop, q_img.stop)
            if qi0 >= qi1:
                return
            local_q = slice(qi0 - q_sl.start, qi1 - q_sl.start)
            img_slice = slice(qi0 - q_img.start, qi1 - q_img.start)
            p_h = probs[cond_rows[0], :, local_q, :].float().mean(dim=0)
            ks0, ks1 = k_sl.start, k_sl.stop
            for t_idx, tname in enumerate(target_names):
                ts0, ts1 = resolved_spans[tname]
                a0 = max(ks0, txt_offset + ts0)
                a1 = min(ks1, txt_offset + ts1)
                if a0 < a1:
                    tag_buf[t_idx, img_slice] += p_h[:, a0 - ks0 : a1 - ks0].sum(dim=-1) / span_div[tname]

        def finish() -> None:
            nonlocal maps_cnt
            maps_gpu.add_(tag_buf)
            maps_cnt += 1

        obs.finish = finish  # type: ignore[attr-defined]
        return obs

    session.add_observer(observer_factory)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        noise = c.sample.prepare_noise(latent, seed)
        sampler_obj = c.samplers.sampler_object(sampler)
        with adapter.attached(session):
            samples = c.sample.sample_custom(
                patcher, noise, cfg, sampler_obj, sigmas, positive, negative_c,
                latent, disable_pbar=True, seed=seed,
            )
        decoded = vae.decode(samples)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

    img_t = decoded
    while img_t.ndim > 3:
        img_t = img_t[0]
    img_np = np.clip(img_t.float().cpu().numpy(), 0.0, 1.0)
    rgb_u8 = (img_np * 255.0).round().astype(np.uint8)

    maps_np = (maps_gpu / max(maps_cnt, 1)).reshape(n_tags, side, side).cpu().numpy()
    maps_dict = {t: maps_np[i] for i, t in enumerate(target_names)}

    accel_meta = {
        "disable_dynamic_vram": True,
        "highvram": True,
        "attention": attention,
        "fast_features": list(fast_features),
        "easycache": easycache,
        "easycache_threshold": ec_thresh if easycache else None,
        "easycache_start": ec_start if easycache else None,
        "easycache_end": ec_end if easycache else None,
        "easycache_steps_skipped": int(getattr(ec_holder, "steps_skipped", 0)) if ec_holder else 0,
        "xai_blocks": list(block_subset) if block_subset is not None else "all",
    }
    meta = {
        "mode": mode_name,
        "model": model_name,
        "prompt": prompt_text,
        "negative_prompt": negative_text if not zero_negative else "<ConditioningZeroOut>",
        "steps": steps,
        "cfg": cfg,
        "shift": shift,
        "sampler": sampler,
        "scheduler": scheduler,
        "seed": seed,
        "width": 1024,
        "height": 1024,
        "weights": {"dit": dit_name, "te": te_name, "vae": vae_name},
        "acceleration": accel_meta,
        "wall_clock_sec": round(elapsed, 2),
        "peak_vram_gb": round(peak_vram, 2),
        "heatmap_hook_calls": maps_cnt,
    }
    graph = build_comfy_prompt_graph(
        model_name=model_name, dit_name=dit_name, te_name=te_name, vae_name=vae_name,
        clip_type=d.clip_type, prompt=prompt_text, negative=negative_text,
        steps=steps, cfg=cfg, shift=shift, sampler=sampler, scheduler=scheduler,
        seed=seed, width=1024, height=1024, accel=accel_meta,
        filename_prefix=f"{model_name}_{mode_name}", zero_negative=zero_negative,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save_png_with_exif(rgb_u8, out_dir / f"{mode_name}_gen.png", meta, graph)
    log(f"[{model_name}/{mode_name}] {elapsed:.1f}s | vram={peak_vram:.2f}GB | hook_calls={maps_cnt}")
    return meta, maps_dict, img_np


def render_phase2_grid(
    title: str,
    mode_entries: Sequence[tuple[str, np.ndarray, Mapping[str, np.ndarray], str]],
    target_names: Sequence[str],
    out_path: Path,
) -> None:
    """Grid: rows = modes (Baseline / Best EC Full / Best EC Selective / Turbo), cols = [Image | Tag1..TagN]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    nrows = len(mode_entries)
    ncols = 1 + len(target_names)
    cmap = plt.get_cmap(DEFAULT_HEATMAP_CMAP)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.3 * nrows), dpi=125)
    if nrows == 1:
        axes = np.expand_dims(axes, 0)

    for r, (mname, img_np, maps, subtitle) in enumerate(mode_entries):
        H, W = img_np.shape[:2]
        axes[r, 0].imshow(img_np)
        axes[r, 0].set_title(f"{mname}\n{subtitle}", fontsize=8.5, fontweight="bold")
        axes[r, 0].axis("off")

        def resize_map(m2d: np.ndarray) -> np.ndarray:
            return np.asarray(Image.fromarray(m2d.astype(np.float32), mode="F").resize((W, H), Image.BILINEAR), dtype=np.float32)

        for c_i, tag in enumerate(target_names, start=1):
            ax = axes[r, c_i]
            mn = prepare_heatmap(maps[tag], DEFAULT_HEATMAP_NORM, DEFAULT_HEATMAP_SMOOTH_SIGMA, DEFAULT_HEATMAP_CLIP_PCT, resize=resize_map)
            ov = np.clip(0.52 * img_np + 0.48 * cmap(mn)[..., :3], 0.0, 1.0)
            ax.imshow(ov)
            ax.set_title(tag.replace("\\(", "(").replace("\\)", ")"), fontsize=8.5)
            ax.axis("off")

    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def run_model_phase2(args: argparse.Namespace) -> int:
    from ..adapters import get_adapter

    model_name = args.model
    d = MODEL_DEFAULTS[model_name]
    prompt_ids = [s.strip() for s in args.prompt_ids.split(",") if s.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # Model-verified Phase 1 process flags
    if model_name == "anima":
        attn, fast_f = "pytorch", ()
        ec_t, ec_s, ec_e = 0.20, 0.20, 0.80
    elif model_name == "moss_image":
        attn, fast_f = "pytorch", SAFE_FAST_FEATURES
        ec_t, ec_s, ec_e = 0.25, 0.20, 0.85
    else:  # krea2
        attn, fast_f = "ck", SAFE_FAST_FEATURES
        ec_t, ec_s, ec_e = 0.20, 0.25, 0.80

    c = bootstrap_with_accel(
        comfyui_dir=args.comfyui_dir,
        attention=attn,
        fast_features=fast_f,
        force_upcast_attention=False,
        dont_upcast_attention=False,
    )
    adapter = get_adapter(model_name)
    models_dir = args.models_dir
    te_path = resolve_weight("te", d.te, models_dir)
    vae_path = resolve_weight("vae", d.vae, models_dir)
    dit_path = resolve_weight("dit", d.dit, models_dir)

    pspecs = [get_prompt(model_name, pid) for pid in prompt_ids]
    all_texts = [t for p in pspecs for t in (p.prompt, p.negative)]
    conds = encode_with_ids(c, adapter, te_path, d.clip_type, all_texts, args.out_dir / "_cond_cache")

    tok = _load_lightweight_tokenizer(model_name)
    spans_by_pid: dict[str, dict[str, tuple[int, int]]] = {}
    for p in pspecs:
        tm = adapter.build_token_map(conds[p.prompt]["token_ids"], branch="cond")
        t_objs = [{"text": t, "occurrence_idx": 0} for t in TARGETS_BY_PROMPT[p.prompt_id]]
        resolved, _ = _match_target_spans_in_S(tok, tm.token_ids, tm.spans["S"], t_objs)
        spans_by_pid[p.prompt_id] = resolved
    del tok

    vae = c.sd.VAE(sd=c.utils.load_torch_file(str(vae_path)))
    base_patcher = c.sd.load_diffusion_model(str(dit_path))

    # Optional Krea 2 Turbo patcher loaded after Raw runs
    all_rows: list[dict[str, Any]] = []
    cached_grid_data: dict[tuple[str, int], list[tuple[str, np.ndarray, dict[str, np.ndarray], str]]] = {}
    cached_ref_maps: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    for p in pspecs:
        tags = TARGETS_BY_PROMPT[p.prompt_id]
        r_spans = spans_by_pid[p.prompt_id]
        for sd in seeds:
            case_dir = args.out_dir / f"{p.prompt_id}_s{sd}"
            # 1. Reference: uncached full-block heatmap
            m_ref, maps_ref, img_ref = run_heatmap_mode(
                c, model_name=model_name, mode_name="ref_full",
                base_patcher=base_patcher, vae=vae, adapter=adapter, conds=conds,
                prompt_text=p.prompt, negative_text=p.negative, target_names=tags, resolved_spans=r_spans,
                dit_name=dit_path.name, te_name=te_path.name, vae_name=vae_path.name,
                seed=sd, steps=d.steps, cfg=d.cfg, shift=d.shift, sampler=d.sampler, scheduler=d.scheduler,
                attention=attn, fast_features=fast_f, easycache=False, ec_thresh=ec_t, ec_start=ec_s, ec_end=ec_e,
                block_subset=None, zero_negative=False, out_dir=case_dir,
            )
            cached_ref_maps[(p.prompt_id, sd)] = maps_ref
            cached_grid_data[(p.prompt_id, sd)] = [("Ref Full (No Cache)", img_ref, maps_ref, f"{m_ref['wall_clock_sec']:.1f}s")]
            all_rows.append({"prompt_id": p.prompt_id, "seed": sd, "mode": "ref_full", "wall_sec": m_ref["wall_clock_sec"], "speedup": 1.0, "mean_pearson": 1.0, "mean_spearman": 1.0, "mean_top15_iou": 1.0, "per_tag": {}})

            # 2. Best EasyCache + All Blocks
            m_ec, maps_ec, img_ec = run_heatmap_mode(
                c, model_name=model_name, mode_name="best_ec_full",
                base_patcher=base_patcher, vae=vae, adapter=adapter, conds=conds,
                prompt_text=p.prompt, negative_text=p.negative, target_names=tags, resolved_spans=r_spans,
                dit_name=dit_path.name, te_name=te_path.name, vae_name=vae_path.name,
                seed=sd, steps=d.steps, cfg=d.cfg, shift=d.shift, sampler=d.sampler, scheduler=d.scheduler,
                attention=attn, fast_features=fast_f, easycache=True, ec_thresh=ec_t, ec_start=ec_s, ec_end=ec_e,
                block_subset=None, zero_negative=False, out_dir=case_dir,
            )
            fid_ec = {t: heatmap_fidelity(maps_ref[t], maps_ec[t]) for t in tags}
            mp_ec = float(np.mean([v["pearson"] for v in fid_ec.values()]))
            ms_ec = float(np.mean([v["spearman"] for v in fid_ec.values()]))
            mi_ec = float(np.mean([v["top15_iou"] for v in fid_ec.values()]))
            spd_ec = m_ref["wall_clock_sec"] / max(m_ec["wall_clock_sec"], 0.01)
            cached_grid_data[(p.prompt_id, sd)].append(("Best EasyCache Full", img_ec, maps_ec, f"{m_ec['wall_clock_sec']:.1f}s ({spd_ec:.2f}x) r={mp_ec:.3f}"))
            all_rows.append({"prompt_id": p.prompt_id, "seed": sd, "mode": "best_ec_full", "wall_sec": m_ec["wall_clock_sec"], "speedup": round(spd_ec, 2), "mean_pearson": round(mp_ec, 4), "mean_spearman": round(ms_ec, 4), "mean_top15_iou": round(mi_ec, 4), "per_tag": fid_ec})

            # 3. Best EasyCache + Selective Blocks (unhooked blocks stay 100% on fast kernel!)
            sel_blocks = SELECTIVE_BLOCKS[model_name]
            m_sel, maps_sel, img_sel = run_heatmap_mode(
                c, model_name=model_name, mode_name="best_ec_selective",
                base_patcher=base_patcher, vae=vae, adapter=adapter, conds=conds,
                prompt_text=p.prompt, negative_text=p.negative, target_names=tags, resolved_spans=r_spans,
                dit_name=dit_path.name, te_name=te_path.name, vae_name=vae_path.name,
                seed=sd, steps=d.steps, cfg=d.cfg, shift=d.shift, sampler=d.sampler, scheduler=d.scheduler,
                attention=attn, fast_features=fast_f, easycache=True, ec_thresh=ec_t, ec_start=ec_s, ec_end=ec_e,
                block_subset=sel_blocks, zero_negative=False, out_dir=case_dir,
            )
            fid_sel = {t: heatmap_fidelity(maps_ref[t], maps_sel[t]) for t in tags}
            mp_sel = float(np.mean([v["pearson"] for v in fid_sel.values()]))
            ms_sel = float(np.mean([v["spearman"] for v in fid_sel.values()]))
            mi_sel = float(np.mean([v["top15_iou"] for v in fid_sel.values()]))
            spd_sel = m_ref["wall_clock_sec"] / max(m_sel["wall_clock_sec"], 0.01)
            cached_grid_data[(p.prompt_id, sd)].append((f"Best EC Selective ({len(sel_blocks)}b)", img_sel, maps_sel, f"{m_sel['wall_clock_sec']:.1f}s ({spd_sel:.2f}x) r={mp_sel:.3f}"))
            all_rows.append({"prompt_id": p.prompt_id, "seed": sd, "mode": "best_ec_selective", "wall_sec": m_sel["wall_clock_sec"], "speedup": round(spd_sel, 2), "mean_pearson": round(mp_sel, 4), "mean_spearman": round(ms_sel, 4), "mean_top15_iou": round(mi_sel, 4), "per_tag": fid_sel})

    # 4. Krea 2 Turbo 8-step heatmap comparison
    if model_name == "krea2":
        del base_patcher
        unload_all(c)
        gc.collect()
        dit_turbo = resolve_weight("dit", "krea2_turbo_fp8_scaled.safetensors", models_dir)
        turbo_patcher = c.sd.load_diffusion_model(str(dit_turbo))
        for p in pspecs:
            tags = TARGETS_BY_PROMPT[p.prompt_id]
            r_spans = spans_by_pid[p.prompt_id]
            for sd in seeds:
                case_dir = args.out_dir / f"{p.prompt_id}_s{sd}"
                maps_ref = cached_ref_maps[(p.prompt_id, sd)]
                ref_wall = next(r["wall_sec"] for r in all_rows if r["prompt_id"] == p.prompt_id and r["seed"] == sd and r["mode"] == "ref_full")
                m_tb, maps_tb, img_tb = run_heatmap_mode(
                    c, model_name=model_name, mode_name="turbo_8step_selective",
                    base_patcher=turbo_patcher, vae=vae, adapter=adapter, conds=conds,
                    prompt_text=p.prompt, negative_text=p.negative, target_names=tags, resolved_spans=r_spans,
                    dit_name=dit_turbo.name, te_name=te_path.name, vae_name=vae_path.name,
                    seed=sd, steps=8, cfg=1.0, shift=d.shift, sampler="euler", scheduler="simple",
                    attention=attn, fast_features=fast_f, easycache=False, ec_thresh=0.20, ec_start=0.25, ec_end=0.80,
                    block_subset=SELECTIVE_BLOCKS["krea2"], zero_negative=True, out_dir=case_dir,
                )
                fid_tb = {t: heatmap_fidelity(maps_ref[t], maps_tb[t]) for t in tags}
                mp_tb = float(np.mean([v["pearson"] for v in fid_tb.values()]))
                ms_tb = float(np.mean([v["spearman"] for v in fid_tb.values()]))
                mi_tb = float(np.mean([v["top15_iou"] for v in fid_tb.values()]))
                spd_tb = ref_wall / max(m_tb["wall_clock_sec"], 0.01)
                cached_grid_data[(p.prompt_id, sd)].append(("Turbo 8-step Selective", img_tb, maps_tb, f"{m_tb['wall_clock_sec']:.1f}s ({spd_tb:.2f}x) r={mp_tb:.3f}"))
                all_rows.append({"prompt_id": p.prompt_id, "seed": sd, "mode": "turbo_8step_selective", "wall_sec": m_tb["wall_clock_sec"], "speedup": round(spd_tb, 2), "mean_pearson": round(mp_tb, 4), "mean_spearman": round(ms_tb, 4), "mean_top15_iou": round(mi_tb, 4), "per_tag": fid_tb})

    # Render grids & markdown summary
    for (pid, sd), entries in cached_grid_data.items():
        render_phase2_grid(
            f"Phase 2 Accelerated XAI Heatmap — {model_name} / {pid} (seed={sd})",
            entries,
            TARGETS_BY_PROMPT[pid],
            args.out_dir / f"grid_{pid}_s{sd}.png",
        )

    lines = [
        f"# Phase 2 Accelerated XAI Heatmap Summary — `{model_name}`",
        "",
        "| Prompt | Seed | Mode | Wall (s) | Speedup | Mean Pearson r | Mean Spearman ρ | Mean Top-15% IoU |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in all_rows:
        lines.append(
            f"| `{r['prompt_id']}` | {r['seed']} | `{r['mode']}` | {r['wall_sec']:.1f} | {r['speedup']:.2f}x | "
            f"{r['mean_pearson']:.4f} | {r['mean_spearman']:.4f} | {r['mean_top15_iou']:.4f} |"
        )
    md = "\n".join(lines) + "\n"
    (args.out_dir / "summary.md").write_text(md, encoding="utf-8")
    write_json(args.out_dir / "summary.json", {"model": model_name, "rows": all_rows})
    print("\n" + md, flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 2 Accelerated XAI Heatmap Fidelity & Robustness runner.")
    p.add_argument("--model", required=True, choices=["anima", "moss_image", "krea2"])
    p.add_argument("--prompt-ids", required=True, help="Comma-separated prompt IDs.")
    p.add_argument("--seeds", default="42,101", help="Comma-separated seeds for robustness.")
    p.add_argument("--comfyui-dir", default=None)
    p.add_argument("--models-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_local_env(REPO_ROOT / "local/.env")
    return run_model_phase2(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
