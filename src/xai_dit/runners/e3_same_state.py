"""E3-0 Same-State Δv [tags x steps] Direct Effect (Logit Knockout) + Attention vs Δv runner.

Fixes the exact base trajectory x_t at the 6 representative steps (full base rollout, state
taken from the k-diffusion callback = x at sigma_i), then evaluates the CFG-combined velocity
difference Δv(x_t) = ||v_base(x_t) - v_cf(x_t)||_2 under Logit Knockout (`direct_effect`,
cond branch, image queries) for each of the 4 manifest targets plus a matched-token-count
`random_ctrl` row. Base and counterfactual both run the manual chunked SDPA on every call
(path-matched), and every evaluation goes through ComfyUI's own ``sample_custom`` path via a
custom ``KSAMPLER`` (process_conds, CFG batching wrapper, full ``sample_sigmas``).

Saves standard outputs under <out_root>/e3_0_<model>_<precision>_<run_id>/:
- visuals/grid_direct_effect_steps.png
- visuals/grid_direct_effect_dsigma_steps.png
- visuals/grid_attn_vs_delta_v.png (if E1-1 maps exist)
- raw/same_state_direct.npz, config.json, metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

from .cli import (
    MODEL_DEFAULTS,
    REPO_ROOT,
    SCHEMA_VERSION,
    _IdentityObserver,
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
from .e1_e4_profile import find_profile_dir


def _pick_random_ctrl_span(
    span_S: Any,
    target_spans: Mapping[str, Sequence[int]],
    seed: int = 42,
) -> tuple[int, int]:
    """Pick a matched-token-count control span inside S disjoint from all target spans."""
    lengths = [int(v[1]) - int(v[0]) for v in target_spans.values() if int(v[1]) > int(v[0])]
    lengths.sort()
    k = lengths[len(lengths) // 2] if lengths else 1
    k = max(1, min(k, max(1, span_S.end - span_S.start)))

    occupied = set()
    for s0, s1 in target_spans.values():
        occupied.update(range(int(s0), int(s1)))

    candidates = [
        (i, i + k)
        for i in range(span_S.start, span_S.end - k + 1)
        if not any(j in occupied for j in range(i, i + k))
    ]
    if candidates:
        return candidates[seed % len(candidates)]
    # Fallback: pick first non-target single token inside S
    free = [i for i in range(span_S.start, span_S.end) if i not in occupied]
    if free:
        i = free[seed % len(free)]
        return (i, i + 1)
    return (span_S.start, min(span_S.start + 1, span_S.end))


def _render_same_state_grid(
    base_img: Any,
    row_names: Sequence[str],
    rep_steps: Sequence[int],
    sigma_table: Sequence[float],
    maps_grid: Any,  # [R, C, H, W] float32 numpy
    out_path: Path,
    title: str,
    unit_label: str,
) -> dict[str, float]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    img_np = base_img.float().clamp(0, 1).cpu().numpy()
    H_img, W_img = img_np.shape[:2]
    cmap = plt.get_cmap("inferno")

    pre_min = float(np.min(maps_grid))
    pre_max = float(np.max(maps_grid))
    vmax = float(np.percentile(maps_grid, 99.5))

    nrows, ncols = len(row_names), len(rep_steps)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.1 * nrows), dpi=130)
    if nrows == 1:
        axes = np.expand_dims(axes, 0)
    if ncols == 1:
        axes = np.expand_dims(axes, 1)

    for r, name in enumerate(row_names):
        for c_idx, step_i in enumerate(rep_steps):
            ax = axes[r, c_idx]
            m_raw = maps_grid[r, c_idx]
            pil = Image.fromarray(m_raw.astype(np.float32), mode="F").resize((W_img, H_img), Image.BILINEAR)
            m = np.asarray(pil, dtype=np.float32)
            mn = np.clip(m / max(vmax, 1e-12), 0.0, 1.0)
            ov = np.clip(0.52 * img_np + 0.48 * cmap(mn)[..., :3], 0.0, 1.0)
            ax.imshow(ov)
            sig = sigma_table[step_i] if step_i < len(sigma_table) else float("nan")
            ax.set_title(f"{name[:18]}\ni={step_i} (σ={sig:.2f}) μ={float(m_raw.mean()):.2e}", fontsize=7.5)
            ax.axis("off")

    fig.suptitle(
        f"{title} | {unit_label} | shared 99.5%={vmax:.2e} (pre-clip [{pre_min:.2e}, {pre_max:.2e}], bilinear)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)
    return {"pre_clip_min": pre_min, "pre_clip_max": pre_max, "vmax_99_5": vmax}


def _render_attn_vs_delta_v(
    base_img: Any,
    target_names: Sequence[str],
    attn_maps: Mapping[str, Any],  # {tag: [H, W]}
    dv_mean_maps: Mapping[str, Any],  # {tag: [H, W]}
    out_path: Path,
    model_title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    img_np = base_img.float().clamp(0, 1).cpu().numpy()
    H_img, W_img = img_np.shape[:2]
    cmap = plt.get_cmap("inferno")

    tags = [t for t in target_names if t in attn_maps and t in dv_mean_maps]
    if not tags:
        return

    attn_vmax = float(np.percentile(np.concatenate([attn_maps[t].ravel() for t in tags]), 99.5))
    dv_vmax = float(np.percentile(np.concatenate([dv_mean_maps[t].ravel() for t in tags]), 99.5))

    fig, axes = plt.subplots(len(tags), 2, figsize=(7.2, 3.3 * len(tags)), dpi=130)
    if len(tags) == 1:
        axes = np.expand_dims(axes, 0)

    for r, tag in enumerate(tags):
        for col, (m_raw, vmax, col_title) in enumerate([
            (attn_maps[tag], attn_vmax, "E1-1 Attention (img→txt)"),
            (dv_mean_maps[tag], dv_vmax, "E3-0 Same-State Δv (direct_effect)"),
        ]):
            ax = axes[r, col]
            pil = Image.fromarray(m_raw.astype(np.float32), mode="F").resize((W_img, H_img), Image.BILINEAR)
            mn = np.clip(np.asarray(pil, dtype=np.float32) / max(vmax, 1e-12), 0.0, 1.0)
            ov = np.clip(0.52 * img_np + 0.48 * cmap(mn)[..., :3], 0.0, 1.0)
            ax.imshow(ov)
            ax.set_title(f"{tag[:22]} — {col_title}\nμ={float(m_raw.mean()):.2e}", fontsize=8.5)
            ax.axis("off")

    fig.suptitle(f"{model_title} — Attention vs Same-State Δv (independent metric scales, 99.5% clip)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    plt.close(fig)


def run_model_same_state(
    c: SimpleNamespace,
    model_name: str,
    prompt_id: str,
    comfy_dir: Path,
    models_dir: Path,
    out_root: Path,
    cache_root: Path,
    run_id: str,
) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from ..adapters import get_adapter
    from ..adapters.base import InterventionSpec, TokenSpan
    from ..core.session import XAISession

    d = MODEL_DEFAULTS[model_name]
    manifest_path = REPO_ROOT / f"experiments/design/manifest_{model_name}.json"
    manifest, manifest_sha = load_manifest(manifest_path)
    entry = pick_prompt(manifest, prompt_id)
    prompt = entry["prompt"]
    negative = entry.get("negative_prompt", manifest.get("default_negative_prompt", d.negative))
    seed = 42

    # Resolved target spans from the finished E1/E4 profile (same model defaults / precision)
    prof_dir = find_profile_dir(out_root, model_name, run_id)
    if prof_dir is None:
        raise FileNotFoundError(f"no finished E1/E4 profile for {model_name} ({d.precision}) under {out_root}; "
                                "run e1_e4_profile first")
    log(f"[{model_name}] target spans from {prof_dir.name}")
    prof_metrics = json.loads((prof_dir / "metrics.json").read_text(encoding="utf-8"))
    target_spans: dict[str, list[int]] = prof_metrics["resolved_target_spans"]
    target_names = list(target_spans.keys())[:4]

    args = SimpleNamespace(
        model=model_name, comfyui_dir=str(comfy_dir), models_dir=models_dir, te=None,
        cache_root=str(cache_root), no_highvram=False, enable_dynamic_vram=False,
    )
    adapter = get_adapter(model_name)
    texts = [prompt, negative]
    with torch.inference_mode():  # TE pass only runs on a cache miss (TE unloaded right after)
        cache, conds = ensure_cond_cache(c, args, d, adapter, texts, manifest_sha)
    tm_cond = adapter.build_token_map(conds[prompt]["token_ids"], branch="cond")
    for t in target_names:
        if int(target_spans[t][1]) > tm_cond.text_len:
            raise ValueError(f"profile span {target_spans[t]} for {t!r} exceeds text length {tm_cond.text_len}; "
                             "profile and E3-0 must use the same TE / conditioning cache")
    ctrl_span = _pick_random_ctrl_span(tm_cond.spans["S"], target_spans, seed=seed)

    row_names = [*target_names, "random_ctrl"]
    row_spans = {t: TokenSpan(int(target_spans[t][0]), int(target_spans[t][1])) for t in target_names}
    row_spans["random_ctrl"] = TokenSpan(ctrl_span[0], ctrl_span[1])
    log(f"[{model_name}] E3-0 rows={[(k, (v.start, v.end)) for k, v in row_spans.items()]}")

    # Quantized int8_convrot / fp8_scaled DiTs load natively (no dtype forced)
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
    rep_steps = [i for i in d.rep_steps if i < d.steps]
    rep_set = set(rep_steps)
    latent = empty_latent(c, patcher)

    positive = to_conditioning(conds[prompt])
    negative_c = to_conditioning(conds[negative])

    def identity_session(interventions: Sequence[InterventionSpec] = ()) -> XAISession:
        """Manual chunked SDPA on every call (cond AND uncond), so base and knockout share one path."""
        sess = XAISession(adapter.capability, interventions=interventions, query_chunk=1024)
        sess.add_observer(lambda call: _IdentityObserver())
        return sess

    # Jobs: path-matched identity reference + 5 rows x 6 rep steps under Logit Knockout (direct_effect).
    ident = "__identity__"
    jobs: list[tuple[str, int, XAISession]] = [(ident, s, identity_session()) for s in rep_steps]
    for rname in row_names:
        spec = InterventionSpec(hook="logit_knockout", cfg_scope="cond_only",
                                params={"keys": (row_spans[rname],), "query": "img"})
        jobs += [(rname, s, identity_session([spec])) for s in rep_steps]

    t0 = time.monotonic()
    x_states: dict[int, Any] = {}
    x0_eval: dict[tuple[str, int], Any] = {}

    def cb(step: int, x0: Any, x: Any, total: int) -> None:
        # k-diffusion fires the callback before the Euler update, so ``x`` is exactly the state at sigma_step
        if step in rep_set:
            x_states[step] = x.detach().clone()
        log(f"[{model_name}] base rollout step {step + 1}/{total} (σ={sigma_table[step]:.3f})"
            f"{' [rep, state saved]' if step in rep_set else ''}")

    def same_state_sampler(model: Any, x: Any, sigmas_: Any, extra_args: Optional[dict] = None,
                           callback: Any = None, disable: Any = None) -> Any:
        """Evaluates the CFG-combined x0 at each stored x_t inside ComfyUI's prepared sampling context
        (process_conds, CFG batching wrapper, full ``sample_sigmas`` -> true step_index)."""
        extra_args = extra_args or {}
        s_in = x.new_ones([x.shape[0]])
        for n, (tag, step_i, sess) in enumerate(jobs, 1):
            x_i = x_states[step_i].to(device=x.device, dtype=x.dtype)
            with adapter.attached(sess):
                x0_eval[(tag, step_i)] = model(x_i, sigmas_[step_i] * s_in, **extra_args).detach().float()
            log(f"[{model_name}] E3-0 eval {n}/{len(jobs)}: {tag} @ step {step_i} (σ={sigma_table[step_i]:.3f}) "
                f"+{time.monotonic() - t0:.1f}s")
        return x

    with torch.inference_mode():
        # 1. Base rollout (fused SDPA, no hooks): exact x_t at the representative steps
        noise = c.sample.prepare_noise(latent, seed)
        samples = c.sample.sample_custom(
            patcher, noise, d.cfg, c.samplers.sampler_object(d.sampler), sigmas, positive, negative_c,
            latent, callback=cb, disable_pbar=True, seed=seed,
        )
        img = vae.decode(samples)
        base_img = img.reshape(-1, *img.shape[-3:])[0].detach().cpu()
        missing = sorted(rep_set - set(x_states))
        if missing:
            raise RuntimeError(f"base rollout did not reach rep steps {missing}")

        # 2. Same-State evaluations through the same sample_custom path (custom KSAMPLER, no integration)
        c.sample.sample_custom(
            patcher, noise, d.cfg, c.samplers.KSAMPLER(same_state_sampler), sigmas, positive, negative_c,
            latent, disable_pbar=True, seed=seed,
        )

        # 3. Δv = v_ident - v_cf = (x0_cf - x0_ident) / σ_i (flow / CONST) -> L2 over latent channels
        dv_grid = np.zeros((len(row_names), len(rep_steps), 64, 64), dtype=np.float32)
        dsigma_grid = np.zeros((len(row_names), len(rep_steps), 64, 64), dtype=np.float32)
        mean_table: dict[str, dict[str, float]] = {}
        for r_idx, rname in enumerate(row_names):
            mean_table[rname] = {}
            for c_idx, step_i in enumerate(rep_steps):
                sig_i = max(sigma_table[step_i], 1e-6)
                dsigma = abs(sigma_table[step_i] - sigma_table[step_i + 1])
                diff = (x0_eval[(rname, step_i)] - x0_eval[(ident, step_i)]) / sig_i
                if diff.ndim == 5:  # Wan21 latents (Anima, Krea 2): [B, C, 1, H, W]
                    diff = diff[:, :, 0]
                spatial = diff.norm(dim=1, keepdim=True)  # [1, 1, H_lat, W_lat]
                spatial_64 = F.interpolate(spatial, size=(64, 64), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
                dv_grid[r_idx, c_idx] = spatial_64
                dsigma_grid[r_idx, c_idx] = spatial_64 * dsigma
                mean_table[rname][f"step_{step_i}"] = float(spatial_64.mean())

    elapsed = time.monotonic() - t0
    log(f"[{model_name}] E3-0 Same-State direct_effect done in {elapsed:.1f}s ({len(jobs)} evals)")

    latent_shape = list(latent.shape)
    del patcher, vae, samples, x_states, x0_eval, jobs
    unload_all(c)

    out_dir = out_root / f"e3_0_{model_name}_{d.precision}_{run_id}"
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "visuals").mkdir(parents=True, exist_ok=True)

    save_png(base_img, out_dir / "visuals" / "generated.png")
    scale_raw = _render_same_state_grid(
        base_img, row_names, rep_steps, sigma_table, dv_grid,
        out_dir / "visuals" / "grid_direct_effect_steps.png",
        f"{model_name} ({prompt_id}) — E3-0 Same-State direct_effect (Logit Knockout, cfg_combined)",
        "||Δv(x_σ)||_2",
    )
    scale_dsig = _render_same_state_grid(
        base_img, row_names, rep_steps, sigma_table, dsigma_grid,
        out_dir / "visuals" / "grid_direct_effect_dsigma_steps.png",
        f"{model_name} ({prompt_id}) — E3-0 Same-State Δσ-weighted direct_effect (cfg_combined)",
        "Δσ_i · ||Δv(x_σ)||_2",
    )

    # Attention vs Δv side-by-side using saved E1-1 maps
    prof_npz = prof_dir / "raw" / "profiles.npz"
    if prof_npz.is_file():
        npz = np.load(prof_npz)
        attn_maps = {t: npz[f"map_all_{i}"] for i, t in enumerate(target_names) if f"map_all_{i}" in npz}
        dv_mean_maps = {t: dv_grid[i].mean(axis=0) for i, t in enumerate(target_names)}
        _render_attn_vs_delta_v(
            base_img, target_names, attn_maps, dv_mean_maps,
            out_dir / "visuals" / "grid_attn_vs_delta_v.png",
            f"{model_name} ({prompt_id})",
        )

    np.savez_compressed(
        out_dir / "raw" / "same_state_direct.npz",
        dv_grid=dv_grid,
        dsigma_grid=dsigma_grid,
        rep_steps=np.array(rep_steps),
        row_names=np.array(row_names),
    )

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model": model_name,
        "prompt_id": prompt_id,
        "intervention": "direct_effect (logit_knockout)",
        "output_type": "cfg_combined",
        "elapsed_sec": elapsed,
        "knockout_scope": {"cfg_scope": "cond_only", "query": "img", "blocks": "all"},
        "reference": "path-matched identity (manual chunked SDPA on cond and uncond)",
        "profile_run": prof_dir.name,
        "row_spans": {k: [v.start, v.end] for k, v in row_spans.items()},
        "mean_delta_v": mean_table,
        "color_scales": {"raw_dv": scale_raw, "dsigma_dv": scale_dsig},
    }
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "config.json", {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "exp_id": "e3_0",
        "model": {"name": model_name, "precision": d.precision, "dit": dit_path.name, "te": d.te,
                  "te_precision": weight_precision(d.te), "vae": vae_path.name,
                  "te_cache_sha256": cache.manifest_hash(texts)},
        "runtime": {"sampler": d.sampler, "scheduler": d.scheduler, "steps": d.steps, "cfg": d.cfg, "shift": d.shift,
                    "cfg_batching_policy": d.batching, "seed": seed, "latent_shape": latent_shape,
                    "representative_steps": [{"i": i, "sigma": sigma_table[i]} for i in rep_steps],
                    "sigma_table": sigma_table},
        "prompt": {"prompt_id": prompt_id, "prompt": prompt, "negative_prompt": negative},
        "token_map": token_map_json(tm_cond),
    })
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run E3-0 Same-State Δv direct_effect grids across models")
    p.add_argument("--models", nargs="+", default=["anima", "moss_image", "krea2"])
    p.add_argument("--comfyui-dir", required=True)
    p.add_argument("--models-dir", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, default=REPO_ROOT / "cache" / "cond")
    p.add_argument("--run-id", default="l4_01")
    args = p.parse_args(argv)

    prompt_map = {"anima": "P-A1", "moss_image": "P-M1", "krea2": "P-K1"}
    unknown = [m for m in args.models if m not in prompt_map]
    if unknown:
        p.error(f"unknown models {unknown}; choose from {list(prompt_map)}")
    boot_args = SimpleNamespace(comfyui_dir=args.comfyui_dir, enable_dynamic_vram=False, no_highvram=False)
    c = bootstrap(boot_args)
    out: dict[str, Any] = {}
    for n, m in enumerate(args.models, 1):
        log(f"=== E3-0 {n}/{len(args.models)}: {m} ({MODEL_DEFAULTS[m].precision}) ===")
        out[m] = run_model_same_state(
            c, m, prompt_map[m], Path(args.comfyui_dir), args.models_dir,
            args.out_root, args.cache_root, args.run_id,
        )
    (args.out_root / "cross_model").mkdir(parents=True, exist_ok=True)
    write_json(args.out_root / "cross_model" / f"summary_e3_0_{args.run_id}.json", out)
    log("ALL E3-0 SAME-STATE GRIDS DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
