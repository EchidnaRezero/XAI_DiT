"""Fast E2 prop-transplant test: Anima BF16, P-A1, prop ``sketchbook`` only (Colab L4).

Pure XAI L4 baseline (``euler`` / ``simple``, 30 steps, cfg 4.0, shift 3.0, seed 42,
``--disable-dynamic-vram`` + ``--highvram``; no ``--fast``, no ``--use-ck-attention``, no EasyCache).

1. ONE hooked pass on P-A1: exact 28-block ``Cont^(l)`` (``raw_mvo``, ``output_proj`` module call,
   cond branch, all steps) for ``sketchbook`` -> ``cont_sketchbook_28blocks.png``.
2. Counterfactual context transplant: the cond cross-attention text context ``[B, 512, D]`` is
   replaced by the one of the counterfactual prompt (``sketchbook`` -> ``--swap-text``) on a block
   set. Implemented as a ``kv_patch`` callable: text K/V of the selected blocks are recomputed from
   the counterfactual context with the block's own ``k_proj/k_norm/v_proj/v_norm`` (identical to
   feeding that context to ``cross_attn``; Q, uncond branch and other blocks untouched; attention
   stays on the native fused path). Contexts come from ``Anima.preprocess_text_embeds`` exactly as
   ``model_base.Anima.extra_conds`` builds them (LLMAdapter + 512 zero pad).

   Arms: Base | Full Target (0..27) | Cont Top-6 | Mid [9..18] | Cont Bottom-6 (control).

Outputs (``--out-dir``): ``cont_sketchbook_28blocks.png``, ``prop_transplant_test.png`` (1x5 strip),
``<arm>.png`` per arm, ``metrics.json``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

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
    replace_occurrence,
    resolve_weight,
    save_png,
    to_conditioning,
    token_map_json,
    unload_all,
    with_shift,
    write_json,
)

MODEL = "anima"
PROMPT_ID = "P-A1"
TARGET_TEXT = "sketchbook"
SET_PATH = REPO_ROOT / "experiments" / "design" / "unified_anima_set.json"
MID_BLOCKS = list(range(9, 19))  # unified set block_thirds["mid"] = [9..18]
TOP_K = 6
ARMS = ("base", "full_target", "cont_top6", "mid_9_18", "cont_bottom6")

C_TOP, C_BOTTOM, C_MID, C_NEUTRAL, C_INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7", "#52514e"


# =========================================================================== context transplant


class ContextTransplant:
    """``kv_patch`` callable: cond-row text K/V <- K/V of the counterfactual 512-padded context.

    Per-block counterfactual K/V are computed once (the Anima text stream is static across
    steps) and cached on the GPU. On the first call of each block, the K computed the same way
    from the *base* context is compared with the live K (path check of the context rebuild).
    """

    def __init__(self, torch: Any, dm: Any, base_entry: dict, cf_entry: dict, dtype: Any) -> None:
        self.torch, self.dm, self.dtype = torch, dm, dtype
        self.base_entry, self.cf_entry = base_entry, cf_entry
        self.ctx: dict[str, Any] = {}
        self.kv: dict[int, tuple[Any, Any]] = {}
        self.path_check: dict[int, float] = {}

    def _context(self, which: str, device: Any) -> Any:
        if which not in self.ctx:  # mirrors comfy.model_base.Anima.extra_conds
            e = self.base_entry if which == "base" else self.cf_entry
            ids = e["extra.t5xxl_ids"].unsqueeze(0).to(device)
            w = e.get("extra.t5xxl_weights")
            if w is not None:
                w = w.unsqueeze(0).unsqueeze(-1).to(device=device, dtype=self.dtype)
            self.ctx[which] = self.dm.preprocess_text_embeds(e["cond"].to(device=device, dtype=self.dtype), ids,
                                                             t5xxl_weights=w)
        return self.ctx[which]

    def _kv(self, block: int, context: Any) -> tuple[Any, Any]:
        attn = self.dm.blocks[block].cross_attn
        h, d = attn.n_heads, attn.head_dim
        k = attn.k_norm(attn.k_proj(context).unflatten(-1, (h, d)))
        v = attn.v_norm(attn.v_proj(context).unflatten(-1, (h, d)))
        return k.transpose(1, 2), v.transpose(1, 2)  # [1, H, 512, d]

    def __call__(self, *, block: int, ctx: Any, k: Any, v: Any, layout: Any, spec: Any) -> tuple[Any, Any]:
        rows = list(spec.rows(ctx))
        if not rows:
            return k, v
        if block not in self.kv:
            cf = self._context("cf", k.device)
            if tuple(cf.shape[1:]) != tuple(self._context("base", k.device).shape[1:]):
                raise ValueError(f"counterfactual context {tuple(cf.shape)} != base {tuple(self.ctx['base'].shape)}")
            self.kv[block] = self._kv(block, cf)
            kb, _ = self._kv(block, self.ctx["base"])
            self.path_check[block] = float((kb[0].float() - k[rows[0]].float()).abs().max())
        kc, vc = self.kv[block]
        if kc.shape[1:] != k.shape[1:]:
            raise ValueError(f"block {block}: transplant K {tuple(kc.shape)} vs live K {tuple(k.shape)}")
        k, v = k.clone(), v.clone()
        for r in rows:
            k[r], v[r] = kc[0].to(k.dtype), vc[0].to(v.dtype)
        return k, v


# =========================================================================== helpers


def _delta(a: Any, b: Any) -> float:
    return float((a.float() - b.float()).abs().mean())


def _plt() -> Any:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _render_cont(raw: Any, top: Sequence[int], bottom: Sequence[int], swap_text: str, path: Path) -> None:
    import numpy as np
    from matplotlib.patches import Patch
    plt = _plt()
    xs = np.arange(len(raw))
    colors = [C_TOP if b in top else C_BOTTOM if b in bottom else C_NEUTRAL for b in xs]
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=130)
    ax.axvspan(MID_BLOCKS[0] - 0.5, MID_BLOCKS[-1] + 0.5, color=C_MID, alpha=0.10, lw=0)
    ax.bar(xs, raw, color=colors, width=0.8, edgecolor="white", linewidth=1.0)
    ax.set_xticks(xs)
    ax.set_xlabel("Block index")
    ax.set_ylabel("Cont^(l) raw_mvo (mean over steps)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(color=C_TOP, label=f"Top-{TOP_K}"), Patch(color=C_BOTTOM, label=f"Bottom-{TOP_K}"),
                       Patch(color=C_MID, alpha=0.25, label="Mid [9..18]"), Patch(color=C_NEUTRAL, label="other")],
              fontsize=8, frameon=False, loc="upper right", ncol=4)
    ax.set_title(f"Anima BF16 ({PROMPT_ID}) — Cont^(l) for '{TARGET_TEXT}' (transplant -> '{swap_text}')", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _render_strip(images: dict[str, Any], labels: dict[str, str], path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, len(ARMS), figsize=(4.0 * len(ARMS), 4.5), dpi=130)
    for ax, arm in zip(axes, ARMS):
        ax.imshow(images[arm].float().clamp(0, 1).cpu().numpy())
        ax.set_title(labels[arm], fontsize=9, color=C_INK2)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# =========================================================================== main


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments" / "e2_prop_transplant_test_anima_bf16")
    p.add_argument("--models-dir", type=Path, default=os.environ.get("XAI_DIT_MODELS_DIR"))
    p.add_argument("--comfyui-dir", default=os.environ.get("COMFYUI_DIR"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--swap-text", default="glowing paper lantern")
    p.add_argument("--cache-root", type=Path, default=REPO_ROOT / "cache" / "cond")
    p.add_argument("--query-chunk", type=int, default=1024)
    p.add_argument("--dry-run", action="store_true", help="print the plan without importing torch/ComfyUI")
    args = p.parse_args(argv)

    d = MODEL_DEFAULTS[MODEL]
    unified, set_sha = load_manifest(SET_PATH)
    entry = pick_prompt(unified, PROMPT_ID)
    prompt = entry["prompt"]
    negative = entry.get("negative_prompt", d.negative)
    target = next(t for t in entry["targets"] if t["text"] == TARGET_TEXT)
    cf_prompt = replace_occurrence(prompt, TARGET_TEXT, args.swap_text, int(target.get("occurrence_idx", 0)))
    log(f"plan: {MODEL} {d.precision} {PROMPT_ID} seed={args.seed} {d.sampler}/{d.scheduler} steps={d.steps} "
        f"cfg={d.cfg} shift={d.shift} | '{TARGET_TEXT}' -> '{args.swap_text}' | arms={list(ARMS)} "
        f"| out={args.out_dir}")
    if args.dry_run:
        log(f"counterfactual prompt: {cf_prompt}")
        return 0
    if args.models_dir is None:
        raise SystemExit("--models-dir (or $XAI_DIT_MODELS_DIR) is required")

    c = bootstrap(SimpleNamespace(comfyui_dir=args.comfyui_dir, enable_dynamic_vram=False, no_highvram=False))
    torch = c.torch
    import numpy as np
    from ..adapters import get_adapter
    from ..adapters.base import InterventionSpec, TokenSpan
    from ..core.session import XAISession
    from .e1_e4_profile import _load_lightweight_tokenizer, _match_target_spans_in_S
    from .e2_e4_interventions import _make_cont_tracker

    flags = c.comfy.cli_args.args
    runtime_flags = {"disable_dynamic_vram": bool(flags.disable_dynamic_vram), "highvram": bool(flags.highvram),
                     "fast": sorted(str(f) for f in (getattr(flags, "fast", None) or [])),
                     "use_ck_attention": bool(getattr(flags, "use_ck_attention", False)), "easycache": False}
    if runtime_flags["fast"] or runtime_flags["use_ck_attention"]:
        raise SystemExit(f"baseline violated: {runtime_flags}")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.monotonic()

    with torch.inference_mode():
        adapter = get_adapter(MODEL)
        cap = adapter.capability
        cc_args = SimpleNamespace(te=None, models_dir=args.models_dir, cache_root=str(args.cache_root))
        log(f"stage 1/4: conditioning ({PROMPT_ID}, counterfactual, negative)")
        _, conds = ensure_cond_cache(c, cc_args, d, adapter, [prompt, cf_prompt, negative], set_sha)
        tm = adapter.build_token_map(conds[prompt]["token_ids"], branch="cond")
        tm_cf = adapter.build_token_map(conds[cf_prompt]["token_ids"], branch="cond")

        log("stage 2/4: target span (tokenizer only)")
        tok = _load_lightweight_tokenizer(MODEL)
        spans, method = _match_target_spans_in_S(tok, tm.token_ids, tm.spans["S"], [target])
        del tok
        span = TokenSpan(*spans[TARGET_TEXT])
        log(f"'{TARGET_TEXT}' -> {span} ({method[TARGET_TEXT]}); Lt base={tm.branch_text_len['cond']} "
            f"cf={tm_cf.branch_text_len['cond']}, padded L={tm.text_len}/{tm_cf.text_len}")

        log("stage 3/4: DiT + VAE")
        dit_path = resolve_weight("dit", d.dit, args.models_dir)
        vae_path = resolve_weight("vae", d.vae, args.models_dir)
        patcher = with_shift(c, c.sd.load_diffusion_model(str(dit_path)), d.shift)
        install_runtime_wrappers(c, patcher, d.batching)
        adapter.bind(patcher)
        vae = c.sd.VAE(sd=c.utils.load_torch_file(str(vae_path)))
        sigmas = c.samplers.calculate_sigmas(patcher.get_model_object("model_sampling"), d.scheduler, d.steps).cpu()
        sigma_table = [float(s) for s in sigmas]
        latent = empty_latent(c, patcher)
        device = c.mm.get_torch_device()
        positive, negative_c = to_conditioning(conds[prompt]), to_conditioning(conds[negative])

        def generate(tag: str, session: Any = None) -> Any:
            t0 = last = time.monotonic()
            log(f">>> {tag} ({'hooked' if session is not None else 'plain'})")

            def cb(step: int, x0: Any, x: Any, total: int) -> None:
                nonlocal last
                now = time.monotonic()
                log(f"    {tag} step {step + 1}/{total} (σ={sigma_table[step]:.3f}) +{now - last:.1f}s")
                last = now

            noise = c.sample.prepare_noise(latent, args.seed)
            run = lambda: c.sample.sample_custom(patcher, noise, d.cfg, c.samplers.sampler_object(d.sampler), sigmas,  # noqa: E731
                                                 positive, negative_c, latent, callback=cb, disable_pbar=True,
                                                 seed=args.seed)
            if session is None:
                samples = run()
            else:
                with adapter.attached(session):
                    samples = run()
            img = vae.decode(samples)
            log(f"<<< {tag} done in {time.monotonic() - t0:.1f}s")
            return img.reshape(-1, *img.shape[-3:])[0].detach().float().cpu()

        log("stage 4/4: experiments")
        # (1) Cont^(l) — ONE hooked pass
        L = cap.num_blocks
        factory, acc = _make_cont_tracker(torch, L, device, span, adapter.output_projection)
        sess = XAISession(cap, query_chunk=args.query_chunk)
        sess.add_observer(factory)
        cont_img = generate("cont_pass", sess)
        raw = acc["raw_mvo"].cpu().numpy() / np.maximum(np.array(acc["count"]), 1)
        order = [int(b) for b in np.argsort(-raw, kind="stable")]
        top, bottom = sorted(order[:TOP_K]), sorted(order[-TOP_K:])
        log(f"Cont^(l) raw_mvo top-{TOP_K}={top} bottom-{TOP_K}={bottom}")
        _render_cont(raw, top, bottom, args.swap_text, out / "cont_sketchbook_28blocks.png")

        # (2) context transplant arms
        transplant = ContextTransplant(torch, adapter.model, conds[prompt], conds[cf_prompt],
                                       patcher.model.get_dtype_inference())
        block_sets = {"full_target": list(range(L)), "cont_top6": top, "mid_9_18": MID_BLOCKS, "cont_bottom6": bottom}
        images = {"base": generate("base")}
        for arm, blocks in block_sets.items():
            spec = InterventionSpec("kv_patch", blocks=blocks, cfg_scope="cond_only", params={"source": transplant})
            images[arm] = generate(arm, XAISession(cap, interventions=[spec], query_chunk=args.query_chunk))
        ctx_shape = {k: list(v.shape) for k, v in transplant.ctx.items()}
        transplant.kv.clear()
        transplant.ctx.clear()
        unload_all(c)

    labels = {"base": f"Base ('{TARGET_TEXT}')", "full_target": f"Full Target (0..{L - 1})",
              "cont_top6": f"Cont Top-6 {top}", "mid_9_18": "Mid [9..18]", "cont_bottom6": f"Cont Bottom-6 {bottom}\n(control)"}
    for arm in ARMS:
        save_png(images[arm], out / f"{arm}.png")
    _render_strip(images, labels, out / "prop_transplant_test.png")

    base, full = images["base"], images["full_target"]
    d_full = _delta(full, base)
    arms_m = {}
    for arm in ARMS:
        dv_base, dv_full = _delta(images[arm], base), _delta(images[arm], full)
        arms_m[arm] = {"blocks": block_sets.get(arm, []), "pixel_mean_abs_vs_base": dv_base,
                       "pixel_mean_abs_vs_full_target": dv_full,
                       "transplant_fraction": None if d_full == 0 else 1.0 - dv_full / d_full}
    metrics = {
        "schema_version": SCHEMA_VERSION, "exp_id": "e2_prop_transplant_test", "model": MODEL, "precision": d.precision,
        "prompt_id": PROMPT_ID, "unified_set_sha256": set_sha, "seed": args.seed,
        "target": TARGET_TEXT, "swap_text": args.swap_text, "counterfactual_prompt": cf_prompt,
        "target_span": [span.start, span.end], "target_span_method": method[TARGET_TEXT],
        "runtime": {"sampler": d.sampler, "scheduler": d.scheduler, "steps": d.steps, "cfg": d.cfg, "shift": d.shift,
                    "cfg_batching_policy": d.batching, "query_chunk": args.query_chunk, "comfy_flags": runtime_flags,
                    "dit": dit_path.name, "vae": vae_path.name, "sigma_table": sigma_table,
                    "torch_version": torch.__version__},
        "cont": {"metric": "raw_mvo", "wo": "output_proj module call", "branch": "cond", "steps": "all",
                 "raw_mvo": raw.tolist(), "count": list(acc["count"]), "top6": top, "bottom6": bottom,
                 "cont_pass_vs_base_pixel_mean_abs": _delta(cont_img, base)},
        "transplant": {"mechanism": "kv_patch callable: cond text K/V <- k/v_proj+norm(counterfactual 512-padded ctx)",
                       "cfg_scope": "cond_only", "context_shape": ctx_shape,
                       "ctx_rebuild_path_check_k_max_abs": {str(b): v for b, v in sorted(transplant.path_check.items())},
                       "token_map_base": token_map_json(tm), "token_map_cf": token_map_json(tm_cf)},
        "arms": arms_m, "wall_clock_sec": time.monotonic() - t_all,
    }
    write_json(out / "metrics.json", metrics)
    log(f"DONE -> {out} | " + " ".join(f"{a}:Δbase={m['pixel_mean_abs_vs_base']:.4f}" for a, m in arms_m.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
