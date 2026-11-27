"""Colab GPU keepalive & exploratory seed/prompt sweep runner.

While code is being edited or results are being inspected during a live Colab session,
run this in the background so the GPU stays active AND produces traceable ComfyUI-EXIF
reference images across seeds and prompts.
"""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path
from typing import Optional, Sequence

from ..runners.cli import MODEL_DEFAULTS, REPO_ROOT, load_local_env, log, resolve_weight
from .presets import PROMPTS_BY_MODEL, presets_by_name
from .run_t2i import bootstrap_with_accel, ensure_conds, run_single_preset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Background Colab keepalive seed/prompt generator with ComfyUI EXIF.")
    p.add_argument("--model", required=True, choices=["anima", "moss_image", "krea2"])
    p.add_argument("--preset", default="baseline", help="Preset name from presets.py.")
    p.add_argument("--start-seed", type=int, default=100)
    p.add_argument("--max-images", type=int, default=30, help="Stop after generating N images.")
    p.add_argument("--sleep-sec", type=float, default=2.0, help="Pause between images.")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--comfyui-dir", default=None)
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--cond-cache-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_local_env(REPO_ROOT / "local/.env")
    args = build_parser().parse_args(argv)
    d = MODEL_DEFAULTS[args.model]
    catalog = presets_by_name(args.model)
    if args.preset not in catalog:
        raise SystemExit(f"Unknown preset {args.preset!r} for {args.model}")
    preset = catalog[args.preset]
    pool = PROMPTS_BY_MODEL[args.model]

    if args.dry_run:
        log(
            f"keepalive dry-run: model={args.model} preset={preset.name} "
            f"prompts={[p.prompt_id for p in pool]} max_images={args.max_images}"
        )
        return 0

    stop = {"flag": False}

    def _handle_stop(_signum: int, _frame: object) -> None:
        log("keepalive: stop signal received; finishing current image...")
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_stop)

    c = bootstrap_with_accel(
        comfyui_dir=args.comfyui_dir,
        attention=preset.attention,
        fast_features=preset.fast_features,
        force_upcast_attention=preset.force_upcast_attention,
        dont_upcast_attention=preset.dont_upcast_attention,
    )

    te_path = resolve_weight("te", d.te, args.models_dir)
    vae_path = resolve_weight("vae", d.vae, args.models_dir)
    dit_name = preset.dit_override or d.dit
    dit_path = resolve_weight("dit", dit_name, args.models_dir)

    texts = [t for p in pool for t in (p.prompt, p.negative)]
    conds = ensure_conds(
        c,
        model_name=args.model,
        te_path=te_path,
        clip_type=d.clip_type,
        texts=texts,
        cache_dir=args.cond_cache_dir or (args.out_dir / "_cond_cache"),
    )
    vae = c.sd.VAE(sd=c.utils.load_torch_file(str(vae_path)))
    base_patcher = c.sd.load_diffusion_model(str(dit_path))
    weights_meta = {"dit": dit_path.name, "te": te_path.name, "vae": vae_path.name}

    for i in range(args.max_images):
        if stop["flag"]:
            break
        pspec = pool[i % len(pool)]
        seed = args.start_seed + i
        run_preset = type(preset)(**{**preset.to_dict(), "fast_features": preset.fast_features, "name": f"{preset.name}_{pspec.prompt_id}_s{seed}"})
        run_single_preset(
            c,
            model_name=args.model,
            preset=run_preset,
            base_patcher=base_patcher,
            vae=vae,
            conds=conds,
            prompt_text=pspec.prompt,
            negative_text=pspec.negative,
            weights_meta=weights_meta,
            clip_type=d.clip_type,
            seed=seed,
            width=args.width,
            height=args.height,
            out_dir=args.out_dir,
            models_dir=args.models_dir,
            baseline_png=None,
            diag_probe=False,
            repeat=1,
            no_highvram=False,
        )
        if args.sleep_sec > 0 and not stop["flag"]:
            time.sleep(args.sleep_sec)

    log("keepalive finished cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
