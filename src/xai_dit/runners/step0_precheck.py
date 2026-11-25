"""Step 0 Quick Pre-Check runner for MOSS-Image & Krea 2 Raw (Generation Only, Full Speedups ON).

Verifies 2D character recognition and ``@artist`` style-tag responsiveness before locking the
shared cross-model prompt set. No XAI hooks are attached; ComfyUI-native acceleration flags
(``--fast``, ``--use-ck-attention``, ``EasyCache``) are enabled by default for fast iteration.

Never uses SageAttention (forbidden on Krea 2 Raw due to NaN/black output).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

from .cli import (
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


@dataclass(frozen=True)
class PromptCase:
    case_id: str
    suite: str
    label: str
    prompt: str
    negative: str


MOSS_NEG_2D = "worst quality, bad anatomy, watermark, jpeg artifact, 3d, realistic"

MOSS_CASES: tuple[PromptCase, ...] = (
    # Suite A: 2D Character Pool (6 characters, no @artist)
    PromptCase(
        "moss_char_tsuyuri_kanao", "chars", "tsuyuri kanao",
        "best quality, highres, tsuyuri kanao, kimetsu no yaiba, 1girl, solo, upper body, looking at viewer, soft lighting, 2d anime illustration",
        MOSS_NEG_2D,
    ),
    PromptCase(
        "moss_char_hoshino_ai", "chars", "hoshino ai",
        "best quality, highres, hoshino ai, oshi no ko, 1girl, solo, upper body, looking at viewer, soft lighting, 2d anime illustration",
        MOSS_NEG_2D,
    ),
    PromptCase(
        "moss_char_frieren", "chars", "frieren",
        "best quality, highres, frieren, sousou no frieren, 1girl, solo, upper body, looking at viewer, soft lighting, 2d anime illustration",
        MOSS_NEG_2D,
    ),
    PromptCase(
        "moss_char_emilia", "chars", "emilia (re:zero)",
        "best quality, highres, emilia \\(re:zero\\), re:zero kara hajimeru isekai seikatsu, 1girl, solo, upper body, looking at viewer, soft lighting, 2d anime illustration",
        MOSS_NEG_2D,
    ),
    PromptCase(
        "moss_char_arona", "chars", "arona (blue archive)",
        "best quality, highres, arona \\(blue archive\\), blue archive, 1girl, solo, upper body, looking at viewer, soft lighting, 2d anime illustration",
        MOSS_NEG_2D,
    ),
    PromptCase(
        "moss_char_raiden_shogun", "chars", "raiden shogun",
        "best quality, highres, raiden shogun, genshin impact, 1girl, solo, upper body, looking at viewer, soft lighting, 2d anime illustration",
        MOSS_NEG_2D,
    ),
    # Suite B: @artist tag shift check on hoshino ai (same seed)
    PromptCase(
        "moss_artist_none", "artists", "no @artist",
        "best quality, highres, hoshino ai, oshi no ko, 1girl, solo, idol dress, upper body, looking at viewer, bright background",
        MOSS_NEG_2D,
    ),
    PromptCase(
        "moss_artist_mika_pikazo", "artists", "@mika pikazo",
        "best quality, highres, hoshino ai, oshi no ko, @mika pikazo, 1girl, solo, idol dress, upper body, looking at viewer, bright background",
        MOSS_NEG_2D,
    ),
    PromptCase(
        "moss_artist_ogipote", "artists", "@ogipote",
        "best quality, highres, hoshino ai, oshi no ko, @ogipote, 1girl, solo, idol dress, upper body, looking at viewer, bright background",
        MOSS_NEG_2D,
    ),
    PromptCase(
        "moss_artist_yoneyama_mai", "artists", "@yoneyama mai",
        "best quality, highres, hoshino ai, oshi no ko, @yoneyama mai, 1girl, solo, idol dress, upper body, looking at viewer, bright background",
        MOSS_NEG_2D,
    ),
)

KREA2_CASES: tuple[PromptCase, ...] = (
    # Suite C: 2D Character Pool via "1girl, <character> cosplay" phrasing
    PromptCase(
        "krea2_cosplay_tsuyuri_kanao", "cosplay", "tsuyuri kanao cosplay",
        "A candid portrait photograph of 1girl, tsuyuri kanao cosplay, kimetsu no yaiba, upper body, looking at camera, natural daylight, 50mm lens",
        "",
    ),
    PromptCase(
        "krea2_cosplay_hoshino_ai", "cosplay", "hoshino ai cosplay",
        "A candid portrait photograph of 1girl, hoshino ai cosplay, oshi no ko, upper body, looking at camera, natural daylight, 50mm lens",
        "",
    ),
    PromptCase(
        "krea2_cosplay_frieren", "cosplay", "frieren cosplay",
        "A candid portrait photograph of 1girl, frieren cosplay, sousou no frieren, upper body, looking at camera, natural daylight, 50mm lens",
        "",
    ),
    PromptCase(
        "krea2_cosplay_emilia", "cosplay", "emilia cosplay",
        "A candid portrait photograph of 1girl, emilia cosplay, re:zero, upper body, looking at camera, natural daylight, 50mm lens",
        "",
    ),
    PromptCase(
        "krea2_cosplay_arona", "cosplay", "arona cosplay",
        "A candid portrait photograph of 1girl, arona cosplay, blue archive, upper body, looking at camera, natural daylight, 50mm lens",
        "",
    ),
    PromptCase(
        "krea2_cosplay_raiden_shogun", "cosplay", "raiden shogun cosplay",
        "A candid portrait photograph of 1girl, raiden shogun cosplay, genshin impact, upper body, looking at camera, natural daylight, 50mm lens",
        "",
    ),
    # Suite D: Direct 2D anime check (2 characters) for contrast against cosplay
    PromptCase(
        "krea2_anime_frieren", "anime", "frieren (2d anime)",
        "2d anime illustration of 1girl, frieren, sousou no frieren, upper body, looking at viewer, soft lighting",
        "",
    ),
    PromptCase(
        "krea2_anime_hoshino_ai", "anime", "hoshino ai (2d anime)",
        "2d anime illustration of 1girl, hoshino ai, oshi no ko, upper body, looking at viewer, soft lighting",
        "",
    ),
)

KREA2_PLAIN_CASES: tuple[PromptCase, ...] = (
    PromptCase(
        "krea2_plain_tsuyuri_kanao", "plain", "tsuyuri kanao (no cosplay)",
        "1girl, tsuyuri kanao, kimetsu no yaiba, upper body, looking at viewer, soft natural lighting",
        "",
    ),
    PromptCase(
        "krea2_plain_hoshino_ai", "plain", "hoshino ai (no cosplay)",
        "1girl, hoshino ai, oshi no ko, upper body, looking at viewer, soft natural lighting",
        "",
    ),
    PromptCase(
        "krea2_plain_frieren", "plain", "frieren (no cosplay)",
        "1girl, frieren, sousou no frieren, upper body, looking at viewer, soft natural lighting",
        "",
    ),
    PromptCase(
        "krea2_plain_emilia", "plain", "emilia (no cosplay)",
        "1girl, emilia, re:zero kara hajimeru isekai seikatsu, upper body, looking at viewer, soft natural lighting",
        "",
    ),
    PromptCase(
        "krea2_plain_arona", "plain", "arona (no cosplay)",
        "1girl, arona, blue archive, upper body, looking at viewer, soft natural lighting",
        "",
    ),
    PromptCase(
        "krea2_plain_raiden_shogun", "plain", "raiden shogun (no cosplay)",
        "1girl, raiden shogun, genshin impact, upper body, looking at viewer, soft natural lighting",
        "",
    ),
)

CASES_BY_MODEL: dict[str, tuple[PromptCase, ...]] = {
    "moss_image": MOSS_CASES,
    "krea2": KREA2_CASES,
    "krea2_plain": KREA2_PLAIN_CASES,
}

BASE_MODEL_KEY: dict[str, str] = {
    "moss_image": "moss_image",
    "krea2": "krea2",
    "krea2_plain": "krea2",
}


def bootstrap_fast(args: argparse.Namespace) -> SimpleNamespace:
    """Import torch + ComfyUI headlessly with Step 0 speedups configured before model_management."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    comfy_dir = args.comfyui_dir or os.environ.get("COMFYUI_DIR")
    if comfy_dir:
        sys.path.insert(0, str(Path(comfy_dir).expanduser().resolve()))
    try:
        import comfy.options
    except ImportError as e:
        raise SystemExit("ComfyUI not importable: pass --comfyui-dir or set COMFYUI_DIR") from e
    comfy.options.enable_args_parsing(False)
    import comfy.cli_args

    comfy.cli_args.args.disable_dynamic_vram = True
    comfy.cli_args.args.highvram = not args.no_highvram
    if args.fast:
        comfy.cli_args.args.fast = set(comfy.cli_args.PerformanceFeature)
    if args.ck_attention:
        comfy.cli_args.args.use_ck_attention = True

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


def apply_easycache(
    patcher: Any,
    reuse_threshold: float,
    start_percent: float,
    end_percent: float,
) -> tuple[Any, Optional[Any]]:
    """Attach ComfyUI's built-in EasyCache wrapper cleanly; returns (patched_model, holder)."""
    try:
        from comfy_extras.nodes_easycache import EasyCacheNode
    except Exception as exc:
        log(f"EasyCache import failed ({exc}); continuing without EasyCache")
        return patcher, None
    out = EasyCacheNode.execute(patcher, reuse_threshold, start_percent, end_percent, verbose=False)
    patched = out.result[0] if hasattr(out, "result") else out[0]
    holder = patched.model_options.get("transformer_options", {}).get("easycache")
    return patched, holder


def build_comfy_prompt_graph(
    model_name: str,
    dit_name: str,
    te_name: str,
    vae_name: str,
    clip_type: str,
    case: PromptCase,
    steps: int,
    cfg: float,
    shift: float,
    sampler: str,
    scheduler: str,
    seed: int,
    width: int,
    height: int,
    accel: dict[str, Any],
) -> dict[str, Any]:
    """ComfyUI-compatible prompt node graph capturing exact weights, HP, and EasyCache settings."""
    clip_type_str = clip_type.lower()
    graph: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": dit_name, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": te_name, "type": clip_type_str, "device": "default"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        "4": {
            "class_type": "ModelSamplingAuraFlow" if model_name == "moss_image" else "ModelSamplingFlux",
            "inputs": (
                {"model": ["1", 0], "shift": shift}
                if model_name == "moss_image"
                else {"model": ["1", 0], "max_shift": shift, "base_shift": shift, "width": width, "height": height}
            ),
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": case.prompt, "clip": ["2", 0]},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": case.negative, "clip": ["2", 0]},
        },
        "7": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
    }
    model_ref: list[Any] = ["4", 0]
    if accel.get("easycache"):
        graph["8"] = {
            "class_type": "EasyCache",
            "inputs": {
                "model": model_ref,
                "reuse_threshold": accel["easycache_threshold"],
                "start_percent": accel["easycache_start"],
                "end_percent": accel["easycache_end"],
                "verbose": False,
            },
        }
        model_ref = ["8", 0]
    graph["9"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref,
            "positive": ["5", 0],
            "negative": ["6", 0],
            "latent_image": ["7", 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "denoise": 1.0,
        },
    }
    graph["10"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["9", 0], "vae": ["3", 0]},
    }
    graph["11"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["10", 0], "filename_prefix": case.case_id},
    }
    return graph


def save_png_with_metadata(
    image: Any,
    path: Path,
    meta: dict[str, Any],
    comfy_prompt: dict[str, Any],
) -> None:
    """Save an image PNG with full xai_dit_meta, A1111 parameters, and ComfyUI prompt PngInfo chunks."""
    import numpy as np
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    arr = (image.float().clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    pnginfo = PngInfo()
    pnginfo.add_text("xai_dit_meta", json.dumps(meta, ensure_ascii=False))
    pnginfo.add_text("prompt", json.dumps(comfy_prompt, ensure_ascii=False))

    accel = meta["acceleration"]
    flags = []
    if accel.get("disable_dynamic_vram"):
        flags.append("--disable-dynamic-vram")
    if accel.get("highvram"):
        flags.append("--highvram")
    if accel.get("fast"):
        flags.append("--fast")
    if accel.get("ck_attention"):
        flags.append("--use-ck-attention")
    if accel.get("easycache"):
        flags.append(
            f"EasyCache(thresh={accel['easycache_threshold']},"
            f"start={accel['easycache_start']},end={accel['easycache_end']},"
            f"skipped={accel.get('easycache_steps_skipped', 0)})"
        )
    params_text = (
        f"{meta['prompt']}\n"
        f"Negative prompt: {meta['negative_prompt']}\n"
        f"Steps: {meta['steps']}, Sampler: {meta['sampler']}, Schedule type: {meta['scheduler']}, "
        f"CFG scale: {meta['cfg']}, Shift: {meta['shift']}, Seed: {meta['seed']}, "
        f"Size: {meta['width']}x{meta['height']}, Model: {meta['weights']['dit']}, "
        f"Text Encoder: {meta['weights']['te']}, VAE: {meta['weights']['vae']}, "
        f"Acceleration: {' '.join(flags)}, Time: {meta['seconds']}s"
    )
    pnginfo.add_text("parameters", params_text)
    Image.fromarray(arr).save(path, pnginfo=pnginfo)


def make_contact_sheet(
    items: Sequence[tuple[str, Path]],
    out_path: Path,
    cols: int,
    tile_size: int = 512,
    header_h: int = 40,
) -> None:
    """Create a labeled PNG contact sheet for visual inspection."""
    from PIL import Image, ImageDraw, ImageFont

    if not items:
        return
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_size, rows * (tile_size + header_h)), (20, 20, 24))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    for idx, (label, img_path) in enumerate(items):
        r, col = divmod(idx, cols)
        x0 = col * tile_size
        y0 = r * (tile_size + header_h)
        draw.rectangle([x0, y0, x0 + tile_size, y0 + header_h], fill=(32, 34, 42))
        draw.text((x0 + 10, y0 + 10), label, fill=(240, 244, 250), font=font)
        with Image.open(img_path) as im:
            resized = im.convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)
            sheet.paste(resized, (x0, y0 + header_h))
    sheet.save(out_path)


def encode_all_prompts(c: SimpleNamespace, te_path: Path, clip_type: str, texts: Sequence[str]) -> dict[str, list]:
    """Encode all unique prompt/negative strings once, then unload the TE completely."""
    unique_texts = list(dict.fromkeys(texts))
    t0 = time.perf_counter()
    log(f"loading TE {te_path.name} ({weight_precision(te_path.name)}) for {len(unique_texts)} texts")
    clip = load_text_encoder(c, te_path, clip_type)
    conds: dict[str, list] = {}
    for text in unique_texts:
        tokens = clip.tokenize(text)
        out = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = out.pop("cond").cpu()
        extras = {k: v.cpu() if isinstance(v, c.torch.Tensor) else v for k, v in out.items()}
        conds[text] = [[cond, extras]]
    del clip
    unload_all(c)
    gc.collect()
    if c.torch.cuda.is_available():
        c.torch.cuda.empty_cache()
    log(f"encoded {len(unique_texts)} texts in {time.perf_counter() - t0:.1f}s (TE unloaded)")
    return conds


def run_model_precheck(
    c: SimpleNamespace,
    args: argparse.Namespace,
    model_name: str,
    out_dir: Path,
    runtime_info: dict[str, Any],
) -> dict[str, Any]:
    base_model = BASE_MODEL_KEY[model_name]
    d = MODEL_DEFAULTS[base_model]
    cases = CASES_BY_MODEL[model_name]
    steps = args.steps if args.steps is not None else d.steps
    seed = args.seed
    img_dir = out_dir / "images" / model_name
    img_dir.mkdir(parents=True, exist_ok=True)

    t_model_start = time.perf_counter()
    te_path = resolve_weight("te", d.te, args.models_dir)
    dit_path = resolve_weight("dit", d.dit, args.models_dir)
    vae_path = resolve_weight("vae", d.vae, args.models_dir)

    all_texts = [t for case in cases for t in (case.prompt, case.negative)]
    with c.torch.inference_mode():
        t_te0 = time.perf_counter()
        conds = encode_all_prompts(c, te_path, d.clip_type, all_texts)
        te_sec = time.perf_counter() - t_te0

        t_load0 = time.perf_counter()
        log(f"loading DiT {dit_path.name} and VAE {vae_path.name}")
        base = c.sd.load_diffusion_model(str(dit_path))
        vae = c.sd.VAE(sd=c.utils.load_torch_file(str(vae_path)))
        vae.throw_exception_if_invalid()
        load_sec = time.perf_counter() - t_load0

        patcher = with_shift(c, base, d.shift)
        easycache_holder = None
        if args.easycache:
            patcher, easycache_holder = apply_easycache(
                patcher, args.easycache_threshold, args.easycache_start, args.easycache_end
            )

        sigmas = c.samplers.calculate_sigmas(patcher.get_model_object("model_sampling"), d.scheduler, steps).cpu()
        latent = empty_latent(c, patcher, args.width, args.height)
        sampler_obj = c.samplers.sampler_object(d.sampler)

        results: list[dict[str, Any]] = []
        decoded_tensors: dict[str, Any] = {}
        suite_items: dict[str, list[tuple[str, Path]]] = {}

        for idx, case in enumerate(cases, start=1):
            img_path = img_dir / f"{case.case_id}.png"
            if easycache_holder is not None:
                easycache_holder.reset()
            noise = c.sample.prepare_noise(latent, seed)
            t0 = time.perf_counter()
            samples = c.sample.sample_custom(
                patcher,
                noise,
                d.cfg,
                sampler_obj,
                sigmas,
                conds[case.prompt],
                conds[case.negative],
                latent,
                disable_pbar=True,
                seed=seed,
            )
            gen_sec = time.perf_counter() - t0
            skipped = int(easycache_holder.total_steps_skipped) if easycache_holder is not None else 0

            img = vae.decode(samples)
            img = img.reshape(-1, *img.shape[-3:])[0].float().cpu()
            accel_meta = {
                "disable_dynamic_vram": True,
                "highvram": not args.no_highvram,
                "fast": args.fast,
                "ck_attention": args.ck_attention,
                "easycache": args.easycache,
                "easycache_threshold": args.easycache_threshold,
                "easycache_start": args.easycache_start,
                "easycache_end": args.easycache_end,
                "easycache_steps_skipped": skipped,
            }
            png_meta = {
                "exp_id": "step0_precheck",
                "case_id": case.case_id,
                "suite": case.suite,
                "label": case.label,
                "model": model_name,
                "weights": {"dit": dit_path.name, "te": te_path.name, "vae": vae_path.name},
                "prompt": case.prompt,
                "negative_prompt": case.negative,
                "seed": seed,
                "steps": steps,
                "cfg": d.cfg,
                "shift": d.shift,
                "sampler": d.sampler,
                "scheduler": d.scheduler,
                "width": args.width,
                "height": args.height,
                "seconds": round(gen_sec, 2),
                "acceleration": accel_meta,
                "runtime": runtime_info,
            }
            comfy_graph = build_comfy_prompt_graph(
                model_name=model_name,
                dit_name=dit_path.name,
                te_name=te_path.name,
                vae_name=vae_path.name,
                clip_type=d.clip_type,
                case=case,
                steps=steps,
                cfg=d.cfg,
                shift=d.shift,
                sampler=d.sampler,
                scheduler=d.scheduler,
                seed=seed,
                width=args.width,
                height=args.height,
                accel=accel_meta,
            )
            save_png_with_metadata(img, img_path, png_meta, comfy_graph)
            decoded_tensors[case.case_id] = img

            mean_lum = float(img.mean())
            is_black = bool(mean_lum < 0.01 or c.torch.isnan(img).any())
            log(
                f"[{model_name} {idx}/{len(cases)}] {case.case_id}: {gen_sec:.1f}s "
                f"(skipped={skipped}/{steps}, mean_lum={mean_lum:.3f}, black_or_nan={is_black})"
            )
            results.append(
                {
                    "case_id": case.case_id,
                    "suite": case.suite,
                    "label": case.label,
                    "prompt": case.prompt,
                    "negative": case.negative,
                    "seconds": round(gen_sec, 2),
                    "easycache_steps_skipped": skipped,
                    "mean_luminance": round(mean_lum, 4),
                    "black_or_nan": is_black,
                    "image": str(img_path.relative_to(out_dir)).replace("\\", "/"),
                }
            )
            suite_items.setdefault(case.suite, []).append((case.label, img_path))

        # Quantitative @artist shift check for MOSS-Image
        artist_diffs: dict[str, dict[str, float]] = {}
        if model_name == "moss_image" and "moss_artist_none" in decoded_tensors:
            ref = decoded_tensors["moss_artist_none"]
            for cid, ten in decoded_tensors.items():
                if cid.startswith("moss_artist_") and cid != "moss_artist_none":
                    diff = (ten - ref).abs()
                    artist_diffs[cid] = {
                        "pixel_mae_vs_none": round(float(diff.mean()), 4),
                        "pixel_rmse_vs_none": round(float((diff.pow(2).mean()).sqrt()), 4),
                    }

        # Build labeled contact sheets per suite
        contact_sheets: dict[str, str] = {}
        for suite, items in suite_items.items():
            cols = 3 if len(items) == 6 else min(4, len(items))
            sheet_name = f"{model_name}_grid.png" if model_name.endswith(f"_{suite}") else f"{model_name}_{suite}_grid.png"
            sheet_path = out_dir / sheet_name
            make_contact_sheet(items, sheet_path, cols=cols)
            contact_sheets[suite] = sheet_path.name
            log(f"saved contact sheet: {sheet_path.name}")

        del base, patcher, vae, decoded_tensors
        unload_all(c)
        gc.collect()
        if c.torch.cuda.is_available():
            c.torch.cuda.empty_cache()

    total_sec = time.perf_counter() - t_model_start
    return {
        "model": model_name,
        "weights": {"dit": dit_path.name, "te": te_path.name, "vae": vae_path.name},
        "hp": {
            "steps": steps,
            "cfg": d.cfg,
            "shift": d.shift,
            "sampler": d.sampler,
            "scheduler": d.scheduler,
            "width": args.width,
            "height": args.height,
            "seed": seed,
        },
        "timing_sec": {
            "te_encode": round(te_sec, 2),
            "dit_vae_load": round(load_sec, 2),
            "total": round(total_sec, 2),
            "mean_per_image": round(sum(r["seconds"] for r in results) / max(1, len(results)), 2),
        },
        "artist_diffs_vs_none": artist_diffs,
        "contact_sheets": contact_sheets,
        "cases": results,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="xai-dit-step0", description="Step 0 quick pre-check (MOSS-Image & Krea 2 Raw)")
    p.add_argument("--models", nargs="+", default=["moss_image", "krea2"], choices=["moss_image", "krea2", "krea2_plain"])
    p.add_argument("--comfyui-dir", help="ComfyUI checkout (default: $COMFYUI_DIR)")
    p.add_argument("--models-dir", type=Path, default=os.environ.get("XAI_DIT_MODELS_DIR"))
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments" / "step0_precheck_l4_01")
    p.add_argument("--env-file", type=Path, default=REPO_ROOT / "local" / ".env")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=None, help="override step count (default: model default 28/32)")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--no-highvram", action="store_true")
    p.add_argument("--no-fast", dest="fast", action="store_false", default=True)
    p.add_argument("--no-ck-attention", dest="ck_attention", action="store_false", default=True)
    p.add_argument("--no-easycache", dest="easycache", action="store_false", default=True)
    p.add_argument("--easycache-threshold", type=float, default=0.20)
    p.add_argument("--easycache-start", type=float, default=0.15)
    p.add_argument("--easycache-end", type=float, default=0.95)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    load_local_env(Path(args.env_file))
    if args.models_dir is None and os.environ.get("XAI_DIT_MODELS_DIR"):
        args.models_dir = Path(os.environ["XAI_DIT_MODELS_DIR"])
    if args.models_dir is not None:
        args.models_dir = Path(args.models_dir).expanduser()

    if args.dry_run:
        plan = {m: [c.case_id for c in CASES_BY_MODEL[m]] for m in args.models}
        log(f"dry-run plan: {json.dumps(plan)}")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c = bootstrap_fast(args)
    gpu_name = c.torch.cuda.get_device_name(0) if c.torch.cuda.is_available() else "cpu"
    t_all = time.perf_counter()

    summary_path = out_dir / "summary.json"
    existing_models: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            existing_models = json.loads(summary_path.read_text(encoding="utf-8")).get("models", {})
        except Exception:
            existing_models = {}

    summary: dict[str, Any] = {
        "exp_id": "step0_precheck",
        "runtime": {
            "gpu_name": gpu_name,
            "torch_version": c.torch.__version__,
            "comfyui_commit": comfyui_commit(c.pkg_dir),
            "fast": args.fast,
            "ck_attention": args.ck_attention,
            "easycache": args.easycache,
            "easycache_threshold": args.easycache_threshold,
            "easycache_start": args.easycache_start,
            "easycache_end": args.easycache_end,
            "disable_dynamic_vram": True,
            "highvram": not args.no_highvram,
        },
        "models": existing_models,
    }

    for model_name in args.models:
        summary["models"][model_name] = run_model_precheck(c, args, model_name, out_dir, summary["runtime"])
        write_json(out_dir / "summary.json", summary)

    summary["total_wall_clock_sec"] = round(time.perf_counter() - t_all, 2)
    write_json(out_dir / "summary.json", summary)
    log(f"Step 0 pre-check complete in {summary['total_wall_clock_sec']}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
