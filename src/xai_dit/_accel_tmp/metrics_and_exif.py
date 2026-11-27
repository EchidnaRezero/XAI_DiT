"""ComfyUI PNG EXIF metadata builder, image quality/noise metrics, and contact-sheet generator.

Pure NumPy + PIL (no torch dependency), runnable and testable on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


def build_comfy_prompt_graph(
    *,
    model_name: str,
    dit_name: str,
    te_name: str,
    vae_name: str,
    clip_type: str,
    prompt: str,
    negative: str,
    steps: int,
    cfg: float,
    shift: float,
    sampler: str,
    scheduler: str,
    seed: int,
    width: int,
    height: int,
    accel: Mapping[str, Any],
    filename_prefix: str,
    zero_negative: bool = False,
) -> dict[str, Any]:
    """Build a ComfyUI-compatible node graph dict for the PNG ``prompt`` chunk."""
    graph: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": dit_name, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": te_name, "type": clip_type.lower(), "device": "default"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
    }
    model_ref: list[Any] = ["1", 0]

    if accel.get("lora"):
        graph["12"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_ref,
                "lora_name": Path(str(accel["lora"])).name,
                "strength_model": float(accel.get("lora_strength", 1.0)),
            },
        }
        model_ref = ["12", 0]

    if model_name == "moss_image":
        graph["4"] = {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": model_ref, "shift": shift},
        }
        model_ref = ["4", 0]
    elif model_name == "anima":
        graph["4"] = {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": model_ref, "shift": shift},
        }
        model_ref = ["4", 0]
    else:
        graph["4"] = {
            "class_type": "ModelSamplingFlux",
            "inputs": {
                "model": model_ref,
                "max_shift": shift,
                "base_shift": shift,
                "width": width,
                "height": height,
            },
        }
        model_ref = ["4", 0]

    if accel.get("easycache"):
        graph["8"] = {
            "class_type": "EasyCache",
            "inputs": {
                "model": model_ref,
                "reuse_threshold": float(accel["easycache_threshold"]),
                "start_percent": float(accel["easycache_start"]),
                "end_percent": float(accel["easycache_end"]),
                "verbose": False,
            },
        }
        model_ref = ["8", 0]

    if accel.get("torch_compile"):
        graph["13"] = {
            "class_type": "TorchCompileModel",
            "inputs": {"model": model_ref, "backend": str(accel["torch_compile"])},
        }
        model_ref = ["13", 0]

    graph["5"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": prompt, "clip": ["2", 0]},
    }
    if zero_negative:
        graph["6"] = {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["5", 0]},
        }
    else:
        graph["6"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["2", 0]},
        }
    graph["7"] = {
        "class_type": "EmptySD3LatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1},
    }
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
        "inputs": {"images": ["10", 0], "filename_prefix": filename_prefix},
    }
    return graph


def save_png_with_exif(
    rgb_uint8: np.ndarray,
    path: Path,
    meta: Mapping[str, Any],
    comfy_prompt: Mapping[str, Any],
) -> None:
    """Save RGB uint8 HxWx3 PNG with ComfyUI ``prompt``, ``xai_dit_meta``, and A1111 ``parameters`` chunks."""
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    pnginfo = PngInfo()
    pnginfo.add_text("prompt", json.dumps(comfy_prompt, ensure_ascii=False))
    pnginfo.add_text("xai_dit_meta", json.dumps(meta, ensure_ascii=False))

    accel = meta.get("acceleration", {})
    flags: list[str] = []
    if accel.get("disable_dynamic_vram"):
        flags.append("--disable-dynamic-vram")
    if accel.get("highvram"):
        flags.append("--highvram")
    attn = accel.get("attention", "pytorch")
    if attn != "pytorch":
        flags.append(f"--use-{attn}-attention")
    fast_feats = accel.get("fast_features") or []
    if fast_feats:
        flags.append("--fast=" + ",".join(fast_feats))
    if accel.get("force_upcast_attention"):
        flags.append("--force-upcast-attention")
    if accel.get("dont_upcast_attention"):
        flags.append("--dont-upcast-attention")
    if accel.get("easycache"):
        flags.append(
            f"EasyCache(t={accel.get('easycache_threshold')},"
            f"s={accel.get('easycache_start')},e={accel.get('easycache_end')},"
            f"skipped={accel.get('easycache_steps_skipped', 0)})"
        )
    if accel.get("torch_compile"):
        flags.append(f"TorchCompile({accel['torch_compile']})")
    if accel.get("lora"):
        flags.append(f"LoRA({Path(str(accel['lora'])).name}:{accel.get('lora_strength', 1.0)})")

    weights = meta.get("weights", {})
    params_text = (
        f"{meta.get('prompt', '')}\n"
        f"Negative prompt: {meta.get('negative_prompt', '')}\n"
        f"Steps: {meta.get('steps')}, Sampler: {meta.get('sampler')}, Schedule type: {meta.get('scheduler')}, "
        f"CFG scale: {meta.get('cfg')}, Shift: {meta.get('shift')}, Seed: {meta.get('seed')}, "
        f"Size: {meta.get('width')}x{meta.get('height')}, Model: {weights.get('dit')}, "
        f"Text Encoder: {weights.get('te')}, VAE: {weights.get('vae')}, "
        f"Preset: {meta.get('preset', 'custom')}, Acceleration: {' '.join(flags)}, "
        f"Time: {meta.get('wall_clock_sec')}s"
    )
    pnginfo.add_text("parameters", params_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_uint8).save(path, pnginfo=pnginfo)


def _laplacian_hf_energy(rgb_f32: np.ndarray) -> float:
    """Mean absolute discrete 4-neighbor Laplacian on luminance (high-frequency noise/graininess proxy)."""
    gray = 0.299 * rgb_f32[..., 0] + 0.587 * rgb_f32[..., 1] + 0.114 * rgb_f32[..., 2]
    lap = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    return float(np.mean(np.abs(lap)))


def compute_image_metrics(
    rgb_float_01: np.ndarray,
    ref_float_01: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Compute collapse/black-image checks, high-freq graininess proxy, and optional diff vs baseline."""
    arr = np.asarray(rgb_float_01, dtype=np.float32)
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    clean = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    clean = np.clip(clean, 0.0, 1.0)

    mean_val = float(clean.mean())
    std_val = float(clean.std())
    hf_energy = _laplacian_hf_energy(clean)
    is_black = bool(mean_val < 0.005 and std_val < 0.005)
    is_collapsed = bool(nan_count > 0 or inf_count > 0 or is_black)

    out: dict[str, Any] = {
        "nan_count": nan_count,
        "inf_count": inf_count,
        "is_black": is_black,
        "is_collapsed": is_collapsed,
        "mean_01": round(mean_val, 5),
        "std_01": round(std_val, 5),
        "hf_laplacian": round(hf_energy, 6),
    }

    if ref_float_01 is not None:
        ref = np.clip(np.nan_to_num(np.asarray(ref_float_01, dtype=np.float32)), 0.0, 1.0)
        if ref.shape == clean.shape:
            diff = clean - ref
            l1 = float(np.mean(np.abs(diff)))
            mse = float(np.mean(diff * diff))
            rmse = float(np.sqrt(mse))
            psnr = float(10.0 * np.log10(1.0 / mse)) if mse > 1e-12 else 99.0
            ref_hf = _laplacian_hf_energy(ref)
            hf_ratio = float(hf_energy / max(ref_hf, 1e-8))
            out.update({
                "l1_vs_base": round(l1, 5),
                "rmse_vs_base": round(rmse, 5),
                "psnr_db": round(psnr, 2),
                "hf_ratio_vs_base": round(hf_ratio, 4),
            })
    return out


def load_png_float01(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def make_comparison_sheet(
    items: Sequence[tuple[str, Path]],
    out_path: Path,
    cols: int = 4,
    tile_size: int = 448,
    header_h: int = 44,
    include_center_crop: bool = True,
) -> None:
    """Create a labeled contact sheet; optionally appends a 1:1 center-crop strip to inspect graininess."""
    from PIL import Image, ImageDraw, ImageFont

    if not items:
        return
    crop_h = 160 if include_center_crop else 0
    cell_h = header_h + tile_size + crop_h
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_size, rows * cell_h), (18, 20, 26))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for idx, (label, img_path) in enumerate(items):
        r, c = divmod(idx, cols)
        x0, y0 = c * tile_size, r * cell_h
        draw.rectangle([x0, y0, x0 + tile_size, y0 + header_h], fill=(30, 34, 44))
        draw.text((x0 + 8, y0 + 12), label[:58], fill=(240, 244, 250), font=font)
        if not img_path.is_file():
            continue
        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
            resized = rgb.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
            sheet.paste(resized, (x0, y0 + header_h))
            if include_center_crop:
                w, h = rgb.size
                cw = min(tile_size, w)
                ch = min(crop_h, h)
                left = (w - cw) // 2
                top = (h - ch) // 2
                crop = rgb.crop((left, top, left + cw, top + ch))
                sheet.paste(crop, (x0 + (tile_size - cw) // 2, y0 + header_h + tile_size))
                draw.line(
                    [(x0, y0 + header_h + tile_size), (x0 + tile_size, y0 + header_h + tile_size)],
                    fill=(90, 160, 255),
                    width=2,
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
