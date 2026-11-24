"""Headless ComfyUI runner for XAI_DiT experiments (Colab L4+; ``plan/m1.md`` §1).

Subcommands
-----------
``encode``  Offline text-encoder pass (per-model default TE: BF16 / INT8 convrot / FP8 scaled)
            over a prompt manifest (prompts, negatives and neutral substitutes) into
            ``cache/cond/<model>_<manifest_sha16>/``; the TE is unloaded before any DiT is loaded.
``e0``      E0 sanity & resource budget: ``reps`` x baseline (seed 42), exact no-op (adapter
            attached, nothing active), path-matched identity (manual chunked SDPA on every
            block), stage-wise VRAM / RSS / ``sec_per_forward``.

Fixed policies: deterministic ``euler`` with an explicit ``sigma_table`` (logged in full),
one CFG batching policy per run (``split`` default; ``CALC_COND_BATCH`` wrapper +
``XAISession.check_batching``), ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``,
``--disable-dynamic-vram`` and ``--highvram`` ComfyUI flags.

Output (``experiments/design/output_spec.md``)::

    <out_root>/<exp_id>_<model>_<precision>_<run_id>/
        config.json  manifest.json  metrics.json  raw/  visuals/

Secrets: ``local/.env`` is parsed into ``os.environ`` without printing names or values;
loaded values are scrubbed from every JSON written. Only weight-file *basenames* are logged.

Torch / ComfyUI are imported lazily, so ``--help`` and ``--dry-run`` work on a CPU box.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "1.0"
_CFG_WRAPPER_KEY = "xai_dit_cfg_batching"
_PROFILE_WRAPPER_KEY = "xai_dit_profile"


# =========================================================================== model defaults


@dataclass(frozen=True)
class ModelDefaults:
    """L4 baseline per ``experiments/design/runtime_and_hp.md`` §1 (single source of truth)."""

    name: str
    dit: str
    te: str
    vae: str
    precision: str
    clip_type: str
    steps: int
    cfg: float
    shift: float
    sampler: str
    scheduler: str
    negative: str
    batching: str
    rep_steps: tuple[int, ...]


MODEL_DEFAULTS: dict[str, ModelDefaults] = {
    "anima": ModelDefaults(
        "anima", "anima-base-v1.0.safetensors", "qwen_3_06b_base.safetensors", "qwen_image_vae.safetensors",
        "bf16", "STABLE_DIFFUSION", 30, 4.0, 3.0, "euler", "simple",
        "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts", "split",
        (1, 5, 10, 15, 22, 29)),
    "moss_image": ModelDefaults(
        "moss_image", "moss_image_sft_int8_convrot.safetensors", "qwen3.5_9b_int8_convrot.safetensors",
        "flux2-vae.safetensors", "int8_convrot", "MOSS_IMAGE", 28, 3.0, 5.0, "euler", "sgm_uniform",
        "worst quality, bad anatomy, watermark, jpeg artifact", "split", (1, 5, 10, 15, 21, 27)),
    "krea2": ModelDefaults(
        "krea2", "krea2_raw_fp8_scaled.safetensors", "qwen3vl_4b_fp8_scaled.safetensors", "qwen_image_vae.safetensors",
        "fp8", "KREA2", 32, 4.0, 1.15, "euler", "simple", "", "split", (1, 5, 10, 16, 23, 31)),
}

# Weight-file precision tags, most specific first. Untagged TE releases (e.g. ``qwen_3_06b_base``) are BF16.
_PRECISION_TAGS = ("int8_convrot", "fp8_scaled", "fp8_e4m3fn", "fp8_e5m2", "nvfp4", "fp8", "bf16", "fp16", "fp32")


def weight_precision(name: str, default: str = "bf16") -> str:
    base = Path(name).name.lower()
    return next((tag for tag in _PRECISION_TAGS if tag in base), default)


_MODEL_SUBDIRS = {"dit": ("diffusion_models", "unet"), "te": ("text_encoders", "clip"), "vae": ("vae",)}


# =========================================================================== secrets / env


_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_SECRET_VALUES: list[str] = []


def load_local_env(path: Path, *, override: bool = False) -> int:
    """Load ``KEY=VALUE`` lines into ``os.environ``; returns the count. Never prints names or values."""
    if not path.is_file():
        return 0
    n = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        elif " #" in val:
            val = val.split(" #", 1)[0].rstrip()
        if val and len(val) >= 6:
            _SECRET_VALUES.append(val)
        if override or key not in os.environ:
            os.environ[key] = val
            n += 1
    return n


def scrub(obj: Any) -> Any:
    """Replace any loaded ``.env`` value inside strings with ``<redacted>`` (recursive)."""
    if isinstance(obj, str):
        for secret in _SECRET_VALUES:
            if secret in obj:
                obj = obj.replace(secret, "<redacted>")
        return obj
    if isinstance(obj, Mapping):
        return {k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v) for v in obj]
    return obj


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(scrub(obj), indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def _json_default(o: Any) -> Any:
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return o.name
    return str(o)


def log(msg: str) -> None:
    print(f"[xai-dit] {scrub(msg)}", flush=True)


# =========================================================================== files / hashes


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def resolve_weight(kind: str, name_or_path: str, models_dir: Optional[Path]) -> Path:
    p = Path(name_or_path).expanduser()
    if p.is_file():
        return p
    roots = [models_dir] if models_dir else []
    for root in roots:
        for sub in (*_MODEL_SUBDIRS[kind], ""):
            cand = root / sub / name_or_path
            if cand.is_file():
                return cand
    raise FileNotFoundError(f"{kind} weights {Path(name_or_path).name!r} not found (set --models-dir or pass a path)")


def comfyui_commit(comfy_pkg_dir: Path) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(comfy_pkg_dir.parent), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def next_run_id(out_root: Path, prefix: str) -> str:
    day = _dt.datetime.now().strftime("%Y%m%d")
    n = 1
    while (out_root / f"{prefix}_{day}_{n:02d}").exists():
        n += 1
    return f"{day}_{n:02d}"


# =========================================================================== manifest


def load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    data = path.read_bytes()
    return json.loads(data.decode("utf-8")), hashlib.sha256(data).hexdigest()


def replace_occurrence(text: str, old: str, new: str, occurrence: int = 0) -> str:
    start = -1
    for _ in range(occurrence + 1):
        start = text.find(old, start + 1)
        if start < 0:
            raise ValueError(f"target {old!r} occurrence {occurrence} not in prompt")
    return text[:start] + new + text[start + len(old):]


def manifest_texts(manifest: Mapping[str, Any]) -> list[str]:
    """Every text needing conditioning: prompts, negatives and neutral substitutes (ordered, unique)."""
    out: dict[str, None] = {}
    default_neg = manifest.get("default_negative_prompt", "")
    out[default_neg] = None
    for p in manifest.get("prompts", []):
        out[p["prompt"]] = None
        out[p.get("negative_prompt", default_neg)] = None
        for t in p.get("targets", []):
            if t.get("neutral_sub"):
                out[replace_occurrence(p["prompt"], t["text"], t["neutral_sub"], int(t.get("occurrence_idx", 0)))] = None
    return list(out)


def pick_prompt(manifest: Mapping[str, Any], prompt_id: Optional[str]) -> Mapping[str, Any]:
    prompts = manifest.get("prompts", [])
    if not prompts:
        raise ValueError("manifest has no prompts")
    if prompt_id is None:
        return prompts[0]
    for p in prompts:
        if p["prompt_id"] == prompt_id:
            return p
    raise KeyError(f"prompt_id {prompt_id!r} not in manifest")


# =========================================================================== ComfyUI bootstrap


def bootstrap(args: argparse.Namespace) -> SimpleNamespace:
    """Import torch + ComfyUI headlessly with the L4 runtime flags applied before model_management loads."""
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

    comfy.cli_args.args.disable_dynamic_vram = not args.enable_dynamic_vram
    comfy.cli_args.args.highvram = not args.no_highvram
    import comfy.model_management
    import comfy.patcher_extension
    import comfy.sample
    import comfy.samplers
    import comfy.sd
    import comfy.utils
    import torch

    return SimpleNamespace(torch=torch, comfy=comfy, mm=comfy.model_management, sd=comfy.sd,
                           sample=comfy.sample, samplers=comfy.samplers, utils=comfy.utils,
                           pe=comfy.patcher_extension, pkg_dir=Path(comfy.sd.__file__).resolve().parent)


def load_text_encoder(c: SimpleNamespace, path: Path, clip_type: str) -> Any:
    """Load the TE with ComfyUI's native dtype handling (no forced dtype).

    The Anima / MOSS-Image / Krea 2 TE wrappers take their dtype from the checkpoint
    (``dtype_llama`` = norm-weight dtype) and build ``mixed_precision_ops`` from the embedded
    quantization metadata, so ``int8_convrot`` / ``fp8_scaled`` layers stay quantized and
    BF16 files load as BF16. A forced ``dtype`` would be overridden anyway and only skews
    ComfyUI's load log and initial-device memory estimate.
    """
    return c.sd.load_clip(ckpt_paths=[str(path)], clip_type=c.sd.CLIPType[clip_type])


def unload_all(c: SimpleNamespace) -> None:
    c.mm.unload_all_models()
    c.mm.soft_empty_cache()


def with_shift(c: SimpleNamespace, patcher: Any, shift: float) -> Any:
    """Clone with a fresh ``model_sampling`` of the same class at ``shift`` (ModelSamplingSD3/Flux pattern)."""
    m = patcher.clone()
    orig = m.get_model_object("model_sampling")
    ms = type(orig)(m.model.model_config)
    kw: dict[str, Any] = {"shift": shift}
    if "multiplier" in inspect.signature(ms.set_parameters).parameters:
        kw["multiplier"] = getattr(orig, "multiplier", 1.0)
    ms.set_parameters(**kw)
    if hasattr(orig, "noise_scale") and hasattr(ms, "set_noise_scale"):
        ms.set_noise_scale(orig.noise_scale)
    m.add_object_patch("model_sampling", ms)
    return m


def empty_latent(c: SimpleNamespace, patcher: Any, width: int = 1024, height: int = 1024) -> Any:
    """Model-native empty latent (EmptyLatentImage + ``fix_empty_latent_channels``).

    Resolves channels, spatial ratio (Flux2 /16 for MOSS-Image) and the 5D Wan21 layout
    (Anima, Krea 2) from the model's ``latent_format``; 1024x1024 -> 64x64 DiT tokens for all 3.
    """
    latent = c.torch.zeros([1, 4, height // 8, width // 8])
    return c.sample.fix_empty_latent_channels(patcher, latent, downscale_ratio_spacial=8)


def install_runtime_wrappers(c: SimpleNamespace, patcher: Any, policy: str) -> None:
    from ..adapters._comfy_hooks import cfg_batching_wrapper
    from ..core.profiling import FORWARD_STAGE, measure

    def profile_forward(executor: Callable[..., Any], *a: Any, **kw: Any) -> Any:
        with measure(FORWARD_STAGE):
            return executor(*a, **kw)

    patcher.add_wrapper_with_key(c.pe.WrappersMP.CALC_COND_BATCH, _CFG_WRAPPER_KEY, cfg_batching_wrapper(policy))
    patcher.add_wrapper_with_key(c.pe.WrappersMP.DIFFUSION_MODEL, _PROFILE_WRAPPER_KEY, profile_forward)


# =========================================================================== conditioning cache


def cache_dir(args: argparse.Namespace, model: str, manifest_sha: str) -> Path:
    return Path(args.cache_root) / f"{model}_{manifest_sha[:16]}"


def encoder_id(model: str, te_name: str) -> str:
    """Cache identity of the TE: file basename + precision (a different TE file never hits old entries)."""
    name = Path(te_name).name
    return f"{model}:{name}:{weight_precision(name)}"


def make_encoder(c: SimpleNamespace, clip: Any, adapter: Any) -> Callable[[str], dict[str, Any]]:
    def encode(text: str) -> dict[str, Any]:
        tokens = clip.tokenize(text)
        out = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        payload: dict[str, Any] = {"cond": out.pop("cond"), "token_ids": adapter.dit_token_ids(tokens)}
        for k, v in out.items():
            if isinstance(v, c.torch.Tensor):
                payload[f"extra.{k}"] = v
        return payload

    return encode


def to_conditioning(entry: Mapping[str, Any]) -> list:
    extras = {k[len("extra."):]: v for k, v in entry.items() if k.startswith("extra.")}
    return [[entry["cond"], extras]]


def ensure_cond_cache(c: SimpleNamespace, args: argparse.Namespace, d: ModelDefaults, adapter: Any,
                      texts: Sequence[str], manifest_sha: str) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Encode missing texts with the model's TE (then unload it); return (cache, {text: entry}).

    Conditioning is stored as BF16 on CPU (``CondCache`` default) whatever the TE weight precision.
    """
    from ..core.patching import CondCache
    from ..core.profiling import measure

    te_name = args.te or d.te
    cache = CondCache(cache_dir(args, d.name, manifest_sha), encoder_id(d.name, te_name))
    missing = [t for t in texts if not cache.path(t).exists()]
    if missing:  # the TE file is only needed when something must be encoded
        te_path = resolve_weight("te", te_name, args.models_dir)
        log(f"encoding {len(missing)}/{len(texts)} texts with {weight_precision(te_path.name)} TE {te_path.name}")
        with measure("te_encode"):
            clip = load_text_encoder(c, te_path, d.clip_type)
            encode = make_encoder(c, clip, adapter)
            for t in missing:
                cache.get_or_encode(t, encode)
            del clip, encode
            unload_all(c)
    return cache, {t: cache.get_or_encode(t, _no_encode) for t in texts}


def _no_encode(text: str) -> dict[str, Any]:
    raise RuntimeError("conditioning cache miss after encode pass")


# =========================================================================== serialisation helpers


def token_map_json(tm: Any) -> dict[str, Any]:
    return {"token_ids": list(tm.token_ids), "occurrence_idx": list(tm.occurrence_idx),
            "spans": {k: [s.start, s.end] for k, s in tm.spans.items()},
            "targets": {k: [s.start, s.end] for k, s in tm.targets.items()},
            "text_len": tm.text_len, "branch_text_len": dict(tm.branch_text_len)}


def stage_resources(summary: Mapping[str, Any], stage: str) -> Optional[dict[str, Any]]:
    st = summary["stages"].get(stage)
    if st is None:
        return None
    return {"gpu_allocated_gb": st["gpu_allocated_gb"], "gpu_reserved_gb": st["gpu_reserved_gb"],
            "cpu_rss_gb": st["cpu_rss_gb"], "seconds": st["seconds"], "calls": st["calls"],
            "sec_per_forward": summary.get("sec_per_forward") if stage == "forward" else st["sec_per_call"],
            "wall_clock_sec": summary["wall_clock_sec"]}


def save_png(image: Any, path: Path) -> bool:
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return False
    arr = (image.float().clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)
    return True


# =========================================================================== commands


def cmd_encode(args: argparse.Namespace) -> int:
    from ..adapters import canonical_model_name, get_adapter

    model = canonical_model_name(args.model)
    d = MODEL_DEFAULTS[model]
    manifest, manifest_sha = load_manifest(args.manifest or REPO_ROOT / f"experiments/design/manifest_{model}.json")
    texts = manifest_texts(manifest)
    if args.dry_run:
        log(f"encode plan: model={model} texts={len(texts)} cache={cache_dir(args, model, manifest_sha).name}")
        return 0
    c = bootstrap(args)
    from ..core.profiling import Profiler

    prof = Profiler("cuda" if c.torch.cuda.is_available() else None)
    with c.torch.inference_mode(), prof.activate():
        cache, _ = ensure_cond_cache(c, args, d, get_adapter(model), texts, manifest_sha)
    summary = prof.summary()
    log(f"cond cache ready: {len(texts)} texts, te_cache_sha256={cache.manifest_hash(texts)[:16]}..., "
        f"te_encode={summary['stages'].get('te_encode', {}).get('seconds', 0.0):.1f}s")
    return 0


class _IdentityObserver:
    """Forces the manual chunked SDPA path without editing anything (path-matched identity)."""

    def __call__(self, probs: Any, q_sl: slice, k_sl: slice) -> None:
        return None

    def finish(self) -> None:
        pass


def cmd_e0(args: argparse.Namespace) -> int:
    from ..adapters import canonical_model_name, get_adapter

    model = canonical_model_name(args.model)
    d = MODEL_DEFAULTS[model]
    manifest_path = Path(args.manifest or REPO_ROOT / f"experiments/design/manifest_{model}.json")
    manifest, manifest_sha = load_manifest(manifest_path)
    entry = pick_prompt(manifest, args.prompt_id)
    prompt = args.prompt if args.prompt is not None else entry["prompt"]
    negative = args.negative if args.negative is not None else entry.get(
        "negative_prompt", manifest.get("default_negative_prompt", d.negative))
    seed = args.seed if args.seed is not None else int(manifest.get("seed_sets", {}).get("sanity", [42])[0])
    steps, cfg = args.steps or d.steps, args.cfg if args.cfg is not None else d.cfg
    shift = args.shift if args.shift is not None else d.shift
    sampler, scheduler = args.sampler or d.sampler, args.scheduler or d.scheduler
    policy = args.cfg_batching or d.batching
    precision = args.precision or d.precision
    out_root = Path(args.out_root)
    prefix = f"e0_{model}_{precision}"
    run_id = args.run_id or next_run_id(out_root, prefix)
    out = out_root / f"{prefix}_{run_id}"
    plan = dict(model=model, prompt_id=entry.get("prompt_id"), seed=seed, steps=steps, cfg=cfg, shift=shift,
                sampler=sampler, scheduler=scheduler, cfg_batching=policy, reps=args.reps, out=out.name)
    if args.dry_run:
        log(f"e0 plan: {json.dumps(plan)}")
        return 0

    c = bootstrap(args)
    torch = c.torch
    from ..core.profiling import Profiler, measure
    from ..core.session import XAISession

    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "visuals").mkdir(exist_ok=True)
    shards: dict[str, dict[str, Any]] = {}
    manifest_json = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "exp_id": "e0", "plan": plan,
                     "shards": shards, "failures": [], "non_crossing": []}

    def shard_done(name: str, **info: Any) -> None:
        shards[name] = {"status": "done", **info}
        write_json(out / "manifest.json", manifest_json)

    cuda = "cuda" if torch.cuda.is_available() else None
    texts = [prompt, negative]
    adapter = get_adapter(model)
    resources: dict[str, Any] = {}

    with torch.inference_mode():
        prof_te = Profiler(cuda)
        with prof_te.activate():
            cache, conds = ensure_cond_cache(c, args, d, adapter, texts, manifest_sha)
        resources["te_encode"] = stage_resources(prof_te.summary(), "te_encode")
        positive, negative_c = to_conditioning(conds[prompt]), to_conditioning(conds[negative])

        dit_path = resolve_weight("dit", args.dit or d.dit, args.models_dir)
        vae_path = resolve_weight("vae", args.vae or d.vae, args.models_dir)
        prof_load = Profiler(cuda)
        with prof_load.activate():
            with measure("dit_load"):
                base = c.sd.load_diffusion_model(str(dit_path))
            vae = c.sd.VAE(sd=c.utils.load_torch_file(str(vae_path)))
            vae.throw_exception_if_invalid()
        resources["dit_load"] = stage_resources(prof_load.summary(), "dit_load")

        patcher = with_shift(c, base, shift)
        install_runtime_wrappers(c, patcher, policy)
        adapter.bind(patcher)
        sigmas = c.samplers.calculate_sigmas(patcher.get_model_object("model_sampling"), scheduler, steps).cpu()
        sigma_table = [float(s) for s in sigmas]
        latent = empty_latent(c, patcher, args.width, args.height)

        tm_cond = adapter.build_token_map(conds[prompt]["token_ids"], branch="cond")
        tm_uncond = adapter.build_token_map(conds[negative]["token_ids"], branch="uncond")

        def run(tag: str, session: Optional[Any] = None) -> dict[str, Any]:
            raw_path = out / "raw" / f"{tag}.pt"
            if args.resume and raw_path.exists():
                log(f"resume: {tag} loaded from raw/")
                return torch.load(raw_path, map_location="cpu", weights_only=True)
            x0s: list[Any] = []

            def cb(step: int, x0: Any, x: Any, total: int) -> None:
                x0s.append(x0.detach().float().cpu())

            prof = Profiler(cuda)
            noise = c.sample.prepare_noise(latent, seed)
            sampler_obj = c.samplers.sampler_object(sampler)
            with prof.activate():
                if session is None:
                    samples = c.sample.sample_custom(patcher, noise, cfg, sampler_obj, sigmas, positive, negative_c,
                                                     latent, callback=cb, disable_pbar=True, seed=seed)
                else:
                    with adapter.attached(session):
                        samples = c.sample.sample_custom(patcher, noise, cfg, sampler_obj, sigmas, positive, negative_c,
                                                         latent, callback=cb, disable_pbar=True, seed=seed)
            result = {"latent": samples.detach().float().cpu(), "x0_steps": torch.stack(x0s)}
            torch.save(result, raw_path)
            shard_done(tag, resources=stage_resources(prof.summary(), "forward"))
            log(f"{tag}: sec/forward={prof.summary()['sec_per_forward']}")
            return result

        def max_diff(a: Any, b: Any) -> float:
            return float((a.float() - b.float()).abs().max())

        def mean_diff(a: Any, b: Any) -> float:
            return float((a.float() - b.float()).abs().mean())

        reps = [run(f"baseline_rep{r}") for r in range(args.reps)]
        resources["dit_baseline"] = shards["baseline_rep0"]["resources"] if "baseline_rep0" in shards else None

        noop = run("exact_noop", XAISession(adapter.capability))
        identity_session = XAISession(adapter.capability, query_chunk=args.query_chunk)
        identity_session.add_observer(lambda call: _IdentityObserver())
        ident = run("path_identity", identity_session)
        resources["dit_hook_identity"] = shards.get("path_identity", {}).get("resources")

        prof_vae = Profiler(cuda)
        images = []
        with prof_vae.activate(), measure("vae_decode"):
            for r in reps[:2]:
                img = vae.decode(r["latent"])
                images.append(img.reshape(-1, *img.shape[-3:])[0])
        resources["vae_decode"] = stage_resources(prof_vae.summary(), "vae_decode")
        resources["offline_evaluators"] = None  # DINOv2 / LPIPS run in the separate offline pass
        if len(images) == 2:
            diff = (images[0].float() - images[1].float()).abs()
            sheet = torch.cat([images[0].float(), images[1].float(), (diff * 10).clamp(0, 1)], dim=1)
            save_png(sheet, out / "visuals" / "baseline_run1_vs_run2.png")
        if images:
            save_png(images[0], out / "visuals" / "generated.png")

    floor = [max_diff(reps[0]["latent"], r["latent"]) for r in reps[1:]]
    pixel_floor = float((images[0].float() - images[1].float()).abs().max()) if len(images) == 2 else None
    noop_diff = max(max_diff(reps[0]["latent"], noop["latent"]), max_diff(reps[0]["x0_steps"], noop["x0_steps"]))
    ident_step0_max = max_diff(reps[0]["x0_steps"][0], ident["x0_steps"][0])
    ident_step0_mean = mean_diff(reps[0]["x0_steps"][0], ident["x0_steps"][0])
    ident_final_max = max_diff(reps[0]["latent"], ident["latent"])
    ident_final_mean = mean_diff(reps[0]["latent"], ident["latent"])
    pass_fail = {
        "exact_noop_max_abs_diff": noop_diff,
        "exact_noop_pass": noop_diff == 0.0,
        "path_matched_identity_step0_max_abs_diff": ident_step0_max,
        "path_matched_identity_step0_mean_abs_diff": ident_step0_mean,
        "path_matched_identity_pass": ident_step0_mean <= 1e-2,
        "path_matched_identity_final_latent_max_abs_diff": ident_final_max,
        "path_matched_identity_final_latent_mean_abs_diff": ident_final_mean,
        "non_determinism_floor": {"latent_max_abs_diff": floor, "pixel_max_abs_diff": pixel_floor,
                                  "dinov2": None, "lpips": None},
    }
    metrics = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "pass_fail": pass_fail, "resources": resources,
        "per_sample": [{"prompt_id": entry.get("prompt_id"), "seed": seed, "condition": cond, "status": "ok"}
                       for cond in [f"baseline_rep{r}" for r in range(args.reps)] + ["exact_noop", "path_identity"]],
        "summary": {"identity_step0_reference": "CFG-combined denoised x0 at step 0 (single forward, same x_T)"},
    }
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    config = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id, "baseline_run_id": None, "exp_id": "e0",
        "model": {"name": model, "variant": precision, "weights_file": dit_path.name, "repo_revision": args.repo_revision,
                  "weights_sha256": None if args.skip_hash else sha256_file(dit_path),
                  "vae_file": vae_path.name, "vae_sha256": None if args.skip_hash else sha256_file(vae_path),
                  "te_file": Path(args.te or d.te).name, "te_precision": weight_precision(args.te or d.te),
                  "te_cache_sha256": cache.manifest_hash(texts)},
        "runtime": {"gpu_name": gpu, "comfyui_commit": comfyui_commit(c.pkg_dir), "attention_path": "fused_sdpa",
                    "attention_paths": {"baseline": "fused_sdpa", "exact_noop": "fused_sdpa",
                                        "path_identity": "manual_chunked_sdpa"},
                    "cfg_batching_policy": policy, "sampler": sampler, "scheduler": scheduler, "cfg": cfg,
                    "shift": shift, "steps": steps, "sigma_table": sigma_table,
                    "representative_steps": [{"i": i, "sigma": sigma_table[i]} for i in d.rep_steps if i < len(sigma_table)],
                    "resolution": [args.width, args.height], "seed": seed, "reps": args.reps,
                    "comfyui_flags": {"disable_dynamic_vram": not args.enable_dynamic_vram, "highvram": not args.no_highvram},
                    "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                    "torch_version": torch.__version__},
        "prompt_manifest_sha256": manifest_sha,
        "prompt": {"prompt_id": entry.get("prompt_id"), "prompt": prompt, "negative_prompt": negative},
        "token_map": {"cond": token_map_json(tm_cond), "uncond": token_map_json(tm_uncond)},
        "evaluator_versions": {"dinov2": None, "lpips": None, "openclip": None, "grounded_sam2": None, "qwen25_vl": None},
    }
    write_json(out / "config.json", config)
    write_json(out / "metrics.json", metrics)
    log(f"E0 done -> {out.name}: exact_noop={noop_diff:.3g} identity_step0(mean/max)={ident_step0_mean:.3g}/{ident_step0_max:.3g} floor={floor}")
    return 0 if pass_fail["exact_noop_pass"] and pass_fail["path_matched_identity_pass"] else 1


# =========================================================================== argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="xai-dit", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--model", required=True, help="anima | moss_image | krea2")
        sp.add_argument("--comfyui-dir", help="ComfyUI checkout (default: $COMFYUI_DIR)")
        sp.add_argument("--models-dir", type=Path, default=os.environ.get("XAI_DIT_MODELS_DIR"),
                        help="root holding diffusion_models/ text_encoders/ vae/ (default: $XAI_DIT_MODELS_DIR)")
        sp.add_argument("--te", help="text-encoder file name or path (default per model)")
        sp.add_argument("--manifest", type=Path, help="prompt manifest (default experiments/design/manifest_<model>.json)")
        sp.add_argument("--cache-root", default=str(REPO_ROOT / "cache" / "cond"))
        sp.add_argument("--env-file", type=Path, default=REPO_ROOT / "local" / ".env")
        sp.add_argument("--no-highvram", action="store_true")
        sp.add_argument("--enable-dynamic-vram", action="store_true")
        sp.add_argument("--dry-run", action="store_true", help="resolve the plan without importing torch/ComfyUI")

    enc = sub.add_parser("encode", help="offline conditioning cache for a manifest")
    common(enc)
    enc.set_defaults(func=cmd_encode)

    e0 = sub.add_parser("e0", help="E0 sanity: non-determinism floor, exact no-op, path identity, budget")
    common(e0)
    e0.add_argument("--dit", help="DiT file name or path (default per model)")
    e0.add_argument("--vae", help="VAE file name or path (default per model)")
    e0.add_argument("--precision", help="precision label for the run folder (default per model)")
    e0.add_argument("--repo-revision", help="HF repo revision of the weights (logged)")
    e0.add_argument("--prompt-id")
    e0.add_argument("--prompt")
    e0.add_argument("--negative")
    e0.add_argument("--seed", type=int)
    e0.add_argument("--steps", type=int)
    e0.add_argument("--cfg", type=float)
    e0.add_argument("--shift", type=float)
    e0.add_argument("--sampler")
    e0.add_argument("--scheduler")
    e0.add_argument("--cfg-batching", choices=("split", "batched"))
    e0.add_argument("--width", type=int, default=1024)
    e0.add_argument("--height", type=int, default=1024)
    e0.add_argument("--reps", type=int, default=3)
    e0.add_argument("--query-chunk", type=int, default=1024)
    e0.add_argument("--out-root", default=str(REPO_ROOT / "experiments"))
    e0.add_argument("--run-id")
    e0.add_argument("--resume", action="store_true", help="reuse finished shards under raw/")
    e0.add_argument("--skip-hash", action="store_true", help="skip sha256 of DiT/VAE weights")
    e0.set_defaults(func=cmd_e0)
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(None if argv is None else list(argv))
    loaded = load_local_env(Path(args.env_file))
    if loaded:
        log(f"loaded {loaded} variable(s) from the local env file")
    if args.models_dir is None and os.environ.get("XAI_DIT_MODELS_DIR"):
        args.models_dir = Path(os.environ["XAI_DIT_MODELS_DIR"])
    if args.models_dir is not None:
        args.models_dir = Path(args.models_dir).expanduser()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
