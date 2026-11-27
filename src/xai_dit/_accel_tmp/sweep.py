"""Phase 1 Multi-Preset Sweep Orchestrator (Anima -> MOSS-Image -> Krea 2).

Groups acceleration presets by process-level ComfyUI flags (attention backend, --fast features,
upcast flags) and launches ``run_t2i.py`` in a clean subprocess per flag-group so global kernel
state never leaks between tests, while reusing cached conditioning on disk.

Outputs:
    <out_dir>/
        <preset>.png               # Full 3-chunk ComfyUI PNG EXIF
        <preset>_result.json       # Timing, VRAM, image metrics, optional diag_probe
        summary.json               # Aggregated table
        summary.md                 # Markdown comparison table (Speedup, VRAM, PSNR, HF ratio)
        contact_sheet.png          # Visual grid + 1:1 center-crop graininess strip
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Sequence

from ..runners.cli import REPO_ROOT, log, write_json
from .metrics_and_exif import make_comparison_sheet
from .presets import AccelPreset, build_presets_for_model, get_prompt, presets_by_name


def select_presets(model: str, stage: str, preset_names: Optional[str], include_compile: bool) -> list[AccelPreset]:
    if preset_names:
        catalog = presets_by_name(model)
        out = []
        for raw in preset_names.split(","):
            name = raw.strip()
            if not name:
                continue
            if name not in catalog:
                raise SystemExit(f"Unknown preset {name!r} for {model}; choices: {sorted(catalog)}")
            out.append(catalog[name])
        return out

    all_p = build_presets_for_model(model)
    if stage == "all":
        return [p for p in all_p if include_compile or p.torch_compile is None]
    wanted_stages = {s.strip() for s in stage.split(",") if s.strip()}
    # Always include baseline first when running single/combo/fix so PSNR & HF ratio are computed
    selected = [p for p in all_p if p.stage in wanted_stages and (include_compile or p.torch_compile is None)]
    if "baseline" not in wanted_stages and not any(p.name == "baseline" for p in selected):
        selected.insert(0, all_p[0])
    return selected


def group_by_process_flags(presets: Sequence[AccelPreset]) -> list[tuple[tuple[Any, ...], list[AccelPreset]]]:
    groups: OrderedDict[tuple[Any, ...], list[AccelPreset]] = OrderedDict()
    for p in presets:
        groups.setdefault(p.process_flag_key(), []).append(p)
    return list(groups.items())


def render_markdown_summary(model: str, results: Sequence[dict[str, Any]]) -> str:
    base_sec = next((r["wall_clock_sec"] for r in results if r.get("preset") == "baseline" and r.get("wall_clock_sec")), None)
    lines = [
        f"# Phase 1 Acceleration Summary — `{model}`",
        "",
        "| Preset | Stage | Status | Wall (s) | Speedup | s/fwd | VRAM (GB) | PSNR (dB) | HF Ratio | Skipped | Notes |",
        "|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        if r.get("error"):
            lines.append(
                f"| `{r['preset']}` | {r.get('stage', '-')} | ERROR | - | - | - | - | - | - | - | {r['error'][:80]} |"
            )
            continue
        m = r.get("metrics", {})
        status = "COLLAPSE" if m.get("is_collapsed") else ("GRAIN?" if m.get("hf_ratio_vs_base", 1.0) > 1.15 else "OK")
        wall = r.get("wall_clock_sec", 0.0)
        spd = f"{base_sec / wall:.2f}x" if (base_sec and wall > 0) else "-"
        psnr = f"{m['psnr_db']:.2f}" if "psnr_db" in m else "-"
        hfr = f"{m['hf_ratio_vs_base']:.3f}" if "hf_ratio_vs_base" in m else "-"
        skipped = r.get("acceleration", {}).get("easycache_steps_skipped", 0)
        lines.append(
            f"| `{r['preset']}` | {r.get('stage', '-')} | **{status}** | {wall:.2f} | {spd} | "
            f"{r.get('sec_per_forward', 0.0):.3f} | {r.get('peak_vram_gb', 0.0):.2f} | "
            f"{psnr} | {hfr} | {skipped} | {r.get('notes', '')} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 1 multi-preset acceleration sweep orchestrator.")
    p.add_argument("--model", required=True, choices=["anima", "moss_image", "krea2"])
    p.add_argument(
        "--stage",
        default="single,combo,fix",
        help="Stage(s) to run: baseline, single, combo, fix, fewstep, or all (comma-separated).",
    )
    p.add_argument("--presets", default=None, help="Explicit comma-separated preset names (overrides --stage).")
    p.add_argument("--include-compile", action="store_true", help="Include TorchCompileModel preset in sweep.")
    p.add_argument("--prompt-id", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--negative", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--comfyui-dir", default=None)
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--cond-cache-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--diag-probe", action="store_true")
    p.add_argument("--list-presets", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_presets:
        for p in build_presets_for_model(args.model):
            print(f"{p.name:28s} [{p.stage:8s}] attn={p.attention:7s} fast={list(p.fast_features)} ec={p.easycache} | {p.notes}")
        return 0

    selected = select_presets(args.model, args.stage, args.presets, args.include_compile)
    groups = group_by_process_flags(selected)
    pspec = get_prompt(args.model, args.prompt_id)

    log(f"sweep model={args.model} prompt_id={pspec.prompt_id} presets={len(selected)} flag_groups={len(groups)}")
    for idx, (fkey, plist) in enumerate(groups, 1):
        log(f"  group {idx}/{len(groups)} flags={fkey} -> {[p.name for p in plist]}")

    if args.dry_run:
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cond_cache = args.cond_cache_dir or (args.out_dir / "_cond_cache")
    baseline_png = args.out_dir / "baseline.png"

    for idx, (fkey, plist) in enumerate(groups, 1):
        names = ",".join(p.name for p in plist)
        cmd = [
            sys.executable,
            "-m",
            "xai_dit._accel_tmp.run_t2i",
            "--model", args.model,
            "--presets", names,
            "--seed", str(args.seed),
            "--width", str(args.width),
            "--height", str(args.height),
            "--out-dir", str(args.out_dir),
            "--cond-cache-dir", str(cond_cache),
            "--baseline-png", str(baseline_png),
        ]
        if args.prompt_id:
            cmd += ["--prompt-id", args.prompt_id]
        if args.prompt is not None:
            cmd += ["--prompt", args.prompt]
        if args.negative is not None:
            cmd += ["--negative", args.negative]
        if args.comfyui_dir:
            cmd += ["--comfyui-dir", args.comfyui_dir]
        if args.models_dir:
            cmd += ["--models-dir", str(args.models_dir)]
        if args.diag_probe:
            cmd.append("--diag-probe")

        log(f"=== launching group {idx}/{len(groups)}: {names} ===")
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        if proc.stdout:
            print(proc.stdout.rstrip(), flush=True)
        if proc.returncode != 0:
            if proc.stderr:
                print(proc.stderr[-1500:], flush=True)
            log(f"WARNING: group {names} exited with code {proc.returncode}; recording error and continuing.")
            for p in plist:
                res_path = args.out_dir / f"{p.name}_result.json"
                if not res_path.is_file():
                    write_json(res_path, {
                        "preset": p.name,
                        "stage": p.stage,
                        "model": args.model,
                        "error": f"subprocess exit code {proc.returncode}",
                    })

    # Aggregate results
    results: list[dict[str, Any]] = []
    sheet_items: list[tuple[str, Path]] = []
    for p in selected:
        res_path = args.out_dir / f"{p.name}_result.json"
        if res_path.is_file():
            r = json.loads(res_path.read_text(encoding="utf-8"))
            results.append(r)
            png = args.out_dir / f"{p.name}.png"
            if png.is_file():
                wall = r.get("wall_clock_sec", 0.0)
                m = r.get("metrics", {})
                tag = "COLLAPSE" if m.get("is_collapsed") else f"{wall:.1f}s"
                hfr = f" HF={m['hf_ratio_vs_base']:.2f}" if "hf_ratio_vs_base" in m else ""
                sheet_items.append((f"{p.name} ({tag}{hfr})", png))

    write_json(args.out_dir / "summary.json", {"model": args.model, "prompt_id": pspec.prompt_id, "results": results})
    md_text = render_markdown_summary(args.model, results)
    (args.out_dir / "summary.md").write_text(md_text, encoding="utf-8")
    if sheet_items:
        make_comparison_sheet(sheet_items, args.out_dir / "contact_sheet.png", cols=4, include_center_crop=True)
    log(f"sweep finished: {args.out_dir / 'summary.md'} & {args.out_dir / 'contact_sheet.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
