"""Experiment track manager (experiments/accel & experiments/case_studies)."""
from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from . import REPO_ROOT

VALID_TRACKS = ("accel", "case_studies")
RUN_ID_RE = re.compile(r"^(\d{8})_(\d{2})_([a-z0-9_]+)$")


def track_dir(track: str, root: Path = REPO_ROOT) -> Path:
    if track not in VALID_TRACKS:
        raise ValueError(f"Unknown track '{track}'. Choose from: {VALID_TRACKS}")
    return root / "experiments" / track


def load_index(track: str, root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    idx_path = track_dir(track, root) / "runs_index.jsonl"
    if not idx_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(idx_path.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        try:
            rows.append(json.loads(s))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{idx_path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def next_run_id(track: str, slug: str, date_str: str | None = None, root: Path = REPO_ROOT) -> str:
    clean_slug = re.sub(r"[^a-z0-9]+", "_", slug.lower()).strip("_")
    if not clean_slug:
        raise ValueError("Slug must contain at least one alphanumeric character.")
    day = (date_str or dt.date.today().strftime("%Y%m%d")).replace("-", "")
    runs_root = track_dir(track, root) / "runs"
    max_seq = 0
    for row in load_index(track, root):
        m = RUN_ID_RE.match(row.get("run_id", ""))
        if m and m.group(1) == day:
            max_seq = max(max_seq, int(m.group(2)))
    if runs_root.exists():
        for d in runs_root.iterdir():
            if d.is_dir():
                m = RUN_ID_RE.match(d.name)
                if m and m.group(1) == day:
                    max_seq = max(max_seq, int(m.group(2)))
    return f"{day}_{max_seq + 1:02d}_{clean_slug}"


def new_run(
    track: str,
    slug: str,
    run_type: str = "t2i_bench",
    models: list[str] | None = None,
    gpu: str = "L4",
    status: str = "verified",
    tags: list[str] | None = None,
    summary: str = "",
    date_str: str | None = None,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    tdir = track_dir(track, root)
    rid = next_run_id(track, slug, date_str=date_str, root=root)
    iso_date = f"{rid[:4]}-{rid[4:6]}-{rid[6:8]}"
    run_path = tdir / "runs" / rid
    run_path.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": rid,
        "date": iso_date,
        "type": run_type,
        "models": models or [],
        "gpu": gpu,
        "status": status,
        "tags": tags or [],
        "summary": summary or "(pending summary)",
        "path": f"runs/{rid}",
    }
    idx_path = tdir / "runs_index.jsonl"
    with idx_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def render_history(track: str, root: Path = REPO_ROOT) -> Path:
    tdir = track_dir(track, root)
    rows = load_index(track, root)
    title = (
        "# Acceleration Experiment History & Changelog"
        if track == "accel"
        else "# XAI Case Studies History (Local-Only / Gitignored)"
    )
    subtitle = (
        "Auto-generated from [`runs_index.jsonl`](runs_index.jsonl) · "
        "Consolidated Report: [`REPORT.md`](REPORT.md) · "
        "Practical Presets: [`docs/acceleration.md`](../../docs/acceleration.md)."
        if track == "accel"
        else "Auto-generated from [`runs_index.jsonl`](runs_index.jsonl) · "
        "Public Showcase Policy: [`README.md`](README.md)."
    )
    lines = [
        title,
        "",
        subtitle,
        "",
        "| Run ID | Date | Type | Models | GPU | Status | Key Outcome |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rid = r["run_id"]
        rel = r["path"].rstrip("/") + "/"
        models_s = ", ".join(r.get("models", []))
        summary_s = str(r.get("summary", "")).replace("|", "\\|")
        lines.append(
            f"| [`{rid}`]({rel}) | {r.get('date', '')} | `{r.get('type', '')}` | "
            f"{models_s} | {r.get('gpu', '')} | `{r.get('status', '')}` | {summary_s} |"
        )
    lines.append("")
    out_path = tdir / "HISTORY.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def check_track(track: str, root: Path = REPO_ROOT) -> list[str]:
    tdir = track_dir(track, root)
    idx_path = tdir / "runs_index.jsonl"
    if not idx_path.exists():
        return []
    errors: list[str] = []
    rows = load_index(track, root)
    seen_ids: set[str] = set()
    indexed_dirs: set[str] = set()
    for r in rows:
        rid = r.get("run_id", "")
        if not rid:
            errors.append(f"[{track}] Row missing run_id: {r}")
            continue
        if rid in seen_ids:
            errors.append(f"[{track}] Duplicate run_id: {rid}")
        seen_ids.add(rid)
        rel = r.get("path", "")
        target = tdir / rel
        if not target.exists():
            errors.append(f"[{track}] Missing path for {rid}: {rel}")
        indexed_dirs.add(Path(rel).name)
    runs_root = tdir / "runs"
    if runs_root.exists():
        for d in sorted(runs_root.iterdir()):
            if d.is_dir() and d.name not in indexed_dirs:
                errors.append(f"[{track}] Unindexed run folder in runs/: {d.name}")
    return errors


def export_sqlite(track: str, db_path: Path | None = None, root: Path = REPO_ROOT) -> Path:
    tdir = track_dir(track, root)
    rows = load_index(track, root)
    out = db_path or (tdir / "registry.sqlite")
    conn = sqlite3.connect(out)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                date TEXT,
                type TEXT,
                models TEXT,
                gpu TEXT,
                status TEXT,
                tags TEXT,
                summary TEXT,
                path TEXT
            )
            """
        )
        conn.execute("DELETE FROM runs")
        conn.executemany(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r.get("run_id", ""),
                    r.get("date", ""),
                    r.get("type", ""),
                    json.dumps(r.get("models", []), ensure_ascii=False),
                    r.get("gpu", ""),
                    r.get("status", ""),
                    json.dumps(r.get("tags", []), ensure_ascii=False),
                    r.get("summary", ""),
                    r.get("path", ""),
                )
                for r in rows
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return out
