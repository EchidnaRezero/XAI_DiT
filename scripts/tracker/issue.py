"""Issue log manager (issues/NNN_<slug>.md + issues_index.jsonl)."""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

from . import REPO_ROOT

ISSUE_RE = re.compile(r"^(\d{3})_([a-z0-9_]+)\.md$")


def issues_dir(root: Path = REPO_ROOT) -> Path:
    return root / "issues"


def scan_issues(root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    idir = issues_dir(root)
    if not idir.exists():
        return []
    records: list[dict[str, Any]] = []
    for p in sorted(idir.glob("*.md")):
        m = ISSUE_RE.match(p.name)
        if not m:
            continue
        title = p.stem
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        records.append(
            {
                "issue_id": f"ISSUE-{m.group(1)}",
                "num": int(m.group(1)),
                "file": p.name,
                "title": title,
            }
        )
    return records


def new_issue(slug: str, title: str | None = None, root: Path = REPO_ROOT) -> Path:
    clean_slug = re.sub(r"[^a-z0-9]+", "_", slug.lower()).strip("_")
    if not clean_slug:
        raise ValueError("Slug must contain at least one alphanumeric character.")
    existing = scan_issues(root)
    next_num = (max((r["num"] for r in existing), default=0)) + 1
    fname = f"{next_num:03d}_{clean_slug}.md"
    out = issues_dir(root) / fname
    tpl_path = issues_dir(root) / "_template.md"
    header = f"# ISSUE-{next_num:03d}: {title or clean_slug.replace('_', ' ')}\n\nDate: {dt.date.today().isoformat()}\n\n"
    if tpl_path.exists():
        body = tpl_path.read_text(encoding="utf-8")
        out.write_text(header + body, encoding="utf-8")
    else:
        out.write_text(
            header + "## Problem\n\n## Observation\n\n## Root Cause\n\n## Fix & Verification\n",
            encoding="utf-8",
        )
    rebuild_index(root)
    return out


def rebuild_index(root: Path = REPO_ROOT) -> Path:
    records = scan_issues(root)
    out = issues_dir(root) / "issues_index.jsonl"
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + ("\n" if records else ""),
        encoding="utf-8",
    )
    return out


def check_issues(root: Path = REPO_ROOT) -> list[str]:
    errors: list[str] = []
    seen: set[int] = set()
    for r in scan_issues(root):
        if r["num"] in seen:
            errors.append(f"[issues] Duplicate issue number: {r['num']:03d}")
        seen.add(r["num"])
    return errors
