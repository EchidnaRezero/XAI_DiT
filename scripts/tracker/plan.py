"""Plan, interrupt, link integrity, and bilingual (*_KR.md) sync checker."""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

from . import REPO_ROOT

LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
PLAN_RE = re.compile(r"^PLAN-(\d{3})_([a-z0-9_]+)\.md$")


def new_active(slug: str, title: str | None = None, root: Path = REPO_ROOT) -> Path:
    clean_slug = re.sub(r"[^a-z0-9]+", "_", slug.lower()).strip("_")
    adir = root / "plan" / "active"
    adir.mkdir(parents=True, exist_ok=True)
    max_num = 0
    for p in adir.glob("PLAN-*.md"):
        m = PLAN_RE.match(p.name)
        if m:
            max_num = max(max_num, int(m.group(1)))
    num = max_num + 1
    out = adir / f"PLAN-{num:03d}_{clean_slug}.md"
    out.write_text(
        f"# PLAN-{num:03d}: {title or clean_slug.replace('_', ' ')}\n\n"
        f"Created: {dt.date.today().isoformat()}  \n"
        f"Status: Active\n\n---\n\n## Goal\n\n## Checklist\n- [ ] \n",
        encoding="utf-8",
    )
    return out


def new_interrupt(slug: str, title: str | None = None, root: Path = REPO_ROOT) -> Path:
    clean_slug = re.sub(r"[^a-z0-9]+", "_", slug.lower()).strip("_")
    idir = root / "plan" / "interrupts"
    idir.mkdir(parents=True, exist_ok=True)
    day = dt.date.today().strftime("%Y%m%d")
    out = idir / f"{day}_{clean_slug}.md"
    out.write_text(
        f"# Interrupt ({day}): {title or clean_slug.replace('_', ' ')}\n\n"
        f"Created: {dt.date.today().isoformat()}  \n"
        f"Status: Active Interrupt (archive or promote to `plan/active/` when resolved)\n\n"
        f"---\n\n## Trigger / Why Now\n\n## Action Items\n- [ ] \n\n## Resolution\n",
        encoding="utf-8",
    )
    return out


def _strip_code_blocks(text: str) -> str:
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    return "\n".join(out)


def check_links(root: Path = REPO_ROOT) -> list[str]:
    targets: list[Path] = []
    for p in (root / "README.md", root / "README_KR.md", root / "AGENTS.md"):
        if p.exists():
            targets.append(p)
    for folder in (root / "docs", root / "plan", root / "experiments" / "accel"):
        if folder.exists():
            for md in sorted(folder.rglob("*.md")):
                if "reference" in md.parts:
                    continue
                if "runs" in md.parts:
                    continue
                targets.append(md)
    errors: list[str] = []
    for md in targets:
        text = _strip_code_blocks(md.read_text(encoding="utf-8", errors="replace"))
        for match in LINK_RE.finditer(text):
            raw = match.group(1).strip().split()[0]
            if not raw or raw.startswith(("http://", "https://", "mailto:", "#", "file://")):
                continue
            path_part = raw.split("#", 1)[0]
            if not path_part:
                continue
            resolved = (md.parent / path_part).resolve()
            if not resolved.exists():
                rel_md = md.relative_to(root)
                errors.append(f"[link] Broken link in {rel_md}: ({raw})")
    return errors


def check_bilingual(root: Path = REPO_ROOT) -> list[str]:
    errors: list[str] = []
    for kr in sorted(root.rglob("*_KR.md")):
        if any(part in kr.parts for part in (".git", ".venv", "reference_repo")):
            continue
        en = kr.with_name(kr.name.replace("_KR.md", ".md"))
        if not en.exists():
            errors.append(f"[bilingual] Missing English companion for {kr.relative_to(root)}")
            continue
        kr_h2 = [l for l in kr.read_text(encoding="utf-8").splitlines() if l.startswith("## ")]
        en_h2 = [l for l in en.read_text(encoding="utf-8").splitlines() if l.startswith("## ")]
        if len(kr_h2) != len(en_h2):
            errors.append(
                f"[bilingual] H2 section mismatch: {en.name} ({len(en_h2)}) vs {kr.name} ({len(kr_h2)})"
            )
    return errors
