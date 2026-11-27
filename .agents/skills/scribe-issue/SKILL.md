---
name: scribe-issue
description: Issue-Scribe SOP for creating numbered issue logs and maintaining issues_index.jsonl. Owns issues/ ONLY.
---

# Issue-Scribe SOP

## File Ownership Boundary
- **Allowed**: `issues/` (`NNN_<slug>.md`, `issues_index.jsonl`).
- **Forbidden**: Never edit `experiments/`, `plan/`, `README*.md`, or `src/`.

## Procedure
1. Create new issue files strictly via `python -m scripts.tracker.cli issue new <slug> --title "<Title>"`.
2. Fill `Problem → Observation → Root Cause → Fix & Verification` with exact file:line citations and evidence links (never duplicate PNGs).
3. Run `python -m scripts.tracker.cli issue index` and `python -m scripts.tracker.cli issue check`, confirming `0 error(s)`.
4. Return a 2–3 line summary (including new `ISSUE-NNN` path) to Orchestrator and exit immediately.
