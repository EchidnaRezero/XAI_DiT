---
name: scribe-plan
description: Plan-Scribe SOP for updating plan/status.md, managing interrupts/archives, syncing README/README_KR, and checking markdown links. Owns plan/ and README*.md ONLY.
---

# Plan-Scribe SOP

## File Ownership Boundary
- **Allowed**: `plan/` (`status.md`, `roadmap.md`, `active/`, `interrupts/`, `archive/`) and `README.md` / `README_KR.md`.
- **Forbidden**: Never edit `experiments/`, `issues/`, or `src/`.

## Procedure
1. Update `plan/status.md` (check off completed items, reflect active plans or active interrupts, update `Last updated`).
2. If an active plan or interrupt is finished, summarize it in `plan/archive/completed_milestones.md` and remove/archive the finished plan file.
3. If `README.md` changed, keep `README_KR.md` 100% synchronized (identical structure, numbers, and links).
4. Run `python -m scripts.tracker.cli plan check` and confirm `0 error(s)` (broken links + bilingual sync).
5. Return a 2–3 line summary to Orchestrator and exit immediately.
