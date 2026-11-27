---
name: scribe-exp
description: Exp-Scribe SOP for logging experiment runs, regenerating HISTORY.md, and updating experiment reports. Owns experiments/accel/ and experiments/case_studies/ ONLY.
---

# Exp-Scribe SOP

## File Ownership Boundary
- **Allowed**: `experiments/accel/{runs_index.jsonl, HISTORY.md, REPORT.md, RESOURCES.md}`, `experiments/case_studies/{runs_index.jsonl, HISTORY.md}`, and `docs/acceleration.md` (only if Top-3 presets changed).
- **Forbidden**: Never edit `issues/`, `plan/`, `README*.md`, or `src/`.

## Procedure
1. Ensure every new run folder uses `YYYYMMDD_NN_<slug>` (created via `python -m scripts.tracker.cli exp new <accel|case_studies> <slug> ...`).
2. Verify `runs_index.jsonl` has an accurate one-line JSON summary for the run.
3. Run `python -m scripts.tracker.cli exp render <track>` to regenerate `HISTORY.md`.
4. If new verified results supersede or extend benchmarks, update `experiments/accel/REPORT.md` (and `RESOURCES.md` / `docs/acceleration.md` if new weights/Top-3 presets).
5. Run `python -m scripts.tracker.cli exp check <track>` and confirm `0 error(s)`.
6. Return a 2–3 line summary to Orchestrator and exit immediately.
