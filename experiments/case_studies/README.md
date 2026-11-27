# XAI Practical Case Studies (Diagnosis & Improvement Track)

This folder stores real-world LoRA / model failure diagnoses and before-after improvement studies conducted with `XAI_DiT`.

## Git Tracking Policy (`.gitignore`)
- **Tracked (Public)**: `README.md` and `showcase/` (curated, privacy-safe 2D/public diagnostic studies ready for presentation).
- **Untracked (Local-Only)**: `runs/`, `HISTORY.md`, `runs_index.jsonl`, and draft reports/previews at the root of `case_studies/`. This keeps private LoRAs, uncurated test runs, and non-public datasets strictly local while preserving the exact same `YYYYMMDD_NN_<name>` + `runs_index.jsonl` (SQLite-ready) structure as `experiments/accel/`.

## Promotion Workflow
1. Run diagnosis and before/after improvement experiments in `runs/YYYYMMDD_NN_<name>/` and log them in `runs_index.jsonl` + `HISTORY.md`.
2. Once verified with clean, privacy-safe visuals, copy only the curated assets and summary into `showcase/<case_name>/` to track in Git and link from `showcase/index.html`.
