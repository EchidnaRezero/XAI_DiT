---
name: project-tracker
description: Orchestrator skill for running XAI_DiT's tracker CLI (exp, issue, plan, sync, to-sqlite) and dispatching the 3 disjoint ephemeral Scribes (scribe-exp, scribe-issue, scribe-plan). Use when creating runs/issues/interrupts, running sync checks, exporting SQLite, or wrapping up a task.
---

# Project Tracker & 3-Scribe Dispatch Skill

Use `.venv\Scripts\python.exe` (Windows) or `python` (Linux/Colab) with `-m scripts.tracker.cli`.

## 1. CLI Command Reference

```powershell
# Experiments (auto-numbers YYYYMMDD_NN_<slug> + updates runs_index.jsonl & HISTORY.md)
.venv\Scripts\python.exe -m scripts.tracker.cli exp new <accel|case_studies> <slug> --type t2i_bench --models anima --gpu L4 --tags tag1,tag2 --summary "..."
.venv\Scripts\python.exe -m scripts.tracker.cli exp render <accel|case_studies>
.venv\Scripts\python.exe -m scripts.tracker.cli exp check <accel|case_studies>
.venv\Scripts\python.exe -m scripts.tracker.cli exp to-sqlite <accel|case_studies>

# Issues (auto-numbers issues/NNN_<slug>.md + rebuilds issues_index.jsonl)
.venv\Scripts\python.exe -m scripts.tracker.cli issue new <slug> --title "Title"
.venv\Scripts\python.exe -m scripts.tracker.cli issue index
.venv\Scripts\python.exe -m scripts.tracker.cli issue check

# Plans & Interrupts (4-bucket plan/ + broken links + *_KR.md sync check)
.venv\Scripts\python.exe -m scripts.tracker.cli plan new-active <slug> --title "Title"
.venv\Scripts\python.exe -m scripts.tracker.cli plan new-interrupt <slug> --title "Title"
.venv\Scripts\python.exe -m scripts.tracker.cli plan check

# Full Sync Gate (all checks + optional table regeneration)
.venv\Scripts\python.exe -m scripts.tracker.cli sync --render --check
```

## 2. 3-Scribe Dispatch Matrix (Disjoint Ownership — Parallel Safe)

Before finishing any task that creates experiments, fixes bugs, or modifies structure/docs (skip on pure Q&A), invoke only the relevant ephemeral `self` subagent(s) (`Gemini Neon`):

| Trigger Event | Subagent(s) to Invoke | Skill SOP | Exclusive Ownership |
|---|---|---|---|
| New experiment / benchmark / diagnostic run | `Exp-Scribe` | `.agents/skills/scribe-exp/SKILL.md` | `experiments/accel/`, `experiments/case_studies/`, `docs/acceleration.md` |
| Bug / collapse diagnosed or fixed | `Issue-Scribe` | `.agents/skills/scribe-issue/SKILL.md` | `issues/` |
| Milestone done, interrupt opened/closed, layout or README changed | `Plan-Scribe` | `.agents/skills/scribe-plan/SKILL.md` | `plan/`, `README.md`, `README_KR.md` |

After Scribes finish, confirm `sync --check` exits with `0 error(s)`.
