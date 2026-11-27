# Project Management & Automated Tracking Architecture

Lightweight, multi-agent-compatible tracking and sync system for `XAI_DiT` (zero external dependencies; works identically across Antigravity, Claude Code, and Codex CLI).

---

## 1. Core Design: 4-Bucket `plan/` + Immutable `runs/` + 3-Module Tracker

### A. 4-Bucket `plan/` Taxonomy
| Bucket | Path | Purpose | Lifecycle |
|---|---|---|---|
| **Status Board** | `plan/status.md` | Single glance at active work, current interrupts, and recent completions | Updated by `Plan-Scribe` after milestones |
| **Roadmap (Future)** | `plan/roadmap.md` | High-level future directions & hardware-gated backlog (no rigid spec yet) | Promote to `active/` when ready |
| **Active (In Progress)** | `plan/active/` | Concrete in-progress execution plans (`PLAN-NNN_<slug>.md`) | Summarize in `archive/` when done |
| **Interrupts (Detours)** | `plan/interrupts/` | Unplanned, event-driven side-quests (`YYYYMMDD_<slug>.md`) that temporarily pause main work | Archive or promote to `active/` when resolved |
| **Archive (Completed)** | `plan/archive/` | Chronological milestone record (`completed_milestones.md`) | Append-only history |

### B. Chronological Experiment & Case-Study Tracks
- **`experiments/accel/`** (public, git-tracked): `REPORT.md` (consolidated technical report), `RESOURCES.md` (weights/nodes catalog), `HISTORY.md` (auto-generated), `runs_index.jsonl` (SQLite-ready metadata), and immutable `runs/YYYYMMDD_NN_<slug>/` folders.
- **`experiments/case_studies/`** (local-only except `README.md` & `showcase/`): Identical `runs/YYYYMMDD_NN_<slug>/` + `runs_index.jsonl` + `HISTORY.md` schema. Only curated, privacy-safe studies promoted to `showcase/` are tracked in Git.
- **`issues/`**: Auto-numbered `NNN_<slug>.md` root-cause investigations indexed in `issues/issues_index.jsonl`.

### C. 3-Module Stdlib CLI (`scripts/tracker/`)
Pure Python standard library (`json`, `sqlite3`, `pathlib`, `argparse`), split cleanly by domain:
- **`scripts/tracker/exp.py`**: Auto-number `YYYYMMDD_NN_<slug>`, append `runs_index.jsonl`, render `HISTORY.md`, detect orphan folders, export `registry.sqlite`.
- **`scripts/tracker/issue.py`**: Auto-number `issues/NNN_<slug>.md`, rebuild `issues_index.jsonl`, check numbering.
- **`scripts/tracker/plan.py`**: Create `plan/active/PLAN-NNN_*` or `plan/interrupts/YYYYMMDD_*`, verify relative markdown links across docs/plans/reports, and verify `*_KR.md` bilingual parity.
- **`scripts/tracker/cli.py`**: Unified entry point (exact CLI commands & dispatch SOP live in [`.agents/skills/project-tracker/SKILL.md`](../.agents/skills/project-tracker/SKILL.md)).

---

## 2. Disjoint 3-Scribe Ephemeral Subagent Automation

To prevent file collisions and eliminate manual logging reminders, the `Scribe` role (`Gemini Neon`) is split into **three ephemeral subagents with strictly disjoint directory ownership** (`Exp-Scribe` → `experiments/`, `Issue-Scribe` → `issues/`, `Plan-Scribe` → `plan/` & `README*`). Because ownership never overlaps, multiple Scribes can run in parallel and exit immediately upon completion. Exact SOPs and trigger rules are maintained in [`.agents/skills/project-tracker/SKILL.md`](../.agents/skills/project-tracker/SKILL.md) and `.agents/skills/scribe-{exp,issue,plan}/SKILL.md`.

---

## 3. Inspiration & Credits
- The fixed-ID auto-numbering, **Plan → Run → Sync** completion gate, and JSONL-backed automated verification pattern were inspired by [**MoAI-ADK** (`modu-ai/moai-adk`)](https://github.com/modu-ai/moai-adk), adapted into a zero-dependency, multi-CLI Python architecture tailored for ML/XAI experimentation.
