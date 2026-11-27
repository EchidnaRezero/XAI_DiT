# XAI_DiT Workspace & Role Guide

Reference guide linked from `AGENTS.md` showing **which document or folder to read/write** by situation and role.

---

## 1. Folder & Document Layout

```text
XAI_DiT/                               # Git repository root
├── README.md                          # Visitor overview · results · docs/folder guide · credits (no agent policies)
├── AGENTS.md                          # Agent instructions · constraints · role & model assignment
├── .gitignore                         # reference_repo/* (!guide), local/* (!.env.example), paper PDFs, etc.
├── src/                               # XAI_DiT core · model adapters · ComfyUI nodes
├── tests/                             # Unit · integration · regression tests (git-tracked)
├── plan/                              # Milestone task checklists & progress status
│   └── status.md                      # Current progress & next steps
├── docs/
│   ├── workspace_guide.md             # (This file) Situational & role routing guide
│   ├── direction.md                   # Project direction · goals · scope
│   ├── architecture.md                # Architecture · model adapters · package layout
│   ├── experiment_design.md           # Planned experiment & verification design
│   ├── setup.md                       # Public installation & execution guide
│   ├── specs/                         # Verified paper method specs (formulas · paper vs custom diffs)
│   │   ├── _template.md
│   │   └── 01_heatmap.md .. 04_verification_mi.md
│   ├── reference_guide.md             # Routing index for docs/reference/
│   └── reference/                     # Read-only reference materials
│       ├── XAI_DiT_조사정리.md        # Literature survey & verification baseline
│       ├── models/                    # anima.md, krea2.md, moss_image.md
│       └── paper/                     # 16 core papers (01~04 dirs + root 1, PDFs gitignored)
├── experiments/                       # Results of planned experiments from docs/experiment_design.md
│   ├── design/                        # Operational details: runtime_and_hp.md, prompts.md, output_spec.md
│   └── _template.md
├── issues/                            # Event-driven unexpected problem investigations (RAW, portfolio basis)
│   └── _template.md
├── showcase/                          # Visual summaries (HTML · slides · portfolio)
├── reference_repo/                    # External reference repos (gitignored, read-only)
│   ├── reference_repo_guide.md        # Repo intent · commit · license · credits (git-tracked)
│   ├── Anima_HEATMAP/                 # v1 original implementation
│   ├── ComfyUI/                       # Reference model loading · inference · custom node hooks
│   └── DiffSynth-Studio/              # Pure-PyTorch architecture & low-VRAM reference
└── local/                             # Local-only env configs · API keys & secrets (gitignored)
    ├── .env.example                   # Required env var names only (git-tracked)
    └── <env-and-keys>                 # Actual .env, API keys, personal paths (never commit/expose)
```

---

## 2. Core Boundaries

### ① `README.md` vs `AGENTS.md`
| File | Audience | Contents | Never Include |
|---|---|---|---|
| **`README.md`** | Visitors · Portfolio | What this project is, results, docs/folder map, credits | Agent instructions, policies, unconfirmed plans |
| **`AGENTS.md`** | Agents | Work rules, constraints, role/model assignment, `workspace_guide.md` link | Marketing prose or long result showcases |

### ② `docs/` vs `plan/`
- **`docs/`**: Direction (`direction.md`), architecture (`architecture.md`), experiment design (`experiment_design.md`), public setup (`setup.md`), and **project background/reference docs** (`reference_guide.md` + `reference/` — target model internals, 16 core papers, survey).
- **`plan/`**: Milestone-level **task checklists and progress (`status.md`)**. Architecture and experiment design belong in `docs/`.

### ③ Recurring/Systemized (`tests/`, `experiments/`) vs Event-Driven (`issues/`)
Anything planned upfront or promoted into ongoing/recurring checks goes into **`tests/`** or **`experiments/`**. Unexpected, event-driven problems and their root-cause investigations go into **`issues/`** (promote recurring regression checks to `tests/` after resolving).

| Folder | Criterion | Nature |
|---|---|---|
| **`tests/`** | Verification code run repeatedly | Unit · integration · regression tests (git-tracked) |
| **`experiments/`** | Planned (`docs/experiment_design.md`) or promoted experiments | Objective run data (config · metrics · output images) |
| **`issues/`** | Unexpected event-driven problem investigations | RAW portfolio stories (`Problem → Observation → Root Cause → Fix → Verification`; link images, don't duplicate) |

### ④ `docs/setup.md` vs `local/`
- **`docs/setup.md`**: Public installation & execution guide for humans/agents (git-tracked).
- **`local/`**: Machine-specific values, API keys, secrets loaded by code (gitignored; only `local/.env.example` tracked).

---

## 3. Situational Routing

| Situation / Question | Open First | Then Check |
|---|---|---|
| Project overview & top-level map | `README.md` | `showcase/` |
| Rules, constraints, role/model assignment | `AGENTS.md` | `docs/workspace_guide.md` |
| Current progress & next milestone | `plan/status.md` | `plan/active/`, `plan/interrupts/`, `plan/roadmap.md`, `plan/archive/` |
| Project tracking, auto-numbering & 3-Scribe automation | `docs/project_management.md` | `scripts/tracker/`, `.agents/skills/scribe-*/` |
| Project direction & scope | `docs/direction.md` | — |
| Architecture & package layout | `docs/architecture.md` | `src/`, `tests/` |
| Exact paper formulas & custom diffs per module | `docs/specs/` | `docs/reference_guide.md` → `docs/reference/paper/` |
| Planned experiments & verification criteria | `docs/experiment_design.md` | `experiments/_template.md` → `experiments/` |
| Practical inference/XAI acceleration presets & links | `docs/acceleration.md` | `experiments/accel/REPORT.md`, `RESOURCES.md`, `HISTORY.md` |
| Real-world LoRA/model XAI diagnosis & improvements | `experiments/case_studies/README.md` | `experiments/case_studies/showcase/` (git-tracked) · `runs/` (local-only) |
| Public setup & execution steps | `docs/setup.md` | `local/.env.example` |
| Target model internals (Anima/Krea 2/MOSS) or 16 papers | `docs/reference_guide.md` | `docs/reference/models/`, `docs/reference/paper/`, `XAI_DiT_조사정리.md` |
| External repo code, licenses, credits | `reference_repo/reference_repo_guide.md` | `reference_repo/<repo>/` |
| Unexpected bugs or anomalies | `issues/_template.md` | `issues/<issue-id>.md` |
| Visual summaries & portfolio views | `showcase/` | `experiments/`, `issues/` (evidence links) |
| Local env vars / API keys | `local/.env.example` | `local/` (never print or copy real values) |

---

## 4. Roles Overview

```text
User ↔ Orchestrator ──┬──► Builder   (code · tests · planned runs)
                      ├──► Reviewer  (cross-review · debugging / different model)
                      ├──► Scribe    (RAW issues · summaries · guides · visuals)
                      └──► Git       (commit · branch · push / Echidna account)
```

- Exact model assignments, reasoning effort, and CLI tools are defined in `AGENTS.md`.
- Orchestrator passes only the required file paths per Section 3 when delegating.
- `Reviewer` diagnoses from code/docs evidence (no guessing); `Scribe` never fabricates metrics outside RAW evidence.

---

## 5. Common Protection Rules

- **Read-Only**: Never modify `docs/reference/` or `reference_repo/`.
- **Secrets (`local/`)**: `local/*` is gitignored (`!local/.env.example` only). Never expose real API keys, tokens, or personal paths anywhere.
- **Ask First**: Confirm before installing packages, downloading external files (trusted paper sources like arXiv are fine), deleting files, or running heavy jobs—especially anything **≥ 1 GB** or straining **RAM/disk (HDD/SSD)**.
