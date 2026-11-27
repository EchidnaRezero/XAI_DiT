# AGENTS.md

Shared instructions for all agents working in `XAI_DiT`.
For situational file/folder routing and role details, refer to [`docs/workspace_guide.md`](docs/workspace_guide.md).
For practical inference/XAI acceleration presets and guardrails, refer to [`docs/acceleration.md`](docs/acceleration.md).
For project tracking (`plan/` 4-bucket layout, `scripts/tracker/`, 3-Scribe automation), refer to [`docs/project_management.md`](docs/project_management.md).

---

## 1. Core Principles

1. **Evidence First**: Never answer from memory. Verify against actual code, `docs/reference/`, `reference_repo/`, or `experiments/`.
2. **Ask Before Side Effects**: Confirm with the user before installing packages, downloading external files (papers from trusted sources like arXiv are fine), deleting files, or running heavy jobs. In particular, always disclose expected size/load and get approval before handling **packages/files ≥ 1 GB** or tasks that strain **RAM or disk (HDD/SSD)**.
3. **Never Silently Alter Agreed Specs**: Never unilaterally change user-approved experiment designs, model/quantization selections, hyperparameters, or protocols (including when applying reviewer feedback). Always explain any proposed change and get explicit user approval before modifying them.
4. **Read-Only Areas**: `docs/reference/` and `reference_repo/` are reference materials—do not modify them.
5. **Secrets & Private Data (`local/`, `experiments/case_studies/`)**: Never copy or print real API keys, tokens, personal paths from `local/`, or non-public/gitignored private case studies (`experiments/case_studies/` outside `showcase/`) into public docs, commits, `showcase/`, or responses.
6. **Document Boundaries**:
   - `README.md` is strictly for visitors (overview, results, document/folder map). Do not put agent instructions, policies, or unconfirmed plans in it.
   - Recurring/systemized assets go in `tests/` and `experiments/`; unexpected, event-driven problem investigations (RAW) go in `issues/`.
7. **Language & Style**:
   - Write project documentation in **English** by default, except `showcase/` and items explicitly specified otherwise by the user.
   - **Bilingual Sync (`*_KR.md`)**: Whenever a `<name>_KR.md` companion file exists (e.g., `README.md` ↔ `README_KR.md`), always keep the English and Korean versions synchronized with identical structure, facts, numbers, and links.
   - **Tone & Audience (Docs & `showcase/`)**: Keep writing **dry, concise, and fact-first** around clear key points—never use hype, dramatic exaggeration, or empty emphasis. Assume an **introductory DL audience** (basic Diffusion / Flow Matching / Transformer level) using standard AI terminology for technical details, while framing the overall purpose, why it was needed, and what was done so even a **non-specialist** can understand the big picture.
   - Keep responses concise, plain, and structured with tables and short bullets rather than long prose.

---

## 2. XAI_DiT Constraints

- **Target Models**: Anima (cross-attn DiT), Krea 2 (base/raw + quantized, distilled, and fine-tuned derivatives such as Turbo and `Krea2-Anime`), MOSS-Image (single-stream joint-attn DiT).
- **Local Environment**: Local machine is CPU-only (no GPU). **Do NOT install PyTorch (`torch`, `torchvision`, `torchaudio`, etc.) locally.** Write GPU inference/experiment scripts for external GPU environments (e.g., Colab L4+).
- **Colab Session Lifecycle**: Always get user approval before starting a paid Colab GPU session. Once started, **keep the session alive** (do NOT auto-stop with `colab stop` or release after a single script) so downloaded weights/environments remain cached; **only terminate a Colab session when the user explicitly asks to stop it**.
- **PDF Reading**: Extract text from PDFs using the project `.venv` (`pymupdf` / `fitz`) or `pdftotext` rather than opening raw PDFs directly.

---

## 3. Agent Roles & Model Assignment

| Role | Responsibility | Model (Priority) |
|---|---|---|
| **Orchestrator** | Talk with user, delegate/review/re-assign work, handle ad-hoc user requests | `Gemini Neon` |
| **Reviewer** | Cross-review design/code/experiments, debugging & root-cause diagnosis | Review: `GPT 6.2 Astra` (primary) or `Claude Fable 5.2`<br>Debugging: `Claude Opus 5.5` (primary); fallback `GPT 6.2 Astra` / `Claude Fable 5.2` |
| **Builder** | Implement `src/` & `tests/`, run planned experiments (`experiments/`) | `Claude Opus 5.5` (primary) or `Gemini Neon` / `GPT 6.2 Astra` |
| **Scribe** (`Exp` / `Issue` / `Plan`) | Disjoint ephemeral subagents (`.agents/skills/scribe-{exp,issue,plan}/`): `Exp-Scribe` (`experiments/`), `Issue-Scribe` (`issues/`), `Plan-Scribe` (`plan/`, `README*`) | `Gemini Neon` (all) |
| **Git** | All Git operations (commit, branch, push, etc.) — verify `tracker sync --check` before committing | `GPT-6.1-Sol` |

- **Reasoning Effort**: Default `medium`; **`high` for Reviewer only**.
- **CLI Execution**: Gemini → **Antigravity CLI**, GPT → **Codex CLI**, Claude → **Claude Code CLI**.
- **Git Account**: Perform all Git operations using the **`Echidna`** account.
- **Auto-Numbering & Sync Gate (`scripts/tracker/`)**: Always create new runs/issues/interrupts via `python -m scripts.tracker.cli <exp|issue|plan> new*` (never invent ad-hoc folder names). Before finishing any task that adds experiments, fixes bugs, or changes structure/docs (skip on pure Q&A), automatically invoke the relevant ephemeral `Scribe` subagent(s) (`Exp-Scribe` / `Issue-Scribe` / `Plan-Scribe`) and pass `python -m scripts.tracker.cli sync --check`.
- **Context Handoff**: Orchestrator passes only the required file paths (per `docs/workspace_guide.md`) when delegating.
- **Tool-Specific Wrappers**: Add thin tool-specific files only when needed (e.g., Claude Code `.claude/agents/*.md`, `CLAUDE.md`).
