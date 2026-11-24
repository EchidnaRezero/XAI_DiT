# AGENTS.md

Shared instructions for all agents working in `XAI_DiT`.
For situational file/folder routing and role details, refer to [`docs/workspace_guide.md`](docs/workspace_guide.md).

---

## 1. Core Principles

1. **Evidence First**: Never answer from memory. Verify against actual code, `docs/reference/`, `reference_repo/`, or `experiments/`.
2. **Ask Before Side Effects**: Confirm with the user before installing packages, downloading external files (papers from trusted sources like arXiv are fine), deleting files, or running heavy jobs. In particular, always disclose expected size/load and get approval before handling **packages/files ≥ 1 GB** or tasks that strain **RAM or disk (HDD/SSD)**.
3. **Read-Only Areas**: `docs/reference/` and `reference_repo/` are reference materials—do not modify them.
4. **Secrets (`local/`)**: Never copy or print real API keys, tokens, or personal paths from `local/` into docs, commits, `showcase/`, or responses.
5. **Document Boundaries**:
   - `README.md` is strictly for visitors (overview, results, document/folder map). Do not put agent instructions, policies, or unconfirmed plans in it.
   - Recurring/systemized assets go in `tests/` and `experiments/`; unexpected, event-driven problem investigations (RAW) go in `issues/`.
6. **Language & Style**:
   - Write project documentation in **English** by default, except `showcase/` and items explicitly specified otherwise by the user.
   - Keep responses concise, plain, and structured with tables and short bullets rather than long prose.

---

## 2. XAI_DiT Constraints

- **Target Models**: Anima (cross-attn DiT), Krea 2 (**base model only**, exclude anime variant), MOSS-Image (single-stream joint-attn DiT).
- **Local Environment**: Local machine is CPU-only (no GPU). **Do NOT install PyTorch (`torch`, `torchvision`, `torchaudio`, etc.) locally.** Write GPU inference/experiment scripts for external GPU environments (e.g., Colab L4+).
- **PDF Reading**: Extract text from PDFs using the project `.venv` (`pymupdf` / `fitz`) or `pdftotext` rather than opening raw PDFs directly.

---

## 3. Agent Roles & Model Assignment

| Role | Responsibility | Model (Priority) |
|---|---|---|
| **Orchestrator** | Talk with user, delegate/review/re-assign work, handle ad-hoc user requests | `Gemini Neon` |
| **Reviewer** | Cross-review design/code/experiments, debugging & root-cause diagnosis | Review: `GPT 6.2 Astra` (primary) or `Claude Fable 5.2`<br>Debugging: `Claude Opus 5.5` (primary); fallback `GPT 6.2 Astra` / `Claude Fable 5.2` |
| **Builder** | Implement `src/` & `tests/`, run planned experiments (`experiments/`) | `Claude Opus 5.5` (primary) or `Gemini Neon` / `GPT 6.2 Astra` |
| **Scribe** | Maintain `issues/` RAW logs, `experiments/` summaries, `plan/status.md`, `README.md`, `*_guide.md`, and `showcase/` visuals | `Gemini Neon` (all) |
| **Git** | All Git operations (commit, branch, push, etc.) | `GPT-6.1-Sol` |

- **Reasoning Effort**: Default `medium`; **`high` for Reviewer only**.
- **CLI Execution**: Gemini → **Antigravity CLI**, GPT → **Codex CLI**, Claude → **Claude Code CLI**.
- **Git Account**: Perform all Git operations using the **`Echidna`** account.
- **Context Handoff**: Orchestrator passes only the required file paths (per `docs/workspace_guide.md`) when delegating.
- **Tool-Specific Wrappers**: Add thin tool-specific files only when needed (e.g., Claude Code `.claude/agents/*.md`, `CLAUDE.md`).
