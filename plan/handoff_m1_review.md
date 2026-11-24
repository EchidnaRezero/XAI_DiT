# Session Handoff: Experiment Design Cross-Review & M1 Implementation Prep

Created: 2026-11-25
Target workspace: `C:\Projects\XAI_DiT`

---

## 1. Completed So Far
- **Official Paper Code Repos (`reference_repo/paper_code/`)**:
  - All 16 paper repos verified & shallow-cloned (`--depth 1`, `GIT_LFS_SKIP_SMUDGE=1`); exact commits, licenses, sizes, and caveats recorded in `reference_repo/reference_repo_guide.md`.
  - Note: `StableFlow` is Non-Commercial Research Only (read-only reference, never copy code); `TemplateRegisters`, `DiTRelationCircuits`, and `DiT-ST` currently have README/assets only (no source code).
- **Verified Paper Specs (`docs/specs/01_heatmap.md` .. `04_verification_mi.md`)**:
  - Extracted directly from all 16 PDFs (`.venv` `pymupdf`) + official repos + `reference_repo/ComfyUI` + `docs/reference/models/{anima,moss_image,krea2}.md`.
  - Independently cross-reviewed at `high` reasoning by **GPT 6.2 Astra** (Codex CLI) and **Claude Fable 5.2** (Claude Code CLI); all critical findings reconciled.
- **Experiment Design & Operational Specs (Korean)**:
  - `docs/experiment_design.md`: Core E0–E4 design, common control rules, Phase 1–4 execution order, routing to `experiments/design/`.
  - `experiments/design/runtime_and_hp.md`: Colab L4 (24 GB) default weights & locked HPs (`Anima` BF16 30 steps; `MOSS-Image` `sft_bf16` + `qwen3.5_9b_int8_convrot` or `int8_convrot + fp32 stream` 28 steps; `Krea 2 Raw` `krea2_raw_fp8_scaled` + `qwen3vl_4b_fp8_scaled` 32 steps `raw_dynamic` shift), `--disable-dynamic-vram`, no `FlexAttention` on L4 (use Listing 2 / chunked SDPA), non-discarded deferred options, and Phase 4 (A100/H100) batch.
  - `experiments/design/prompts.md`: Danbooru-verified 2D artist pool (`@ogipote`, `@mika pikazo`, `@ciloranko`, `@mochizuki kei`, `@yoneyama mai`) & character pool (`tsuyuri kanao`, `emilia (re:zero)`, `arona (blue archive)`, `frieren`, `hoshino ai`, `raiden shogun`), real community prompts in ` ```text ` blocks for Anima / MOSS-Image / Krea 2 Raw, plus spatial-relation & length-matched counterfactual/PCI pairs.
  - `experiments/design/output_spec.md`: Standard `experiments/<exp_id>_<model>/` layout, Dawn-style Same-State `[tags × steps]` shared-color-scale overlay grid (`grid_delta_v_steps.png`) & `grid_attn_vs_delta_v.png`, and per-experiment PNG/JSON specs.

---

## 2. Immediate Goal for Next Session (Step 1 First!)
1. **Cross-Review `docs/experiment_design.md` + `experiments/design/{runtime_and_hp,prompts,output_spec}.md`** using **GPT 6.2 Astra** (Codex CLI, `high`, `-s read-only`) and **Claude Fable 5.2** (Claude Code CLI, `high`, `--allowedTools "Read,Grep,Glob"`):
   - Focus broadly on **scientific & engineering soundness of the experiment design** (not just optimization flags):
     1. Alignment between `docs/experiment_design.md`, `experiments/design/*`, `docs/specs/01..04.md`, and `docs/architecture.md`.
     2. Soundness of E0–E4 protocols, controls, metrics, and statistical/reproducibility criteria (including Same-State $\Delta v$ vs final-$x_0$ causality boundary).
     3. Token-length alignment, CFG `cond_or_uncond` routing, prompt pool coverage, and output schema completeness.
     4. Practical L4 vs Phase 4 (A100/H100) split and any remaining gaps before M1 coding.
2. **Reconcile** reviewer findings directly into `docs/experiment_design.md` and `experiments/design/*`.

---

## 3. Next After Review Reconciliation
1. **Git Initialization**: Confirm with the user and delegate to `GPT-6.1-Sol` using the **`Echidna`** account.
2. **Begin M1 Implementation (`src/xai_dit/` + `tests/`)**:
   - Reuse/evolve proven `reference_repo/Anima_HEATMAP` designs: `compute.py` (query-chunked + OOM key-chunked `logsumexp` softmax), `aggregate.py` (streaming `RunningMap`), `profiling.py` (`CaptureProfile`), `phrases.py` (ComfyUI prompt/tag span matching).
   - Build clean `ModelAdapter` contract (`adapters/{base,anima,moss_image,krea2}.py`) + 4-direction core (`core/`) + thin ComfyUI wrapper (`comfyui/nodes.py`) + Colab CLI runner (`runners/cli.py`).

---

## 4. Hard Constraints (Do Not Violate)
- **No local code execution** except `.venv` `pymupdf` PDF reading (everything runs bundled on Colab when ready).
- **No local PyTorch install** (CPU-only machine).
- **Read-Only**: Never modify `docs/reference/` or `reference_repo/` (except `reference_repo_guide.md` if explicitly updated).
- **Secrets**: Never expose anything from `local/`.
- **Workflow preference**: Orchestrator (Gemini Neon) focuses on reading/understanding/designing; delegate mechanical prep/searches to parallel Gemini Neon subagents.
