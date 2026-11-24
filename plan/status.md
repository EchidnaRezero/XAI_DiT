# Project Status

Last updated: 2026-11-25

---

## Completed
- [x] Architecture & hook-point analysis of 3 target models: Anima, MOSS-Image, Krea 2 (`docs/reference/models/`)
- [x] 2025–2026 DiT XAI/MI literature survey & 16 core papers organized into 4 directions (`docs/reference/paper/`, `XAI_DiT_조사정리.md`)
- [x] Official code repositories for all 16 papers verified & shallow-cloned into `reference_repo/paper_code/` (`reference_repo/reference_repo_guide.md`)
- [x] Workspace structure, guides, templates, `.gitignore`, and `AGENTS.md` setup
- [x] `docs/direction.md` and `docs/architecture.md` drafted (Korean)
- [x] Local `.venv` (`pymupdf` 2.1, no PyTorch) and `docs/specs/` slots prepared
- [x] Extracted verified paper specs (`docs/specs/01_heatmap.md` .. `04_verification_mi.md`) and completed independent `high`-reasoning cross-review by **GPT 6.2 Astra** & **Claude Fable 5.2** (all critical findings reconciled)
- [x] `docs/experiment_design.md` and `experiments/design/{runtime_and_hp,prompts,output_spec}.md` drafted (Korean)
- [x] Completed independent `high`-reasoning cross-review of experiment design 4 docs by **GPT 6.2 Astra** & **Claude Fable 5.2** and reconciled all Critical/Major/Minor findings into `docs/experiment_design.md`, `experiments/design/*`, `docs/architecture.md`, `docs/specs/04_verification_mi.md`, `experiments/_template.md`, and `plan/m1.md`
- [x] Initialized Git repository (`main`) and pushed initial commit (`8cea79b`) to `EchidnaRezero/XAI_DiT` via `GPT-6.1-Sol` (`Echidna` account)
- [x] Implemented M1 core (`src/xai_dit/core/`), 3 model adapters (`anima`, `moss_image`, `krea2`), Headless ComfyUI CLI runner (`runners/cli.py`), thin ComfyUI nodes (`comfyui/nodes.py`), 4 frozen prompt manifests (`experiments/design/manifest_*.json`), and 8 CPU unit test suites (`tests/`) — pushed (`fd210e8`, `888fd7c`) via `GPT-6.1-Sol` (`Echidna`)
- [x] Passed all M1 CPU unit tests on **free Colab CPU runtime** (`Python 3.13.18`, `Torch 2.14.1+cpu`, `CUDA: False`): **`210 passed in 6.15s`** (0 failed, 0 skipped, `XAI_DIT_REQUIRE_MANIFESTS=1`)
- [x] Completed **Phase 1 Anima E0 Sanity & Resource Budget** on Colab NVIDIA L4 (`experiments/e0_anima_bf16_l4_01/`): 3-rep floor `0.0`, exact no-op `0.0` (PASS), peak VRAM `4.70 GB` (+0.04 GB manual chunked SDPA), `0.363 s/fwd` baseline / `0.558 s/fwd` hooked; L4 session terminated immediately after download

## Next Up
- [ ] Commit `cli.py` namespace fix + `experiments/e0_anima_bf16_l4_01/` (`GPT-6.1-Sol`, `Echidna` account) and refine E0 per-block vs end-to-end `x0` identity reporting
- [ ] Request user approval before next paid Colab L4 session (MOSS-Image E0 / Krea 2 Raw E0 or Anima Phase 1 E1–E4)
