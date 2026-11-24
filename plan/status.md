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
- [x] Completed **E0 Sanity & Resource Budget for ALL 3 target models** on Colab NVIDIA L4 (24 GB) with quantized L4 defaults:
  - **Anima BF16** (`experiments/e0_anima_bf16_l4_01/`): floor `0.0`, exact no-op `0.0` (PASS), `0.363 s/fwd`
  - **MOSS-Image SFT `int8_convrot`** (`experiments/e0_moss_image_int8_convrot_l4_quant_01/`): floor `0.0`, exact no-op `0.0` (PASS), `0.539 s/fwd` (~30s/img)
  - **Krea 2 Raw `fp8_scaled`** (`experiments/e0_krea2_fp8_l4_quant_01/`): floor `0.0`, exact no-op `0.0` (PASS), `2.117 s/fwd`
- [x] Completed **3-Model E1-1 + E1-3 + E4-2 + E4-3 Profile Suite** on L4 (all steps, quantized defaults):
  - Heatmaps `[4 tags × 6 steps]` for Anima (`P-A1`), MOSS-Image (`P-M1`, 2m 29s), Krea 2 Raw (`P-K1`, 5m 14s)
  - Cross-model Template Share (`experiments/cross_model/e4_2/e4_2_template_share.png`): Anima `8.3%` vs Krea 2 `46.1%` vs MOSS-Image `92.3%`
  - Cross-model M-Act & Krea 2 G1 Gate (`experiments/cross_model/e4_3/e4_3_mact_compare.png`, `e4_3_krea2_gate.png`): Krea 2 Block 0 gate 96% closed (`mean ≈ 0.03`), late spike `~1,920`
- [x] Completed **3-Model E3-0 Same-State $\Delta v$ (`direct_effect` Logit Knockout + `random_ctrl` + `grid_attn_vs_delta_v.png`)** on L4 (`experiments/e3_0_{anima_bf16,moss_image_int8_convrot,krea2_fp8}_l4_01/`)
- [x] Simplified `docs/experiment_design.md` & `experiments/design/runtime_and_hp.md` from paper-scale multi-hundred-image sweeps to practical personal-project L4 scale (1–2 representative prompts × 1 seed)
- Persistent Colab L4 session `xai-l4` kept alive (per `AGENTS.md` Colab Session Lifecycle rule)

## Next Up
- [ ] Run practical Before/After intervention demos (`E2-2`, `E3-1`, `E3-3`, `E4-3` gate ablation) on `xai-l4` and build user-facing `showcase/` visual dashboard
