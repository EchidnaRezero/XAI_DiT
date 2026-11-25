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
- [x] Completed **All 4 Remaining Core Before/After Intervention Experiments** on L4 across Anima, MOSS-Image (`int8_convrot`), and Krea 2 Raw (`fp8_scaled`):
  - **E2-2 (`Cont^(l)` + Top-20% vs Bottom-20%)**: `experiments/e2_2_{anima_bf16,moss_image_int8_convrot,krea2_fp8}_l4_01/`
  - **E3-3 (TACA $\sigma \ge 0.97, \gamma_0=1.2$)**: `experiments/e3_3_{anima_bf16,moss_image_int8_convrot,krea2_fp8}_l4_01/`
  - **E3-1 (K/V Patching Red→Blue Scarf Early/Mid/Late)**: `experiments/e3_1_{anima_bf16,moss_image_int8_convrot,krea2_fp8}_l4_01/`
  - **E4-3 (Krea 2 Early-Block `[0..5]` Gate Ablation `gate<-1.0` & `gate<-mean`)**: `experiments/e4_3_ablation_krea2_fp8_l4_01/`
- [x] Terminated Colab L4 session `xai-l4` per user instruction (`No active sessions found on server`)
- [x] Built user-facing `showcase/` interactive HTML dashboard (`showcase/index.html`), portfolio summary (`showcase/README.md`), and updated visitor-facing `README.md`
- [x] Fixed 3-model heatmap aggregation & display scaling (`ISSUE-001`: `.mean()`, `per_tag`, `turbo`, `p99` clip + display-only pure-NumPy Gaussian `σ=1.5`; `17/17` CPU tests passed; `E3-0` Δv grids kept shared-scale)
- [x] Completed **Step 0 Cross-Model Recognition Pre-Check** on L4 (`experiments/step0_precheck_l4_01/`, 24 PNGs with 3-chunk `PngInfo`): confirmed `@artist` Anima-only, MOSS 2D chars `6/6`, Krea 2 plain `3/6` (`emilia`, `arona`, `raiden shogun`) + `cosplay` `5/6`, `--fast`+`--use-ck-attention` graininess caveat (OFF for XAI)
- [x] Locked **Step 1 Unified Anima 5-Prompt Set** (`experiments/design/unified_anima_set.json`) and completed **Step 2 Anima 7-Stage Suite** + clean `P-A3` (`experiments/step2_anima_l4_01/`)
- [x] Completed **Anima Prop-Transplant Control Test** (`experiments/e2_prop_transplant_anima_l4_01/`): causal `Mid [9..18]` per-block `kv_patch` cleanly transplanted `sketchbook → glowing paper lantern` (`+0.483`), whereas raw `Cont^(l)` Top-6 failed (`-0.128`)
- [x] Completed **3-Model Identical Prompt Heatmap Comparison** (`Emilia + anime coloring`, `experiments/cross_model_emilia_anime_coloring_l4_01/`): verified `anime coloring` as a genuine 3-model shared style tag across Anima, MOSS-Image, and Krea 2 Raw
- [x] Built, visually refined, and promoted **v3 4-Tab Visual Showcase** (`showcase/index.html`: CAM vs DiT structural comparison, 3-model 3-tier 3D architecture diagrams, interactive 5-chapter Canvas code-pipeline animation, results & control) alongside `showcase/preview_anima_step2.html`; archived v1/v2 pages into `showcase/archive/` and routed `README.md` & `showcase/README.md`
- [x] Audited `docs/` vs actual implementation/experiments (`issues/002_docs_vs_implementation_drift.md`, 12 drift items) and drafted **v2 Practical Control Replan** (`plan/v2_practical_control_replan.md`)

## Next Up
- [ ] Execute **v2 Practical Semantic-Unit Control Replan** (see `plan/v2_practical_control_replan.md`):
  - Shift core experiments from destructive collapse ablations to non-destructive semantic-unit manipulation (StableFlow Stage 2, `Mid` `kv_patch` prompt routing, `S`-only TACA & additive per-tag log-bias, early-$\sigma$ SAP, head/layer-level attribute swap)
  - Replace toy RGB proxies with offline GPU semantic evaluators (OpenCLIP/SigLIP, DINOv2, LPIPS, Grounded-SAM2/VQA)
  - Roll out to MOSS-Image & Krea 2 Raw on the verified shared prompt pool (`emilia`, `arona`, `raiden shogun`, `anime coloring`) + `E1-2` ConceptAttention
  - Validate & adapt v2 XAI/control methods under real-world ComfyUI optimization stacks (`docs/acceleration.md`, `plan/accel_xai_eval.md`: Turbo/eDMD LoRA, EasyCache/Spectrum, `--use-ck-attention`/`--fast`, `TorchCompileModel`) and package into practical ComfyUI nodes
- [ ] Reconcile `issues/002_docs_vs_implementation_drift.md` & `plan/v2_practical_control_replan.md` into `docs/` and `experiments/design/` upon user approval
- [ ] Commit & push updated results, `plan/`, `issues/`, and `showcase/` via `GPT-6.1-Sol` (`Echidna` account) upon user approval
