# Pending Design-Doc Updates (2026-11-25 Session Changelog)

Staging checklist of all user-approved spec changes, operational adjustments, and empirical findings from the **2026-11-25** execution session, prepared for structured propagation into `docs/` and `experiments/design/`.

> **Scope**: Staging record only. Frozen manifests (`experiments/design/manifest_*.json`, `experiments/design/unified_anima_set.json`), `docs/`, `src/`, `docs/reference/`, and `reference_repo/` are untouched.

---

## 1. Heatmap Aggregation & Visual Scaling Spec Changes (User-Approved)

Implemented in `src/xai_dit/runners/e1_e4_profile.py` (L54–209, L320–400, L518–521, L652–658) and `src/xai_dit/runners/e3_same_state.py` (L93–211, L420–424). Raw `.npz` maps and numerical metrics remain **unsmoothed**; smoothing and percentile clipping are display-only (`display_only: true`).

| Parameter | Previous Spec | Approved New Spec | CLI Flag & Default | Implementation Evidence |
|---|---|---|---|---|
| **Token-span aggregation** | `.sum(dim=-1)` (scales with token count) | `.mean(dim=-1)` (exact across OOM key chunks via `span_agg_divisor` / `span_div`) | `--span-agg mean\|sum` (default: `mean`) | `e1_e4_profile.py` L56–59, L71–81, L519, L657 |
| **Color normalization** | Shared `0 .. 99.5%` (`vmax`) across tags/panels | Per-tag relative `min .. p<clip_pct>` per map | `--heatmap-norm per_tag\|shared` (default: `per_tag`) | `e1_e4_profile.py` L57–60, L91–107, L189–191 |
| **Colormap** | `inferno` | `turbo` (matches `Anima_HEATMAP` v1) | `--heatmap-cmap turbo\|inferno` (default: `turbo`) | `e1_e4_profile.py` L58–61, L192–193 |
| **Upper-percentile clip** | None (exact max in relative mode) | `p99` upper-percentile clip (`100` = exact `min..max`; ignored in `shared`) | `--heatmap-clip-pct 99.0` | `e1_e4_profile.py` L67, L101–104, L197–199 |
| **Display smoothing** | None | Pure-NumPy separable 2D Gaussian blur (`GAUSSIAN_TRUNCATE=4.0`, `mode="symmetric"`) on native `64×64` patch grid before bilinear upsampling & normalization (`prepare_heatmap`) | `--heatmap-smooth-sigma 1.5` (`0` disables) | `e1_e4_profile.py` L66–68, L110–160, L194–196 |

- **Empirical Rationale**: Across all 5 Anima prompts (`P-A1`–`P-A4`, `P-A-SHORT`), `@artist` tags exhibit isolated corner/border sink spikes with `max/p99 = 3.91x–6.09x`, which crushed the dynamic range of the rest of the heatmap under unclipped `min..max`. Applying `σ = 1.5` patch-grid smoothing + `p99` clipping recovers readable spatial structure without altering saved raw arrays.
- **Control Integrity Exception (`E3-0`)**: `grid_direct_effect_steps.png` and `grid_direct_effect_dsigma_steps.png` (`e3_same_state.py` L93–149, L420–424) **always keep the unsmoothed shared `0..99.5%` scale** (`dv_grids_norm: "shared"`) while using `turbo`, because `random_ctrl` is a magnitude control and per-row relative scaling would visually inflate noise to look like signal. `grid_attn_vs_delta_v.png` uses the full display pipeline (`--heatmap-norm`, `--heatmap-cmap`, `--heatmap-smooth-sigma`, `--heatmap-clip-pct`).
- **Target Doc(s) to Update**:
  - `experiments/design/output_spec.md`
  - `docs/specs/01_heatmap.md`
  - `docs/experiment_design.md`

---

## 2. Step 0 Cross-Model Recognition & Acceleration Findings

Evidence from `src/xai_dit/runners/step0_precheck.py`, `experiments/step0_precheck_l4_01/summary.json`, and `experiments/step0_precheck_l4_01/summary_krea2_plain.json` (NVIDIA L4, `1024×1024`, seed `42`, `0` NaN/black across all 24 generations).

### 2.1 Character & `@artist` Recognition Matrix

| Model & Weights | Suite | Result | Exact Evidence |
|---|---|---|---|
| **MOSS-Image** (`moss_image_sft_int8_convrot.safetensors`, TE `qwen3.5_9b_int8_convrot.safetensors`, 28 steps, cfg 3.0, shift 5.0, `euler`/`sgm_uniform`) | 2D Characters (6) | **6/6 PASS** | `tsuyuri kanao`, `hoshino ai`, `frieren`, `emilia (re:zero)`, `arona (blue archive)`, `raiden shogun` |
| **MOSS-Image** | `@artist` shift on `hoshino ai` (3) | **0/3 FAIL** | Pixel MAE / RMSE vs `no @artist`: `@mika pikazo` `0.0778` / `0.1426`, `@ogipote` `0.0479` / `0.0940`, `@yoneyama mai` `0.0412` / `0.0820` → **`@artist` confirmed Anima-only** |
| **Krea 2 Raw** (`krea2_raw_fp8_scaled.safetensors`, TE `qwen3vl_4b_fp8_scaled.safetensors`, 32 steps, cfg 4.0, shift 1.15, `euler`/`simple`, neg `""`) | `cosplay` phrasing (6) | **5/6 PASS** | `tsuyuri kanao`, `frieren`, `emilia`, `arona`, `raiden shogun` PASS; `hoshino ai` FAIL |
| **Krea 2 Raw** (`krea2_plain`) | Plain character, no `cosplay` (6) | **3/6 clean PASS + 1 partial** | Clean PASS: `emilia`, `arona`, `raiden shogun`; Partial: `tsuyuri kanao` (2-person duplication); FAIL: `hoshino ai`, `frieren` |
| **Krea 2 Raw** | `2d anime illustration` (2) | **0/2 FAIL** | `frieren`, `hoshino ai` FAIL |

- **Cross-Model Shared 2D Character Pool**:
  - **Core 3 (all 3 models, even without `cosplay` on Krea 2)**: `emilia`, `arona`, `raiden shogun`.
  - **Extended +2 (all 3 models when using `cosplay` phrasing on Krea 2)**: `tsuyuri kanao`, `frieren`.
  - **Anima + MOSS-Image only**: `hoshino ai` (exclude from Krea 2 shared pool).

### 2.2 Acceleration & Metadata Conventions

- **Step 0 Speedups (`--fast`, `--use-ck-attention`, `EasyCache` thresh `0.20`, start `0.15`, end `0.95`; `--disable-dynamic-vram --highvram`; SageAttention strictly OFF)**:
  - **MOSS-Image**: steady-state `~16.2s/img` (`15.80–16.38s`, mean `17.70s` incl. `31.26s` warmup; **~1.85×** speedup), `0` NaN/black.
  - **Krea 2 Raw**: steady-state `~53.0s/img` (`49.17–57.64s`, mean `57.42s` cosplay/anime incl. `86.62s` warmup, `55.31s` plain; **~2.55×** speedup), `0` NaN/black.
  - **EasyCache**: skipped `0` steps across all 24 images at threshold `0.20`.
  - **Quality Caveat**: `--fast` + `--use-ck-attention` introduced visible high-frequency graininess relative to official ComfyUI workflows → **reinforces policy of keeping all accelerators OFF for XAI measurement (`E1–E4`)** (use Step 0 speedups for quick visual pre-checks only).
- **PNG Metadata Convention**: `save_png_with_metadata()` (`step0_precheck.py` L347–390) embeds three `PngInfo` `tEXt` chunks in every individual PNG:
  - `xai_dit_meta`: structured JSON (weights, HP, seed, timing, acceleration flags, runtime commit/torch/GPU).
  - `prompt`: ComfyUI-compatible node graph (`build_comfy_prompt_graph`, L255–344).
  - `parameters`: A1111-style human-readable parameter block.
- **Target Doc(s) to Update**:
  - `experiments/design/prompts.md`
  - `docs/experiment_design.md`
  - `docs/acceleration.md`
  - `experiments/design/output_spec.md`

---

## 3. Step 1 & Step 2 Anima Unified Prompt Set & Practical L4 Budget

Evidence from `experiments/design/unified_anima_set.json` (`unified_anima_set_v1`) and `src/xai_dit/runners/step2_anima_suite.py` (L76–91, L249–292, L1519–1546).

### 3.1 Frozen 5-Prompt Unified Anima Set (`1024×1024`, 4,096 tokens)

All 5 prompts run at `1024×1024` (`64×64 = 4,096` image tokens for cross-model parity) on the pure XAI L4 baseline (`anima-base-v1.0.safetensors`, `euler`/`simple`, 30 steps, cfg 4.0, shift 3.0, `--disable-dynamic-vram --highvram`, all accelerators OFF).

| Prompt ID | Words / Bucket | Artist & Character(s) | Primary Role Across Axes |
|---|---|---|---|
| `P-A1` | 43w (`long`) | `@ogipote`, `tsuyuri kanao` | Flagship v1 prompt; paired with `P-A-SHORT` (E4-2); E3-2 PCI global style (`@ogipote`) |
| `P-A2` | 40w (`medium_long`) | `@yoneyama mai`, `frieren` | Full-body outdoor anchor; E3-2 PCI local prop (`wooden staff`) |
| `P-A3` | 46w (`long_multi_char`) | `@ciloranko`, `2girls` (`emilia`, `arona`) | Multi-character spatial anchor for E4-4 L/R swap, E3-4 2-stage SAP, E3-1 K/V swap |
| `P-A4` | 31w (`medium`) | `@mika pikazo`, `hoshino ai` | 4th artist + MOSS-Image `P-M1` character match; secondary E3-1 swap |
| `P-A-SHORT` | 9w (`short`) | `@ogipote`, `tsuyuri kanao` | Minimal length-control pair with `P-A1` for E4-2 S/R/P consistency |

*(Note: `unified_anima_set.json` L280 still contains a `"Pending user approval"` string for E1-1 heatmap aggregation/display; this is now approved and implemented per §1.)*

### 3.2 Practical L4 Run Budget Defaults (`step2_anima_suite.py`)

To fit the full 7-stage Anima suite into a single **~100-generation / 30–35 min** Colab L4 run (`step2_anima_suite.py` L88–91, L1535–1538), two heavy per-block/per-third stages default to representative prompt subsets (CLI-overridable to the full set):

| Stage | Full Set in `unified_anima_set.json` | L4 Default in `step2_anima_suite.py` | Override Flag |
|---|---|---|---|
| `block_third_knockout` (`E3-0`) | `P-A1, P-A2, P-A3, P-A4, P-A-SHORT` | `P-A1, P-A2` | `--knockout-prompts` |
| `e2_1_stableflow` (`E2-1`) | `P-A1, P-A2, P-A3` (+ optional `P-A4, P-A-SHORT`) | `P-A1` | `--stableflow-prompts` |

### 3.3 `P-A3` (`2girls` Emilia + Arona) Framing Lesson

- **Observed Issue**: Two-sentence spatial phrasing (`On the left, ... On the right, ...`) combined with no explicit shot-size tag triggered Danbooru `split screen / 2koma` vertical dividers and cropped heads on Anima.
- **Mitigation for Clean-Seed Follow-Up**: Add `upper body, sitting side by side on bench`, consolidate into single-sentence phrasing, and append `split screen, comic, border, panel` to the negative prompt — **without modifying** the frozen `experiments/design/manifest_anima.json` (or `unified_anima_set.json` v1).
- **Target Doc(s) to Update**:
  - `experiments/design/prompts.md`
  - `docs/experiment_design.md`

---

## 4. MI & Implementation Notes for Docs & Showcase

- **`E4-1` + `E4-2` Conceptual Framing (Non-Semantic ≠ Disposable)**:
  - `R` (`</s>`) and `P` (0-padding) tokens are **non-semantic** — they must be separated from semantic prompt span `S` in heatmap analysis (`E1-1`, `E4-2`) so they do not confound content attribution.
  - However, they are **functionally essential structural pressure valves / attention registers**, *not* "disposable garbage": on `P-A1`, dynamic top-1 sink masking in blocks `[7, 14, 21]` collapses the image with `μ|Δ| = 0.2191` vs `random-1` control `0.0596` (**~3.7×** effect). Ensure all docs and `showcase/` narratives explicitly preserve this distinction.
- **`E4-1` Cond-Only Sink Masking Rationale**:
  - `stage_e4_1_sinks` (`step2_anima_suite.py` L1268–1296) sets `cfg_scope="cond_only"` with `groups=(tm.spans["P"],)`. Because Anima's zero-padding `P` span is grouped as one sink unit using the **cond** prompt's `TokenMap` indices, applying those same indices to the uncond (negative) prompt would misalign against a different token length/padding boundary.
- **Online Proxy Metrics vs Offline Evaluator Pass**:
  - `step2_anima_suite.py` (L28–29, L148–170, L1051, L1471) logs training-free online proxies on L4: `mean_abs` pixel delta, `latent_cos_vitality` (`1 - cos` on final latents), `color_layout` (saturated red/blue pixel left-share for `P-A3`), and 128×128 pixel CIS proxy (`1 - d(x_k, P_c) / d(P_b, P_c)`).
  - Model-based evaluators (**DINOv2 CLS**, **CLIP**, **VQA**, and detector-based spatial accuracy) run in a separate offline pass over the saved PNGs/latents.
- **Target Doc(s) to Update**:
  - `docs/specs/04_verification_mi.md`
  - `docs/experiment_design.md`
  - `showcase/` (narrative guidelines & figure captions)
