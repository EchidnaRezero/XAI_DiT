# ISSUE-001: Heatmap Token-Count Bias & Corner-Spike Outlier Crushing

- **Date**: 2026-11-25
- **Status**: Promoted to Tests
- **Model / Module**: `anima` (cross-attn DiT, `1024×1024` / `64×64` tokens); `src/xai_dit/runners/e1_e4_profile.py`, `src/xai_dit/runners/e3_same_state.py`, `src/xai_dit/runners/step2_anima_suite.py`
- **Related Experiments**: [`experiments/step2_anima_l4_01/multiprompt_profile/`](../experiments/step2_anima_l4_01/multiprompt_profile/), [`heatmap_smooth_compare_P-A1.png`](../experiments/step2_anima_l4_01/multiprompt_profile/heatmap_smooth_compare_P-A1.png), [`all_prompts_p99_s15_overlays.png`](../experiments/step2_anima_l4_01/multiprompt_profile/all_prompts_p99_s15_overlays.png)

---

## 1. Problem

Two compounding visualization defects degraded E1-1 `img→txt` attention heatmaps across the unified Anima prompt set:

1. **Token-Count Bias & Cross-Tag Crushing**: Under the legacy pipeline (`.sum(dim=-1)` over target span tokens + shared `0..99.5%` `vmax` across tags + `inferno` colormap), short 2-token tags (e.g., `sketchbook`, `wooden staff`, `red umbrella`) appeared near-black whenever displayed alongside 6–9 token spans (e.g., `tsuyuri kanao`, `emilia \(re:zero\)`).
2. **Corner/Border Sink Spike Crushing**: Even after switching to `.mean(dim=-1)` + per-tag `min..max` + `turbo`, `@artist` style-tag heatmaps across all 5 Anima prompts remained visually flat (almost entirely dark blue) with a single bright dot at the image corner/border.

---

## 2. Observation & Evidence

Inspecting raw `64×64` mean-aggregated attention maps in [`experiments/step2_anima_l4_01/multiprompt_profile/*/profiles.npz`](../experiments/step2_anima_l4_01/multiprompt_profile/) revealed two distinct numerical patterns:

- **Token-count spread**: Target spans in the 5-prompt unified set range from **2 tokens** (`sketchbook`, `wooden staff`, `red scarf`, `silver microphone`, `frieren`, `red umbrella`) to **9 tokens** (`tsuyuri kanao`, `emilia \(re:zero\)`) — a **4.5×** multiplicative bias under `.sum(dim=-1)`.
- **Extreme `max/p99` outlier spikes on `@artist` tags (`3.91x–6.09x`)** vs mild tails on localizable content tags (`1.45x–2.54x`):

| Prompt | `@artist` Tag (Tokens) | Mean | `p99` | `max` | `max / p99` | Content Tags `max / p99` Range |
|---|---:|---:|---:|---:|---:|---:|
| `P-A-SHORT` | `@ogipote` (6) | `1.22e-03` | `1.72e-03` | `6.70e-03` | **3.91×** | `1.76× – 2.27×` |
| `P-A1` | `@ogipote` (6) | `1.00e-03` | `1.36e-03` | `6.32e-03` | **4.65×** | `1.51× – 2.34×` |
| `P-A2` | `@yoneyama mai` (5) | `1.08e-03` | `1.51e-03` | `7.05e-03` | **4.67×** | `1.68× – 2.54×` |
| `P-A4` | `@mika pikazo` (6) | `8.91e-04` | `1.23e-03` | `6.35e-03` | **5.17×** | `1.57× – 1.86×` |
| `P-A3` | `@ciloranko` (5) | `1.01e-03` | `1.32e-03` | `8.02e-03` | **6.09×** | `1.45× – 2.28×` |

Visual before/after evidence:
- Single-prompt smoothing & clipping sweep (`P-A1`): [`heatmap_smooth_compare_P-A1.png`](../experiments/step2_anima_l4_01/multiprompt_profile/heatmap_smooth_compare_P-A1.png)
- All 5 prompts rendered with `p99` clip + Gaussian `σ = 1.5`: [`all_prompts_p99_s15_overlays.png`](../experiments/step2_anima_l4_01/multiprompt_profile/all_prompts_p99_s15_overlays.png)

---

## 3. Root Cause / Hypothesis

1. **Aggregation & Shared-Scale Bias (`e1_e4_profile.py`)**: Summing attention probabilities over target tokens scales linearly with tokenizer split length (`4.5×` between 2-token and 9-token spans), and a shared `vmax` compresses lower-mass spans into the bottom ~10% of the `inferno` colormap.
2. **Patch-Grid Sink Spikes on Global Style Tokens**: Global `@artist` style tokens have relatively uniform diffuse attention over the foreground/background, but 1–2 isolated corner/border latent patches absorb massive attention spikes (`3.91×–6.09×` above `p99`, consistent with ViT/DiT spatial register/sink artifacts). Under exact `min..max` normalization, that single patch consumes ~75–85% of the `[0, 1]` color dynamic range, crushing the true semantic distribution into the bottom of `turbo`.

---

## 4. Fix / Method Applied

Implemented in `src/xai_dit/runners/e1_e4_profile.py` (user-approved spec change, keeping raw `.npz` and numerical metrics strictly unsmoothed):

1. **Span Mean Aggregation (`--span-agg mean|sum`, default `mean`)**: Replaced `.sum(dim=-1)` with division by full span length (`span_agg_divisor` / `span_div`, exact even when OOM key-chunking splits a span across chunks).
2. **Per-Tag Relative Normalization (`--heatmap-norm per_tag|shared`, default `per_tag`)**: Each map normalizes to its own `[min, hi]` instead of a shared cross-tag `vmax` (matching `reference_repo/Anima_HEATMAP` v1).
3. **Colormap (`--heatmap-cmap turbo|inferno`, default `turbo`)**: Switched default display colormap from `inferno` to `turbo` for perceptual contrast across mid-range values.
4. **Display-Only Smoothing + Upper-Percentile Clipping (`prepare_heatmap`)**:
   - Pure-NumPy separable 2D Gaussian blur (`smooth_heatmap`, `GAUSSIAN_TRUNCATE = 4.0`, `mode="symmetric"` reflect borders) on the native `64×64` patch grid with `--heatmap-smooth-sigma 1.5` (`0` disables) **before** bilinear upsampling and normalization.
   - Upper-percentile clip `--heatmap-clip-pct 99.0` (`100` = exact `min..max`) for `per_tag` mode.
5. **Control Integrity Exception (`e3_same_state.py`)**: `grid_direct_effect_steps.png` and `grid_direct_effect_dsigma_steps.png` retain unsmoothed shared `0..99.5%` scaling (`dv_grids_norm: "shared"`, with `turbo`) so the `random_ctrl` magnitude-control row is never inflated to look like signal.

---

## 5. Verification & Improvement

- **Visual Verification**:
  - [`heatmap_smooth_compare_P-A1.png`](../experiments/step2_anima_l4_01/multiprompt_profile/heatmap_smooth_compare_P-A1.png) confirms that `p99` + Gaussian `σ = 1.5` removes corner-spike crushing on `@ogipote` while preserving sharp localization on `tsuyuri kanao`, `demon slayer uniform`, and `sketchbook`.
  - [`all_prompts_p99_s15_overlays.png`](../experiments/step2_anima_l4_01/multiprompt_profile/all_prompts_p99_s15_overlays.png) confirms consistent readability across all 5 Anima prompts (`P-A1`–`P-A4`, `P-A-SHORT`).
- **Promoted Regression Suite**: [`tests/test_runner_heatmap_display.py`](../tests/test_runner_heatmap_display.py) (CPU-only, 13 unit tests) verifies:
  - Exact chunk-invariant span `mean` removing token-count scaling (`test_span_mean_removes_token_count_scaling`, `test_span_mean_exact_across_key_chunks`).
  - Separable Gaussian kernel exactness, mass conservation on corner impulses, and reflect-border behavior (`test_gaussian_kernel_matches_closed_form`, `test_smooth_is_separable_gaussian_and_mass_preserving`).
  - Synthetic `4.65×` corner-spike crushing under `min..max` (`< 0.25` at true blob peak) vs recovery under `p99` + `σ = 1.5` (`> 0.9` at true blob peak) (`test_corner_spike_crushes_exact_min_max_but_not_p99`, `test_prepare_heatmap_smooths_before_clipping`).
