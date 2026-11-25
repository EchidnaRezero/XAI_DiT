# ISSUE-002: Planned Docs vs Actual Implementation & Empirical Findings Drift

- **Date**: 2026-11-25
- **Status**: Investigating *(audit complete; doc reconciliation pending user approval per `AGENTS.md` §1.3)*
- **Model / Module**: `anima`, `moss_image`, `krea2`; `docs/`, `experiments/design/`, `src/xai_dit/runners/{e1_e4_profile,e3_same_state,e2_e4_interventions,e2_prop_transplant_test,step0_precheck,step2_anima_suite}.py`
- **Related Experiments & Issues**:
  - [`ISSUE-001`](001_heatmap_outlier_crushing.md) (Heatmap outlier crushing)
  - [`plan/design_updates_pending.md`](../plan/design_updates_pending.md)
  - [`experiments/step0_precheck_l4_01/`](../experiments/step0_precheck_l4_01/)
  - [`experiments/cross_model_emilia_anime_coloring_l4_01/`](../experiments/cross_model_emilia_anime_coloring_l4_01/)
  - [`experiments/step2_anima_l4_01/`](../experiments/step2_anima_l4_01/)
  - [`experiments/e2_prop_transplant_anima_l4_01/`](../experiments/e2_prop_transplant_anima_l4_01/)
  - [`experiments/e3_3_{anima,moss_image,krea2}_*`](../experiments/)

---

## 1. Problem

Following M1 implementation and Colab L4 execution across Anima, MOSS-Image, and Krea 2 Raw, **12 concrete specification drifts** have accumulated between the pre-execution design documents (`docs/`, `experiments/design/`) and what is actually implemented in `src/xai_dit/runners/` and empirically observed in `experiments/`.

If left unreconciled, the docs will misstate:
- Heatmap aggregation/rendering defaults (`E1-1`, `E3-0`),
- Cross-model prompt compatibility (`@artist`, Krea 2 `cosplay`, `anime coloring`, `P-A3` framing),
- Which layer-selection methods actually localize editable concept layers (`E2-1`, `E2-2` vs causal `Mid [9..18]` context swap), and
- Critical intervention/MI behaviors (`E3-3` MOSS explosion, `E3-4` late `s_rel`, `E4-1` `cond_only` & register essentiality, online proxy metrics, `Step 0` acceleration & `PngInfo`).

---

## 2. Observation & Evidence

### 2.1 Heatmap Aggregation & Rendering (`E1-1`, `E3-0`)

*(Cross-reference: [`ISSUE-001`](001_heatmap_outlier_crushing.md) & [`plan/design_updates_pending.md`](../plan/design_updates_pending.md) §1.)*

| Parameter | Planned Spec (`output_spec.md` / legacy) | Actual Implementation & Verified Standard | Evidence |
|---|---|---|---|
| **Token-span aggregation** | `.sum(dim=-1)` (4.5× bias across 2–9 token spans) | `.mean(dim=-1)` (chunk-invariant via `span_agg_divisor` / `span_div`, `--span-agg mean`) | `e1_e4_profile.py`, [`ISSUE-001`](001_heatmap_outlier_crushing.md) |
| **Color normalization** | Shared `0..99.5%` `vmax` across tags | Per-tag relative `min..p99` (`--heatmap-norm per_tag`, `--heatmap-clip-pct 99.0`) | [`all_prompts_p99_s15_overlays.png`](../experiments/step2_anima_l4_01/multiprompt_profile/all_prompts_p99_s15_overlays.png) |
| **Colormap** | `inferno` | `turbo` (`--heatmap-cmap turbo`) | `e1_e4_profile.py`, `e3_same_state.py` |
| **Display smoothing** | None (raw patch grid) | Pure-NumPy separable 2D Gaussian `σ=1.5` (`GAUSSIAN_TRUNCATE=4.0`, `mode="symmetric"`) on `64×64` grid before upsampling (`display_only: true`; `.npz` unsmoothed) | [`heatmap_smooth_compare_P-A1.png`](../experiments/step2_anima_l4_01/multiprompt_profile/heatmap_smooth_compare_P-A1.png) |
| **`E3-0` Δv grids** | Uniform scale across all grids | `grid_direct_effect_steps.png` & `grid_direct_effect_dsigma_steps.png` **intentionally stay unsmoothed shared `0..99.5%`** (`dv_grids_norm: "shared"`, `turbo`) so `random_ctrl` is never inflated; `grid_attn_vs_delta_v.png` uses the full display pipeline | `e3_same_state.py` |

- **Key metric**: `@artist` tags exhibit `max / p99 = 3.91×–6.09×` corner/border sink spikes across all 5 Anima prompts vs `1.45×–2.54×` on content tags (`ISSUE-001` §2).

---

### 2.2 Cross-Model Prompts & Shared Style/Character Pool (`Step 0`, `Emilia + anime coloring`, `P-A3`)

| Assumption in `experiments/design/prompts.md` | Empirical Finding | Exact Evidence |
|---|---|---|
| `@artist` tags shared by Anima & MOSS-Image (`§0`, `P-M1`, `P-M2`) | **`@artist` is Anima-only** (`0/3` FAIL on MOSS-Image; pixel MAE / RMSE vs `no @artist`: `@mika pikazo` `0.0778` / `0.1426`, `@ogipote` `0.0479` / `0.0940`, `@yoneyama mai` `0.0412` / `0.0820`). MOSS 2D characters pass **6/6**. | [`step0_precheck_l4_01/summary.json`](../experiments/step0_precheck_l4_01/summary.json) |
| Krea 2 Raw requires photorealistic phrasing or `cosplay` for all 2D characters | Krea 2 `cosplay` passes **5/6** (`hoshino ai` FAIL); plain phrasing **without `cosplay`** passes **3/6 cleanly** (`emilia`, `arona`, `raiden shogun`) + 1 partial (`tsuyuri kanao`, 2-person duplication); `2d anime illustration` **0/2 FAIL**. | [`step0_precheck_l4_01/summary.json`](../experiments/step0_precheck_l4_01/summary.json), [`summary_krea2_plain.json`](../experiments/step0_precheck_l4_01/summary_krea2_plain.json) |
| No 3-model shared style tag defined | **`anime coloring`** works as a genuine 3-model shared style tag alongside `emilia` (no `cosplay` required): concentrates on **face/eyes** while `emilia` focuses on **hair/ornaments** across all 3 architectures. | [`anima_moss_krea2_emilia_anime_coloring_overlays.png`](../experiments/cross_model_emilia_anime_coloring_l4_01/anima_moss_krea2_emilia_anime_coloring_overlays.png), [`{anima,moss_image,krea2}/metrics.json`](../experiments/cross_model_emilia_anime_coloring_l4_01/) |
| `P-A3` (`2girls`) prompt in `manifest_anima.json` / `prompts.md` uses 2 sentences (`On the left... On the right...`) with no shot-size tag | Triggered Danbooru `split screen / 2koma` vertical divider + cropped heads on Anima. Mitigated in `p_a3_clean_seed_42` using `upper body, sitting side by side on bench`, single-sentence phrasing, and negative `split screen, comic, border, panel` (frozen manifests untouched). | [`step2_anima_l4_01/p_a3_seed_check/p_a3_seed_check_grid.png`](../experiments/step2_anima_l4_01/p_a3_seed_check/p_a3_seed_check_grid.png), [`p_a3_clean_seed_42/run_config.json`](../experiments/step2_anima_l4_01/p_a3_clean_seed_42/run_config.json) |

**3-Model `emilia + anime coloring` Internal Profile (`experiments/cross_model_emilia_anime_coloring_l4_01/`)**:

| Model | `m_S_mean` | `m_R_prefix` | `m_R_suffix` | `m_P_mean` | `normalized_template_share` | `e1_3_vnorm` Peak Block | `mact_max` (Peak Block) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Anima** (`28` blks) | `0.1495` | `0.0000` | `0.0157` | `0.8348` | `0.0952` | Blk `27` (`4.69`) | `17.0` (Blk `21`) |
| **MOSS-Image** (`42` blks) | `0.0222` | `0.1466` | `0.1093` | `0.0000` | **`0.9200`** | Blk `4` (`5.99`) | `1288.0` (Blk `41`) |
| **Krea 2 Raw** (`28` blks) | `0.0774` | `0.0000` | `0.0721` | `0.0000` | `0.4825` | Blk `11` (`1.29`) | `1784.0` (Blk `26`) |

---

### 2.3 Layer Selection vs Actual Causal Concept Control (`E2-1`, `E2-2`, `e2_prop_transplant_test.py`) & L4 Budget

1. **`E2-1` StableFlow Stage 1 (whole-block bypass) captures global skeleton collapse, not editable concept layers**:
   - On `P-A1` ([`step2_anima_l4_01/e2_1_stableflow/metrics.json`](../experiments/step2_anima_l4_01/e2_1_stableflow/metrics.json)), `true_block_bypass` (`block_body_calls == 0` verified across all 28 blocks) selected `selected_vital = [0, 1, 12]` (`latent_vitality = 0.8966, 0.9450, 0.8630`; `pixel_mean_abs = 0.2612, 0.3579, 0.2700`), with safest blocks `[24, 25, 26]` (`0.0573, 0.0931, 0.1099`). Bypassing `[0, 1, 12]` destroys overall image structure rather than isolating editable semantic concepts.
2. **`E2-2` `Cont^(l)` (`raw_mvo` averaged over all 30 steps) fails to localize causal prop-editing layers**:
   - In [`e2_2_anima_bf16_l4_01/metrics.json`](../experiments/e2_2_anima_bf16_l4_01/metrics.json) (`tsuyuri kanao`), `raw_mvo` spiked at Block `27` (`0.2358`, `3.46×` higher than runner-up Block `19` at `0.0682`), yielding `top = [2, 4, 19, 22, 23, 27]` and an inverted `top_minus_bottom_mean_abs = -0.0332` (`top20 = 0.0178` vs `bottom20 = 0.0510`, under `logit_knockout` fallback after span-length mismatch).
   - In [`e2_prop_transplant_anima_l4_01/metrics.json`](../experiments/e2_prop_transplant_anima_l4_01/metrics.json) (`sketchbook -> glowing paper lantern`, [`prop_transplant_test.png`](../experiments/e2_prop_transplant_anima_l4_01/prop_transplant_test.png)), `raw_mvo` peaked bimodally at early+late blocks (`top6 = [4, 5, 8, 9, 23, 27]`, incl. Block 27 output-norm spike) and **failed to transplant** the prop, whereas **causal `Mid [9..18]` per-block cross-attention `kv_patch` (`cond_only`, `ctx_rebuild_path_check_k_max_abs = 0.0` across all 28 blocks)** cleanly transplanted the lantern while preserving Kanao's pose, face, and composition:

| Arm (`e2_prop_transplant_anima_l4_01`) | Swapped Blocks | `pixel_mean_abs_vs_base` | `pixel_mean_abs_vs_full_target` | `transplant_fraction` | Visual Outcome |
|---|---|---:|---:|---:|---|
| `base` | `[]` | `0.0000` | `0.1766` | `0.000` | Original sketchbook |
| `full_target` | `[0..27]` | `0.1766` | `0.0000` | `1.000` | Full counterfactual target |
| `cont_top6` (`E2-2 raw_mvo`) | `[4, 5, 8, 9, 23, 27]` | `0.1748` | `0.1993` | **`-0.128`** | **FAIL** — no clean lantern transplant; composition drift |
| `cont_bottom6` | `[0, 3, 13, 14, 16, 17]` | `0.1786` | `0.2040` | **`-0.155`** | **FAIL** |
| **`mid_9_18`** *(causal mid-block swap)* | **`[9..18]`** | `0.1526` | **`0.0913`** | **`+0.483`** | **PASS** — clean lantern transplant, pose/face preserved |

3. **L4 Practical Budget Defaults (`step2_anima_suite.py`) vs Full Manifest Sweeps (`docs/experiment_design.md`)**:
   - To fit inside a single Colab L4 session, `step2_anima_suite.py` defaults heavy stages to representative subsets: `--knockout-prompts P-A1 P-A2` (vs all 5 prompts in `unified_anima_set.json`) and `--stableflow-prompts P-A1` (vs `P-A1..P-A3`), taking `58` generations / `1626.2s` (`27.1 min`) for those two stages alone ([`step2_anima_l4_01/summary.json`](../experiments/step2_anima_l4_01/summary.json)).

---

### 2.4 Intervention & MI Protocol Drifts (`E3-3`, `E3-4`, `E4-1`, `E4-2`, Proxies, `Step 0`)

| Area | Planned Spec | Observed Behavior / Actual Implementation | Exact Evidence |
|---|---|---|---|
| **`E3-3` TACA (`γ0=1.2`, `σ≥0.97`)** | Scale all text keys uniformly across all 3 models | Works cleanly on **Anima** (`P-A1`, steps `[0,1,2]`, `pixel_mean_abs = 0.0799`) & **Krea 2** (`P-K1`, steps `[0,1,2]`, `0.0534`), but **explodes/oversaturates on MOSS-Image** (`P-M1`, steps `[0,1,2,3]`, `pixel_mean_abs = 0.2674`, `max_abs = 1.0`) because **`92.3%`** (`0.9233`) of MOSS I2T attention goes to chat-template tokens `R` (`m_R_prefix=0.1455`, `m_R_suffix=0.1103` vs `m_S=0.0213`), so scaling all text amplifies `R` registers. | [`e3_3_anima_bf16_l4_01/metrics.json`](../experiments/e3_3_anima_bf16_l4_01/metrics.json), [`e3_3_krea2_fp8_l4_01/metrics.json`](../experiments/e3_3_krea2_fp8_l4_01/metrics.json), [`e3_3_moss_image_int8_convrot_l4_01/metrics.json`](../experiments/e3_3_moss_image_int8_convrot_l4_01/metrics.json), [`e1_e4_profile_moss_image_int8_convrot_l4_01/metrics.json`](../experiments/e1_e4_profile_moss_image_int8_convrot_l4_01/metrics.json) |
| **`E3-4` SAP auto-switch (`DiT-ST s_rel`)** | Auto-detect layout→detail boundary `s_rel` for 2-stage prompt switching | `s_rel` fired late at **`step 25` (`σ = 0.3756`)** on both original `P-A3` (`delta_two_vs_stage1 = 0.0137`) and clean `P-A3` (`delta_two_vs_stage1 = 0.0174`, `delta_two_vs_one = 0.2490`) — far too late for Stage 2 prompt uptake given **`E3-2` PCI** shows lock-in by **steps `3–6`** (`σ = 0.9643–0.9231`: `@ogipote` `cis_proxy` drops `1.0 → 0.3084 → 0.1074`; `wooden staff` drops `1.0 → 0.0748 → -0.0060`). | [`step2_anima_l4_01/e3_4_sap/metrics.json`](../experiments/step2_anima_l4_01/e3_4_sap/metrics.json), [`p_a3_clean_seed_42/e3_4_sap/metrics.json`](../experiments/step2_anima_l4_01/p_a3_clean_seed_42/e3_4_sap/metrics.json), [`step2_anima_l4_01/e3_2_pci/metrics.json`](../experiments/step2_anima_l4_01/e3_2_pci/metrics.json) |
| **`E4-1` / `E4-2` Sink Masking & `R/P` Framing** | Dynamic top-$k$ sink masking; separate `R/P` from `S` | Implemented as **`cfg_scope="cond_only"`** (not both CFG branches) with `groups=(tm.spans["P"],)` because `P` indices follow the cond `TokenMap` length. Masking top-1 sink (`P` group, selected `100%` of calls, `sink_mass_before = 0.81–0.88`, `residual_ratio = 0.0`) in blocks `[7, 14, 21]` collapses `P-A1` by **`0.2191` vs `random_k` `0.0596` (`3.68×`)**, `P-A2` `0.2368` vs `0.1249` (`1.90×`), `P-A3` `0.2173` vs `0.1146` (`1.90×`). **`R/P` are non-semantic yet functionally essential structural registers**, not disposable noise. | [`step2_anima_l4_01/e4_1_sinks/metrics.json`](../experiments/step2_anima_l4_01/e4_1_sinks/metrics.json) |
| **Online Proxies vs Offline Evaluators** | `output_spec.md` / `experiment_design.md` specify DINOv2, LPIPS, OpenCLIP, Grounded-SAM2, VQA | `step2_anima_suite.py` computes fast training-free online proxies on L4 (`mean_abs` pixel delta, `latent_cos_vitality = 1 - cos` on final latents, `color_layout` red/blue left-share for `P-A3`, 128×128 pixel CIS proxy); model-based evaluators run in a separate offline pass. | `step2_anima_suite.py`, [`e2_1_stableflow/metrics.json`](../experiments/step2_anima_l4_01/e2_1_stableflow/metrics.json), [`e3_2_pci/metrics.json`](../experiments/step2_anima_l4_01/e3_2_pci/metrics.json) |
| **`Step 0` Acceleration & `PngInfo` Metadata** | `output_spec.md` specifies JSON/PNG layout without embedded `PngInfo`; `acceleration.md` lacked L4 empirical speedup/quality numbers | `step0_precheck.py` embeds 3 `PngInfo` `tEXt` chunks (`xai_dit_meta`, `prompt`, `parameters`) in every PNG. `--fast` + `--use-ck-attention` achieved **`1.85×`** on MOSS (`~16.2s/img` steady-state) and **`2.55×`** on Krea 2 (`~53.0s/img` steady-state) with `0` NaN/black (`EasyCache` thresh `0.20` skipped `0` steps), but introduced high-frequency graininess — reinforcing keeping all accelerators **OFF** for XAI measurement (`E1–E4`). | `step0_precheck.py`, [`step0_precheck_l4_01/summary.json`](../experiments/step0_precheck_l4_01/summary.json), [`summary_krea2_plain.json`](../experiments/step0_precheck_l4_01/summary_krea2_plain.json) |

---

## 3. Root Cause / Hypothesis

1. **Architecture-specific register & output-norm scaling**:
   - DiT final blocks (e.g., Anima Block `27`, `vnorm = 4.69`) have large `W_o` / residual output norms that dominate unnormalized `raw_mvo` (`E2-2`), biasing `Cont^(l)` away from the mid-network semantic editing band (`[9..18]`).
   - MOSS-Image's bidirectional 2-block refiner routes `92.0%–92.3%` of I2T attention mass into chat-template tokens `R`, causing naive all-text TACA (`E3-3`) to amplify structural registers rather than semantic prompt tokens `S`.
   - Anima's 0-padding `P` tokens act as attention pressure-valve registers (`81%–88%` mass in mid/late blocks); masking them destabilizes softmax allocation (`3.68×` collapse), while unclipped `min..max` heatmaps get crushed by spatial corner-sink spikes (`3.91×–6.09×` `max/p99`).
2. **Flow-matching noise-schedule curvature (`E3-4` vs `E3-2`)**:
   - Under `euler` + `shift=3.0`, $\Delta\bar{A}$ over SNR exhibits its sharpest curvature near late low-$\sigma$ steps (`step 25`, `σ=0.3756`), whereas semantic concept identity locks in during the first `3–6` high-$\sigma$ steps (`σ ≥ 0.92`).
3. **Training-data priors & practical L4 constraints**:
   - Model pretraining priors diverged from initial prompt assumptions (`@artist` Anima-only; `anime coloring` + `emilia` 3-model shared; Danbooru `split screen` prior on `P-A3`).
   - Single-L4 VRAM and wall-clock budgets required online training-free proxies, `cond_only` sink index alignment, and default prompt subsets for multi-pass stages (`E2-1`, `E3-0` knockout).

---

## 4. Fix / Method Applied

Code-level mitigations are already implemented in `src/xai_dit/runners/` and staged in [`plan/design_updates_pending.md`](../plan/design_updates_pending.md). Per `AGENTS.md` §1.3, **no `docs/` or `experiments/design/` files have been silently altered**. The table below maps every drift item to its target document(s) for structured reconciliation once user-approved:

### Reconciliation Checklist Table

| # | Drift Item | Summary of Update Needed | Target Doc(s) |
|---|---|---|---|
| **1** | `E1-1` heatmap aggregation & rendering | Document `.mean()`, `per_tag`, `turbo`, `p99` clip, display-only Gaussian `σ=1.5` (`.npz` raw), cross-link `ISSUE-001` | `experiments/design/output_spec.md`, `docs/specs/01_heatmap.md`, `docs/experiment_design.md` |
| **2** | `E3-0` Δv grid scaling exception | Explicitly separate `dv_grids_norm: "shared"` (`0..99.5%`, unsmoothed, `turbo`) for `random_ctrl` integrity vs `grid_attn_vs_delta_v.png` display pipeline | `experiments/design/output_spec.md`, `docs/specs/03_tag_influence.md` |
| **3** | `@artist` & 3-model character/style pool | Mark `@artist` Anima-only (`0/3` MOSS); update shared 2D core pool (`emilia`, `arona`, `raiden shogun` no-`cosplay` + `tsuyuri kanao`, `frieren` with `cosplay`; exclude `hoshino ai` from Krea 2); add **`anime coloring`** as verified 3-model shared style tag | `experiments/design/prompts.md`, `docs/experiment_design.md`, `docs/direction.md` |
| **4** | `P-A3` (`2girls`) framing lesson | Document `split screen / 2koma` failure mode & clean-seed mitigation (`upper body, sitting side by side on bench` + negative `split screen, comic, border, panel`) without altering frozen `manifest_anima.json` | `experiments/design/prompts.md`, `docs/experiment_design.md` |
| **5** | `E2-1` StableFlow scope & interpretation | Clarify Stage 1 whole-block bypass identifies global skeleton collapse (`[0, 1, 12]`), not localized concept-editing layers | `docs/specs/02_layer_selection.md`, `docs/experiment_design.md` |
| **6** | `E2-2` `Cont^(l)` vs causal mid-block swap | Record Block 27 norm-spike bias & `-0.128` prop-transplant failure of raw `Cont^(l)` vs `+0.483` causal `Mid [9..18]` cross-attn `kv_patch`; note `Cont^(l)` needs output-norm normalization or causal verification | `docs/specs/02_layer_selection.md`, `docs/experiment_design.md`, `docs/architecture.md` |
| **7** | `step2_anima_suite.py` L4 budget defaults | Document CLI-overridable L4 defaults (`--knockout-prompts P-A1 P-A2`, `--stableflow-prompts P-A1`) alongside full unified set | `docs/experiment_design.md`, `experiments/design/runtime_and_hp.md` |
| **8** | `E3-3` TACA MOSS-Image `R`-register caveat | Document MOSS `92.3%` `R`-share oversaturation under all-text scaling (`γ0=1.2`) and specify **`S`-only scaling** requirement for MOSS-Image | `docs/specs/03_tag_influence.md`, `docs/experiment_design.md`, `docs/architecture.md` |
| **9** | `E3-4` SAP `s_rel` vs `E3-2` PCI lock-in | Document late `s_rel` (`step 25`, `σ=0.3756`) vs early PCI lock-in (`steps 3–6`, `σ≥0.92`) and recommend early-σ override / schedule-aware switch rule | `docs/specs/03_tag_influence.md`, `docs/experiment_design.md` |
| **10** | `E4-1` `cond_only` & `E4-2` register framing | Document `cfg_scope="cond_only"` rationale for Anima `P`-group alignment and preserve **"non-semantic ≠ disposable"** register narrative (`3.68×` collapse on `P-A1`) | `docs/specs/04_verification_mi.md`, `docs/experiment_design.md`, `docs/architecture.md` |
| **11** | Online proxy metrics vs offline evaluators | Distinguish L4 online training-free proxies (`mean_abs`, `latent_cos_vitality`, `color_layout`, 128×128 pixel CIS) from deferred offline DINOv2/LPIPS/CLIP/VQA pass | `experiments/design/output_spec.md`, `experiments/design/runtime_and_hp.md`, `docs/experiment_design.md` |
| **12** | `Step 0` acceleration & `PngInfo` convention | Record `1.85×` (MOSS) / `2.55×` (Krea 2) `--fast` + `--use-ck-attention` speedup, graininess caveat (OFF for `E1–E4`), `EasyCache` `0`-skip finding, and 3-chunk `PngInfo` (`xai_dit_meta`, `prompt`, `parameters`) convention | `docs/acceleration.md`, `experiments/design/output_spec.md`, `experiments/design/runtime_and_hp.md` |

---

## 5. Verification & Improvement

- **Existing Regression Coverage**:
  - Heatmap display pipeline (`#1`–`#2`) is covered by [`tests/test_runner_heatmap_display.py`](../tests/test_runner_heatmap_display.py) (13 CPU-only unit tests, all passing; see [`ISSUE-001`](001_heatmap_outlier_crushing.md)).
  - `E2` prop-transplant causal path integrity verified at `ctx_rebuild_path_check_k_max_abs == 0.0` across all 28 blocks ([`e2_prop_transplant_anima_l4_01/metrics.json`](../experiments/e2_prop_transplant_anima_l4_01/metrics.json)).
  - `E2-1` bypass invariant (`block_body_calls == 0`) and `E4-1` sink elimination (`residual_ratio == 0.0`) verified in [`step2_anima_l4_01/`](../experiments/step2_anima_l4_01/).
- **Next Steps Upon User Approval**:
  1. Propagate checklist items `#1`–`#12` into the listed `docs/` and `experiments/design/` files without modifying frozen `manifest_*.json`.
  2. Test `S`-only TACA on MOSS-Image (`#8`) and early-σ SAP switch (`#9`) in follow-up L4 runs, then promote `ISSUE-002` status to `Resolved`.
