# ISSUE-006: `EasyCache` + `split` CFG Wrapper Registration Order Causes Asymmetric Speckled Noise Collapse

- **Date**: 2026-11-26
- **Status**: Resolved
- **Model / Module**: `anima` (and any model using `split` CFG batching with `EasyCache`); `src/xai_dit/_accel_tmp/phase2_accel_heatmap.py`, `src/xai_dit/adapters/_comfy_hooks.py`, ComfyUI `comfy_extras/nodes_easycache.py`
- **Related Experiments**: [`experiments/phase2_xai/anima/grid_anima_kanao_s42.png`](../experiments/phase2_xai/anima/grid_anima_kanao_s42.png), [`experiments/phase2_xai_fixed/anima/summary.md`](../experiments/phase2_xai_fixed/anima/summary.md), [`experiments/phase2_xai_fixed/anima/grid_anima_kanao_s42.png`](../experiments/phase2_xai_fixed/anima/grid_anima_kanao_s42.png)

---

## 1. Problem

In Phase 2 (`src/xai_dit/_accel_tmp/phase2_accel_heatmap.py`), combining `EasyCache` with XAI_DiT's `install_runtime_wrappers(c, patcher, "split")` produced severe high-frequency speckled noise collapse across the entire generated image on Anima (`anima_kanao`, seed `42`, `best_ec_full` and `best_ec_selective`), even though `EasyCache` was completely clean in Phase 1 and attention heatmap correlation remained `r = 0.978`.

---

## 2. Observation & Evidence

- **Visual inspection** of [`experiments/phase2_xai/anima/grid_anima_kanao_s42.png`](../experiments/phase2_xai/anima/grid_anima_kanao_s42.png):
  - Row 1 (`ref_full`, no cache): crystal clear.
  - Rows 2–3 (`best_ec_full`, `best_ec_selective`): heavy speckled noise / static across the entire image.
- **Timing anomaly**: `best_ec_full` achieved only `1.31–1.42x` speedup (`28.7s -> 21.7s`) compared to Phase 1's `1.66–1.81x` (`25.4s -> 15.0s`), indicating that half of the forward passes (`uncond`) were not being skipped.
- **Heatmap metric**: Attention heatmap correlation remained `r = 0.978`, because the `cond` branch still followed the cached trajectory while the CFG combination was corrupted.

---

## 3. Root Cause / Hypothesis

ComfyUI executes `WrappersMP.CALC_COND_BATCH` wrappers in registration order (first-registered = outermost).

`phase2_accel_heatmap.py` previously registered `install_runtime_wrappers(..., "split")` (`cfg_batching_wrapper` in `src/xai_dit/adapters/_comfy_hooks.py:446-457`) **before** `apply_easycache(...)` (`easycache_calc_cond_batch_wrapper` in `comfy_extras/nodes_easycache.py:158-163`).

Because `cfg_batching_wrapper("split")` ran **outermost**, it invoked `executor(cond)` and `executor(uncond)` as two separate calls into `easycache_calc_cond_batch_wrapper`:

1. **On `cond`**: `easycache_calc_cond_batch_wrapper` set `skip_current_step = False`, then `EasyCache` decided to skip and set `skip_current_step = True`, returning the **cached `cond`** from the previous step.
2. **On `uncond`**: `easycache_calc_cond_batch_wrapper` ran a second time and **reset `skip_current_step = False`**, causing `uncond` to execute **fresh** on the current step.
3. **Asymmetric CFG**: CFG then combined `uncond_NEW + cfg * (cond_OLD - uncond_NEW)`, injecting massive asymmetric high-frequency noise on every skipped step and cutting the effective skip rate in half.

---

## 4. Fix / Method Applied

In `src/xai_dit/_accel_tmp/phase2_accel_heatmap.py` (L181–L190), reordered wrapper registration so `apply_easycache(...)` is attached **before** `install_runtime_wrappers(c, patcher, d.batching)`:

```python
ec_holder = None
if easycache:
    patcher, ec_holder = apply_easycache(patcher, ec_thresh, ec_start, ec_end)

install_runtime_wrappers(c, patcher, d.batching)
```

With `easycache_calc_cond_batch_wrapper` running **outermost**, `skip_current_step = False` is reset once per step, and `cfg_batching_wrapper("split")` splits `cond`/`uncond` inside it so both branches skip synchronously.

---

## 5. Verification & Improvement

Re-ran Anima Phase 2 (`2 prompts × 2 seeds = 4 cases`) after registering `apply_easycache(...)` before `install_runtime_wrappers(..., "split")` ([`experiments/phase2_xai_fixed/anima/summary.md`](../experiments/phase2_xai_fixed/anima/summary.md)):

- **Image collapse 100% eliminated**: Visually verified in [`experiments/phase2_xai_fixed/anima/grid_anima_kanao_s42.png`](../experiments/phase2_xai_fixed/anima/grid_anima_kanao_s42.png) — all 3 rows (`ref_full`, `best_ec_full`, `best_ec_selective`) are crystal clear with zero speckled noise.
- **Speedup restored** (both `cond` and `uncond` now skip synchronously):
  - `best_ec_full`: improved from `1.31–1.42x` (`21.7–22.5s`) to **`1.65–1.75x` (`16.2–17.4s`)**.
  - `best_ec_selective` (6 mid blocks): improved from `1.45–1.59x` (`19.7–20.1s`) to **`1.87–1.99x` (`14.2–15.1s`)**.
- **Heatmap fidelity**:
  - `best_ec_full`: Mean Pearson `r = 0.9764–0.9879`, Mean Spearman `ρ = 0.9712–0.9832`, Mean Top-15% IoU `0.8088–0.8728`.
  - `best_ec_selective`: Mean Pearson `r = 0.9554–0.9699`, Mean Spearman `ρ = 0.9198–0.9596`, Mean Top-15% IoU `0.7583–0.8159`.
