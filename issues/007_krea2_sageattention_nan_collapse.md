# ISSUE-007: Krea 2 Global `--use-sage-attention` NaN / Black Image Collapse (Code-Level Diagnosis)

- **Date**: 2026-11-26
- **Status**: Diagnosed (Verification Probe Scheduled in Phase 3)
- **Model / Module**: `krea2` (`krea2_raw_fp8_scaled` / `krea2_turbo`), ComfyUI `comfy/ldm/modules/attention.py`, `comfy/ldm/krea2/model.py`
- **Related Docs / Plans**: [`docs/acceleration.md`](../docs/acceleration.md), [`docs/reference/models/krea2.md`](../docs/reference/models/krea2.md), [`plan/accel_phase3_extended.md`](../plan/accel_phase3_extended.md)

---

## 1. Problem

Enabling ComfyUI's global `--use-sage-attention` flag on **Krea 2 Raw** produces `NaN` activations and completely black output images (also reported across community workflows when forcing SageAttention globally via CLI flags rather than per-module patching).

---

## 2. Observation & Evidence

- **Observed failure**: Global `--use-sage-attention` on Krea 2 Raw produces all-black / `NaN` outputs, forcing exclusion of global SageAttention in [`docs/acceleration.md`](../docs/acceleration.md) and [`plan/accel_phase1_phase2.md`](../plan/accel_phase1_phase2.md).
- **Community corroboration**: `civitai.red` (`articles/37212`) explicitly warns against forcing global `--use-sage-attention` across all models due to black-image outputs, recommending per-model/selective patching instead.
- **Model internal baseline (`experiments/cross_model/e4_3/e4_3_mact_compare.png`)**: Krea 2 exhibits Massive Activations up to `~1,920` and relies on 12 raw un-normalized Qwen3-VL-4B hidden layers fused inside `TextFusionTransformer`.

---

## 3. Root Cause / Hypothesis (Code-Level Verification in `reference_repo/ComfyUI`)

Code audit of `reference_repo/ComfyUI` reveals **three compounding causes** in ComfyUI's SageAttention integration and Krea 2 model implementation:

1. **`smooth_k=False` Hardcoded in ComfyUI (`comfy/ldm/modules/attention.py:740`)**:
   ```python
   sage_kwargs = {"is_causal": False, "tensor_layout": tensor_layout, "sm_scale": kwargs.get("scale", None), "smooth_k": False}
   ```
   Hardcoding `smooth_k=False` disables SageAttention's per-channel K-mean subtraction, removing its primary defense against channel-wise outliers prior to INT8 Q/K quantization.
2. **Global Dispatch into `TextFusionTransformer` Without `low_precision_attention=False` (`comfy/ldm/krea2/model.py:99-100, 157-166`)**:
   - Unlike `comfy/image_encoders/dino3.py:97` (which explicitly passes `low_precision_attention=False` to bypass `attention_sage`), `krea2/model.py:99-100` unconditionally calls `optimized_attention_masked(...)` for **both** `TextFusionTransformer` and all 28 DiT blocks.
   - Inside `TextFusionTransformer.forward` (`krea2/model.py:157-166`), `layerwise_blocks` run attention over `seq_len = n = 12` (the 12 raw, un-normalized Qwen3-VL-4B hidden layers per [`docs/reference/models/krea2.md`](../docs/reference/models/krea2.md)), which combine extreme LLM channel outliers with a tiny sequence length (`12`, smaller than SageAttention's tiled kernel block size).
3. **Un-Normalized `V` Projections Exposed to Massive Activations (`comfy/ldm/krea2/model.py:80-84`)**:
   - While `q` and `k` pass through FP32 `QKNorm` (RMSNorm), **`v = self.wv(x)` has NO normalization**. Combined with Krea 2's Massive Activations (`~1,920`) and `smooth_k=False`, SageAttention's INT8/FP8/FP16-accum tiled kernels overflow to `Inf`/`NaN`, collapsing the latent and decoded image to black.

---

## 4. Fix / Method Applied (Proposed & Pending Phase 3 Verification)

- **Current safe policy**: Keep global `--use-sage-attention` **OFF** on Krea 2; use `--use-ck-attention` or PyTorch SDPA (`fix_ck_fast_safe`).
- **Targeted code-level mitigation to verify in Phase 3 (`plan/accel_phase3_extended.md` §2.3)**:
  1. Pass `low_precision_attention=False` inside `TextFusionTransformer` so the 12-layer Qwen3-VL fusion module stays on exact SDPA.
  2. Test enabling `smooth_k=True` on Krea 2 DiT blocks (or keeping high-`M-Act` blocks on SDPA) to determine whether the collapse is caused solely by `TextFusionTransformer` + `smooth_k=False` or also by un-normalized `V` in late DiT blocks.

---

## 5. Verification & Improvement

- **Pending GPU verification**: Scheduled as Probe P3-C in [`plan/accel_phase3_extended.md`](../plan/accel_phase3_extended.md) on Colab L4 (isolating `TextFusionTransformer` SDPA bypass and `smooth_k=True` separately while logging first-NaN block/step).
