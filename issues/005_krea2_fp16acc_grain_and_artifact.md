# ISSUE-005: Krea 2 Raw FP8 `--fast fp16_accumulation` High-Frequency Grain & Right-Edge Hallucination

- **Date**: 2026-11-26
- **Status**: Resolved
- **Model / Module**: `krea2` (`krea2_raw_fp8_scaled.safetensors`, Colab L4); ComfyUI `--fast fp16_accumulation` (`PerformanceFeature.Fp16Accumulation`), `--use-ck-attention`
- **Related Experiments**: [`experiments/accel_p1_krea2_idol_cafe/summary.md`](../experiments/accel_p1_krea2_idol_cafe/summary.md), [`experiments/accel_p1_krea2_idol_cafe/contact_sheet.png`](../experiments/accel_p1_krea2_idol_cafe/contact_sheet.png)

---

## 1. Problem

Combining `--use-ck-attention` + full `--fast` (`combo_ck_fast_all`) on Krea 2 Raw FP8 (`krea2_idol_cafe`, seed `42`, 32 steps `euler/simple`) slowed per-forward latency (`2.138 s/fwd` vs `2.035 s/fwd` for `attn_ck` alone), increased high-frequency graininess (`HF Ratio 1.068`), dropped PSNR to `23.01 dB`, and introduced a hallucinated jacket/shoulder artifact on the right edge of the generated image.

---

## 2. Observation & Evidence

Measurements from [`experiments/accel_p1_krea2_idol_cafe/summary.md`](../experiments/accel_p1_krea2_idol_cafe/summary.md) (`krea2_idol_cafe`, seed `42`, `13.43 GB` VRAM across all presets) and visual inspection of [`contact_sheet.png`](../experiments/accel_p1_krea2_idol_cafe/contact_sheet.png):

| Preset | Wall (s) | Speedup | s/fwd | PSNR (dB) | HF Ratio | Visual / Notes |
|---|---:|---:|---:|---:|---:|---|
| `baseline` (SDPA) | 176.57 | 1.00x | 2.104 | - | - | Unaccelerated L4 FP8 baseline |
| `fast_no_fp16acc` | 140.36 | 1.26x | 2.101 | **58.88** | **0.980** | `cublas_ops, fp8_matrix_mult, autotune`; pixel-identical |
| `fast_fp16acc` | 148.13 | 1.19x | **2.222** | **26.90** | **1.040** | Slower per-forward; fidelity drop & elevated HF grain |
| `fast_all` | 147.91 | 1.19x | **2.213** | **26.90** | **1.040** | Matches `fast_fp16acc` degradation |
| `attn_ck` | 172.34 | 1.02x | **2.035** | 29.90 | 1.002 | Faster per-forward; clean output |
| `combo_ck_fast_all` | 142.71 | 1.24x | 2.138 | **23.01** | **1.068** | Grain + hallucinated jacket/shoulder artifact on right edge |
| `fix_ck_fast_safe` | 135.93 | 1.30x | **2.031** | **29.90** | **1.002** | Clean; hallucinated artifact completely removed |
| `fix_full_safe_ec25` | **61.60** | **2.87x** | **2.020** | **27.16** | **0.917** | `ck + safe --fast + EasyCache(0.20, s=0.25)` |

---

## 3. Root Cause / Hypothesis (Code-Level Verification in `reference_repo/ComfyUI`)

1. **Silent BF16 → FP16 `compute_dtype` Override via `PRIORITIZE_FP16 = True`**:
   - Enabling `--fast fp16_accumulation` sets **`PRIORITIZE_FP16 = True`** in `comfy/model_management.py:551-558`.
   - `Krea2` declares `supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]` (`comfy/supported_models.py:2042`). *(By contrast, `MOSSImage` at `supported_models.py:2071` omits `torch.float16`, making MOSS immune to this override.)*
   - Because `krea2_raw_fp8_scaled` carries `quant_config`, `comfy/sd.py:2556-2557` calls `unet_manual_cast(...)` (`comfy/model_management.py:1197-1199`), where `PRIORITIZE_FP16 = True` **silently switches `mixed_precision_ops` `compute_dtype` from `torch.bfloat16` to `torch.float16`**.
2. **Why Per-Forward Latency Regresses (`2.104 → 2.222 s/fwd`, `+5.6%`)**:
   - In `comfy/quant_ops.py:101-106` (`_TensorCoreFP8LayoutBase.quantize`), dynamic FP8 input quantization checks:
     ```python
     if tensor.dtype not in [torch.float32, torch.bfloat16]:  # Prevent scale from being too small
         tensor_info = torch.finfo(tensor.dtype)
         scale = (1.0 / torch.clamp((1.0 / scale), min=tensor_info.min, max=tensor_info.max))
     ```
     Forcing `compute_dtype = float16` pushes every dynamically quantized FP8 linear input through this extra reciprocal/clamp path, while `CastBiasWeightContext` casts non-FP8 BF16 weights (`txtfusion`, embeddings, `modulation.lin`, `final_layer`) to FP16 on every forward pass.
3. **Why High-Frequency Grain (`1.040`/`1.068`) & Right-Edge Hallucination Occur**:
   - Non-FP8 linear layers execute with `allow_fp16_accumulation = True` (10-bit mantissa accumulation), and the residual stream / QKV / `RMSNorm` output (`comfy/ldm/krea2/model.py:35`) runs in `float16` (max 65,504). Under Krea 2's extreme Massive Activations (`~1,920`) and 96%-closed Block 0 `G1` gate, FP16 dynamic range + FP16 accumulation injects high-frequency drift (`PSNR 26.90 dB`, `HF Ratio 1.040`).
   - Stacking `--use-ck-attention` (`comfy/ldm/modules/attention.py:672-695`, which quantizes Q/K/V to **INT8** via `comfy_kitchen.prequantize_int8_attention`) compounds INT8 attention quantization error on top of FP16 accumulation error, degrading PSNR to `23.01 dB`, raising HF Ratio to `1.068`, and triggering the right-edge jacket/shoulder structural hallucination.

---

## 4. Fix / Method Applied

Dropped `fp16_accumulation` from `--fast` while keeping `--use-ck-attention` and the safe `--fast` subset (`cublas_ops`, `fp8_matrix_mult`, `autotune`) (`fix_ck_fast_safe`), optionally combined with delayed `EasyCache(threshold=0.20, start=0.25)` (`fix_full_safe_ec25`).

---

## 5. Verification & Improvement

- Verified in [`experiments/accel_p1_krea2_idol_cafe/summary.md`](../experiments/accel_p1_krea2_idol_cafe/summary.md) and [`contact_sheet.png`](../experiments/accel_p1_krea2_idol_cafe/contact_sheet.png):
  - `fix_ck_fast_safe` restores `2.031 s/fwd` (vs `2.138 s/fwd` in `combo_ck_fast_all`), improves PSNR from `23.01 dB` to `29.90 dB`, lowers HF Ratio from `1.068` to `1.002`, and completely removes the right-edge jacket/shoulder hallucination.
  - Adding delayed `EasyCache(0.20, s=0.25)` (`fix_full_safe_ec25`) reaches `61.60s` (`2.87x` speedup, `2.020 s/fwd`) at `27.16 dB` PSNR and `0.917` HF Ratio.
