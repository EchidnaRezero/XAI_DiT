# ISSUE-004: Anima BF16 `--fast fp16_accumulation` Slowdown/Drift & `TorchCompileModel + EasyCache` Dynamo Recompile Limit

- **Date**: 2026-11-26
- **Status**: Resolved
- **Model / Module**: `anima` (BF16 cross-attn DiT, `1024×1024`, Colab L4); ComfyUI `--fast fp16_accumulation`, `TorchCompileModel(inductor)`, `comfy_extras/nodes_easycache.py`
- **Related Experiments**: [`experiments/accel_p1_anima_kanao/summary.md`](../experiments/accel_p1_anima_kanao/summary.md), [`experiments/accel_p1_anima_round2/summary.md`](../experiments/accel_p1_anima_round2/summary.md)

---

## 1. Problem

Two unexpected regressions were identified during Phase 1 acceleration benchmarking of Anima (BF16) on Colab L4 (`anima_kanao`, seed `42`, 30 steps `euler/simple`):

1. **ComfyUI `--fast fp16_accumulation` slows down Anima BF16 by 37% and degrades image fidelity**: Enabling `fp16_accumulation` (alone or via full `--fast`) reduced throughput from `1.00x` to `0.73x` while dropping PSNR to `21.57 dB`.
2. **`TorchCompileModel(inductor) + EasyCache` hits `torch._dynamo` `config.recompile_limit (8)`**: Combining Inductor compilation with EasyCache caused repeated Dynamo recompilations, making steady-state execution slower than uncompiled EasyCache.

---

## 2. Observation & Evidence

### A. `--fast fp16_accumulation` Regression (`experiments/accel_p1_anima_kanao/summary.md`)

| Preset | Wall (s) | Speedup | s/fwd | VRAM (GB) | PSNR (dB) | HF Ratio |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` (SDPA) | 25.39 | 1.00x | 0.815 | 5.37 | - | - |
| `fast_no_fp16acc` | 27.46 | 0.92x | 0.827 | 5.37 | 59.14 | 0.991 |
| `fast_cublas` | 27.62 | 0.92x | 0.826 | 5.37 | 59.14 | 0.991 |
| `fast_fp8mm` | 27.74 | 0.92x | 0.830 | 5.37 | 59.14 | 0.991 |
| `fast_fp16acc` | 34.64 | **0.73x** | 1.069 | 5.37 | **21.57** | **1.033** |
| `fast_all` | 34.95 | **0.73x** | 1.076 | 5.37 | **21.57** | **1.033** |

- Non-`fp16_accumulation` `--fast` variants (`fast_cublas`, `fast_fp8mm`, `fast_no_fp16acc`) stayed pixel-identical (`59.14 dB` PSNR) at `27.46–27.74s` (`0.92x`).
- Enabling `fp16_accumulation` (`fast_fp16acc`, `fast_all`) regressed wall time by ~37% (`34.64s` / `34.95s`, `1.069–1.076 s/fwd`), dropped PSNR to `21.57 dB`, and raised the HF Laplacian ratio to `1.033`.

### B. `TorchCompileModel(inductor) + EasyCache` Recompile Limit (`experiments/accel_p1_anima_round2/summary.md`)

| Preset | Rep 1 Wall (s) | Rep 2 Steady-State Wall (s) | Speedup | s/fwd | VRAM (GB) | PSNR (dB) | HF Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 25.39 | 25.39 | 1.00x | 0.815 | 5.37 | - | - |
| `ec_t20_s20_e80` (uncompiled) | 15.31 | 15.31 | **1.66x** | 0.797 | 5.39 | 24.11 | 0.942 |
| `compile_inductor` | 63.22 | 23.48 | 1.08x | 0.769 | 5.40 | 23.45 | 1.007 |
| `compile_inductor_ec20` | 22.07 | **16.00** | 1.59x | 0.743 | 5.40 | 22.51 | 0.927 |

- `compile_inductor_ec20` hit `torch._dynamo` `config.recompile_limit (8)` with `___check_type_id` guard failures.
- Rep 2 steady-state (`16.00s`) was slower than uncompiled `ec_t20_s20_e80` (`15.31s`), after paying a `22.07s` rep 1 warmup.
- `compile_inductor` alone required a `63.22s` rep 1 warmup for only a `1.08x` steady-state gain (`23.48s`).

---

## 3. Root Cause / Hypothesis (Code-Level Verification in `reference_repo/ComfyUI`)

1. **Hidden `PRIORITIZE_FP16 = True` Override + 168 `FP32 ↔ FP16` Casts/Forward**:
   - In `comfy/model_management.py:551-558`, enabling `PerformanceFeature.Fp16Accumulation` does **not** merely toggle `torch.backends.cuda.matmul.allow_fp16_accumulation = True`; it also globally sets **`PRIORITIZE_FP16 = True`**.
   - Because `Anima` declares `supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]` (`comfy/supported_models.py:1184`), `unet_dtype()` (`comfy/model_management.py:1162-1164`, called from `comfy/sd.py:2552`) **silently overrides BF16 and loads/runs Anima in `torch.float16`** (and `Anima.set_inference_dtype` at `supported_models.py:1197-1198` bumps `memory_usage_factor *= 1.4`).
   - Inside `MiniTrainDIT._forward()` (`comfy/ldm/cosmos/predict2.py:913-918`), ComfyUI includes an explicit FP16 overflow workaround that upcasts the residual stream to **FP32**:
     ```python
     if x_B_T_H_W_D.dtype == torch.float16:
         x_B_T_H_W_D = x_B_T_H_W_D.float()
     ```
   - Inside every `Block.forward()` (`predict2.py:487-488, 529-597`), all 3 LayerNorms, AdaLN modulations, and residual `torch.addcmul` updates run in **FP32** (2× memory bandwidth vs BF16), and each block executes **6 explicit full-tensor `FP32 ↔ FP16` casts** (`lines 538, 548, 562, 581, 588, 596`) × 28 blocks = **168 extra full-tensor dtype conversions per forward pass**, directly causing the 37% throughput regression (`0.815 → 1.069 s/fwd`).
   - Simultaneously, Attention/MLP GEMMs execute in FP16 (5-bit exponent) with FP16 accumulation (10-bit mantissa) on a BF16-trained model, causing the `21.57 dB` numerical fidelity drop.
2. **EasyCache State Type Oscillation Inside Compiled `diffusion_model` Boundary**:
   - `TorchCompileModel` (`comfy_extras/nodes_torch_compile.py:29` → `comfy_api/torch_helpers/torch_compile.py:58-67`) compiles `diffusion_model`, while `MiniTrainDIT.forward()` (`comfy/ldm/cosmos/predict2.py:856`) invokes `WrappersMP.DIFFUSION_MODEL` (`easycache_forward_wrapper`) **inside** `diffusion_model.forward()`.
   - In `comfy_extras/nodes_easycache.py:74, 210`, `easycache.cumulative_change_rate` is initialized/reset to Python `float` (`0.0`), whereas `:57-60` (`easycache.cumulative_change_rate += approx_output_change_rate`) adds a 0-D CUDA `torch.Tensor`, promoting the attribute to `Tensor`.
   - On every non-skipped step, `cumulative_change_rate` oscillates `float ↔ Tensor`, causing Dynamo `___check_type_id` guard failures until `config.recompile_limit (8)` is exhausted. Changing `:60` to `+= approx_output_change_rate.item()` eliminates the type oscillation (with zero extra GPU sync since `:61` already branches in Python), though cleanly avoiding graph breaks requires moving `DIFFUSION_MODEL` wrappers outside `torch.compile`.

*(Note on `--use-ck-attention` on Anima BF16: `comfy/ldm/modules/attention.py:672-695, 945-951` routes `--use-ck-attention` to `comfy_kitchen.prequantize_int8_attention(q, k, v)`, quantizing Q/K/V to INT8 on every call. Because Anima runs 56 attention calls/fwd—half of which are short-`S_k` cross-attentions with un-normalized `v_norm = nn.Identity()` at `predict2.py:135-141`—INT8 quantization error shifts the 30-step ODE trajectory to `15.05 dB` PSNR while 56 prequantize calls cancel Tensor Core gains at `0.96x`.)*

---

## 4. Fix / Method Applied

1. **Keep `--fast` OFF on Anima BF16**: Do not enable `--fast` (especially `fp16_accumulation`) for Anima BF16 runs on L4.
2. **Do Not Combine `TorchCompileModel` with `EasyCache` on Anima L4**: Use uncompiled PyTorch SDPA + `EasyCache(threshold=0.20, start=0.20, end=0.80)`.

---

## 5. Verification & Improvement

- Verified in [`experiments/accel_p1_anima_kanao/summary.md`](../experiments/accel_p1_anima_kanao/summary.md) and [`experiments/accel_p1_anima_round2/summary.md`](../experiments/accel_p1_anima_round2/summary.md):
  - Uncompiled PyTorch SDPA + `EasyCache(threshold=0.20, start=0.20, end=0.80)` (`ec_t20_s20_e80`) achieves `15.31s` (`1.66x` speedup, `24.11 dB` PSNR, HF ratio `0.942`) with zero compile warmup overhead and zero Dynamo recompile failures.
  - Confirmed on a second prompt (`emilia_ec_t20_s20_e80`): `14.00s` vs `24.70s` baseline (`1.81x` speedup, `29.43 dB` PSNR).
