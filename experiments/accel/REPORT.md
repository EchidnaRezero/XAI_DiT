# XAI_DiT Acceleration & Accelerated XAI Master Report (Phase 1 → Phase 3)

Last updated: 2026-11-27  
Hardware (Tier 1): Colab NVIDIA L4 (24 GB VRAM, `sm_89`, 58 SMs, PyTorch `2.14.1+cu130`, `comfy-kitchen 0.2.42`, `--disable-dynamic-vram --highvram`)  
Target Models: **Anima** (`BF16`, `INT8 ConvRot`, `Turbo v1.1`, `4-step LoRA`) · **MOSS-Image** (`sft_int8_convrot`, `DMD 8-step`, `eDMD 4-step`) · **Krea 2** (`raw_fp8_scaled`, `raw_int8_convrot`, `turbo_fp8_scaled`) · **Krea2-Anime v3.0** (`fp8_e4m3`, `int8_convrot`, `int4_w4a4`)

> **Quick Navigation**: Practical Top-3 Presets → [`docs/acceleration.md`](../../docs/acceleration.md) · Weights & Nodes Catalog → [`RESOURCES.md`](RESOURCES.md) · Chronological Run Index → [`HISTORY.md`](HISTORY.md) · SQLite-Ready Metadata → [`runs_index.jsonl`](runs_index.jsonl) · Raw Outputs → [`runs/`](runs/).

---

## 1. Executive Summary

Across **15 chronological L4 runs** (`20261126_01` → `20261127_07`), we benchmarked built-in ComfyUI and 2026 community acceleration stacks for both **plain T2I inference** and **accelerated XAI (`E1-1` attention heatmaps)**, diagnosed **7 code-level collapse/slowdown mechanisms** in `reference_repo/ComfyUI`, and verified exact patches:

- **Non-Quantized / Non-LoRA Speedups**: Safe `EasyCache` (`t=0.20–0.25, s=0.20–0.25, e=0.80–0.85`) and `Scheduled CFG` (`60–65% → CFG 1`) deliver **`1.7–2.9×`** T2I speedup at near-lossless fidelity (`PSNR 24–48 dB`). Combined with `Selective Block Hooking` (5–6 mid blocks), XAI reaches **`1.99–4.95×`** (`r = 0.955–0.990`).
- **Quantization & Distillation**:
  - **Anima**: `INT8 ConvRot` drops VRAM to **`3.89 GB`** (real `int8_gemm 249 ms`); `Turbo v1.1` runs in **`5.99 s` (`4.30×`)** and XAI in **`3.70 s` (`8.01×, r=0.852`)**.
  - **MOSS-Image**: `SchedCFG 65%` is lossless (`48.20 dB`, XAI `r=1.0000`); `SchedCFG + EasyCache` hits **`18.27 s` (`2.59×, 36.14 dB`)**; `DMD 8-step` hits **`12.15 s`** / XAI **`13.16 s` (`3.84×, r=0.926`)**; `eDMD 4-step` hits **`8.14 s` (`5.80×`)**.
  - **Krea 2 / Krea2-Anime v3.0**: `Krea 2 Turbo` runs in **`16.9–17.4 s` (`~8×` vs Raw)** / XAI **`24.6 s` (`12.22×`)**; `Krea2-Anime v3.0 INT4 W4A4` runs in **`32.0–32.7 s` (`1.3–1.6×` vs FP8)** at **`8.38 GB` VRAM (`-40%`)** with 0 typography typos and XAI **`r = 0.9341, ρ = 0.9682`**.
- **Code-Level Root Causes Proven & Fixed**: `PRIORITIZE_FP16=False` (`59.14 / 58.88 dB`, eliminating `2,831×` `dtype_cast`), `@torch.compiler.disable` (`0` Dynamo recompiles on `TorchCompile + EasyCache`), Krea 2 `SageAttention 2` `TextFusion` bypass + `smooth_k=True` (`+13.82 dB`), `sol-attn` `start_percent=0.35` (100% eliminates extra-limb collapse), `Spectrum d3_safe` (`23.37 dB` vs `d6_aggr` `3.61 dB`), and `EasyCache` outermost before `split` CFG (`issues/006`).

---

## 2. Code-Level Root-Cause Verification & Collapse Fixes

### A. Probe P3-A: `PRIORITIZE_FP16 = True` (`--fast fp16_accumulation`) — **100% Proven**
| Model | Mode | `PRIORITIZE_FP16` | `allow_fp16_acc` | `weight / compute` | Step-0 `x.dtype` | `dtype_cast` / fwd | Denoise (s) | PSNR (dB) |
|---|---|:---:|:---:|---|---|---:|---:|---:|
| **Anima BF16** | `baseline` | `False` | `False` | `bf16 / bf16` | `bfloat16` | `0.07 ms` | `26.22 s` | Ref |
| **Anima BF16** | `stock_fp16acc` | **`True`** | `True` | **`fp16 / fp16`** | **`float32`** (`predict2.py:913`) | **`202.39 ms` (`2,831×`)** | **`33.85 s` (`0.77×`)** | **`21.57 dB`** |
| **Anima BF16** | `patch_no_prioritize_fp16` | **`False`** | **`True`** | **`bf16 / bf16`** | **`bfloat16`** | **`0.08 ms`** | **`26.73 s`** | **`59.14 dB`** |
| **Krea 2 FP8** | `baseline` | `False` | `False` | `fp8 / bf16` | — | `426.1 ms` | `166.98 s` | Ref |
| **Krea 2 FP8** | `stock_fp16acc` | **`True`** | `True` | **`fp8 / fp16`** | — | `426.2 ms` | `143.42 s` | **`26.90 dB`** |
| **Krea 2 FP8** | `patch_no_prioritize_fp16` | **`False`** | **`True`** | **`fp8 / bf16`** | — | `427.1 ms` | **`135.10 s`** | **`58.88 dB`** |

- **Code Path**: `model_management.py:554` sets `PRIORITIZE_FP16=True` → `unet_dtype` (`:1162-1164`) forces FP16 load on Anima (`predict2.py:913-918` upcasts residual to FP32 → 168 full-tensor casts/fwd) and switches Krea 2 `compute_dtype` BF16→FP16 (`:1197-1199`) causing clamp overhead & grain/jacket hallucination under M-Act `~1,920`. See [`issues/004`](../../issues/004_anima_accel_fp16acc_and_compile_easycache.md), [`issues/005`](../../issues/005_krea2_fp16acc_grain_and_artifact.md).

### B. Probe P3-C: Krea 2 + SageAttention 2 — **`+13.82 dB` Recovery Proven**
| Mode | Denoise (s) | PSNR vs SDPA | Code-Level Toggle |
|---|---:|---:|---|
| `baseline_sdpa` | `141.75 s` | Ref | PyTorch SDPA |
| `sage_stock_global` | `146.25 s` | **`20.20 dB`** | Stock ComfyUI (`TextFusion` un-bypassed + `smooth_k=False` at `attention.py:740`) |
| `sage_bypass_text_fusion` | `142.51 s` | `22.70 dB` (`+2.50 dB`) | `TextFusionTransformer` `low_precision_attention=False` |
| **`sage_bypass_and_smooth_k`** | `143.49 s` | **`34.02 dB` (`+13.82 dB`)** | `TextFusion` bypass + **`smooth_k=True`** ([`issues/007`](../../issues/007_krea2_sageattention_nan_collapse.md)) |

### C. Additional Code-Level Mechanisms & Verified Fixes
1. **Probe P3-B (`TorchCompile + EasyCache` Dynamo `recompile_limit(8)`)**: Beyond `nodes_easycache.py:60` (`float↔Tensor`), live L4 tracing found a **2nd recompile trigger** at **`nodes_easycache.py:85`** (`approx_output_change_rates.append(...)` growing a Python list). Decorating `easycache_forward_wrapper` with **`@torch.compiler.disable`** dropped recompiles to **`0`** across 25 steps ([`runs/20261127_05_collapse_fixes_dynamo_sol/`](runs/20261127_05_collapse_fixes_dynamo_sol/)).
2. **Probe P3-D (`EasyCache` `subsample_factor=8` vs `1`)**: Identical `14/30 fwds`, **`23.35 dB` vs `23.35 dB`** (`13.36s` vs `13.15s`) — late-step smoothing is driven by zero-order residual hold (`cache_diff`), not `::8` spatial striding.
3. **`EasyCache` + `split` CFG Wrapper-Order Bug ([`issues/006`](../../issues/006_easycache_split_cfg_wrapper_order_collapse.md))**: Registering `split` CFG outermost resets `skip_current_step=False` on `uncond`, mixing cached `cond` with fresh `uncond` (speckled noise collapse; legacy evidence in [`runs/20261126_07_xai_anima_buggy_issue006/`](runs/20261126_07_xai_anima_buggy_issue006/)). Registering `EasyCache` **outermost (first)** eliminates collapse 100%.
4. **`BlockSparseAttention (sol-attn, τ=1.3)` Extra-Limb Collapse**: `start_percent=0.0` (`11.11 dB`) and `0.25` (`14.79 dB`) hallucinate an extra 3rd arm & 4th hand. Delaying to **`start_percent=0.35, end_percent=0.85`** (`22.36 s`, `17.11 dB`) **100% eliminates extra-limb hallucinations**.
5. **`Spectrum` Chebyshev Forecasting (`d6_aggr` vs `d3_safe`)**:
   - **Anima** (`baseline 24.38s`): `EasyCache 0.20` = `14.30s` (`1.70×`, `21.46 dB`, `HF 0.922`); **`Spectrum d3_safe`** (`degree=3, ridge=0.50, warmup=8, stop=0.80, window=1, blend_w=0.5`) = **`16.88s` (`1.44×`, 8 skips, `23.37 dB`, `HF 0.955` — higher fidelity than `EasyCache`)**; `Spectrum d6_aggr` (`degree=6, ridge=0.05`) = `11.45s`, **`3.61 dB`** (Runge white-out collapse).
   - **MOSS-Image**: Tested Spectrum wrapper recorded `skips=0` (`30.29s`, `99.0 dB` = baseline); `EasyCache 0.20` achieved `19.38s` (`1.66×`, `36.48 dB`).
6. **Hardware / Runtime Limits on L4**:
   - L4 has **58 SMs** (< 80 SM threshold): PyTorch Inductor disables `max_autotune_gemm`; `FlexAttention` hits Triton SMEM OOM (`131KB > 101KB`).
   - **Quantized LoRA Requantization VRAM Spike (`ops.py:1683`)**: LoRAs on `comfy_kitchen` quantized weights trigger `requantize_from_float` (MOSS `5.87 → 10.30 GB`; 13.1 GB `Krea 2 Turbo FP8 + 4-step LoRA` hits 22.03 GiB OOM on L4).

---

## 3. T2I Acceleration Benchmarks & CUDA Kernel Profiling (`torch.profiler`)

### A. Phase 1 & 1.5 Validated Non-Distilled Presets & Multi-Seed Robustness
| Model | Base Weights (L4) | Validated Best Preset | Wall-Clock (vs Unaccel) | Speedup | Multi-Seed/Prompt Fidelity |
|---|---|---|---|---:|---|
| **Anima** | `anima-base-v1.0` (`BF16`) | SDPA · `--fast` OFF · `EasyCache(t=0.20, s=0.20, e=0.80)` · `euler/simple` (30s) | `14.0–15.3s` (vs `25.4s`) | **`1.66–1.81×`** | PSNR `21.2–29.4 dB`, HF `0.83–0.94`, 0 collapses |
| **Anima (Fast)** | `anima-base-v1.0` (`BF16`) | SDPA · `20-step dpmpp_2m/simple` · `EasyCache(0.20, 0.20, 0.85)` | `11.0s` (vs `25.4s`) | **`2.31×`** | Crisp eyes/hands, 0 collapses |
| **MOSS-Image** | `moss_image_sft_int8_convrot` | SDPA · safe `--fast` · `EasyCache(t=0.25, s=0.20, e=0.85)` · `euler/sgm_uniform` (28s) | `18.1–19.7s` (vs `38.5s`) | **`1.95–2.12×`** | Photo PSNR `28.5–30.7 dB`, HF `0.94–1.01` |
| **MOSS (Quality)** | `moss_image_sft_int8_convrot` | `25-step er_sde/simple`, CFG `2.5` · safe `--fast` · `EasyCache(0.20, 0.20, 0.85)` | `23.2s` (vs `38.5s`) | **`1.66×`** | Sharpest knit/hair texture (`HF 1.13`) |
| **Krea 2 Raw** | `krea2_raw_fp8_scaled` | `--use-ck-attention` · safe `--fast` · `EasyCache(t=0.20, s=0.25, e=0.80)` · `euler/simple` (32s) | `61.6–79.0s` (vs `137–177s`) | **`1.74–2.87×`** | PSNR `24.7–27.2 dB`, HF `0.92–0.97` |
| **Krea 2 Turbo** | `krea2_turbo_fp8_scaled` | `--use-ck-attention` · safe `--fast` · `8-step, CFG 1.0, euler/simple` | `16.9–17.4s` (vs `135s` Raw) | **`~8×`** | High-detail photos & 2D `emilia` (+EC `33.94 dB`) |

### B. Phase 3 Extended Options & `torch.profiler` Kernel Breakdown
| Model | Mode | Denoise (s) | Speedup | VRAM | `int8_gemm` / `fp8_gemm` | `bf16_gemm` | PSNR (dB) |
|---|---|---:|---:|---:|---:|---:|---:|
| **Anima** (`civitai_02`) | `bf16_baseline` (30s, CFG 4) | `25.75 s` | `1.00×` | `5.37 GB` | `0.0 / 0.0 ms` | `841.3 ms` | Ref |
| **Anima** | `cfg_scheduled_65` / `60` | `20.00 / 19.33 s` | `1.25 / 1.30×` | `5.37 GB` | `0.0 ms` | `909 / 927 ms` | **`25.56 / 24.86 dB`** |
| **Anima** | **`cfg_sched60_ec20`** | **`12.96 s`** | **`1.93×`** | `5.37 GB` | — | — | `19.50 dB` |
| **Anima** | **`int8_convrot_sdpa` / `_ck`** | `22.56 / 21.90 s` | `1.14 / 1.18×` | **`3.89 GB`** | **`255 / 249 ms` (INT8!)** | `280 / 275 ms` (`-67%`) | `17.75 / 16.84 dB` |
| **Anima** | **`turbo_v11_8step`** / **`4step_lora`** | **`5.99 / 5.18 s`** | **`4.30 / 4.97×`** | `5.37 / 8.90 GB` | `0.0 ms` | `433 / 431 ms` | *(distilled)* |
| **MOSS-Image** (`civitai_01`) | `baseline` (`sft_int8_convrot`, 28s) | `47.24 s` | `1.00×` | `5.87 GB` | **`246.0 ms` (INT8!)** | `297.0 ms` | Ref |
| **MOSS-Image** | **`schedcfg65`** (`65%→CFG 1`) | **`38.62 s`** | **`1.22×`** | `5.87 GB` | `252.5 ms` | `304.8 ms` | **`48.20 dB`** |
| **MOSS-Image** | **`schedcfg65_ec20`** | **`18.27 s`** | **`2.59×`** | `5.87 GB` | `256.2 ms` | `309.1 ms` | **`36.14 dB`** |
| **MOSS-Image** | **`dmd_8step`** / **`edmd_4step`** | **`12.15 / 8.14 s`** | **`3.89 / 5.80×`** | `10.30 GB` | `259 / 261 ms` | `312 / 314 ms` | *(distilled)* |
| **Krea2-Anime v3.0** (8s) | `fp8_e4m3` (13.49 GB) | `42.3–52.9 s` | `1.00×` | `14.04 GB` | `fp8 1676 ms` | — | Ref |
| **Krea2-Anime v3.0** (8s) | `int8_convrot` (13.49 GB) | `52.6–53.1 s` | `0.80–1.01×` | `14.04 GB` | `int8 834 ms` | `752 ms` | — |
| **Krea2-Anime v3.0** (8s) | **`int4_w4a4`** (**7.42 GB**) | **`32.0–32.7 s`** | **`1.3–1.6×`** | **`8.38 GB`** (`-40%`) | `W4A4 cutlass` | `358 ms` | 0 typos on `"LOOK HOW FUN IT IS!"` |

---

## 4. Accelerated XAI (`E1-1` Heatmap) Speed & Fidelity

| Model | XAI Mode | Wall (s) | Speedup | VRAM | Pearson `r` | Spearman `ρ` | Top-15% IoU | Run Evidence |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **Anima** | `bf16_ref_full` | `28.3–29.6 s` | `1.00×` | `5.37 GB` | `1.0000` | `1.0000` | `1.0000` | [`20261126_08`](runs/20261126_08_xai_3model_ec_selective/) |
| **Anima** | **`best_ec_full`** / **`best_ec_sel6`** | `16.2 / 14.2 s` | `1.75 / 1.99×` | `5.37 GB` | `0.988 / 0.970` | — | `0.873 / 0.816` | [`20261126_08`](runs/20261126_08_xai_3model_ec_selective/) |
| **Anima** | **`int8_convrot_sched65_sel6`** | **`17.77 s`** | **`1.67×`** | **`3.89 GB`** | **`0.9548`** | `0.9469` | `0.7474` | [`20261127_02`](runs/20261127_02_anima_quant_turbo_xai/) |
| **Anima** | **`turbo_v11_8step_sel6`** | **`3.70 s`** | **`8.01×`** | `5.37 GB` | **`0.8516`** | `0.8602` | `0.5568` | [`20261127_02`](runs/20261127_02_anima_quant_turbo_xai/) |
| **MOSS-Image** | `ref_full` / `ref_sel6` | `145.7 / 50.5 s` | `1.00×` | `6.51 GB` | `1.0000` | `1.0000` | `1.0000` | [`20261126_08`](runs/20261126_08_xai_3model_ec_selective/) · [`20261127_04`](runs/20261127_04_moss_dmd_schedcfg_xai/) |
| **MOSS-Image** | **`best_ec_sel6`** *(2D: use 8b)* | **`29.4 s`** | **`4.95×`** | `6.51 GB` | **`0.958–0.976`** | — | `0.776–0.823` | [`20261126_08`](runs/20261126_08_xai_3model_ec_selective/) |
| **MOSS-Image** | **`sched65_sel6`** | **`44.32 s`** | **`1.14×`** | `6.51 GB` | **`1.0000`** | **`1.0000`** | **`0.9976`** | [`20261127_04`](runs/20261127_04_moss_dmd_schedcfg_xai/) |
| **MOSS-Image** | **`dmd8_sel6`** | **`13.16 s`** | **`3.84×`** | `10.95 GB` | **`0.9261`** | **`0.9524`** | `0.6823` | [`20261127_04`](runs/20261127_04_moss_dmd_schedcfg_xai/) |
| **Krea 2 Raw** | `best_ec_sel5` (FP8) | **`94.8 s`** | **`3.68×`** | `15.02 GB` | **`0.967–0.990`** | — | `0.836–0.854` | [`20261126_08`](runs/20261126_08_xai_3model_ec_selective/) |
| **Krea 2 Raw** | `int8_convrot_sel5` | `172.12 s` | `0.95×` | `15.58 GB` | **`0.9991`** | **`0.9991`** | **`0.9658`** | [`20261127_03`](runs/20261127_03_krea2_probes_quant_xai/) |
| **Krea 2 Turbo** | **`turbo_8step_sel5`** | **`24.6 s`** | **`12.22×`** | `14.0 GB` | *(pinpoint verified)* | — | — | [`20261126_08`](runs/20261126_08_xai_3model_ec_selective/) |
| **Krea2-Anime v3.0** | **`int4_w4a4_sel5`** | **`38.9 s`** | **`1.27×`** | **`9.63 GB`** | **`0.9341`** | **`0.9682`** | `0.7059` | [`20261127_07`](runs/20261127_07_krea2_anime_v3_quant_xai/) |

---

## 5. GPU Architecture & XAI 4-Axis Compatibility Matrices

### A. GPU Architecture Compatibility
| Feature / Technique | **Colab L4** (`sm_89`, 58 SMs) | **Ampere** (`sm_86`, 3060/3090) | **High-SM Ada / Hopper** (`sm_89`/`90`, ≥80 SMs) | **Blackwell** (`sm_120`, 5070–5090) |
|---|:---:|:---:|:---:|:---:|
| `INT8 ConvRot` (`comfy-kitchen`, `cu130`) | **Yes (Fast)** | **Yes (Best)** | **Yes (Fast)** | **Yes (Fast)** |
| `FP8 Scaled` (`e4m3fn`) | **Yes (Native)** | Slow (dequant) | **Yes (Native)** | **Yes (Native)** |
| `INT4 W4A4` (`ConvRot W4A4` / `Q4_CR`) | **Yes** (`Krea2-Anime 1.3–1.6×`) | **Yes** | **Yes** | **Yes** |
| `NVFP4` / `MXFP8` | No | No | No | **Yes (Native)** |
| `SageAttention 2 / 2++` | **Yes** *(Krea 2 needs patch)* | Yes | **Yes** | Nightly CU12.8/13.0 |
| `SageAttention 3` (FP4) | No | No | No | **Yes (3–5× attn)** |
| `FlexAttention` | **No** (SMEM OOM) | No | **Yes (A100/H100)** | Yes |
| `torch.compile(max-autotune)` + `cudagraphs` | **No (`58 < 80 SMs`)** | 3090 (82 SMs) Yes | **Yes (≥80 SMs)** | **Yes** |
| `EasyCache` / `Spectrum` / `SchedCFG` | **Yes** | **Yes** | **Yes** | **Yes** |

### B. XAI 4-Axis Hook Compatibility
| Acceleration Method | `E1` Heatmap | `E2` Layer Sel. | `E3` Tag Influence | `E4` MI Verif. | Notes |
|---|:---:|:---:|:---:|:---:|---|
| **Quantization (`int8_convrot`, `fp8_scaled`, `int4_w4a4`)** | Full | Full | Full | Full | Verified on L4 (`exact_noop = 0.0`); `Cont^(l)` calls `wo(x)`. |
| **Turbo / DMD / eDMD / Acc LoRAs (4–16 steps)** | Full | Full | Full | Full | Standard DiT forward; directly scales XAI wall-clock. |
| **Scheduled CFG (`CFG → 1` after 60–65%)** | Full | Full | Full | Full | Drops `uncond` only; `cond` hooks run on every step untouched. |
| **`--fast` safe subset (`cublas_ops`, `fp8_matrix_mult`, `autotune`)** | Full | Full | Full | Full | Never include `fp16_accumulation` without `PRIORITIZE_FP16=False`. |
| **`--use-ck-attention` / `Flash` / patched `SageAttention`** | Partial | Partial | Partial | Partial | Hooked blocks fall back to chunked SDPA; unhooked blocks stay fast. |
| **`EasyCache` / `Spectrum` / `TeaCache`** | Partial | Partial | Partial | Partial | Register cache **outermost** before XAI `split` CFG (`issues/006`). |
| **`TorchCompileModel`** | Benchmark | Benchmark | Benchmark | Benchmark | Requires `@torch.compiler.disable` on `EasyCache` wrapper (`issues/004`). |

---

## 6. Pending Tier 2 (Blackwell `sm_120`, ≥80 SMs)
Deferred until Blackwell hardware is available (`plan/accel_phase3_extended.md`): `SageAttention 3` (FP4), `NVFP4`, `MXFP8`, `FlexAttention`, and `torch.compile(max-autotune)` + `cudagraphs`.
