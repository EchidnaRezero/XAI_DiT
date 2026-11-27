# Phase 1 Acceleration Summary — `moss_image`

| Preset | Stage | Status | Wall (s) | Speedup | s/fwd | VRAM (GB) | PSNR (dB) | HF Ratio | Skipped | Notes |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline` | baseline | **OK** | 38.45 | 1.00x | 0.574 | 5.87 | - | - | 0 | Unaccelerated L4 quantized baseline (SDPA) |
| `attn_ck` | single | **OK** | 35.50 | 1.08x | 0.575 | 5.87 | 25.88 | 0.957 | 0 | Comfy Kitchen attention only |
| `fast_all` | single | **OK** | 34.79 | 1.11x | 0.577 | 5.87 | 58.66 | 0.988 | 0 | Full --fast (all 4 features) |
| `fast_cublas` | single | **OK** | 34.70 | 1.11x | 0.576 | 5.87 | 58.92 | 0.988 | 0 | --fast cublas_ops only |
| `fast_fp8mm` | single | **OK** | 34.75 | 1.11x | 0.577 | 5.87 | 58.92 | 0.988 | 0 | --fast fp8_matrix_mult only |
| `fast_fp16acc` | single | **OK** | 34.66 | 1.11x | 0.576 | 5.87 | 58.66 | 0.988 | 0 | --fast fp16_accumulation only (graininess suspect) |
| `fast_no_fp16acc` | fix | **OK** | 34.71 | 1.11x | 0.576 | 5.87 | 58.92 | 0.988 | 0 | --fast without fp16_accumulation |
| `ec_t20_s00` | single | **OK** | 14.10 | 2.73x | 0.569 | 5.87 | 22.12 | 0.787 | 0 | EasyCache from step 0 (early-drift risk) |
| `ec_t20_s15` | single | **OK** | 17.37 | 2.21x | 0.565 | 5.87 | 25.37 | 0.843 | 0 | EasyCache default start=0.15 |
| `ec_t20_s25` | fix | **OK** | 19.60 | 1.96x | 0.564 | 5.87 | 27.66 | 0.890 | 0 | EasyCache delayed start=0.25 (protects early layout) |
| `ec_t30_s15` | single | **OK** | 16.22 | 2.37x | 0.564 | 5.87 | 24.98 | 0.815 | 0 | Aggressive EasyCache thresh=0.30 |
| `sampler_ersde` | single | **OK** | 32.18 | 1.19x | 0.567 | 5.87 | 13.51 | 0.970 | 0 | er_sde/simple sampler |
| `sampler_dpmpp2m` | single | **GRAIN?** | 32.03 | 1.20x | 0.564 | 5.87 | 19.24 | 1.164 | 0 | dpmpp_2m/simple sampler |
| `combo_ck_fast_all` | combo | **OK** | 34.71 | 1.11x | 0.576 | 5.87 | 25.87 | 0.958 | 0 | Step 0 graininess suspect: ck_attention + full --fast |
| `combo_ec_ersde` | combo | **OK** | 24.18 | 1.59x | 0.565 | 5.87 | 13.64 | 0.917 | 0 | Stochastic er_sde + EasyCache interaction test |
| `combo_ck_fast_all_ec` | combo | **OK** | 17.40 | 2.21x | 0.566 | 5.87 | 24.73 | 0.847 | 0 | Aggressive 3-way stack: ck + fast(all) + EasyCache |
| `fix_ck_fast_safe` | fix | **OK** | 34.63 | 1.11x | 0.574 | 5.87 | 25.88 | 0.957 | 0 | ck_attention + --fast excluding fp16_accumulation |
| `fix_ck_fast_all_upcast` | fix | **OK** | 34.68 | 1.11x | 0.575 | 5.87 | 25.87 | 0.958 | 0 | ck_attention + fast(all) + --force-upcast-attention |
| `fix_full_safe_ec25` | fix | **OK** | 18.66 | 2.06x | 0.569 | 5.87 | 24.94 | 0.851 | 0 | Candidate Best Stack: ck + safe --fast + delayed EasyCache(s=0.25) + euler |
