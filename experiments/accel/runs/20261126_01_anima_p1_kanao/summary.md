# Phase 1 Acceleration Summary — `anima`

| Preset | Stage | Status | Wall (s) | Speedup | s/fwd | VRAM (GB) | PSNR (dB) | HF Ratio | Skipped | Notes |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline` | baseline | **OK** | 25.39 | 1.00x | 0.815 | 5.37 | - | - | 0 | Unaccelerated L4 quantized baseline (SDPA) |
| `attn_ck` | single | **OK** | 26.50 | 0.96x | 0.795 | 5.37 | 15.05 | 1.003 | 0 | Comfy Kitchen attention only |
| `fast_all` | single | **OK** | 34.95 | 0.73x | 1.076 | 5.37 | 21.57 | 1.033 | 0 | Full --fast (all 4 features) |
| `fast_cublas` | single | **OK** | 27.62 | 0.92x | 0.826 | 5.37 | 59.14 | 0.991 | 0 | --fast cublas_ops only |
| `fast_fp8mm` | single | **OK** | 27.74 | 0.92x | 0.830 | 5.37 | 59.14 | 0.991 | 0 | --fast fp8_matrix_mult only |
| `fast_fp16acc` | single | **OK** | 34.64 | 0.73x | 1.069 | 5.37 | 21.57 | 1.033 | 0 | --fast fp16_accumulation only (graininess suspect) |
| `fast_no_fp16acc` | fix | **OK** | 27.46 | 0.92x | 0.827 | 5.37 | 59.14 | 0.991 | 0 | --fast without fp16_accumulation |
| `ec_t20_s00` | single | **OK** | 11.12 | 2.28x | 0.825 | 5.37 | 18.20 | 0.901 | 0 | EasyCache from step 0 (early-drift risk) |
| `ec_t20_s15` | single | **OK** | 11.80 | 2.15x | 0.815 | 5.37 | 23.35 | 0.911 | 0 | EasyCache default start=0.15 |
| `ec_t20_s25` | fix | **OK** | 13.44 | 1.89x | 0.816 | 5.37 | 25.14 | 0.891 | 0 | EasyCache delayed start=0.25 (protects early layout) |
| `ec_t30_s15` | single | **OK** | 10.89 | 2.33x | 0.808 | 5.37 | 22.67 | 0.889 | 0 | Aggressive EasyCache thresh=0.30 |
| `sampler_ersde` | single | **OK** | 24.67 | 1.03x | 0.808 | 5.37 | 10.25 | 1.037 | 0 | er_sde/simple sampler |
| `sampler_dpmpp2m` | single | **OK** | 24.86 | 1.02x | 0.815 | 5.37 | 20.54 | 1.044 | 0 | dpmpp_2m/simple sampler |
| `combo_ck_fast_all` | combo | **OK** | 34.03 | 0.75x | 1.044 | 5.37 | 21.00 | 1.028 | 0 | Step 0 graininess suspect: ck_attention + full --fast |
| `combo_ec_ersde` | combo | **OK** | 17.55 | 1.45x | 0.816 | 5.37 | 10.43 | 0.931 | 0 | Stochastic er_sde + EasyCache interaction test |
| `combo_ck_fast_all_ec` | combo | **OK** | 15.98 | 1.59x | 1.039 | 5.37 | 21.20 | 0.917 | 0 | Aggressive 3-way stack: ck + fast(all) + EasyCache |
| `fix_ck_fast_safe` | fix | **OK** | 26.54 | 0.96x | 0.796 | 5.37 | 15.05 | 1.003 | 0 | ck_attention + --fast excluding fp16_accumulation |
| `fix_ck_fast_all_upcast` | fix | **OK** | 33.84 | 0.75x | 1.042 | 5.37 | 21.00 | 1.028 | 0 | ck_attention + fast(all) + --force-upcast-attention |
| `fix_full_safe_ec25` | fix | **OK** | 13.26 | 1.91x | 0.804 | 5.37 | 15.65 | 0.917 | 0 | Candidate Best Stack: ck + safe --fast + delayed EasyCache(s=0.25) + euler |
