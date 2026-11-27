# Phase 1 Acceleration Summary — `krea2`

| Preset | Stage | Status | Wall (s) | Speedup | s/fwd | VRAM (GB) | PSNR (dB) | HF Ratio | Skipped | Notes |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline` | baseline | **OK** | 176.57 | 1.00x | 2.104 | 13.43 | - | - | 0 | Unaccelerated L4 quantized baseline (SDPA) |
| `attn_ck` | single | **OK** | 172.34 | 1.02x | 2.035 | 13.43 | 29.90 | 1.002 | 0 | Comfy Kitchen attention only |
| `fast_all` | single | **OK** | 147.91 | 1.19x | 2.213 | 13.43 | 26.90 | 1.040 | 0 | Full --fast (all 4 features) |
| `fast_fp16acc` | single | **OK** | 148.13 | 1.19x | 2.222 | 13.43 | 26.90 | 1.040 | 0 | --fast fp16_accumulation only (graininess suspect) |
| `fast_no_fp16acc` | fix | **OK** | 140.36 | 1.26x | 2.101 | 13.43 | 58.88 | 0.980 | 0 | --fast without fp16_accumulation |
| `ec_t20_s00` | single | **OK** | 42.78 | 4.13x | 2.091 | 13.43 | 14.77 | 0.942 | 0 | EasyCache from step 0 (early-drift risk) |
| `ec_t20_s15` | single | **OK** | 55.30 | 3.19x | 2.092 | 13.43 | 25.42 | 0.868 | 0 | EasyCache default start=0.15 |
| `ec_t20_s25` | fix | **OK** | 63.61 | 2.78x | 2.091 | 13.43 | 26.83 | 0.912 | 0 | EasyCache delayed start=0.25 (protects early layout) |
| `sampler_ersde` | single | **OK** | 134.88 | 1.31x | 2.093 | 13.43 | 7.58 | 1.133 | 0 | er_sde/simple sampler |
| `sampler_dpmpp2m` | single | **OK** | 134.90 | 1.31x | 2.094 | 13.43 | 19.00 | 1.014 | 0 | dpmpp_2m/simple sampler |
| `combo_ck_fast_all` | combo | **OK** | 142.71 | 1.24x | 2.138 | 13.43 | 23.01 | 1.068 | 0 | Step 0 graininess suspect: ck_attention + full --fast |
| `combo_ec_ersde` | combo | **OK** | 92.99 | 1.90x | 2.091 | 13.43 | 7.72 | 1.088 | 0 | Stochastic er_sde + EasyCache interaction test |
| `fix_ck_fast_safe` | fix | **OK** | 135.93 | 1.30x | 2.031 | 13.43 | 29.90 | 1.002 | 0 | ck_attention + --fast excluding fp16_accumulation |
| `fix_full_safe_ec25` | fix | **OK** | 61.60 | 2.87x | 2.020 | 13.43 | 27.16 | 0.917 | 0 | Candidate Best Stack: ck + safe --fast + delayed EasyCache(s=0.25) + euler |
| `krea2_turbo_8step` | fewstep | **OK** | 59.57 | 2.96x | 2.087 | 13.43 | 9.63 | 0.661 | 0 | Krea 2 Turbo FP8 (8 steps, CFG 1.0, SDPA) |
| `krea2_turbo_8step_ck_safe` | fewstep | **OK** | 62.67 | 2.82x | 2.012 | 13.43 | 9.48 | 0.760 | 0 | Krea 2 Turbo FP8 + ck_attention + safe --fast |
