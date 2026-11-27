# Phase 1 Acceleration Summary — `krea2_robustness`

| Preset | Stage | Status | Wall (s) | Speedup | s/fwd | VRAM (GB) | PSNR (dB) | HF Ratio | Skipped | Notes |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| `raw_neon_base_s42` | baseline | **OK** | 137.08 | - | 2.070 | 18.82 | - | - | 0 | Raw neon baseline (ck+safe_fast) |
| `raw_neon_best_ec_s42` | fix | **OK** | 78.95 | - | 2.061 | 18.82 | 24.66 | 0.973 | 0 | Raw Best Preset (e=0.80) on neon night |
| `turbo_cafe_s42_steady` | fewstep | **OK** | 48.12 | - | 2.062 | 18.82 | - | - | 0 | Turbo 8-step steady-state timing check |
| `turbo_neon_s101` | fewstep | **OK** | 17.34 | - | 2.108 | 18.82 | - | - | 0 | Turbo 8-step neon night seed=101 |
| `turbo_neon_s202` | fewstep | **OK** | 17.41 | - | 2.117 | 18.82 | - | - | 0 | Turbo 8-step neon night seed=202 |
| `turbo_emilia_s42` | fewstep | **OK** | 16.92 | - | 2.057 | 18.82 | - | - | 0 | Turbo 8-step verified 2D emilia |
| `turbo_neon_ec_test` | fewstep | **OK** | 14.89 | - | 2.060 | 18.82 | 33.94 | 0.967 | 0 | Turbo 8-step + EasyCache collapse check |
