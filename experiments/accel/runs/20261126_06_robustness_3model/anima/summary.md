# Phase 1 Acceleration Summary — `anima_robustness`

| Preset | Stage | Status | Wall (s) | Speedup | s/fwd | VRAM (GB) | PSNR (dB) | HF Ratio | Skipped | Notes |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| `kanao_s101_cfg35_sh25_base` | baseline | **OK** | 24.27 | - | 0.791 | 15.75 | - | - | 0 | Base CFG=3.5 shift=2.5 |
| `kanao_s101_cfg35_sh25_best` | fix | **OK** | 14.99 | - | 0.811 | 15.75 | 21.17 | 0.938 | 0 | Best EasyCache CFG=3.5 shift=2.5 |
| `kanao_s202_cfg50_sh35_base` | baseline | **OK** | 24.42 | - | 0.801 | 15.75 | - | - | 0 | Base CFG=5.0 shift=3.5 |
| `kanao_s202_cfg50_sh35_best` | fix | **OK** | 15.35 | - | 0.787 | 15.75 | 21.90 | 0.904 | 0 | Best EasyCache CFG=5.0 shift=3.5 |
| `emilia_s101_cfg40_sh30_base` | baseline | **OK** | 23.99 | - | 0.786 | 15.75 | - | - | 0 | Base CFG=4.0 shift=3.0 |
| `emilia_s101_cfg40_sh30_best` | fix | **OK** | 14.69 | - | 0.794 | 15.75 | 22.10 | 0.834 | 0 | Best EasyCache CFG=4.0 shift=3.0 |
