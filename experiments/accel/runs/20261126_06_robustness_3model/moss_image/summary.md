# Phase 1 Acceleration Summary — `moss_robustness`

| Preset | Stage | Status | Wall (s) | Speedup | s/fwd | VRAM (GB) | PSNR (dB) | HF Ratio | Skipped | Notes |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| `hoshino_s101_cfg30_base` | baseline | **OK** | 50.97 | - | 0.550 | 11.03 | - | - | 0 | Base CFG=3.0 |
| `hoshino_s101_cfg30_best` | fix | **OK** | 19.34 | - | 0.555 | 11.03 | 22.53 | 0.936 | 0 | Best EasyCache CFG=3.0 |
| `photo_s101_cfg25_base` | baseline | **OK** | 31.24 | - | 0.550 | 11.03 | - | - | 0 | Base CFG=2.5 |
| `photo_s101_cfg25_best` | fix | **OK** | 19.10 | - | 0.549 | 11.03 | 30.52 | 1.008 | 0 | Best EasyCache CFG=2.5 |
| `photo_s202_cfg35_base` | baseline | **OK** | 31.37 | - | 0.552 | 11.03 | - | - | 0 | Base CFG=3.5 |
| `photo_s202_cfg35_best` | fix | **OK** | 18.08 | - | 0.551 | 11.03 | 28.54 | 0.976 | 0 | Best EasyCache CFG=3.5 |
