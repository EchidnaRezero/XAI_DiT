# Phase 1 Acceleration Summary — `moss_image_round2`

| Preset | Stage | Status | Wall (s) | Speedup | s/fwd | VRAM (GB) | PSNR (dB) | HF Ratio | Skipped | Notes |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline` | baseline | **OK** | 34.31 | 1.00x | 0.572 | 5.89 | - | - | 0 | Photo baseline (28step euler/sgm_uniform + safe --fast) |
| `photo_ec_t20_s20_e80` | fix | **OK** | 20.97 | 1.64x | 0.570 | 5.89 | 30.81 | 0.970 | 0 | Protect early 20% + late 20% skin/hair |
| `photo_ec_t25_s20_e85` | fix | **OK** | 19.70 | 1.74x | 0.567 | 5.89 | 30.70 | 0.972 | 0 | Balanced EasyCache t=0.25 s=0.20 e=0.85 |
| `photo_ec_t45_s00_e98` | combo | **OK** | 10.59 | 3.24x | 0.564 | 5.89 | 22.54 | 0.915 | 0 | Extreme EasyCache plastic-skin/collapse test |
| `photo_template_ersde` | single | **OK** | 28.84 | 1.19x | 0.568 | 5.89 | 8.83 | 1.138 | 0 | ComfyUI official MOSS template (25step CFG 2.5 er_sde/simple) |
| `photo_template_ersde_ec` | fix | **OK** | 23.18 | 1.48x | 0.568 | 5.89 | 8.90 | 1.130 | 0 | MOSS template er_sde + delayed EasyCache |
| `photo_dpmpp2m_simple` | single | **OK** | 32.28 | 1.06x | 0.569 | 5.89 | 17.34 | 1.131 | 0 | Check dpmpp_2m HF=1.16 grain on photorealistic skin |
| `photo_s20_euler_ec20` | fix | **OK** | 16.31 | 2.10x | 0.568 | 5.89 | 24.03 | 0.938 | 0 | 20step euler/sgm_uniform + EasyCache(s=0.20,e=0.80) |
| `emilia_ec_t20_s20_e80` | fix | **OK** | 20.34 | 1.69x | 0.553 | 5.89 | 25.75 | 0.862 | 0 | 2D Emilia EasyCache end=0.80 lineart check |
