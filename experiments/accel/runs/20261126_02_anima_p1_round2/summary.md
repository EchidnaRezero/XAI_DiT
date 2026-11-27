# Phase 1 Acceleration Summary — `anima_round2`

| Preset | Stage | Status | Wall (s) | Speedup | s/fwd | VRAM (GB) | PSNR (dB) | HF Ratio | Skipped | Notes |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline` | baseline | **OK** | 25.39 | 1.00x | 0.815 | 5.37 | - | - | 0 | Unaccelerated L4 quantized baseline (SDPA) |
| `ec_t20_s20_e80` | fix | **OK** | 15.31 | 1.66x | 0.797 | 5.39 | 24.11 | 0.942 | 0 | Protect early 20% + late 20% lineart |
| `ec_t25_s20_e85` | fix | **OK** | 14.95 | 1.70x | 0.809 | 5.39 | 24.11 | 0.942 | 0 | Balanced EasyCache t=0.25 s=0.20 e=0.85 |
| `ec_t45_s00_e98` | combo | **OK** | 7.75 | 3.28x | 0.817 | 5.39 | 16.17 | 0.901 | 0 | Extreme EasyCache collapse boundary test |
| `ec_t20_s20_e85_ersde` | fix | **OK** | 19.56 | 1.30x | 0.832 | 5.39 | 10.36 | 0.965 | 0 | Delayed EasyCache + er_sde |
| `ec_t20_s20_e85_dpmpp2m` | fix | **OK** | 15.16 | 1.68x | 0.820 | 5.39 | 20.53 | 0.950 | 0 | Delayed EasyCache + dpmpp_2m |
| `s20_euler_simple` | single | **OK** | 16.60 | 1.53x | 0.812 | 5.39 | 17.81 | 1.015 | 0 | 20 steps euler/simple |
| `s20_dpmpp2m_simple` | single | **OK** | 16.55 | 1.53x | 0.809 | 5.39 | 15.27 | 0.987 | 0 | 20 steps dpmpp_2m/simple |
| `s20_dpmpp2m_beta` | single | **OK** | 16.65 | 1.52x | 0.814 | 5.39 | 14.38 | 0.984 | 0 | 20 steps dpmpp_2m/beta |
| `s20_ersde_simple` | single | **OK** | 16.73 | 1.52x | 0.817 | 5.39 | 10.46 | 0.948 | 0 | 20 steps er_sde/simple |
| `s20_dpmpp2m_ec20` | fix | **OK** | 11.01 | 2.31x | 0.818 | 5.39 | 15.82 | 0.906 | 0 | 20 steps dpmpp_2m + EasyCache |
| `compile_inductor` | single | **OK** | 23.48 | 1.08x | 0.769 | 5.40 | 23.45 | 1.007 | 0 | TorchCompileModel(inductor) rep2 steady-state |
| `compile_inductor_ec20` | fix | **OK** | 16.00 | 1.59x | 0.743 | 5.40 | 22.51 | 0.927 | 0 | TorchCompile(inductor) + EasyCache |
| `emilia_baseline` | baseline | **OK** | 24.70 | 1.03x | 0.810 | 5.39 | - | - | 0 | 2nd prompt baseline |
| `emilia_ec_t20_s20_e80` | fix | **OK** | 14.00 | 1.81x | 0.801 | 5.39 | 29.43 | 0.876 | 0 | 2nd prompt EasyCache fix |
