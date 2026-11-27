# Phase 3 Extended Acceleration Summary — `anima` (tier: `l4`)

| Preset | Track | Status | Denoise (s) | Speedup | s/fwd | Skip % | VRAM (GB) | PSNR (dB) | HF Ratio | Notes |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline` | baseline | **OK** | 25.04 | 1.00x | 0.835 | 50.0% | 5.37 | - | - | Anima BF16 SDPA baseline |
| `cfg_scheduled_60` | cfg_attn | **OK** | 19.33 | 1.30x | 0.644 | 50.0% | 5.37 | 24.86 | 0.984 | Scheduled CFG: drop CFG->1.0 at 60% |
| `cfg_scheduled_65` | cfg_attn | **OK** | 20.00 | 1.25x | 0.667 | 50.0% | 5.37 | 25.56 | 0.989 | Scheduled CFG: drop CFG->1.0 at 65% |
| `cfg_sched60_ec20` | cfg_attn | **OK** | 12.96 | 1.93x | 0.589 | 63.3% | 5.37 | 19.50 | 0.908 | Scheduled CFG (60%) + EasyCache (t=0.20) |
| `attn_block_sparse_sol13` | cfg_attn | **OK** | 22.15 | 1.13x | 0.739 | 50.0% | 5.37 | 15.40 | 1.073 | ComfyUI BlockSparseAttention (sol-attn, tau=1.3) |
