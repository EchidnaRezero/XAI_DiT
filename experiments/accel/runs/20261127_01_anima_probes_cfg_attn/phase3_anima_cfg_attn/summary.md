# Phase 3 Extended Acceleration Summary — `anima` (tier: `l4`)

| Preset | Track | Status | Denoise (s) | Speedup | s/fwd | Skip % | VRAM (GB) | PSNR (dB) | HF Ratio | Notes |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline` | baseline | **OK** | 25.01 | 1.00x | 0.834 | 50.0% | 5.37 | - | - | Anima BF16 SDPA baseline |
| `cfg_scheduled_60` | cfg_attn | **OK** | 20.20 | 1.24x | 0.673 | 50.0% | 5.37 | 24.86 | 0.984 | Scheduled CFG: drop CFG->1.0 at 60% |
| `cfg_scheduled_65` | cfg_attn | **OK** | 20.49 | 1.22x | 0.683 | 50.0% | 5.37 | 25.56 | 0.989 | Scheduled CFG: drop CFG->1.0 at 65% |
| `cfg_sched60_ec20` | cfg_attn | **ERROR** | - | - | - | - | - | - | - | missing result json |
| `attn_block_sparse_sol13` | cfg_attn | **ERROR** | - | - | - | - | - | - | - | missing result json |
| `attn_sage2_kj` | cfg_attn | **SKIPPED** | - | - | - | - | - | - | - | custom node 'kjnodes' not installed in ComfyUI/custom_nodes |
