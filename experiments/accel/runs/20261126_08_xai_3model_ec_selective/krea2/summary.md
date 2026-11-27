# Phase 2 Accelerated XAI Heatmap Summary — `krea2`

| Prompt | Seed | Mode | Wall (s) | Speedup | Mean Pearson r | Mean Spearman ρ | Mean Top-15% IoU |
|---|---:|---|---:|---:|---:|---:|---:|
| `krea2_idol_cafe` | 42 | `ref_full` | 349.1 | 1.00x | 1.0000 | 1.0000 | 1.0000 |
| `krea2_idol_cafe` | 42 | `best_ec_full` | 178.4 | 1.96x | 0.9954 | 0.9905 | 0.9289 |
| `krea2_idol_cafe` | 42 | `best_ec_selective` | 94.8 | 3.68x | 0.9666 | 0.9325 | 0.8358 |
| `krea2_idol_neon` | 42 | `ref_full` | 300.0 | 1.00x | 1.0000 | 1.0000 | 1.0000 |
| `krea2_idol_neon` | 42 | `best_ec_full` | 178.2 | 1.68x | 0.9935 | 0.9900 | 0.9061 |
| `krea2_idol_neon` | 42 | `best_ec_selective` | 94.8 | 3.17x | 0.9901 | 0.9636 | 0.8544 |
| `krea2_idol_cafe` | 42 | `turbo_8step_selective` | 65.0 | 5.37x | 0.6455 | 0.6642 | 0.4690 |
| `krea2_idol_neon` | 42 | `turbo_8step_selective` | 24.6 | 12.22x | 0.6849 | 0.8693 | 0.5368 |
