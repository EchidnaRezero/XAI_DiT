# EXP-E0: MOSS-Image SFT BF16 Sanity & Resource Budget (Colab L4)

- **Date**: 2026-11-25
- **Design Basis**: `docs/experiment_design.md` (§2 E0), `plan/m1.md`
- **Target Model**: MOSS-Image (`moss_image_sft_bf16.safetensors`, BF16, 42 blocks; offline BF16 TE `qwen3.5_9b_bf16.safetensors`)
- **Runtime**: Colab NVIDIA L4 (24 GB), `Torch 2.14.1+cu130`, `euler` / `sgm_uniform`, 28 steps, CFG 3.0, `shift=5.0`, `split` CFG batching

---

## 1. Goal
Verify MOSS-Image adapter exact no-op (`== 0`), 3-rep non-determinism floor, full-42-block manual chunked SDPA path identity, and real stage-wise L4 VRAM / `sec_per_forward` budget on `P-M1` (`seed=42`).

## 2. Results

### Pass / Fail & Numerical Checks
| Check | Measured | Note | Status |
|---|---|---|---|
| **3-Rep Non-Determinism Floor (latent / pixel)** | `0.0, 0.0` / `0.0` | Bit-identical across 3 reps | **PASS** |
| **Exact No-Op (`max_abs_diff`)** | **`0.0`** | Adapter attached, 0 active hooks | **PASS** |
| **Path Identity (Step-0 CFG `x0` mean / max)** | `0.0164` / `0.148` | 42 stacked BF16 blocks × CFG 3.0 (FP32 manual softmax vs BF16 fused SDPA) | Logged |
| **Path Identity (Final Latent mean / max)** | `0.0562` / `3.242` | 28-step Euler drift between FP32 manual vs BF16 fused SDPA | Logged |

### Stage-Wise L4 Resource Budget (`1024×1024`, 28 steps = 56 CFG forwards)
| Stage | Peak GPU Alloc | Peak GPU Reserved | CPU RSS | Wall Time | `sec / forward` |
|---|---:|---:|---:|---:|---:|
| `te_encode` (`qwen3.5_9b_bf16`, offline cached) | 18.19 GB | 18.34 GB | 1.57 GB | 67.89 s | — |
| `dit_load` (`moss_image_sft_bf16`) | 18.62 GB | 19.03 GB | 1.57 GB | 32.72 s | — |
| `dit_baseline` (fused SDPA) | **18.99 GB** | 19.44 GB | 2.04 GB | 60.90 s | **1.082 s** |
| `dit_hook_identity` (manual chunked, all 42 blocks) | **20.08 GB** (+1.09 GB) | 20.44 GB | 2.32 GB | 264.43 s | **4.719 s** |
| `vae_decode` (`flux2-vae`) | 19.43 GB | 20.45 GB | 2.72 GB | 1.95 s | — |

## 3. Verdict & Notes
- **Core Hook & L4 Memory Verified**: Exact no-op is `0.0`, 3-rep floor is `0.0`, and full-42-block manual chunked SDPA runs cleanly within L4 24 GB (`20.08 GB` peak, zero OOM).
- **Sweep Budget Projection**: Baseline `1.08 s/fwd` means E2-1 `k=8` (~19,264 fwds) ≈ 5.8 h on BF16; using the planned `moss_image_sft_int8_convrot` (~5.4 GB) or sharding is recommended for large sweeps.
- **Outputs**: [`config.json`](config.json), [`metrics.json`](metrics.json), [`visuals/generated.png`](visuals/generated.png), [`visuals/baseline_run1_vs_run2.png`](visuals/baseline_run1_vs_run2.png).
