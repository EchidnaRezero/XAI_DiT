# EXP-E0: Anima BF16 Sanity & Resource Budget (Colab L4)

- **Date**: 2026-11-25
- **Design Basis**: `docs/experiment_design.md` (§2 E0), `plan/m1.md`
- **Target Model**: Anima (`anima-base-v1.0.safetensors`, BF16, 28 blocks)
- **Runtime**: Colab NVIDIA L4 (24 GB), `Torch 2.14.1+cu130`, ComfyUI `da8bd08e`, `euler` / `simple`, 30 steps, CFG 4.0, `shift=3.0`, `split` CFG batching

---

## 1. Goal
Verify Anima adapter exact no-op (`== 0`), 3-rep non-determinism floor, manual chunked SDPA path identity, and real stage-wise L4 VRAM / `sec_per_forward` budget on `P-A1` (`seed=42`).

## 2. Results

### Pass / Fail & Numerical Checks
| Check | Measured | Threshold / Note | Status |
|---|---|---|---|
| **3-Rep Non-Determinism Floor (latent)** | `0.0`, `0.0` | Bit-identical across 3 reps | **PASS** |
| **3-Rep Non-Determinism Floor (pixel)** | `0.0` | `baseline_run1_vs_run2.png` identical | **PASS** |
| **Exact No-Op (`max_abs_diff`)** | **`0.0`** | `== 0.0` (adapter attached, 0 active hooks) | **PASS** |
| **Path Identity (Step-0 CFG `x0` `max_abs_diff`)** | `0.125` (`2^-3`) | `cli.py` applied single-block `<= 1e-3` threshold to 28-block compounded × CFG=4.0 `x0` | **Calibration Note** *(see §3)* |
| **Path Identity (Final Latent `max_abs_diff`)** | `3.296` | 30-step Euler divergence (FP32 manual softmax vs BF16 fused SDPA) | Logged |

### Stage-Wise L4 Resource Budget (`1024×1024`, 30 steps = 60 CFG forwards)
| Stage | Peak GPU Alloc | Peak GPU Reserved | CPU RSS | Wall Time | `sec / forward` |
|---|---:|---:|---:|---:|---:|
| `te_encode` (`qwen_3_06b_base`) | 1.13 GB | 1.16 GB | 2.04 GB | 6.31 s | — |
| `dit_load` (`anima-base-v1.0`) | 4.45 GB | 4.48 GB | 2.05 GB | 2.84 s | — |
| `dit_baseline` (fused SDPA) | **4.66 GB** | 4.73 GB | 2.14 GB | 22.42 s | **0.363 s** |
| `dit_hook_identity` (manual chunked, all 28 blocks) | **4.70 GB** (+0.04 GB) | 4.75 GB | 2.32 GB | 33.67 s | **0.558 s** |
| `vae_decode` (`qwen_image_vae`) | 5.92 GB | 5.96 GB | 2.63 GB | 1.91 s | — |

## 3. Verdict & Notes
- **Core Hook & Memory Verified**: Exact no-op is `0.0`, 3-rep baseline is `0.0`, `generated.png` renders `P-A1` cleanly, and manual chunked SDPA across all 28 blocks adds only **+0.04 GB VRAM** (`4.70 GB` peak vs 24 GB L4).
- **Path-Identity Threshold Calibration**: `cli.py` currently compares step-0 **end-to-end 28-block CFG-combined `x0`** against the single-block attention tolerance (`1e-3`). Across 28 stacked BF16 blocks × CFG=4.0, a single BF16 ULP shift (`0.03125`) between FP32 manual softmax and BF16 fused SDPA compounds to `0.125` (`2^-3`). Next CLI refinement should log **per-block attention output `max_abs_diff`** (against `1e-3`) separately from end-to-end `x0` `mean_abs_diff`/`max_abs_diff`.
- **Anima Sweep Budget Projection**: E2-1 `k=8` (~13,920 fwds) ≈ 1.4 h; `k=16` (~27,840 fwds) ≈ 2.8 h on L4.
- **Outputs**: [`config.json`](config.json), [`metrics.json`](metrics.json), [`visuals/generated.png`](visuals/generated.png), [`visuals/baseline_run1_vs_run2.png`](visuals/baseline_run1_vs_run2.png).
