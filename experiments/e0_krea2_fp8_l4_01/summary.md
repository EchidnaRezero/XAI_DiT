# EXP-E0: Krea 2 Raw FP8 Scaled Sanity & Resource Budget (Colab L4)

- **Date**: 2026-11-25
- **Design Basis**: `docs/experiment_design.md` (§2 E0), `plan/m1.md`
- **Target Model**: Krea 2 Raw (`krea2_raw_fp8_scaled.safetensors`, FP8 scaled, 28 blocks, GQA 48Q/12KV; offline BF16 TE `qwen3vl_4b_bf16.safetensors`)
- **Runtime**: Colab NVIDIA L4 (24 GB), `Torch 2.14.1+cu130`, `euler` / `simple`, 32 steps, CFG 4.0, `shift=1.15`, `split` CFG batching

---

## 1. Goal
Verify Krea 2 Raw adapter exact no-op (`== 0`), 3-rep non-determinism floor, full-28-block GQA manual chunked SDPA path identity, and real stage-wise L4 VRAM / `sec_per_forward` budget on `P-K1` (`seed=42`).

## 2. Results

### Pass / Fail & Numerical Checks
| Check | Measured | Note | Status |
|---|---|---|---|
| **3-Rep Non-Determinism Floor (latent / pixel)** | `0.0, 0.0` / `0.0` | Bit-identical across 3 reps | **PASS** |
| **Exact No-Op (`max_abs_diff`)** | **`0.0`** | Adapter attached, 0 active hooks | **PASS** |
| **Path Identity (Step-0 CFG `x0` mean / max)** | `0.0117` / `0.172` | 28 stacked FP8/BF16 GQA blocks × CFG 4.0 (FP32 manual softmax vs fused SDPA) | Logged |
| **Path Identity (Final Latent mean / max)** | `0.0993` / `2.868` | 32-step Euler drift between FP32 manual vs fused SDPA | Logged |

### Stage-Wise L4 Resource Budget (`1024×1024`, 32 steps = 64 CFG forwards)
| Stage | Peak GPU Alloc | Peak GPU Reserved | CPU RSS | Wall Time | `sec / forward` |
|---|---:|---:|---:|---:|---:|
| `te_encode` (`qwen3vl_4b_bf16`, offline cached) | 8.39 GB | 8.43 GB | 1.77 GB | 31.68 s | — |
| `dit_load` (`krea2_raw_fp8_scaled`) | 4.41 GB | 5.82 GB | 1.79 GB | 1.79 s | — |
| `dit_baseline` (fused SDPA) | **17.17 GB** | 17.64 GB | 2.15 GB | 170.84 s | **2.120 s** |
| `dit_hook_identity` (manual chunked, all 28 GQA blocks) | **19.36 GB** (+2.19 GB) | 19.85 GB | 2.36 GB | 442.62 s | **6.912 s** |
| `vae_decode` (`qwen_image_vae`) | 18.11 GB | 19.85 GB | 2.68 GB | 1.90 s | — |

## 3. Verdict & Notes
- **Core Hook & L4 Memory Verified**: Exact no-op is `0.0`, 3-rep floor is `0.0`, and full-28-block GQA (48Q/12KV) manual chunked SDPA runs cleanly within L4 24 GB (`19.36 GB` peak, zero OOM).
- **Sweep Budget Projection**: Baseline `2.12 s/fwd` confirms the design doc's expectation — Krea 2 E2-1 `k=8` (~14,848 fwds) ≈ 8.7 h, so `k=16` is rightly assigned to **Phase 4 (A100/H100)**!
- **Outputs**: [`config.json`](config.json), [`metrics.json`](metrics.json), [`visuals/generated.png`](visuals/generated.png), [`visuals/baseline_run1_vs_run2.png`](visuals/baseline_run1_vs_run2.png).
