# Project Roadmap (Future Direction & Backlog)

High-level future directions and hardware-gated tracks not yet in active execution.  
When a direction is ready for concrete implementation, promote it into `plan/active/`.

---

## 1. v2 Practical Semantic-Unit Control & ComfyUI Nodes
- Generalize non-destructive semantic-unit controls (`Mid` `kv_patch`, StableFlow Stage 2, `S`-only TACA / additive per-tag log-bias, early-$\sigma$ SAP) across **Anima**, **MOSS-Image**, and **Krea 2**.
- Replace training-free RGB proxies with offline GPU semantic evaluators (**OpenCLIP / SigLIP**, **DINOv2**, **LPIPS**, **Grounded-SAM2 / VQA**).
- Package verified XAI heatmaps and non-destructive controls into drop-in **ComfyUI custom nodes** (`src/xai_dit/comfyui/`) compatible with [`docs/acceleration.md`](../docs/acceleration.md) presets.

---

## 2. Tier 2 Acceleration Verification (Blackwell `sm_120`, ≥80 SMs)
*Deferred pending Blackwell GPU access (RTX 5080 / 5090 / B200). Tier 1 (Colab L4) is complete in [`experiments/accel/REPORT.md`](../experiments/accel/REPORT.md).*

| Track | Options | Goal |
|---|---|---|
| **T2-1: `SageAttention 3` (FP4)** | `sageattn3_blackwell` on all 3 models (w/ Krea 2 `TextFusion` bypass + `smooth_k=True`) | Verify 3–5× attention speedup & FP4 stability under M-Act `~1,920` |
| **T2-2: `NVFP4`** | `krea2_turbo_nvfp4` (w/ vs w/o `full_precision_matrix_mult` metadata bug check) | Verify `~4.6s` recovery & low-light `LPIPS` |
| **T2-3: `MXFP8`** | `krea2_turbo_mxfp8` & `anima-*-mxfp8` vs `INT8 ConvRot` | Compare eye/banding fidelity vs `INT8 ConvRot` |
| **T2-4: `FlexAttention` & `max-autotune + cudagraphs`** | ≥80-SM GPUs (blocked on L4 by 58 SMs / 101 KB SMEM) | Steady-state `s/fwd` & XAI hook compatibility |
| **T2-5: Bandwidth Scaling** | Re-run Tier 1 Top-3 presets on Blackwell | Separate L4 `300 GB/s` bandwidth ceiling from kernel overhead |

---

## 3. Curated Public XAI Case Studies (`experiments/case_studies/showcase/`)
- Re-run privacy-safe 2D diagnostic examples (clean single-character seed, zero real photos) and promote verified Before/After LoRA improvement loops into `experiments/case_studies/showcase/` and `showcase/`.
