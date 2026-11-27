# Practical Acceleration & Optimization Guide (ComfyUI / L4+)

Lightweight cheat sheet of **Top-3 verified presets** and **mandatory guardrails** for **Anima · MOSS-Image · Krea 2 Raw/Turbo · Krea2-Anime v3.0**.

> **Deep-Dive Routing**:
> - Full Technical Report (Phase 1–3 benchmarks, `torch.profiler`, code-level root causes, GPU & XAI matrices): [`experiments/accel/REPORT.md`](../experiments/accel/REPORT.md)
> - Full Weights, LoRAs & Custom Nodes Catalog: [`experiments/accel/RESOURCES.md`](../experiments/accel/RESOURCES.md)
> - Chronological Run History & SQLite-Ready Index: [`experiments/accel/HISTORY.md`](../experiments/accel/HISTORY.md) · [`experiments/accel/runs_index.jsonl`](../experiments/accel/runs_index.jsonl)
> - Raw Experiment Outputs: [`experiments/accel/runs/`](../experiments/accel/runs/)
> - *Baseline Policy*: Reference XAI measurements (`E0–E4`) keep step-skipping, distillation LoRAs, and experimental fast flags **OFF**.

---

## 1. Model-by-Model Top-3 Presets (Colab L4 24GB Verified)

### A. Anima (`BF16` baseline `25.4–25.8s` / XAI `29.2–29.6s`)
| Purpose | Rank | Recommended Stack | Time (Speedup) | Fidelity / VRAM |
|---|---:|---|---:|---|
| **T2I** | **1** | `SDPA` · `--fast` **OFF** · `EasyCache(t=0.20, s=0.20, e=0.80)` *(+ `SchedCFG 60%` opt.)* | **`13.0–15.3s`** (`1.7–1.9×`) | `24–29 dB` *(max detail: `Spectrum d3_safe` `16.9s`, `23.4 dB`)* |
| **T2I** | **2** | `anima-turbo-v1.1` (8s) or [`sorryhyun/anima-turbo-4step`](https://huggingface.co/sorryhyun/anima-turbo-4step) (`er_sde`) | **`5.99s` / `5.18s`** (`4.3–5.0×`) | Distilled fast draft (`5.37 / 8.90 GB`) |
| **T2I** | **3** | `20-step dpmpp_2m + EasyCache` or [`INT8 ConvRot`](https://huggingface.co/ruwwww/Anima-ConvRot) | `11.0s` (`2.31×`) / `21.9s` | **`3.89 GB`** VRAM (`INT8 ConvRot`) |
| **XAI** | **1** | `BF16` · `EasyCache(0.20)` · **`Selective 6 blocks`** | **`14.2–15.1s`** (`1.99×`) | **`r = 0.955–0.970`** *(full blocks: `16–17s`, `r=0.98+`)* |
| **XAI** | **2** | [`INT8 ConvRot`](https://huggingface.co/ruwwww/Anima-ConvRot) · `SchedCFG 65%` · **`Selective 6 blocks`** | **`17.77s`** (`1.67×`) | **`r = 0.9548`** · **`3.89 GB`** VRAM |
| **XAI** | **3** | `anima-turbo-v1.1` (8s) · **`Selective 6 blocks`** | **`3.70s`** (`8.01×`) | `r = 0.8516` (instant preview XAI) |

### B. MOSS-Image (`sft_int8_convrot` baseline `38.5–47.2s` / XAI `50.5–145.7s`)
| Purpose | Rank | Recommended Stack | Time (Speedup) | Fidelity / VRAM |
|---|---:|---|---:|---|
| **T2I** | **1** | `SDPA` · safe `--fast` *(no `fp16_acc`)* · **`SchedCFG 65%` + `EasyCache(0.20–0.25)`** | **`18.3s`** (`2.59×`) | **`36.14 dB`** (`5.87 GB`) |
| **T2I** | **2** | `SchedCFG 65%` alone or `25-step er_sde + EasyCache(0.20)` | `38.6s` / `23.2s` | **`48.20 dB`** (lossless) / sharpest texture |
| **T2I** | **3** | [`DMD 8-step`](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd) or [`eDMD 4-step`](https://huggingface.co/lvladikov/MOSS-SFT-Distill-4step-LoRA) | **`12.15s` / `8.14s`** (`3.9–5.8×`) | Distilled *(requant VRAM `10.30 GB`)* |
| **XAI** | **1** | [`DMD 8-step`](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd) · **`Selective 6 blocks`** | **`13.16s`** (`3.84×`) | **`r = 0.9261, ρ = 0.9524`** |
| **XAI** | **2** | `EasyCache(0.25)` · **`Selective 6 blocks`** *(2D: 8 blocks)* | **`29.4–29.6s`** (`4.95×`) | Photo **`r = 0.958–0.976`** |
| **XAI** | **3** | **`SchedCFG 65%`** · **`Selective 6 blocks`** | **`44.32s`** (`1.14×`) | **`r = 1.0000, IoU = 0.9976`** (lossless XAI) |

### C. Krea 2 Raw / Turbo & Krea2-Anime v3.0
| Purpose | Rank | Recommended Stack | Time (Speedup) | Fidelity / VRAM |
|---|---:|---|---:|---|
| **T2I** | **1** | **`Krea 2 Turbo (FP8, 8-step)`** · `--use-ck-attention` · safe `--fast` | **`16.9–17.4s`** (`~8×` vs Raw) | High-detail photo & 2D |
| **T2I** | **2** | [`Krea2-Anime v3.0 (INT4 W4A4)`](https://civitai.com/models/2908999?modelVersionId=3435333) · `--use-ck-attention` · safe `--fast` | **`32.0–32.7s`** (`1.3–1.6×` vs FP8) | **`8.38 GB`** VRAM (`-40%`, fits 12 GB GPUs) · 0 typos |
| **T2I** | **3** | `Krea 2 Raw (FP8)` · `--use-ck-attention` · safe `--fast` · `EasyCache(0.20, 0.25, 0.80)` | **`61.6–79.0s`** (`1.7–2.9×`) | `24.7–27.2 dB` |
| **XAI** | **1** | **`Krea 2 Turbo (8-step)`** · **`Selective 5 blocks`** | **`24.6s`** (`12.22×` vs Raw) | Pinpoint localization verified |
| **XAI** | **2** | [`Krea2-Anime v3.0 (INT4 W4A4)`](https://civitai.com/models/2908999?modelVersionId=3435333) · **`Selective 5 blocks`** | **`38.9s`** (`1.27×` vs FP8) | **`r = 0.9341, ρ = 0.9682`** · `9.63 GB` |
| **XAI** | **3** | `Krea 2 Raw (FP8)` · `--use-ck-attention` · safe `--fast` · `EasyCache(0.20)` · **`Sel 5b`** | **`94.8s`** (`3.68×`) | **`r = 0.967–0.990`** |

*(Additional weights, community LoRAs, custom nodes, and ComfyUI CLI flags → [`experiments/accel/RESOURCES.md`](../experiments/accel/RESOURCES.md))*

---

## 2. Forbidden Combinations & Mandatory Guardrails

1. **Never use stock `--fast fp16_accumulation`** on Anima BF16 or Krea 2 FP8: causes **37% slowdown** on Anima (`2,831×` `dtype_cast`, [`issues/004`](../issues/004_anima_accel_fp16acc_and_compile_easycache.md)) and grain + jacket hallucination on Krea 2 ([`issues/005`](../issues/005_krea2_fp16acc_grain_and_artifact.md)). Fix: set `PRIORITIZE_FP16 = False` while keeping `allow_fp16_accumulation = True`.
2. **Never use global `--use-sage-attention` on Krea 2**: produces NaN/black collapse unless `TextFusionTransformer` is bypassed (`low_precision_attention=False`) **and** `smooth_k=True` is enabled ([`issues/007`](../issues/007_krea2_sageattention_nan_collapse.md)).
3. **Never register XAI `split` CFG before `EasyCache`**: causes speckled noise collapse ([`issues/006`](../issues/006_easycache_split_cfg_wrapper_order_collapse.md)). Always register `EasyCache` **outermost (first)**.
4. **Never pair stock `TorchCompileModel` + `EasyCache`**: hits Dynamo `recompile_limit(8)` ([`issues/004`](../issues/004_anima_accel_fp16acc_and_compile_easycache.md)). Fix: decorate `easycache_forward_wrapper` with `@torch.compiler.disable`.
5. **Protect early/late steps**: `EasyCache` (`start=0.20–0.25, end=0.80–0.85`), `sol-attn` (`start_percent=0.35, end_percent=0.85`), `Spectrum` (`degree=3, ridge=0.50, warmup=8, stop=0.80`).
