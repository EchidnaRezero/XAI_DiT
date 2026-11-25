# Inference Acceleration & Optimization Reference (ComfyUI / L4+)

Catalog of verified inference acceleration options for **Anima · MOSS-Image · Krea 2 Raw** in ComfyUI (`reference_repo/ComfyUI`), organized by **common options**, **model-specific differences**, and **XAI hook compatibility**.

> **Baseline Policy (`experiments/design/runtime_and_hp.md`)**: All XAI reference measurements (`E0–E4`) keep step-skipping, distillation LoRAs, and experimental fast flags **OFF** (PyTorch SDPA + quantized L4 weights). Acceleration stacks below are used for (1) quick generation pre-checks (e.g., character/tag recognition) and (2) dedicated real-world accelerated XAI compatibility evaluations (`plan/accel_xai_eval.md`).

---

## 1. Common Acceleration Options (All 3 Models)

### A. ComfyUI Attention Backends (`comfy/cli_args.py`, `comfy/ldm/modules/attention.py`)
| Option / Flag | Backend | Notes |
|---|---|---|
| `--use-pytorch-cross-attention` | `attention_pytorch` (PyTorch SDPA) | **XAI baseline default** on PyTorch 2.x. |
| `--use-ck-attention` | `attention_comfy_kitchen_int8` / `comfy_kitchen_sol` | ComfyUI-native accelerated attention (non-Sage alternative). |
| `--use-flash-attention` | `attention_flash` (`flash_attn_func`) | Requires `flash-attn` installed. |
| `--use-sage-attention` | `attention_sage` / `sage3` | Requires `sageattention`. **Forbidden globally on Krea 2 Raw** (NaN / black images). |
| `FlexAttention` | PyTorch FlexAttention | **Excluded on L4** (Triton shared-memory OOM `131072 > 101376`); A100/H100 only. |

### B. ComfyUI Runtime, Compiler & Step-Caching Options
| Option / Node | Location | Effect & Caveats |
|---|---|---|
| `--disable-dynamic-vram` | `comfy/cli_args.py` | **Required on Colab L4 High-RAM**: speeds up MOSS-Image and prevents Krea 2 `comfy-aimdo` NaN/black output. |
| `--fast [OPTIONS]` | `comfy/cli_args.py` | Enables `fp16_accumulation`, `fp8_matrix_mult`, `cublas_ops`, `autotune` (experimental; may slightly shift numerics). |
| `EasyCache` / `LazyCache` | `comfy_extras/nodes_easycache.py` | ComfyUI built-in step-skipping via output-difference reuse (`reuse_threshold`, `start_percent`, `end_percent`). |
| `TorchCompileModel` | `comfy_extras/nodes_torch_compile.py` | Wraps DiT with `torch.compile` (`inductor` / `cudagraphs`). High first-run compile cost; check for graph breaks with Python hooks. |
| `Spectrum` / `TeaCache` | External ComfyUI nodes | Hidden-state forecasting / feature caching (~1.5–2.1× speedup without cutting total step count). |

---

## 2. Model-Specific Options & Differences

| Category | Anima (2B, Cross-Attn) | MOSS-Image (5B+9B, Joint-Attn) | Krea 2 Raw (12.8B+4B, Gated Joint-Attn) |
|---|---|---|---|
| **L4 Weight Precision** | **BF16** (~4 GB, `0.363 s/fwd`); quantization unnecessary | **`sft_int8_convrot`** (~5.4 GB, `0.539 s/fwd`, **2.0× faster** than `sft_bf16` `1.082 s/fwd`) | **`krea2_raw_fp8_scaled`** (12.2 GB, `2.122 s/fwd`); alternative `int8_convrot` (13.5 GB, Hadamard caveat) |
| **Distilled LoRAs / Checkpoints** | 4–8 step `Anima-Turbo` / `DMD` / `HaDeS` LoRAs | 4–8 step `eDMD` / `DMD` / `Turbo` LoRAs (keep SFT family consistent) | 8-step `Krea-2-Turbo` / `DMD` LoRA |
| **Sampler / Guidance Notes** | `er_sde`/`simple` auxiliary fast/quality sampler | `SLG (Layer 22)` is a guidance-quality node, **not** a speedup (OFF in baseline) | Fixed `sigma_table` (`shift=1.15` vs official dynamic shift) |
| **Forbidden / Avoid on L4** | **FP16 forced** (NaN / black images) | `w4a8` / GGUF (slower than `int8_convrot` on L4); mixing `base` & `sft` | **Global `SageAttention`** (NaN / black); `nvfp4` (Blackwell-only) |

---

## 3. XAI 4-Axis Hook Compatibility Matrix

| Acceleration Method | `E1` Heatmap | `E2` Layer Sel. | `E3` Tag Influence | `E4` MI Verif. | Notes |
|---|:---:|:---:|:---:|:---:|---|
| **Quantization (`int8_convrot`, `fp8_scaled`)** | Full | Full | Full | Full | Already verified on L4 (`exact_noop = 0.0`); `Cont^(l)` uses `wo(x)` module call. |
| **Turbo / DMD / eDMD LoRAs (4–8 steps)** | Full | Full | Full | Full | Standard DiT forward; fewer steps directly speed up XAI sweeps (evaluate fidelity vs teacher). |
| **`--fast` (`cublas_ops`, `fp8_matrix_mult`)** | Full | Full | Full | Full | Accelerates linear layers without breaking attention hooks. |
| **`--use-ck-attention` / `FlashAttention`** | Partial | Partial | Partial | Partial | Hooked blocks requiring `probs` or logit edits fall back to chunked SDPA; unhooked blocks/steps & `uncond` remain accelerated. |
| **`EasyCache` / `Spectrum` / `TeaCache`** | Partial | Partial | Partial | Partial | Skipped steps do not execute block forwards, so hooks on skipped steps are bypassed; requires fidelity comparison vs uncached baseline. |
| **`TorchCompileModel`** | Benchmark | Benchmark | Benchmark | Benchmark | Python hooks / dynamic branches may trigger graph breaks or recompilation; benchmark wall-clock before adopting. |
