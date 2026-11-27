# Acceleration Weights, LoRAs & Custom Nodes Catalog

Complete reference of verified **quantized weights**, **distillation/acceleration LoRAs**, **step-caching/attention custom nodes**, and **built-in ComfyUI flags** across Anima, MOSS-Image, and Krea 2.  
For Top-3 practical presets see [`docs/acceleration.md`](../../docs/acceleration.md); for empirical benchmark results see [`REPORT.md`](REPORT.md).

---

## 1. Model-by-Model Weights, LoRAs & Community Nodes

| Model | Category | Name / Link | Settings & Caveats |
|---|---|---|---|
| **Anima** | Quantization | [`ruwwww/Anima-ConvRot`](https://huggingface.co/ruwwww/Anima-ConvRot) · [`Bedovyy/Anima-INT8`](https://huggingface.co/Bedovyy/Anima-INT8) · `ComfyUI-INT8-Fast` | `INT8 ConvRot` (`3.89 GB` VRAM, keeps `blocks.0/1/27, adaln` BF16); `Rowwise` is slightly faster with lower fidelity |
| **Anima** | Distilled / LoRA | `anima-turbo-v1.0/v1.1` · [`sorryhyun/anima-turbo-4step`](https://huggingface.co/sorryhyun/anima-turbo-4step) · [16-step Acc LoRA](https://civitai.com/models/2967964) · `RDBT` (4/8/16s) | `v1.1`: 8–12s CFG 1 `euler/simple`; `4step`: 4s `er_sde`, `strength=1.0`; `16-step Acc`: ~4× without Turbo overcooking |
| **Anima** | Step Caching | [`sorryhyun/ComfyUI-Spectrum-KSampler`](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler) · [`CocyNoric/ComfyUI-Anima-TeaCache`](https://github.com/CocyNoric/ComfyUI-Anima-TeaCache) · [`BlackSnowSkill/ANIMA_BOOSTER`](https://github.com/BlackSnowSkill/ANIMA_BOOSTER) | Spectrum safe: `degree=3, ridge=0.50, warmup=8, stop=0.80`; connect TeaCache **after** LoRAs |
| **Anima** | CFG / Attn | `KJNodes` (`Scheduled CFG`) · `ComfyUI-Anima-NAG` · `PatchSageAttentionKJ` | `Scheduled CFG` `4→1` at 60–65% (`+15–20%`); `NAG` restores negatives at `CFG=1` |
| **MOSS-Image** | Quantization | [`Comfy-Org` `sft_int8_convrot`](https://huggingface.co/Comfy-Org) · [MOSS-Image HMS v1](https://civitai.com/models/3061739) · [`KBlueLeaf/MOSS-Image-heal`](https://huggingface.co/KBlueLeaf) | `int8_convrot`: `5.87 GB`, `2×` vs BF16 (LPIPS `0.058`); `HMS v1`: W6A8+W4A8 (`3.68 GB`, `~2.2×`) |
| **MOSS-Image** | Distilled LoRA | [`lodestones/MOSS-Image-SFT-8-steps-dmd`](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd) · [`lvladikov/MOSS-SFT-Distill-4step-LoRA`](https://huggingface.co/lvladikov/MOSS-SFT-Distill-4step-LoRA) · [Fast-12 OPD](https://civitai.com/models/3063613) · [`akhaliq/MOSS-Image-SFT-4-steps`](https://huggingface.co/akhaliq/MOSS-Image-SFT-4-steps) | Keep DMD LoRA FP32 (BF16 merge erases delta); `eDMD` needs manual sigmas; `OPD` uses `euler/normal` (not `simple`) |
| **MOSS-Image** | Step / KV Caching | `comfyui-moss-accel` · [`Rkkss/ComfyUI-Spectrum-MOSS`](https://github.com/Rkkss/ComfyUI-Spectrum-MOSS) · [`p1atdev/MOSS-Image-SFT-KV-Cache-LoRA`](https://huggingface.co/p1atdev/MOSS-Image-SFT-KV-Cache-LoRA) · `LC Model Optimizer` | Avoid `LC Speed Boost` on MOSS (waxy faces); KV-Cache requires correction LoRA & >500 tokens |
| **Krea 2** | Quant / Fine-tune | [`Comfy-Org` `krea2_{raw,turbo}_{fp8_scaled,int8_convrot}`](https://huggingface.co/Comfy-Org) · [`Krea2-Anime v3.0`](https://civitai.com/models/2908999?modelVersionId=3435333) (`fp8_e4m3`, `int8_convrot`, `int4_w4a4`) · `Q4_CR` GGUF | `Krea2-Anime v3.0 INT4 W4A4` (`7.42 GB`): `32s`, `8.38 GB` VRAM, 0 typos; `int8_convrot` LPIPS `0.066` vs FP8 `0.120` |
| **Krea 2** | Distilled LoRA | [`lvladikov/Krea2-Turbo-Distill-4step-LoRA`](https://huggingface.co/lvladikov) · [`TheDivergentAI/Krea2-Turbo-Distill-8step-LoRA`](https://huggingface.co/TheDivergentAI) | Stacking LoRA onto 13 GB FP8 triggers requant OOM on L4 (merge offline in FP32) |
| **Krea 2** | Step Caching | `LC Model Optimizer` · `Q0809/ComfyUI-Krea2-Accel` · `ComfyUI-CacheDiT` · `RS Progressive Resolution Sampler` | Raw 52-step Spectrum needs `>=20%` warmup |

---

## 2. Built-in ComfyUI Flags & Nodes (`reference_repo/ComfyUI`)

| Flag / Node | Exact Location | Effect & Caveats |
|---|---|---|
| `--use-pytorch-cross-attention` | `cli_args.py:150`, `attention.py:594` | PyTorch SDPA (**XAI baseline default**). |
| `--use-ck-attention` | `cli_args.py:153`, `attention.py:654-695` | `comfy_kitchen` INT8 attention (`cu130` required). Great on Krea 2; avoid on Anima BF16. |
| `--use-sage-attention` | `cli_args.py:151`, `attention.py:716, 740` | Hardcodes `smooth_k=False`. **Forbidden globally on Krea 2** without `TextFusion` bypass + `smooth_k=True` (`issues/007`). |
| `--fast fp8_matrix_mult cublas_ops autotune` | `cli_args.py:197-203` | Safe `--fast` subset. **Omit `fp16_accumulation`** unless `PRIORITIZE_FP16=False` is patched (`issues/004`, `005`). |
| `--disable-dynamic-vram --highvram` | `cli_args.py:182-183` | Required on Colab L4 High-RAM (prevents Krea 2 `comfy-aimdo` NaN/black output). |
| `EasyCache` / `LazyCache` | `nodes_easycache.py:20-520` | Zero-order residual reuse. Protect early `20–25%` & late `15–20%`; register **outermost** before XAI `split` CFG (`issues/006`). |
| `BlockSparseAttention` (`sol-attn`) | `nodes_sparse_attention.py:379-455` | Use `start_percent=0.35, end_percent=0.85` to prevent extra-limb collapse. |
| `TorchCompileModel` | `nodes_torch_compile.py:8-30` | Decorate `easycache_forward_wrapper` with `@torch.compiler.disable` (`issues/004`); avoid `cudagraphs` on <80 SM GPUs. |
