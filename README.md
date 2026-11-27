# XAI_DiT

**English** · [한국어](README_KR.md)

XAI / Mechanistic Interpretability (MI) framework and empirical study for image-generation Diffusion Transformers (DiTs).

> [**Interactive Visual Showcase →**](https://echidnarezero.github.io/XAI_DiT/showcase/) · [**Anima 4-Stage Diagnosis Preview (WIP) →**](https://echidnarezero.github.io/XAI_DiT/showcase/preview_anima_step2.html)

---

## Target Models

| Model | Backbone (DiT) | Text Encoder | VAE | Practical XAI Preset (L4) | Notes |
|---|---|---|---|---|---|
| **[Anima](https://huggingface.co/circlestone-labs/Anima)** | Cross-Attn DiT (28 blocks, ~2B) | Qwen3-0.6B + LLMAdapter (6b) | Wan2.1 VAE (16ch, 8×) | `BF16` + `EasyCache(0.20)` + `Sel 6b` (`14.2s`, `r=0.97`) | [`anima.md`](docs/reference/models/anima.md) |
| **[MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)** ([SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)) | Single-Stream Joint-Attn DiT (42 blocks, ~5B) | Qwen3.5-9B + Text Refiner (2b) | Flux2 VAE (128ch, 16×) | `int8_convrot` + `DMD 8s` + `Sel 6b` (`13.2s`, `r=0.93`) | [`moss_image.md`](docs/reference/models/moss_image.md) |
| **[Krea 2](https://huggingface.co/krea/Krea-2-Raw)** ([Turbo](https://huggingface.co/krea/Krea-2-Turbo) · [Krea2-Anime](https://civitai.com/models/2908999?modelVersionId=3435333)) | Gated Single-Stream Joint-Attn DiT (28 blocks, ~12.8B) | Qwen3-VL-4B (12-layer) + TextFusion | Qwen-Image VAE (16ch, 8×) | `Turbo FP8` + `Sel 5b` (`24.6s`) · `Anime INT4` (`38.9s`, `r=0.93`) | [`krea2.md`](docs/reference/models/krea2.md) |

- **Acceleration Presets & Guardrails**: [`docs/acceleration.md`](docs/acceleration.md) · **Full Acceleration Report**: [`experiments/accel/REPORT.md`](experiments/accel/REPORT.md)

---

## Next Priorities (Work in Progress)

1. **v2 Practical Semantic-Unit Control**: Non-destructive control (`Mid`-block prompt routing, `S`-only TACA / per-tag strength control, early-$\sigma$ SAP, StableFlow Stage 2) across all 3 models.
2. **Offline Semantic Evaluators & ComfyUI Node Packaging**: GPU semantic evaluators (OpenCLIP/SigLIP, DINOv2, LPIPS) and ComfyUI custom node packaging.
3. **Tier 2 Hardware Verification (Blackwell `sm_120`, ≥80 SMs)**: `SageAttention 3` (FP4), `NVFP4`, `MXFP8`, `FlexAttention`, and `torch.compile(max-autotune)` + `cudagraphs`.

---

## Project Structure

| Path | Description |
|---|---|
| [`showcase/`](showcase/) | Interactive visual guide and previews |
| [`src/xai_dit/`](src/xai_dit/) | Core XAI/MI engine, 3 model adapters, CLI runner, and ComfyUI nodes |
| [`tests/`](tests/) | CPU unit tests (210 tests covering hooks, token mapping, and invariants) |
| [`docs/`](docs/) | Project design, verified paper specs, and reference literature |
| [`experiments/`](experiments/) | XAI/MI experiment outputs, acceleration track ([`accel/`](experiments/accel/REPORT.md)), and case studies ([`case_studies/`](experiments/case_studies/README.md)) |
| [`issues/`](issues/) | Code-level root-cause investigations and verified fixes |

---

## Credits

- Built on verified methods from 16 core 2025–2026 DiT XAI/MI papers (see [`docs/reference_guide.md`](docs/reference_guide.md) and [`docs/specs/`](docs/specs/)).
- External reference repositories, commits, and licenses are documented in [`reference_repo/reference_repo_guide.md`](reference_repo/reference_repo_guide.md).
