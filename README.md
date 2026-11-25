# XAI_DiT

XAI / Mechanistic Interpretability (MI) framework and empirical study for modern image-generation Diffusion Transformers (DiTs).
Target models: **Anima** (Cross-Attn DiT) · **MOSS-Image** (Single-Stream Joint-Attn DiT) · **Krea 2 Raw** (Gated Single-Stream Joint-Attn DiT).

---

## Results

> **Interactive Visual Showcase**: Open [`showcase/index.html`](showcase/index.html) in a browser (or read [`showcase/README.md`](showcase/README.md)) for interactive model-comparison tabs, SVG charts, and zoomable experiment figures.

### Key Verified Findings (Colab NVIDIA L4 24GB)

- **Sanity & Unit Verification**: **`210 passed`** CPU unit tests (`0 failed, 0 skipped`) and **E0 Exact No-Op `0.0` / Non-Determinism Floor `0.0`** across all 3 target models.
- **Template / Register Share (`E4-2`)**: Single-stream joint-attention models route substantially more attention mass to chat-template/register tokens than cross-attention DiTs:
  - **Anima (BF16)**: `8.3%` · **Krea 2 Raw (`fp8_scaled`)**: `46.1%` · **MOSS-Image (`int8_convrot`)**: `92.3%`.
- **Massive Activations & G1 Gate (`E4-3`)**:
  - Peak `|attn_out|`: Anima **`17.75`** vs MOSS-Image **`1,256.0`** vs Krea 2 Raw **`1,920.0`**.
  - Krea 2 Block 0 G1 gate is **95.7% closed** (`mean = 0.029`). Removing per-token gating (`gate ← mean` on Blocks `[0..5]`) causes residual peak activations to **explode 3.01×** (`21,376 → 64,256`).
- **Causal Interventions (`E2-2`, `E3-0`, `E3-1`, `E3-3`)**:
  - **E3-1 K/V Patching (Red→Blue Scarf)**: Anima concentrates attribute transfer in **Mid blocks (`0.746`)**, MOSS-Image splits across Mid/Late (`0.365` / `0.330`), and Krea 2 distributes across depth (`0.102–0.124`).
  - **E2-2 `Cont^(l)` Top-20% vs Bottom-20%**: Confirmed on MOSS-Image (`+0.1094`) and Krea 2 `gated_mvo` (`+0.0357`), with an explicit counterexample on Anima `raw_mvo` (`−0.0332`).
  - **E3-3 TACA (`σ ≥ 0.97, γ₀ = 1.2`)**: Strongest pixel shift on MOSS-Image (`0.2674`) vs Anima (`0.0799`) and Krea 2 (`0.0534`).

---

## Next Priorities (Work in Progress)

1. **v2 Showcase Preview Completion**: Finalize and promote the 2-page practical visual showcase — [`showcase/v2_01_heatmap.html`](showcase/v2_01_heatmap.html) (multi-model XAI heatmaps) and [`showcase/v2_02_control_wip.html`](showcase/v2_02_control_wip.html) (causal window diagnosis → non-destructive semantic-unit control).
2. **v2 Practical Control & Real-World Optimization Compatibility**: Expand non-destructive semantic-unit control (`Mid`-block prompt routing, `S`-only TACA / per-tag strength control, early-$\sigma$ SAP, StableFlow Stage 2) across all 3 models, with **full compatibility under real-world ComfyUI optimization stacks** (quantized weights, 4–8 step `Turbo`/`eDMD` LoRAs, `EasyCache`/`Spectrum` step caching, `--use-ck-attention`/`--fast`, and `TorchCompile` — see [`docs/acceleration.md`](docs/acceleration.md)).

---

## Docs & Folder Guide

| Path | Description |
|---|---|
| [`showcase/`](showcase/) | Interactive HTML dashboard ([`index.html`](showcase/index.html)) and visual portfolio summary ([`README.md`](showcase/README.md)) |
| [`src/xai_dit/`](src/xai_dit/) | Pure-PyTorch core (`core/`), 3 model adapters (`adapters/`), headless ComfyUI CLI runner (`runners/cli.py`), and thin ComfyUI nodes (`comfyui/nodes.py`) |
| [`tests/`](tests/) | 8 CPU unit test suites (210 tests covering hooks, TokenMap, E1–E4 invariants) |
| [`docs/`](docs/) | Direction (`direction.md`), architecture (`architecture.md`), experiment design (`experiment_design.md`), verified paper specs (`specs/01~04`), and reference literature |
| [`experiments/`](experiments/) | Raw L4 experiment outputs (`metrics.json`, `visuals/*.png`), frozen prompt manifests (`design/`), and cross-model summaries (`cross_model/`) |

---

## Credits

- Built on verified methods from 16 core 2025–2026 DiT XAI/MI papers (see [`docs/reference_guide.md`](docs/reference_guide.md) and [`docs/specs/`](docs/specs/)).
- External reference repositories, commits, and licenses are documented in [`reference_repo/reference_repo_guide.md`](reference_repo/reference_repo_guide.md).
