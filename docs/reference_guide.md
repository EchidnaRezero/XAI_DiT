# Reference Guide

Routing index for `docs/reference/`—check here first to find the exact document or paper needed.

---

## 1. Survey & Verification Baseline
- [`reference/XAI_DiT_조사정리.md`](reference/XAI_DiT_조사정리.md)
  - **When to read**: Method selection rationale, V-ablation → counterfactual patching transition, verification protocols (random control, no-op, VQA, 1-step correlation), and per-model recommendations (Section 6).

---

## 2. Target Model Specs (`reference/models/`)
| Model | File | When to Read |
|---|---|---|
| **Anima** | [`reference/models/anima.md`](reference/models/anima.md) | Qwen3-0.6B + LLMAdapter (T5 query), 28 cross-attn blocks, hook points & shapes |
| **Krea 2 (base)** | [`reference/models/krea2.md`](reference/models/krea2.md) | Qwen3-VL-4B (12 layers) + txtfusion, 28 single-stream blocks, `attn.gate` · GQA · 5 tail template tokens |
| **MOSS-Image** | [`reference/models/moss_image.md`](reference/models/moss_image.md) | Qwen3.5-9B + 2-block bidirectional refiner, 42 single-stream blocks, 4-axis RoPE · front/back chat templates |

---

## 3. Core Papers (`reference/paper/`, 16 PDFs · gitignored)

| Direction / Folder | File | arXiv / Venue | When to Read |
|---|---|---|---|
| **01_heatmap/** | `2025_ICML_ConceptAttention.pdf` | `2502.04320` (ICML 2025) | Output-space concept maps for joint-attn models (Krea 2, MOSS-Image) |
| | `2025_ICCV_ExploringMMDiT.pdf` | `2508.07519` (ICCV 2025) | 4-block MMDiT attention decomposition · Grounded-SAM2 spatial evaluation |
| | `2025_NeurIPS_Seg4Diff.pdf` | NeurIPS 2025 | Norm-based contribution · map-vs-perturbation layer alignment |
| **02_layer_selection/** | `2025_CVPR_StableFlow_VitalLayers.pdf` | CVPR 2025 | Vital layer selection & layer-wise control baselines |
| **03_tag_influence/** | `2025_ICLR_TextGenParamLocalization.pdf` | ICLR 2025 | Block-level text K/V cache patching · skipping first 2 steps |
| | `2026_ICLR_PCI_TemporalConceptDynamics.pdf` | `2512.08486` (ICLR 2026) | Counterfactual interventions · temporal concept dynamics · multi-VQA judging |
| | `2025_ICCV_TACA.pdf` | `2506.07986` (ICCV 2025) | Early-step cross-modal attention adjustment |
| | `2025_NeurIPS_DiT-ST.pdf` | `2505.19261` (NeurIPS 2025) | Split-text timestep-aware conditioning |
| | `2026_CVPR_SAP.pdf` | `2506.01929` (CVPR 2026) | Early (layout/color) / mid (shape) / late (identity/texture) stage split |
| **04_verification_mi/** | `2026_ICML_AttentionSinksDiT.pdf` | `2605.09313` (ICML 2026) | Attention sinks · logit knockout · random control · no-op · bootstrap CI |
| | `2026_CVPR_DiTRelationCircuits.pdf` | `2601.06338` (CVPR 2026) | Head × token circuit knockout |
| | `2026_arXiv_TemplateRegisters.pdf` | `2607.19139` (arXiv 2026) | Chat-template tokens as implicit semantic registers in Qwen-based DiTs |
| | `2025_NeurIPS_GatedAttention.pdf` | `2505.06708` (NeurIPS 2025 Best) | Post-SDPA sigmoid gate (Krea 2 `attn.gate`) & attention-sink-free behavior |
| | `2026_CVPR_DiffusionProbe.pdf` | `2602.23783` (CVPR 2026) | 1-step cross-attn vs final image quality correlation (SRCC) |
| | `2026_CVPR_ProbeSelect.pdf` | `2603.02829` (CVPR 2026) | Early $x_0$ estimate vs final image correlation |
| **Root (shared)** | `2025_NeurIPS_LocalizingKnowledgeDiT.pdf` | `2505.18832` (NeurIPS 2025) | Core basis for both **layer selection** and **counterfactual prompt patching** |
