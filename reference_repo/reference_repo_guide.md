# Reference Repo Guide

External reference repositories stored under `reference_repo/`.
Cloned repos are gitignored; **only this guide file is tracked in Git** for code-reference routing, license tracking, and future credits.

---

| Repo | Source / Author | Reference Commit / Version | License | Purpose / When to Check |
|---|---|---|---|---|
| **Anima_HEATMAP** | Own repo ([HisameOgasahara/Anima_HEATMAP](https://github.com/HisameOgasahara/Anima_HEATMAP)) | `main` (+ `origin/feature/tag-influence`) | MIT (see repo `LICENSE`) | v1 implementation; reference for Q/K capture, streaming aggregation, ComfyUI node patterns, and profiler logic |
| **ComfyUI** | [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI) | `da8bd08e48de` (2026-11-23, v0.46.0+46) | GPL-3.0 | Reference for Anima (`comfy/ldm/anima/`, `cosmos/predict2.py`), Krea 2 (`comfy/ldm/krea2/`), MOSS-Image (`comfy/ldm/moss_image/`) loading, inference, and custom node hooks |
| **DiffSynth-Studio** | [modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) | Local clone | Apache-2.0 | Pure-PyTorch architecture cross-check (without `comfy_kitchen` kernels) and low-VRAM inference reference |

---

## License & Credit Notes
- ComfyUI is **GPL-3.0**: do not copy ComfyUI source code directly; integrate only via adapter/custom-node interfaces.
- `StableFlow` is **Non-Commercial Research Only** (Snap Inc. + FLUX.1 [dev] license): read for algorithm verification only; do not copy code.
- Record any additional referenced repos or official paper implementations here with source URL and license.

---

## Official Paper Code Repositories (`reference_repo/paper_code/`)

All 16 paper repositories are shallow-cloned (`--depth 1`, `GIT_LFS_SKIP_SMUDGE=1`) for cross-checking paper equations and hook positions during spec extraction (`docs/specs/01..04`).

| Direction | Paper (`docs/reference/paper/`) | Local Folder (`paper_code/`) | Official Repo | Commit | License | Disk Size | Notes |
|---|---|---|---|---|---|---|---|
| 01 Heatmap | `01_heatmap/2025_ICML_ConceptAttention.pdf` | `ConceptAttention` | [helblazer811/ConceptAttention](https://github.com/helblazer811/ConceptAttention) | `0db4b55` | MIT | 40.7 MB | Full package (`concept_attention/`, FLUX/SD3/CogVideoX) |
| 01 Heatmap | `01_heatmap/2025_ICCV_ExploringMMDiT.pdf` | `ExploringMMDiT` | [SNU-VGILab/exploring-mmdit](https://github.com/SNU-VGILab/exploring-mmdit) | `41b626d` | Unspecified | 83.9 MB | Attention visualization & editing notebooks (SD3/SD3.5/FLUX.1) |
| 01 Heatmap | `01_heatmap/2025_NeurIPS_Seg4Diff.pdf` | `Seg4Diff` | [cvlab-kaist/Seg4Diff](https://github.com/cvlab-kaist/Seg4Diff) | `bb500ca` | Unspecified | 53.2 MB | Zero-shot seg & LoRA scripts + caption JSONs |
| 02 Layer | `02_layer_selection/2025_CVPR_StableFlow_VitalLayers.pdf` | `StableFlow` | [snap-research/stable-flow](https://github.com/snap-research/stable-flow) | `c99ac0e` | Non-Commercial Research | 14.9 MB | Reference only (non-commercial license) |
| 02 / 03 | `2025_NeurIPS_LocalizingKnowledgeDiT.pdf` | `LocalizingKnowledgeDiT` | [ArmanZarei/DiT-Knowledge-Localization](https://github.com/ArmanZarei/DiT-Knowledge-Localization) | `c7e99e7` | Unspecified | 7.9 MB | Code repo (separate from project-page repo) |
| 03 Tag | `03_tag_influence/2025_ICCV_TACA.pdf` | `TACA` | [Vchitect/TACA](https://github.com/Vchitect/TACA) | `a22a7ec` | MIT (per README) | 36.3 MB | Inference scripts + teaser assets (LoRA weights on HF) |
| 03 Tag | `03_tag_influence/2025_ICLR_TextGenParamLocalization.pdf` | `TextGenParamLocalization` | [luk-st/t2i-text-localization](https://github.com/luk-st/t2i-text-localization) | `338ea2f` | Unspecified | 32.1 MB | Localization & editing scripts (SDXL/SD3/IF) |
| 03 Tag | `03_tag_influence/2025_NeurIPS_DiT-ST.pdf` | `DiT-ST` | [zy-ansel/DiT-ST](https://github.com/zy-ansel/DiT-ST) | `48bd7d0` | MIT | 5.7 MB | Split-text conditioning code |
| 03 Tag | `03_tag_influence/2026_CVPR_SAP.pdf` | `SAP` | [TDPC2025/SAP](https://github.com/TDPC2025/SAP) | `0cfa5b0` | MIT | 3.5 MB | Cloned with `GIT_LFS_SKIP_SMUDGE=1` (code + prompts only) |
| 03 Tag | `03_tag_influence/2026_ICLR_PCI_TemporalConceptDynamics.pdf` | `PCI` | [adagorgun/PCI-Prompt-Controlled-Interventions](https://github.com/adagorgun/PCI-Prompt-Controlled-Interventions) | `ff7c7dc` | MIT | 28.6 MB | Training-free PCI code & notebooks |
| 04 MI | `04_verification_mi/2025_NeurIPS_GatedAttention.pdf` | `GatedAttention` | [qiuzh20/gated_attention](https://github.com/qiuzh20/gated_attention) | `f4c2a5f` | MIT | 1.6 MB | Code & visualization scripts (weights on HF) |
| 04 MI | `04_verification_mi/2026_arXiv_TemplateRegisters.pdf` | `TemplateRegisters` | [Met4physics/DiT-Interpretability](https://github.com/Met4physics/DiT-Interpretability) | `4716dec` | Unspecified | 0.6 MB | **Code not yet released** (README + teaser only) |
| 04 MI | `04_verification_mi/2026_CVPR_DiffusionProbe.pdf` | `DiffusionProbe` | [Alibaba-VELLDEPTH/DiffusionProbe](https://github.com/Alibaba-VELLDEPTH/DiffusionProbe) | `0e56916` | Apache-2.0 | 28.7 MB | Probe configs/models/scripts (weights on ModelScope) |
| 04 MI | `04_verification_mi/2026_CVPR_DiTRelationCircuits.pdf` | `DiTRelationCircuits` | [Animadversio/DiT-Relation-Circuits](https://github.com/Animadversio/DiT-Relation-Circuits) | `f4a4832` | Unspecified | 0.03 MB | **Code not yet released** (README only) |
| 04 MI | `04_verification_mi/2026_CVPR_ProbeSelect.pdf` | `ProbeSelect` | [Guhuary/ProbeSelect](https://github.com/Guhuary/ProbeSelect) | `db75742` | Unspecified | 94.5 MB | Includes two `.ckpt` probes (`SD3M_IR_ema.ckpt`, `SD3L_IR_ema.ckpt`) |
| 04 MI | `04_verification_mi/2026_ICML_AttentionSinksDiT.pdf` | `AttentionSinksDiT` | [wfz666/ICML26-attention-sink](https://github.com/wfz666/ICML26-attention-sink) | `c021b56` | MIT | 3.9 MB | Training-free attention-sink intervention scripts |
