# Plan: Real-World Accelerated XAI Compatibility & Fidelity Evaluation

> **Scope**: Core v2 production-compatibility evaluation (integrated into [`plan/v2_practical_control_replan.md`](v2_practical_control_replan.md) §4–§5).
> **Reference**: See [`docs/acceleration.md`](../docs/acceleration.md) for the full catalog of common and model-specific acceleration options and hook compatibility.

---

## 1. Goal
Evaluate whether XAI_DiT's 4 core capabilities (**E1 Heatmap · E2 Layer Selection · E3 Tag Influence · E4 MI Verification**) remain **accurate, stable, and fast** when combined with real-world ComfyUI acceleration stacks on a single GPU (L4 24 GB).

---

## 2. Test Matrix (vs Unaccelerated XAI Baseline)

| Stack | Components | Target Models | Key Questions |
|---|---|---|---|
| **A. Distilled Few-Step** | 4–8 step **Turbo / DMD / eDMD LoRA** (`Anima-Turbo`, MOSS `eDMD`, `Krea-2-Turbo`) | Anima, MOSS, Krea 2 | Do heatmaps (`E1-1`), block rankings (`E2-2`), `Early/Mid/Late` localization (`E3`), and `S/R/P` + Gate profiles (`E4`) preserve the teacher model's patterns at 4–8 steps? |
| **B. Step Caching** | **`EasyCache` / `LazyCache`** or **`Spectrum`** / **`TeaCache`** | Anima, MOSS, Krea 2 | How much does skipping cached steps distort accumulated heatmaps (`E1-1`) or block-third `Logit Knockout` (`E3`)? What speedup is achieved? |
| **C. Kernel / Runtime Fast Path** | **`--use-ck-attention`** + **`--fast`** + **`TorchCompileModel`** | Anima, MOSS, Krea 2 | Does `exact_noop == 0.0` (or bounded numerical tolerance) hold? Does `TorchCompileModel` avoid hook graph-break regressions? |
| **D. Combined Real-World Stack** | Quantized L4 weight + (`Turbo LoRA` OR `EasyCache`/`Spectrum`) + `--use-ck-attention` / `--fast` | Anima, MOSS, Krea 2 | End-to-end wall-clock latency & VRAM for practical ComfyUI user workflows (`XAIDiTSession` + `XAIDiTHeatmap`). |

---

## 3. Pass / Evaluation Criteria
1. **Stability**: Zero NaNs, zero black images, zero OOM on L4 24 GB.
2. **Speedup**: Measure `sec_per_forward` and total wall-clock time (`core/profiling.py`) against the unaccelerated baseline.
3. **XAI Fidelity (vs Unaccelerated Baseline)**:
   - `E1-1` Heatmap spatial correlation (Pearson / Spearman or top-k IoU vs baseline map).
   - `E2-2` Top-20% block overlap (Jaccard / Spearman rank correlation).
   - `E3` Block-third (`Early / Mid / Late`) dominant-region agreement.
   - `E4-2 / E4-3` `S/R/P` template share & Krea 2 Gate sparsity drift.
