# EXP-XXX: <Experiment Title>

- **Date**: YYYY-MM-DD
- **Design Basis**: `docs/experiment_design.md` (section/item)
- **Target Model**: Anima / Krea 2 / MOSS-Image
- **Runtime**: (e.g., Colab L4 24GB, precision, ComfyUI commit / branch)

---

## 1. Goal
*(1–2 lines on what this experiment verifies)*

## 2. Setup
- **Prompt / Counterfactual Prompt**:
- **Seed / Steps / CFG**:
- **Intervention**: (blocks, step window, replacement value, controls)

## 3. Results
| Condition | Pixel Metric (e.g., LPIPS) | Semantic Metric (e.g., VQA) | Output Path |
|---|---|---|---|
| Baseline (no-op) | 0.000 | — | `./orig.png` |
| Intervention | | | `./patched.png` |
| Random Control | | | `./rand.png` |

## 4. Verdict & Notes
*(2–3 lines grounded in numbers. If an unexpected anomaly occurred, log it in `issues/` and link here.)*
