# Method Spec Template

Use this template when extracting verified implementation specs directly from `docs/reference/paper/` PDFs (via `.venv` `pymupdf`).

---

## Paper: `<Filename>` (`<arXiv / Venue>`)

- **Verified Pages / Sections**: (e.g., Sec 3.2, Eq. 4–6, Appendix B)
- **Official Repo Checked**: (URL or N/A)

### 1. Exact Mechanism & Equations
- **Input / Hook Point**: (exact tensor, pre/post-norm, pre/post-softmax, pre/post-gate)
- **Formula / Algorithm**: (exact formula from paper)
- **Key Hyperparameters**: (layers, step window, scale factors, CFG handling)

### 2. Paper vs XAI_DiT Customizations
| Item | Original Paper | XAI_DiT Adaptation (Anima / MOSS-Image / Krea 2) | Reason |
|---|---|---|---|
| | | | |

### 3. Verification Invariants
- **Unit Test (CPU)**: (e.g., no-op error = 0, masked prob = 0, reference math match)
- **Sanity Reproduction (Colab GPU)**: (key qualitative/quantitative check from paper)
