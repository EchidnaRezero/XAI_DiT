# v2 Replan: Pretrained Semantic-Unit Control & Paper-Faithful Implementation

Created: 2026-11-25
Related: [`issues/002_docs_vs_implementation_drift.md`](../issues/002_docs_vs_implementation_drift.md), [`plan/design_updates_pending.md`](design_updates_pending.md), [`plan/accel_xai_eval.md`](accel_xai_eval.md), [`docs/acceleration.md`](../docs/acceleration.md), [`plan/status.md`](status.md)

---

## 1. Strategic Reframing: Pretrained Working Models ≠ From-Scratch Collapse Ablations

### Problem with M1/Step-2 Framing
Several M1 experiments focused on **destructive image-collapse ablations** (`E2-1` whole-block bypass collapsing `[0, 1, 12]`, `E4-1` hard `-1e4` sink masking causing `3.68×` meltdown, `E4-3` early-block gate-open distortion).
- Destructive ablations are useful when **training or debugging a model from scratch** (or as a one-time architectural sanity check).
- For **already-pretrained, high-quality image generation models** (Anima, MOSS-Image, Krea 2 Raw), breaking the generated image is neither a useful feature nor an actionable XAI goal.

### v2 Core Principle
All core features, experiments, and showcase stories must operate **while keeping the generated image intact and high-quality**, focusing on **interpretable observation and manipulation of meaningful semantic units**:

1. **Observe (1st Showcase — Complete)**:
   - Per-tag / per-concept attention heatmaps (`E1-1`, `E1-2` ConceptAttention) to verify where each prompt unit lands, check multi-object separation, and compare model reactions 1:1.
2. **Localize Causal Control Windows (2nd Showcase Core)**:
   - Identify **which block depth** (e.g., Anima `Mid [9..18]`) and **which denoising step window** (e.g., `PCI` early `steps 0–6`, `σ ≥ 0.92`) causally control a specific semantic unit (style · character · outfit · prop · spatial layout) **without disturbing the rest of the image**.
3. **Manipulate Semantic Units Non-Destructively (Practical Tooling)**:
   - **Per-block / per-step prompt routing (`kv_patch` / `ctx` swap)**: Swap a prop, outfit, or style cleanly while preserving pose, face, and composition (validated on Anima `P-A1`: `Mid [9..18]` `sketchbook → glowing paper lantern`, `transplant_fraction = +0.483`).
   - **Per-tag strength steering (`S`-only TACA / additive log-bias $\delta_j$)**: Strengthen or soften individual semantic tags without amplifying non-semantic template registers (`R`).
   - **Early-window multi-stage / regional control (SAP / relation control)**: Fix multi-character framing or attribute leakage during early layout formation while keeping final quality intact.

### Demoted to Optional One-Time Diagnostics (Out of Core Feature Scope)
- `E2-1` Stage 1 whole-block bypass collapse (`[0, 1, 12]`)
- `E4-1` hard `-1e4` sink-masking meltdown (`3.68×` collapse)
- `E4-3` destructive early-block gate-open (`gate <- 1.0`) ablation

*(Keep existing M1 results in `experiments/` as architectural reference records; do not expand them into core features or showcase demos.)*

---

## 2. Audit of Downscaled / Simplified Paper Implementations

During M1/Step-2 execution, several paper methods were arbitrarily downscaled or substituted with toy proxies relative to their verified specs in `docs/specs/01~04`. Each item below is explicitly categorized for v2 remediation:

| Method & Paper | Verified Paper Mechanism (`docs/specs/`) | Arbitrary Downscaling / Shortcut in Current Code | v2 Action |
|---|---|---|---|
| **`E2-1` StableFlow** (`CVPR 2025`) | **2-Stage**: Stage 1 vitality (DINOv2 ViT-g/14, $k=64$ prompts) → **Stage 2 Vital-Layer Image K/V Injection** ($K_{\text{img}}^{\text{tgt}} \leftarrow K_{\text{img}}^{\text{src}}, V_{\text{img}}^{\text{tgt}} \leftarrow V_{\text{img}}^{\text{src}}$ at $\mathcal{V}$) for structure-preserving editing | Stopped at Stage 1 whole-block bypass on 1 prompt (`P-A1`) with `latent_cos` proxy; **Stage 2 editing was never implemented** | **Upgrade**: Implement Stage 2 vital-layer K/V injection for structure-preserving semantic editing and benchmark against `Mid` cross-attn `kv_patch` |
| **`E2-2` LocalizingKnowledgeDiT** (`NeurIPS 2025`) | Separate **ranking vs held-out eval** prompt sets; neutral-prompt text-state swap at $K \in \{10\%, 30\%, 40\%, 50\%\}$ evaluated with CLIP-T | Ranked & evaluated on the **same single prompt**; fell back to `logit_knockout` on length mismatch; unnormalized `raw_mvo` biased by Block 27 `W_o` norm spike (`-0.128` prop transplant) | **Fix**: Normalize `Cont^(l)` by layer output/residual norm (or use causal `Mid` scoring), separate ranking/eval prompts, and verify via actual prop/attribute swap |
| **`E3-1` TextGenParamLocalization** (`ICLR 2025`) | **Layer & head-level** text K/V patching with `i >= timestep_start_patching` (skipping initial noisy steps to preserve layout) | Ran only coarse 1-prompt `red → blue scarf` Early/Mid/Late block-third sweeps on an unrelated prompt; no head-level localization | **Upgrade**: Run layer + head-level K/V patching on the **unified prompt set** with proper `i >= start` step-window sweep for clean attribute/prop swaps |
| **`E3-2` PCI** (`ICLR 2026`) | Multi-seed ($8\text{–}16$ filtered seeds) insertion curves scored by **VQA (yes/no)** to compute $\tau_{50}, \tau_{70}, W$ | Single seed (`seed=42`) scored with a `128×128` RGB pixel distance `cis_proxy` | **Fix**: Run multi-seed (4–8 seeds) and replace `128×128` pixel proxy with offline **VQA / OpenCLIP** concept-presence scoring |
| **`E3-3` TACA** (`ICCV 2025`) | `img_to_txt` logit scaling with explicit text-key/template handling (plus additive log-bias $\delta_j$ for sign-independent per-tag steering) | Uniform multiplicative $\gamma_0=1.2$ over **all** text tokens (including `R` template tokens on MOSS-Image → `0.2674` oversaturation/explosion); additive per-tag steering never run | **Upgrade**: Implement **`S`-only scaling** (fixes MOSS `92.3%` `R` explosion) and **additive per-tag log-bias $\delta_j$** for controlled tag strength boosting/softening |
| **`E3-4` SAP + DiT-ST** (`CVPR 2026` / `NeurIPS 2025`) | SAP: multi-stage proxy→full schedule in $\sigma$ fractions; DiT-ST `s_rel` calibrated on trained 40-step split-text DiT | Wired uncalibrated `DiT-ST s_rel` directly into SAP on pretrained Anima → fired at `step 25/30` (`σ=0.3756`), making Stage 2 a near no-op (`delta = 0.0137`) | **Fix**: Decouple SAP from late `s_rel`; use PCI-aligned early-$\sigma$ switch windows (`steps 2–6`, `σ ≥ 0.90`) for multi-character / complex layout control |
| **`E4-1` AttentionSinksDiT** (`ICML 2026`) | Both dynamic sink masking AND **non-destructive static score/value interventions** (`logit_bias`, `prob_scale`, `V` lerp) with CLIP-T/LPIPS equivalence checks | Only ran destructive `-1e4` masking on grouped `P` span (`cond_only`, `[7,14,21]`) | **Demote**: Archive destructive masking as one-time diagnostic; only revisit soft `logit_bias` if needed for attention-map cleanup |
| **`E4-4` DiT Relation Circuits** (`CVPR 2026`) | **Object segmentation mask** inner product (Attention Map Synopsis, `cond`/`uncond` separated) + **token-specific single-head ablation** scored on relation & object accuracy | Replaced segmentation masks with a hardcoded left/right red-vs-blue pixel heuristic (`color_layout`) on 1 seed | **Fix**: Use **Grounded-SAM2 / detector masks** for Attention Map Synopsis and test non-destructive relation/entity steering on clean `P-A3` (`2girls`) |
| **Offline Evaluators** (`output_spec.md`) | **DINOv2 ViT-g/14**, **OpenCLIP / SigLIP**, **LPIPS**, **Grounded-SAM2**, **VQA** | Replaced entirely with training-free RGB `mean_abs` / `latent_cos` / `128×128` proxies | **Build**: Implement standalone offline GPU evaluator script (`eval_offline.py`) running sequentially over saved PNGs |

---

## 3. Showcase-to-Production Continuum (Toward the Practical End-Goal)

Everything in v2 extends directly along the storyline established in the current 2-page showcase preview, moving from single-case proof-of-concept to practical ComfyUI tooling:

| Stage | Deliverable | Practical User Capability | Status |
|---|---|---|---|
| **Stage 1** | [`showcase/v2_01_heatmap.html`](../showcase/v2_01_heatmap.html) | **Observe**: Multi-model XAI heatmaps (`Anima` · `MOSS-Image` · `Krea 2 Raw`) — see where every prompt tag lands, verify multi-object separation, and compare model reactions 1:1 | **Complete** (preview verified) |
| **Stage 2** | [`showcase/v2_02_control_wip.html`](../showcase/v2_02_control_wip.html) | **Diagnose → First Control**: Anima Kanao (`P-A1`) 3-step story (`1. Observe → 2. Find Causal Window [Mid 9..18, Steps 0–6] → 3. Swap Prop [sketchbook → lantern]`) | **WIP** (Anima PoC complete) |
| **Stage 3 (v2 Core)** | Completed 2nd Showcase + 3-Model Control Runners | **Generalized Semantic Control**: Expand from 1 prop swap on Anima to **prop · outfit · style · multi-character layout control** across **Anima · MOSS-Image · Krea 2 Raw** (`Mid` `kv_patch`, StableFlow Stage 2, `S`-only TACA / additive tag bias $\delta_j$, early-$\sigma$ SAP) with real semantic metrics | **Next Up** |
| **Stage 4 (End-Goal)** | Production ComfyUI Nodes & Workflows (`src/xai_dit/comfyui/`) | **Real-World Drop-In Tooling**: Practical ComfyUI nodes (`Heatmap Debugger`, `Block/Step Prompt Router`, `Per-Tag Strength Dial`, `Early-Window Layout Switch`) that work out-of-the-box under users' actual **optimized/accelerated** inference stacks | **v2 Target** |

---

## 4. Real-World Production Optimization Compatibility (v2 Core Requirement)

Real ComfyUI users do not run unaccelerated 30-step baselines — they run quantized weights, fast attention kernels, step-caching, `torch.compile`, and 4–8 step distilled LoRAs. Making v2 XAI observation and semantic control work cleanly **with optimizations enabled** is a first-class v2 requirement.

- **Optimization Catalog & Per-Model Constraints**: Route to [`docs/acceleration.md`](../docs/acceleration.md) (common flags, model-specific precision/LoRAs, forbidden combinations such as global `SageAttention` on Krea 2 Raw or forced FP16 on Anima).
- **Compatibility & Fidelity Test Matrix**: Route to [`plan/accel_xai_eval.md`](accel_xai_eval.md) (Stacks A–D, stability/speedup/XAI-fidelity pass criteria).

### How Each v2 Method Adapts Under Production Optimizations

| Optimization Stack (`docs/acceleration.md`) | Strategy for v2 Observation & Semantic Control | Key Constraint / Verification |
|---|---|---|
| **Quantization (`int8_convrot`, `fp8_scaled`)** | Native support across all v2 methods; route `Cont^(l)` / `kv_patch` through adapter module calls | Already verified `exact_noop = 0.0` on L4 |
| **Fast Attention (`--use-ck-attention`, `FlashAttention`, `SageAttention`*)** | **Zero-overhead for `kv_patch` / `SAP`**: prompt routing & stage switching do not need `probs`, so they stay **100% on the fast kernel**.<br>**Selective fallback for `E1-1` / `TACA`**: fall back to chunked SDPA **only** on specifically hooked blocks/steps (`cond` branch only), keeping unhooked blocks/steps & `uncond` accelerated | *`SageAttention` strictly forbidden on Krea 2 Raw (NaN/black); verify `--fast` + `--use-ck-attention` graininess tradeoff |
| **Step Caching (`EasyCache` / `LazyCache`, `TeaCache`, `Spectrum`)** | **Protect early causal window**: set cache `start_percent` after the early lock-in window (`steps 0–6`, `σ ≥ 0.90`) so early `SAP` / `TACA` / layout controls execute on real steps, while caching accelerates mid/late denoising | Measure heatmap & control-edit fidelity vs uncached baseline (`plan/accel_xai_eval.md` Stack B) |
| **Distilled Few-Step LoRAs (4–8 step `Anima-Turbo`, MOSS `eDMD`, `Krea-2-Turbo`)** | Express all v2 temporal windows (`Mid` swap window, `TACA` threshold, `SAP` switch) in **normalized $\sigma$ / step-fraction coordinates** rather than hardcoded 30-step indices | Verify whether `Mid` block routing, `S`-only TACA, and heatmaps retain teacher-like localization at 4–8 steps (`Stack A`) |
| **`--fast` & `TorchCompileModel`** | Isolate Python hooks so non-hooked blocks remain compiled without graph-break thrashing; prefer tensor-level `ctx`/`K,V` swaps that are compile-friendly | Benchmark end-to-end wall-clock & VRAM (`Stack C/D`) |

---

## 5. v2 Execution Roadmap

### Track A — 1st Showcase Finalization (Local, No GPU)
1. Iterate on [showcase/v2_01_heatmap.html](../showcase/v2_01_heatmap.html) and [showcase/v2_02_control_wip.html](../showcase/v2_02_control_wip.html) per user feedback.
2. Once approved, promote to `showcase/index.html` (and clean up temporary preview files).

### Track B — Practical Semantic Control Suite (Next Colab L4+ Session, Upon User Approval)
1. **Semantic Editing Core (`Anima` → `MOSS-Image` & `Krea 2 Raw`)**:
   - Generalize `e2_prop_transplant_test.py` into a clean **3-model semantic-unit control runner** supporting:
     - **Block-window cross-attn / joint-attn `kv_patch`** (prop swap, outfit swap, style swap while keeping character/pose/composition locked).
     - **StableFlow Stage 2** vital-layer image K/V injection (side-by-side comparison vs `Mid` `kv_patch`).
     - **`S`-only TACA & additive per-tag log-bias $\delta_j$** (non-destructive per-tag strength dial; resolves MOSS `R`-register explosion).
     - **Early-$\sigma$ SAP (`steps 2–6`)** on clean multi-character prompts (`P-A3` clean variant).
2. **Cross-Model Rollout on Verified Shared Pool**:
   - Use the verified 3-model shared pool (`emilia`, `arona`, `raiden shogun` + `anime coloring`; `@artist` Anima-only).
   - Run deferred model-specific observation methods where useful for practical control:
     - `E1-2` **ConceptAttention** ($o_c o_x^T$ output-space heatmaps on MOSS-Image & Krea 2 Raw).
     - `E4-2` `R→I` causal masking (only if needed to verify MOSS/Krea 2 `S`-only control behavior).
3. **Offline Semantic Evaluator (`src/xai_dit/eval/offline_metrics.py`)**:
   - Score saved PNGs with **OpenCLIP / SigLIP** (target concept presence vs source suppression), **DINOv2** (structure/identity preservation), **LPIPS** (non-target region stability), and **Grounded-SAM2 / VQA** (spatial & object accuracy).
4. **Production Optimization Compatibility Pass (`plan/accel_xai_eval.md` + `docs/acceleration.md`)**:
   - Validate v2 heatmaps and semantic-unit controls (`kv_patch`, `S`-only TACA, early-$\sigma$ SAP) under real-world ComfyUI acceleration stacks (`Turbo/eDMD LoRA`, `EasyCache`/`Spectrum`, `--use-ck-attention` + `--fast`, `TorchCompileModel`) and package validated controls into ComfyUI nodes (`src/xai_dit/comfyui/`).

### Track C — Docs & Git Reconciliation (Upon User Approval)
1. Reconcile [`issues/002_docs_vs_implementation_drift.md`](../issues/002_docs_vs_implementation_drift.md) (`#1–#12`) + this v2 reframing into `docs/` and `experiments/design/` (leaving frozen `manifest_*.json` untouched).
2. Update visitor-facing `README.md` and commit/push via `GPT-6.1-Sol` (`Echidna` account).
