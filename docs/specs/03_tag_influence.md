# Direction 03: Tag Influence & Temporal/Causal Intervention Specs

Verified method specifications extracted directly from `docs/reference/paper/2025_NeurIPS_LocalizingKnowledgeDiT.pdf` and `docs/reference/paper/03_tag_influence/*.pdf`, cross-checked against `reference_repo/paper_code/`, `reference_repo/ComfyUI/`, and `docs/reference/models/{anima,moss_image,krea2}.md`, and **cross-reviewed by GPT 6.2 Astra & Claude Fable 5.2 (`high`)**.

> **Common Execution & Test Policy**:
> - **Local vs Colab Testing**: Per `AGENTS.md`, PyTorch is NOT installed locally; CPU unit tests run either on a Colab CPU runtime (PyTorch) or via pure-NumPy reference math checks locally.
> - **Colab L4 (24 GB) Feasibility**: Anima (~2 GB BF16) fits natively; MOSS-Image (5B DiT ~10 GB + Qwen3.5-9B ~18 GB) requires pre-encoding prompts and unloading the text encoder before DiT (and swapping VQA/CLIP evaluators in/out); **Krea 2 base (12.8B DiT ~25.6 GB BF16 > L4 24 GB)** requires FP8/GGUF quantization or block CPU offload on L4 (disclosing numerical precision change; note FP8 breaks bitwise GPU equality) or A100-80GB / H100.
> - **CFG Branch Routing**: Always identify `cond` vs `uncond` via adapter metadata (`transformer_options["cond_or_uncond"]` in ComfyUI), never by hard-coding batch index `[1]` or `[0]`.

---

## Paper 1: `2025_NeurIPS_LocalizingKnowledgeDiT.pdf` (Tag/Concept Influence Role)

- **Verified Pages / Sections**: Sec 3.2 (pp. 4–5, Fig. 3), Sec 4.2 (p. 7)
- **Official Repo Checked**: `reference_repo/paper_code/LocalizingKnowledgeDiT` (commit `c7e99e7`, `attention_processor.py:L315-560`, `calc_attn_cont_varients.py:L1-93`)
- **Cross-Reference**: Core attention-contribution formula $\text{Cont}^{(t,\ell)}(\kappa)$ is fully specified in [`02_layer_selection.md`](02_layer_selection.md). In Direction 03, we apply it **per tag/phrase $\kappa \in \text{Prompt}$** to produce a 2D **(tag × block)** and **(tag × timestep)** influence matrix, and verify causality via **neutral-prompt attention-input text-state swapping** (`attention_processor.py:L477`) at Top-$K$ blocks $\mathcal{B}_K^\kappa$.
- **Token-Length Alignment Policy (Mandatory for MOSS-Image & Krea 2)**:
  - Replacing a prompt with a neutral prompt that omits $\kappa$ changes token length $L$, which shifts MOSS-Image's image RoPE (`seq_pos = L`) and breaks Krea 2's `[:T]` shape.
  - **Policy**: Use **token-length-matched neutral substitutes** (assert `len(tok(p)) == len(tok(p_neutral))` with the model's exact tokenizer), overwriting attention-input text states at $\mathcal{B}_K^\kappa$ (not block output residuals, which would alter downstream blocks). On Krea 2 only (text RoPE = `(0,0,0)`), rectangular K/V-level swap is also mathematically valid; exclude/track the 5 tail template tokens separately.
- **Verification Invariants**: Zero-contribution when tag attention/value is zero; neutral-prompt swap on empty block set $\mathcal{B}=\emptyset$ is a strict no-op (`max_abs_diff == 0`).

---

## Paper 2: `2025_ICLR_TextGenParamLocalization.pdf` (ICLR 2025)

- **Title**: *Precise Parameter Localization for Textual Generation in Diffusion Models* (Staniszewski & Cywiński, arXiv:2502.09935)
- **Verified Pages / Sections**: Sec 4.1–4.3 (pp. 4–6, Fig. 1), Appendix B (p. 15), Appendix I (p. 21)
- **Official Repo Checked**: `reference_repo/paper_code/TextGenParamLocalization` (`luk-st/t2i-text-localization`, commit `338ea2f`)
  - `diffusers/src/diffusers/models/attention_processor.py` (L1206–1287 for `JointAttnProcessor2_0`, L1797–1840 for `AttnProcessor2_0`)
  - `diffusers/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py` (L842–863), `src/edit_sd3_glyph_simple.py` (L43)

### 1. Exact Mechanism & Equations
- **Input / Hook Point**:
  - **Activation Patching of Text Keys & Values** (`attention_processor.py:L1212-1275`):
    - Original SD3 processor (no QK-norm) caches text `K_T = add_k_proj(e_T)` and `V_T = add_v_proj(e_T)` reshaped to `[B, H, L, d_h]` for layer-level (`attn_idx_to_patch`) or head-level (`attn_heads_idx_to_patch`) patching.
    - Corresponding hooks in our 3 models:
      - **Anima**: `cross_attn.k_proj` / `v_proj` (`predict2.py:178`) — since `ctx` is static across steps, caching once (or swapping `ctx`) suffices (Fig. 1B).
      - **MOSS-Image**: text slice `[N:N+L]` immediately after QKV split (`moss_image/model.py:141`), pre-RoPE (QK-norm is per-token/per-head, so pre- or post-QK-norm is equivalent).
      - **Krea 2**: text slice `[:T]` after `wk`/`wv` (or ComfyUI `attn1_patch`, which sits post-QK-norm / pre-RoPE / pre-GQA-repeat; `krea2/model.py:80-90`).
- **Formula / Algorithm (Sec 4.1, App. B; `pipeline_stable_diffusion_3.py:L842-863`, `attention_processor.py:L1262`)**:
  - **CRITICAL Timestep Convention**: In the official code, `timestep` is the **0-indexed generation step counter `i` counted from the noise end** (`i = 0, 1, ..., S-1`), and patching activates for **`i >= timestep_start_patching`** (e.g., `TIMESTEP_START_PATCHING = 2` of 28 steps in `edit_sd3_glyph_simple.py:L43`, matching App. B *"patch from $t_s=26$ down to $0$"* — i.e., **skipping the first `start` noisy steps** to preserve source layout, NOT patching only early steps!).
  1. Cache $K_T^{(i,\ell,h)}, V_T^{(i,\ell,h)}$ during target prompt $p_T$ generation.
  2. During source prompt $p_S$ generation (same initial noise), for `i >= timestep_start_patching`, overwrite $K_S^{(i,\ell,h)} \leftarrow K_T^{(i,\ell,h)}$ and $V_S^{(i,\ell,h)} \leftarrow V_T^{(i,\ell,h)}$ at selected layers/heads.
- **CFG Handling**: Official TextGen code caches/patches the full `[uncond, cond]` batch (`pipeline:L847`, `processor:L1287`). Note that on MOSS-Image and Krea 2, deep text states absorb image information, so `uncond` K/V can differ between trajectories even with identical negative prompts; explicitly specify `cond`-only (via `cond_or_uncond`) vs `both` as an intervention flag.

### 2. Paper vs XAI_DiT Customizations
| Item | Original Paper | XAI_DiT Adaptation (Anima / MOSS-Image / Krea 2) | Reason |
|---|---|---|---|
| **Patching Scope** | Glyph/rendered text swaps | Generalize to **any token-length-matched tag/attribute swap** ($p_S \to p_T$) at layer and head granularity. | General tag-influence localization. |
| **Krea 2 GQA Granularity** | Standard MHA (`H_q = H_kv`) | Krea 2 uses `repeat_interleave(4, dim=1)` (`krea2/model.py:94-97`), mapping 12 KV heads to 48 Q heads (`h_kv = h_q // 4`). Patching KV head $g \in \{0..11\}$ affects Q-head group $\{4g, 4g+1, 4g+2, 4g+3\}$. | Matches exact Krea 2 GQA layout. |

### 3. Verification Invariants
- **Unit Test (CPU)**:
  1. **Self-Patching No-Op**: Caching from $p_S$ and patching back into $p_S$ yields `max_abs_diff == 0.0`.
  2. **Step-Window Boundary**: Verify patching is inactive for `i < start` and active for `i >= start` (catching inverted step-index bugs).
  3. **GQA Head-Group Isolation (Krea 2)**: Observe per-head output of the patched layer **before `attn.gate` and `Wo`** (same input); patching KV head $g$ leaves all 44 Q heads outside $\{4g..4g+3\}$ bitwise identical (`max_abs_diff == 0.0`).
- **Sanity Reproduction (Colab GPU)**:
  - **Pass criterion**: Unpatched baseline reproduces identically on same seed; hook fires strictly at specified `(layers, heads, i >= start)`.
  - **Empirical measurement**: Measure attribute transfer (CLIP/VLM) and layout preservation (LPIPS) for a token-length-matched attribute swap across early/middle/late layer subsets (logged as experimental observation, not assumed).

---

## Paper 3: `2026_ICLR_PCI_TemporalConceptDynamics.pdf` (ICLR 2026)

- **Title**: *Temporal Concept Dynamics in Diffusion Models via Prompt-Conditioned Interventions* (Görgün et al.)
- **Verified Pages / Sections**: Sec 3.1–3.2 (pp. 4–5, Eq. 1–2), Sec 4.2 (p. 6, Key Findings L336–339), Appendix A.2/A.4/B.1/B.2 (pp. 15–19)
- **Official Repo Checked**: `reference_repo/paper_code/PCI` (`adagorgun/PCI-Prompt-Controlled-Interventions`, commit `ff7c7dc`, MIT)
  - `pci/core/pci_runner.py` (L8–89), `pci/models/flux.py` (L50–105), `pci/core/utils.py` (L186–207 — note: `tau_from_sigmas` / `first_crossing` are unused helper functions!)

### 1. Exact Mechanism & Equations
- **Input / Hook Point**: Zero internal DiT hooks — denoising-loop conditioning switch (`pci_runner.py:L65-67`).
- **Formula / Algorithm (Sec 3.1–3.2, App. B.1–B.2)**:
  - Given base prompt $P_b$ (without concept $c$) and concept prompt $P_c$ (with $c$):
    1. **Seed resampling & negative guidance (App. A.2)**: Filter seeds so $P_b$ alone does not generate $c$. Note that in `pci_runner.py:L42, L48`, `base_ctx` receives `negative_prompt` (concept-as-negative) while `concept_ctx` receives `None` (in `flux.py:L50-67` this turns off CFG for the concept segment!). In `XAI_DiT`, explicitly configure and log `base_neg` vs `concept_neg` per experiment.
    2. For generation step index $k \in \{0, \dots, S-1\}$ (0 = noise end): run steps $i < k$ under $P_b$, then **switch at step $i = k$** (step $k$ itself uses $P_c$; `flux.py:L99-101`) and continue $i \ge k$ under $P_c$ preserving latent and scheduler state:
       $$x_0(P_b \xrightarrow{k} P_c) = \text{Denoise}\big(\text{Denoise}(x_T, P_b; 0 \to k-1),\; P_c; k \to S-1\big)$$
    3. **Concept Insertion Success (CIS) & Lock-In Metrics (App. B.1–B.2, p. 19)**:
       - Normalized noise axis: paper uses $\tau = t / 1000 \in [0, 1]$ (`1` = pure noise / early, `0` = clean / late). For our 3 rectified-flow models, use **$\tau := \sigma \in [0, 1]$** directly (**do NOT use** the unused `utils.py:tau_from_sigmas` which breaks at $\sigma = 0$ and reverses direction!).
       - Sort $(\tau, \text{CIS}(\tau))$ in **ascending $\tau$** (from clean `0` to noisy `1`) and compute:
         $$\tau_q = \min\{\tau : \text{CIS}(\tau) \ge q\} \quad (q \in \{0.5, 0.7\}), \qquad W = \tau_{0.7} - \tau_{0.5}$$
         *(Handles monotonic rising curves in $\tau$, non-crossing cases returning `NaN` / sentinel, and avoids the `first_crossing` direction bug when indexed by generation step $k$.)*
- **Key Hyperparameters & Compute Optimization**:
  - **Prefix Caching (Mandatory for Colab L4)**: Official `pci_runner.py:L61-67` re-runs from `latents0` for every $k$, costing $O(S^2)$ forwards per `(seed, concept)` (`~784 × 2` at $S=28$ with CFG). `XAI_DiT` **caches the single base trajectory $\{x_0, x_1, \dots, x_{S-1}\}$ once per seed**, cutting cost by **2×** (and uses `timestep_stride = 2` with $8\text{–}16$ seeds on Colab L4).

### 2. Paper vs XAI_DiT Customizations
| Item | Original Paper | XAI_DiT Adaptation (Anima / MOSS-Image / Krea 2) | Reason |
|---|---|---|---|
| **Axis & Lock-In Metric** | $\tau = t/1000$, $\tau_{50}, \tau_{70}, W$ | Use **$\tau := \sigma$** natively for Anima, MOSS-Image, and Krea 2; report $\tau_{50}, \tau_{70}, W$. | Exact rectified-flow noise scale; avoids unused `tau_from_sigmas` bug. |
| **Evaluation Metric** | Qwen-VL VQA binary yes/no | Primary: VQA yes/no (loaded **sequentially** after unloading DiT on Colab L4); Auxiliary: delta CLIP/SigLIP score. | Fits 24 GB L4 VRAM. |
| **CDS (Deletion)** | App. A.4 notes deletion locks in *earlier* than insertion | Optional `delete` mode ($P_c \to P_b$) alongside primary `insert` mode ($P_b \to P_c$). | Accurate to App. A.4. |

### 3. Verification Invariants
- **Unit Test (CPU)**:
  1. **Boundary & Interior Step Switch**: Switching at $k = 0$ equals pure $P_c$; switching at $k = S$ equals pure $P_b$; switching at interior $k = k^*$ verifies exact conditioning sequence (`k*` calls of $P_b$ followed by `S - k*` calls of $P_c$, total `S` updates).
  2. **$\tau_q$ Crossing Logic**: Test rising CIS in ascending $\tau$, falling CIS in generation-step order, and non-crossing curves (all above or all below $q$) to verify exact $\tau_{50}, \tau_{70}, W$ and sentinel handling.
- **Sanity Reproduction (Colab GPU)**:
  - **Empirical measurement (consistent with PCI p. 3 & p. 6 Key Findings)**: Compare $\tau_{50}, \tau_{70}$ for **global style/color tags** (which the paper finds lock in **early**, high $\sigma$) vs **local accessories** (e.g., `"wearing a hat"`, which the paper finds lock in **mid-to-late**, lower $\sigma$) — report observed $\tau_{50}, \tau_{70}, W$ across Anima, MOSS-Image, and Krea 2 (note paper shows lock-in also varies with prompt context).

---

## Paper 4: `2025_ICCV_TACA.pdf` (ICCV 2025)

- **Title**: *Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers* (Lv et al., Vchitect)
- **Verified Pages / Sections**: Sec 3.2–3.3 (pp. 4–5, Eq. 7–11), Sec 4.1 (p. 5), Appendix A Listing 2 (p. 11, Table 6)
- **Official Repo Checked**: `reference_repo/paper_code/TACA` (`Vchitect/TACA`, commit `a22a7ec` — *note: no `LICENSE` file in checkout*)
  - `infer/pipeline_taca_flux.py` (L53–148, L894), `infer/pipeline_taca_sd3.py` (L85–193, L156–164, L1234–1250)

### 1. Exact Mechanism & Equations
- **Input / Hook Point**:
  - **Pre-softmax scaled attention logits** $s_{ij} = \langle q_i, k_j \rangle / \sqrt{d_h}$ (post-QK-norm, post-RoPE, pre-softmax) in the **`img_to_txt` sub-block** (`token_q ∈ img_indices` AND `token_kv ∈ selected_txt_indices`).
  - **Fast Equivalent Path (App. A Listing 2 — Required on Colab L4)**: Scaling `K_txt` by $\gamma$ **after QK-norm & RoPE** (and restoring text-query outputs) works with fused SDPA/GQA without `flex_attention`'s 0.74× slowdown (Table 6). **Important L4 hardware constraint**: PyTorch `flex_attention` hits Triton shared-memory OOM on NVIDIA L4 (`Required: 131072 > Hardware limit: 101376`), so **Listing 2 or query-chunked manual softmax is mandatory on L4** (defer `flex_attention` comparison to A100/H100).
- **Formula / Algorithm (Eq. 10–11, Sec 4.1; `pipeline_taca_flux.py:L894`, `pipeline_taca_sd3.py:L1234`)**:
  $$P_{\text{vis}\to\text{txt}}^{(i,j)} = \frac{\exp\!\big(\gamma(t)\, s_{ij}^{vt}\big)}{\sum_{k=1}^{N_{\text{txt}}} \exp\!\big(\gamma(t)\, s_{ik}^{vt}\big) + \sum_{k=1}^{N_{\text{vis}}} \exp\!\big(s_{ik}^{vv}\big)}, \quad \gamma(t) = \begin{cases} \gamma_0 & \sigma_t \ge \sigma_{\text{thresh}} \text{ (or } i < 3\text{)} \\ 1 & \text{otherwise} \end{cases}$$
- **Key Hyperparameters (Original Paper)**:
  - $\gamma_0 = 1.2$ (`pipeline_taca_flux.py:L116`).
  - **Threshold**: Paper Sec 4.1 uses **$t_{\text{thresh}} = 970$** (sensitivity range `930–970`, i.e., **first ~10% / 3 of 30 steps**; both official pipelines enforce `if i < 3`). Express threshold in **$\sigma$** ($\sigma \ge 0.97$) so it transfers cleanly across schedules.
  - **CFG Handling**: Official SD3 code (`pipeline_taca_sd3.py:L164, L1250`) scales logits for **both `[uncond, cond]` branches**. Scaling `cond`-only is an **XAI_DiT ablation** (default = both branches for paper match).
  - **LoRA Caveat**: Paper's full benchmark loads TACA LoRA during $\gamma$ steps (Eq. 12); training-free `XAI_DiT` runs without LoRA are not directly comparable to LoRA benchmark tables.

### 2. Paper vs XAI_DiT Customizations
| Item | Original Paper | XAI_DiT Adaptation (Anima / MOSS-Image / Krea 2) | Reason |
|---|---|---|---|
| **Exact Text-Key Masking** | `token_kv < 512` (FLUX) / `-333:` (SD3) | Mask must intersect `img_queries` with **explicit valid text-key indices** (in MOSS-Image, `key >= N` alone would erroneously include trailing reference-image keys if present; `moss_image.md` L72). Treating chat-template / 5 tail tokens as excluded vs included is an **explicit ablation** (Krea 2's 5 tail tokens carry whole-prompt state, `krea2.md` L86–87). | Prevents unintended scaling of reference images or unexamined template exclusion. |
| **Anima Cross-Attn & RMSNorm Caveat** | Joint attention only | Anima's `cross_attn.k_norm` is bias-free RMSNorm (`predict2.py:137-138, 196-197`), which **cancels any scaling applied to `ctx` or `K` before `k_norm`**! Apply TACA strictly to **post-`k_norm` `K` or pre-softmax logits**. Furthermore, because multiplicative $\gamma$ depends on logit sign (negative logits shrink under $\gamma > 1$), also support **additive log-bias** $s'_{ij} = s_{ij} + \delta_j$ for monotone per-tag steering across all 3 models. | Prevents silent no-op bug on Anima and enables sign-independent per-tag steering. |

### 3. Verification Invariants
- **Unit Test (CPU)**:
  1. **No-Op & Exact Logit Selection**: $\gamma_0 = 1.0$ matches baseline (`max_abs_diff < 1e-6`). For $\gamma_0 \neq 1$, verify `s' == gamma * s` on selected `(img_q, sel_txt_k)` entries and `s' == s` everywhere else (note: unselected tokens' post-softmax probabilities still change via denominator!).
  2. **Sign-Aware Monotonicity**: On synthetic logits with **all selected logits $> 0$**, total selected probability mass increases strictly for $\gamma_0 > 1$; with **all selected logits $< 0$**, total selected mass decreases strictly. (For additive bias $\delta > 0$, selected mass increases strictly regardless of logit sign.)
- **Sanity Reproduction (Colab GPU)**:
  - Verify $\gamma_0 = 1.0$ reproduces baseline on same seed and hook fires only on $\sigma \ge 0.97$ (or `i < 3`); compare `both` vs `cond-only` and `include_template` vs `exclude_template` ablations on MOSS-Image and Krea 2.

---

## Paper 5: `2025_NeurIPS_DiT-ST.pdf` (NeurIPS 2025)

- **Title**: *Enhancing Text-to-Image Diffusion Transformer via Split-Text Conditioning* (Zhang et al., arXiv:2505.19261)
- **Verified Pages / Sections**: Sec 3.1–3.4 (pp. 4–7, Eq. 2–9), Sec 4.1 (p. 7–8, L508–516)
- **Official Repo Checked**: `reference_repo/paper_code/DiT-ST` (`zy-ansel/DiT-ST`, commit `48bd7d0`, MIT — **README + assets only; no source code**)

### 1. Exact Mechanism & Equations (Training-Free Phase Boundary Detector, Sec 3.3.1, Eq. 5–9)
- **Input / Hook Point**: Post-softmax `img_to_txt` attention maps $A_i^{(m,h)}$ (on Anima sliced to `[:Lt]` to avoid storing 512 pad cols; on MOSS-Image/Krea 2 state explicitly whether raw unrenormalized joint-softmax slice is used) computed on the `cond` branch in **`float32`** (so $\theta = 10^{-8}$ does not underflow in fp16).
- **Formula / Algorithm (Step Index `i = 0..S-1` from noise end)**:
  1. **Attribute-Stage Boundary $i^* = s_{\text{attr}}$ (Eq. 5–7)**:
     $$\Delta_i^{(m,h)} = \frac{\|A_i^{(m,h)} - A_{i-1}^{(m,h)}\|_F}{\|A_{i-1}^{(m,h)}\|_F + \theta} \quad (\theta = 10^{-8},\;\text{fp32}), \qquad \Delta_i^{(\text{sample})} = \frac{1}{MH}\sum_{m,h} \Delta_i^{(m,h)}$$
     Average $\Delta_i^{(\text{sample})}$ across batch samples to get $\bar{\Delta}_i$, then apply moving average $\hat{\Delta}_i = \frac{1}{w}\sum_{k=0}^{w-1}\bar{\Delta}_{i-k}$ and find $i^* = \min\{i : \hat{\Delta}_i < \tau\}$.
     - **Paper defaults** (p. 8 L508–516): $w = 3$, $\theta = 10^{-8}$, $\tau = 10^{-4}$ (calibrated on 40-step trained DiT-ST; **XAI_DiT fallback**: if $\hat{\Delta}_i$ never drops below $10^{-4}$ on our pretrained models, fall back to elbow / relative drop $\hat{\Delta}_i < 0.1\,\hat{\Delta}_{\max}$).
  2. **Relation-Stage Boundary $i^\# = s_{\text{rel}} < s_{\text{attr}}$ (Eq. 8–9)**:
     - For rectified flow ($x_\sigma = (1-\sigma)x_0 + \sigma\epsilon$), define $\text{snr}_i = ((1-\sigma_i)/\sigma_i)^2$ (clamping $\sigma_i \ge \epsilon_\sigma$ to avoid $\infty$ at $\sigma=0$, and skipping duplicate coordinates). On the semantic-planning prefix $i \in \{0, \dots, s_{\text{attr}} - 1\}$, compute Eq. 8–9 discrete curvature $\kappa_i = |\Delta^2 y_i| / (1 + (\Delta y_i)^2)^{3/2}$ and take $i^\# = \arg\max \kappa_i$.
     - **Memory note**: Storing previous-step $A_{i-1}$ across all layers/heads costs $M \cdot H \cdot N \cdot L \cdot \text{bytes}$ (e.g., Anima unsliced `28×16×4096×512` fp16 = **1.75 GiB**; Krea 2 `T=200` ≈ **2.2 GiB**). Keep `prev_attn` on **CPU pinned memory** (or slice Anima to `[:Lt]` and compute via query-chunked logits + log-sum-exp over all `N+L` keys).

### 2. Paper vs XAI_DiT Customizations
| Item | Original Paper | XAI_DiT Adaptation (Anima / MOSS-Image / Krea 2) | Reason |
|---|---|---|---|
| **Adopted Scope** | Split-text parsing + 3-encoder concat + training (`L_CFM + lambda L_attn`) | Adopt **only the training-free temporal phase-boundary detector (Sec 3.3.1, Eq. 5–9)** in unified generation-step index `i = 0..S-1`. Training is out of scope. | Resolves paper's mixed step/timestep notation and keeps `XAI_DiT` training-free. |

### 3. Verification Invariants
- **Unit Test (CPU)**:
  1. **Constant Attention & Stabilizer**: $A_i = A_{i-1} \Rightarrow \Delta_i = 0.0$; $A_{i-1} = 0 \Rightarrow$ finite output in fp32 ($\theta = 10^{-8}$).
  2. **Uniform-Grid Linearity**: On a **uniformly spaced** $\text{snr}$ grid, $y = a \cdot \text{snr} + b \Rightarrow \Delta^2 y_i == 0$ and $\kappa_i == 0$ at all interior points (note: Eq. 8's uniform-grid second difference is non-zero on non-uniform grids!).
- **Sanity Reproduction (Colab GPU)**:
  - Plot $\hat{\Delta}_i$ vs step $i$ on Anima, MOSS-Image, and Krea 2 (logging whether $\tau=10^{-4}$ triggers or fallback elbow is used) and compare observed $s_{\text{attr}}$ against PCI's empirical $\tau_{50}$ window.

---

## Paper 6: `2026_CVPR_SAP.pdf` (CVPR 2026)

- **Title**: *Image Generation from Contextually Contradictory Prompts* (Huberman et al., TAU & BRIA AI)
- **Verified Pages / Sections**: Sec 4.1–4.3 (pp. 3–5, Fig. 5–7), Sec 5.3 (p. 7), Supp. (p. 11–12; GPT-4o primary, LLaMA-3.1-8B-Instruct comparison)
- **Official Repo Checked**: `reference_repo/paper_code/SAP` (`TDPC2025/SAP`, commit `0cfa5b0`, MIT)
  - `SAP_pipeline_flux.py` (L18–50, L245+)

### 1. Exact Mechanism & Equations
- **Input / Hook Point**: Zero internal DiT hooks — denoising-loop conditioning schedule (`SAP_pipeline_flux.py:L24-50, L245`).
- **Formula / Algorithm (Sec 4.2–4.3)**:
  - Given proxy prompts $\{p_1, \dots, p_n\}$ and sorted switch step indices (or normalized $\sigma$ fractions, since 50-step FLUX steps do not transfer directly): at step $i$, select the **entire conditioning bundle** for active prompt $p_k$ together (`prompt_embeds`, `pooled_prompt_embeds`, `text_ids` / sequence length / RoPE offsets — never mix embeddings from $p_k$ with length/RoPE from $p_1$!).
  - **CFG Handling**: Pre-encode all `{p_k}` bundles once; swap `cond` bundle at switch steps while keeping `uncond` fixed (ComfyUI's `ConditioningSetTimestepRange` in `nodes.py:L305` provides native reference semantics).

### 2. Paper vs XAI_DiT Customizations
| Item | Original Paper | XAI_DiT Adaptation (Anima / MOSS-Image / Krea 2) | Reason |
|---|---|---|---|
| **Prompt Decomposition & Units** | GPT-4o (or LLaMA-3.1-8B-Instruct) with 20 in-context examples; 50-step FLUX indices | Accept deterministic user-defined `StagePromptSchedule` specified in **$\sigma$ thresholds or step fractions** (no LLM dependency in core). Swap full `(text_emb, token_len, rope_ids)` bundle cleanly across Anima, MOSS-Image, and Krea 2. | Deterministic, schedule-agnostic, handles variable text lengths. |

### 3. Verification Invariants
- **Unit Test (CPU)**:
  1. **Single-Stage & Interior-Switch Check**: 1-stage schedule matches standard generation (`max_abs_diff == 0`); 2-stage schedule with interior switch $k^*$ executes exact conditioning bundle sequence (`k*` of $p_1$, `S - k*` of $p_2$, total `S` updates) with matching `(len, rope_ids)` per stage.
  2. **Schedule Validator**: Rejects unsorted switch points, out-of-range steps/$\sigma$, or empty prompt list.
- **Sanity Reproduction (Colab GPU)**:
  - Verify 1-stage schedule reproduces baseline on same seed, and run a 2-stage layout-proxy → full-prompt schedule on MOSS-Image / Krea 2 logging layout vs identity alignment.
