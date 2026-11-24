# Direction 01: Heatmap & Saliency Extraction Specs

Verified method specifications extracted directly from `docs/reference/paper/01_heatmap/*.pdf`, cross-checked against official code in `reference_repo/paper_code/` and target model docs (`docs/reference/models/{anima,moss_image,krea2}.md`), and **cross-reviewed by GPT 6.2 Astra & Claude Fable 5.2 (`high`)**.

> **Cross-Paper Terminology & Softmax-Denominator Note**:
> - **ExploringMMDiT** (Sec 3.1, Eq. 2–3; `utils/dit_attn_viz.py:L153-160`) labels the **image-query × text-key** sub-block ($Q_{\text{img}} K_{\text{txt}}^T \in \mathbb{R}^{N \times L}$) as **`T2I`** and $Q_{\text{txt}} K_{\text{img}}^T$ as **`I2T`**, reporting that `T2I` outperforms `I2T` in *most blocks* (Fig. 7, p. 5) because image queries do not compete within a single softmax row.
> - **Seg4Diff** (Sec 3.2, Eq. 4; `attention_processor.py:L1098-1111`) labels the **exact same sub-block** ($Q_{\text{img}} K_{\text{txt}}^T$) as **`I2T`** and qualitatively observes more complete object masks than `T2I` (App. B.2, p. 17, explicitly noting this is qualitative, not a formal proof).
> - Throughout `XAI_DiT`, we use the unambiguous canonical name **`img_to_txt`** ($Q_{\text{img}} K_{\text{txt}}^T$, shape `[B, H, N, L]`).
> - **Denominator difference across target models**: Anima's cross-attention softmax normalizes over **text slots only** (`L = max(512, Lt)` including 0-pad & `</s>`), whereas MOSS-Image and Krea 2 joint-attention softmax normalizes over **all `N + L` keys** (`img + txt`). Raw probability magnitudes are therefore **not** directly comparable across Anima vs MOSS-Image/Krea 2 without noting denominator scope.

---

## Paper 1: `2025_ICML_ConceptAttention.pdf` (ICML 2025)

- **Title**: *ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features* (Helbling et al.)
- **Verified Pages / Sections**: Sec 3.2 (p. 4), Sec 4.1–4.2 (pp. 4–6, Eq. 3–12), Sec 5.1–5.3 (pp. 6–9, Tables 1–4, Fig. 7), Appendix A (p. 13, Fig. 9)
- **Official Repo Checked**: `reference_repo/paper_code/ConceptAttention` (`helblazer811/ConceptAttention`, commit `0db4b55`, MIT)
  - `concept_attention/flux/dit_block.py` (L70–214), `flux/dit.py` (L110–151), `flux/pipeline.py` (L29–132)
  - `concept_attention/attention_utils.py` (L8–37), `heatmap_utils.py` (L12–195), `utils.py` (L53–67)

### 1. Exact Mechanism & Equations
- **Input / Hook Point**:
  - Multi-head attention **output vectors BEFORE output projection $W_O$ (`attn.proj`) and BEFORE AdaLN residual gate $\alpha_1$** (`flux/dit_block.py:L170-204`):
    - Image attention output: $o_x \in \mathbb{R}^{B \times N \times D}$ (rearranged from `[B, H, N, d_h]` to `[B, N, H \cdot d_h]`)
    - Concept attention output: $o_c \in \mathbb{R}^{B \times R \times D}$
  - **One-directional Concept Residual Stream** (Sec 4.1, Eq. 4–10; `flux/dit_block.py:L99-168, L210-212`):
    - Each concept $r \in \{1, \dots, R\}$ is encoded **individually** by T5, taking the **first token** (`utils.py:L55-67`), with zeroed concept positions and zeroed CLIP conditioning vector (`flux/dit.py:L110-119`).
    - At each dual-stream block $L$, concepts reuse the text branch's norm/mod/QKV/QK-norm/RoPE/proj/MLP. Attention is one-directional (Eq. 7 in paper omits $1/\sqrt{d_h}$; official code uses `F.scaled_dot_product_attention` with $1/\sqrt{d_h}$):
      $$o_c = \text{SDPA}\big(q_c,\; [k_x; k_c],\; [v_x; v_c]\big)$$
      $$c_{L+1} = c_L + \alpha_1 (P o_c), \quad c_{L+1} \leftarrow c_{L+1} + \alpha_2\,\text{MLP}\big((1+\gamma)\,\text{lnorm}(c_{L+1}) + \beta\big) \quad (\text{Eq. 9–10})$$
    - Image and prompt tokens attend **only** to `[img, txt]` (Eq. 8) and never to `concepts`, leaving the generated image unperturbed.
- **Formula / Algorithm (Saliency Map)**:
  - Unscaled dot product (no $1/\sqrt{D}$; `attention_utils.py:L31-35` returns `[B, R, N]`, transpose of `[B, N, R]`):
    $$S^{(t,L)} = o_c^{(t,L)} \big(o_x^{(t,L)}\big)^T \in \mathbb{R}^{B \times R \times N}$$
  - **Concept-axis softmax** ("pixelwise softmax" p. 9; `flux/pipeline.py:L66` `dim=-2` over $R$, NOT spatial $N$):
    $$\phi(o_x, o_c) = \text{softmax}_R\!\left(o_c o_x^T\right)$$
  - **Paper vs Repo Aggregation-Order Discrepancy**:
    - *Paper Eq. (11) (layer mean) & `flux/pipeline.py:L65-75` (layer + timestep mean)*: apply `softmax` over $R$ **per `(t, L)` first**, then average over selected layers $\mathcal{L}$ and timesteps $\mathcal{T}$.
    - *Repo `heatmap_utils.py:L94-114`*: average raw scores over $\mathcal{T}, \mathcal{L}$ **first**, then apply `softmax(avg_scores / T, dim=concept)`.
- **Key Hyperparameters**:
  - **Layers**: Paper Sec 5.1 (p. 6) reports last 10 of 18 MMATTN layers on Flux-Schnell (latent adversarial diffusion distillation); note Flux actually has **19** double-stream blocks (`0..18`), and repo `generate_image` defaults to `range(15, 19)` (`flux/pipeline.py:L114, L128`). Single-stream blocks still execute for generation, but the concept stream is not carried through them (`flux/dit.py:L123-151`).
  - **Timesteps**: All 4 steps for Flux-Schnell (`flux/pipeline.py:L33`); single timestep `500/1000` for real-image encoding (p. 9).
  - **Concept Set**: Requires $R \ge 2$ concepts for non-trivial concept-axis softmax; both concept CA and concept SA are required (Table 4: CA+SA best).
  - **CFG Handling**: Run ConceptAttention on the **conditional (`cond`)** batch slice only.

### 2. Paper vs XAI_DiT Customizations
| Item | Original Paper | XAI_DiT Adaptation (Anima / MOSS-Image / Krea 2) | Reason |
|---|---|---|---|
| **Single-Stream Extension** | Dual-stream MMATTN only (`Kp, Qp, Vp` separate from image) | **Anima**: **N/A** for native output-space ConceptAttention (no text residual stream / no $o_c$ inside DiT; use `cross_attn.probs`).<br>**MOSS-Image & Krea 2** (**XAI_DiT Extension**): Reuse shared single-stream block weights for two modes: (1) `prompt_output_space` (existing prompt tokens $o_{\text{txt}}$ vs $o_x$, excluding template tokens, 0 extra SDPA); (2) `open_set_concept_stream` (one-directional concept stream). | Paper explicitly restricts to dual-stream layers; single-stream reuse is an unvalidated XAI_DiT extension to be verified on Colab. |
| **Open-Set Concept Encoding (Causal LLM + Templates)** | Encodes each concept individually with T5, takes token 0 | Encode each concept **independently** through the LLM encoder + `refiner` (MOSS-Image) / `txtfusion` (Krea 2) so concepts do not cross-mix pre-DiT. Strip chat-template tokens and **mean-pool** multi-subtoken concept spans into 1 vector per concept. | Prevents template tokens from being selected as token 0 and prevents pre-DiT concept mixing. |
| **MOSS-Image Hook & 4-Axis RoPE** | Flux `[txt \| img]` double-stream | Sequence `[img N \| txt L]`. Hook `block[i].attn.out` (`[B, N+L, 3072]`, pre-`layers[i].proj["o"]`, pre-`gate1`). In open-set mode, assign concept 4-axis RoPE `(ref=0, seq_pos=0..R-1, row=0, col=0)` while keeping original image `seq_pos = L` **unchanged** (also allow ablation with `seq_pos=0` for all concepts). | Preserves exact image RoPE (`moss_image.md` L41–42) so generation is unperturbed. |
| **Krea 2 Hook, Gate & GQA** | Flux ungated SDPA output `attn_out` | Sequence `[txt T \| img N]`; exclude 5 tail tokens (`<\|im_end\|>`, `\n`, `<\|im_start\|>`, `assistant`, `\n`); text/concept RoPE = `(0,0,0)`; GQA 48Q/12KV (`h_kv = h_q // 4`).<br>• **Saliency readout**: expose `pre_gate` (`block[i].attn.ctx`) vs `post_gate` (`attn.ctx * sigmoid(attn.gate_logits)` — note `krea2/model.py:106` gate module output is pre-sigmoid!).<br>• **Concept residual update**: ALWAYS applies concept's own `sigmoid(gate) → Wo → g1` regardless of readout mode. | Separates saliency readout choice from valid residual-stream propagation (`krea2.md` L23–36). |
| **Aggregation Order & Layer Window** | Paper Eq. 11: `per_layer`; `heatmap_utils.py`: `after_mean`. Paper Fig. 7 favors deeper layers. | Expose `softmax_order: Literal["per_layer", "after_mean"]` (default `"per_layer"`). Default layer window for ConceptAttention = **last 50% of blocks** (consistent with Fig. 7 favoring deeper layers; distinct from ExploringMMDiT's Top-5 middle-block rule because output-space representations are less noisy than raw QK maps), refined by Direction 02. | Transparently reconciles paper/repo differences and explains why ConceptAttention & ExploringMMDiT use different default windows. |

### 3. Verification Invariants
- **Unit Test (CPU)**:
  1. **Zero-Perturbation Invariant (Open-Set Mode)**: Running a toy single-stream block with $R$ concepts yields identical image/prompt outputs vs without concepts (`max_abs_diff < 1e-6`, fp32).
  2. **Aggregation-Order Differentiation**: On synthetic logits with distinct per-layer scales where `per_layer` and `after_mean` produce different outputs, verify both sum to `1.0` over $R$ AND match their respective analytical ground truths (`max_abs_diff < 1e-6`).
  3. **Scale-Free Dot Product Match**: Output `[B, R, N]` matches `concept_attention.attention_utils.compute_concept_attention_scores` (`einsum("b r d, b n d -> b r n")`, no $1/\sqrt{D}$) within `1e-6`.
  4. **Krea 2 Gate Readout & Update Check**: With non-uniform `gate_logits`, verify `Wo(ctx * sigmoid(gate_logits))` matches block `attn_out` (`max_abs_diff < 1e-6`), and `post_gate` saliency differs from `pre_gate` as expected.
- **Sanity Reproduction (Colab GPU)**:
  - Fixed seed, same initial latent, deterministic mode (or allow bf16 tolerance `max_abs_diff < 1e-3` if SDPA kernel differs): verify unperturbed generation vs unhooked baseline, and quantitatively compare object-mask sharpness (mIoU / foreground-background contrast) of ConceptAttention (`pre_gate` vs `post_gate` on Krea 2) against raw `img_to_txt` attention.

---

## Paper 2: `2025_ICCV_ExploringMMDiT.pdf` (ICCV 2025)

- **Title**: *Exploring Multimodal Diffusion Transformers for Enhanced Prompt-based Image Editing* (Shin et al., SNU-VGILab)
- **Verified Pages / Sections**: Sec 3.1–3.3 (pp. 3–5, Eq. 1–3, Fig. 4–8), Sec 4.1–4.3 (pp. 5–7, Alg. 1, Fig. 9–12)
- **Official Repo Checked**: `reference_repo/paper_code/ExploringMMDiT` (`SNU-VGILab/exploring-mmdit`, commit `41b626d`)
  - `utils/attn_processors.py` (L18–40, L44–157, L364–554), `utils/dit_attn_viz.py` (L91–276), `utils/dit_utils.py` (L636–652), `utils/common.py` (L224–256)

### 1. Exact Mechanism & Equations
- **Input / Hook Point**:
  - Post-QK-norm, **post-RoPE where applicable** (Flux applies RoPE at `attn_processors.py:L521-524`; SD3 has **no RoPE**; Anima cross-attn has **no RoPE**), **post-full-row-softmax** attention probabilities (`dit_attn_viz.py:L131-161`):
    $$A^{(t,L)} = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_h}},\,\text{dim}=-1\right)$$
  - **Explicit CFG `cond` Slicing (XAI_DiT Rule)**: `dit_attn_viz.py:L144` hard-codes `batch_size=1` (if passed a CFG batch, `L151` would mix `cond` and `uncond` across heads!). In `XAI_DiT`, ALWAYS slice the **`cond` batch index** (via adapter CFG metadata) **BEFORE** averaging over heads $\bar{A} = \frac{1}{H}\sum_h A_{:,h,:,:}$ and slicing `img_to_txt`:
    - `[img | txt]` (SD3 / MOSS-Image): `A[:, :N, N:N+L]`
    - `[txt | img]` (Flux / Krea 2): `A[:, T:T+N, :T]`
    *(Note: Flux inspection in `attn_processors.py:L461-462` already covers single-stream blocks, supporting transfer to MOSS-Image and Krea 2.)*
- **Formula / Algorithm (Curated Heatmap & Noise Mitigation, Sec 3.3)**:
  1. **Selective Top-$K$ Block Aggregation**:
     - Average `img_to_txt` over selected timesteps $\mathcal{T}$ and Top-$K$ blocks $\mathcal{B}_{\text{top}K}$. (Note: repo inspection path computes all blocks manually; keeping non-selected blocks on fused SDPA is from Sec 4.2 editing path and adopted here for speed.)
     - Top-5 blocks are selected by **average rank** (not raw metric mean) across BCE, Soft mIoU, and MSE against Grounded SAM2 masks over 100 PARTI prompts; rank **with** Gaussian smoothing enabled if smoothing is used (Sec 3.3, pp. 4–5).
  2. **Multi-Token Word Aggregation** (`dit_attn_viz.py:L264-267`): mean over sub-token indices $\mathcal{I}_w$ of target word/tag.
  3. **2D Gaussian Smoothing** (`utils/common.py:L224-256`): normalized 2D Gaussian kernel `kernel_size = 7`, `sigma = 1.0`, `padding = 3` on `(h, w)`.
  4. **Local Blending Mask (Editing Mode, Sec 4.2, `dit_utils.py:L636-652`)**:
     - Store `img_to_txt` during early steps ($t > \eta$, $\eta = 0.5T$).
     - **Important repo normalization before thresholding**: SD3 applies `smoothing → 3×3 max-pool → interpolate → divide by spatial max → (> θ)`; Flux applies `divide by spatial max → (> θ)`. Threshold $\theta$ is therefore relative to the spatial max (`[0, 1]`), NOT raw probability!
- **Key Hyperparameters**:
  - Top-$K$ blocks: $K = 5$; Gaussian smoothing: `k = 7, sigma = 1.0`; editing windows: $\tau = 0.8T$ (first 20% steps for $q_i, k_i$ replacement), $\eta = 0.5T$ (first 50% steps for local blending).

### 2. Paper vs XAI_DiT Customizations
| Item | Original Paper | XAI_DiT Adaptation (Anima / MOSS-Image / Krea 2) | Reason |
|---|---|---|---|
| **Anima Cross-Attn** | Joint attention `[N+L, N+L]` | Hook `block[i].cross_attn.probs` (`[B, 16, N, L]`, no RoPE, softmax over full `L`). Keep full `L` in softmax denominator, then exclude `Lt..L-1` (0-pad) and `</s>` from word-map display. | Preserves exact model softmax (`anima.md` L42, L76). |
| **MOSS-Image & Krea 2** | SD3 `[img \| txt]` & Flux `[txt \| img]` | **MOSS-Image**: `[img N \| txt L]`, slice `[:N, N:N+L]`, exclude chat-template tokens.<br>**Krea 2**: `[txt T \| img N]`, slice `[T:T+N, :T]`, exclude 5 tail tokens, GQA `h_kv = h_q // 4`. | Matches `moss_image.md` & `krea2.md` (`krea2/model.py:94`). |
| **Colab L4 Peak VRAM & Query Chunking** | Allocates full `[B, H, N+L, N+L]` score tensor (`attn_processors.py:L24-39`) | Running mean keeps **stored** map $O(N \cdot L)$, **NOT** peak computation VRAM! On Krea 2 (`B=1, H=48, N=4096`), one full score tensor is **1.5 GiB fp16** (3.0 GiB fp32), and Krea 2 DiT BF16 weights alone are ~25.6 GB (> L4 24 GB). Therefore:<br>1. Compute manual attention via **query-chunked softmax** (chunk over `Q_img` rows, keeping **all** `img + txt` keys in the softmax denominator);<br>2. Specify weight quantization (FP8/GGUF) or CPU offloading + sequential text-encoder unload for MOSS-Image (5B+9B) & Krea 2 (12.8B) on L4 (24 GB); measure actual peak VRAM on Colab before full sweeps. | Eliminates OOM risk on Colab L4 (24 GB). |
| **Fallback Block Window** | Pre-calibrated Top-5 per model | Before Direction 02 calibration, fallback = middle 60% of blocks (excluding first/last 20% noisy blocks, Sec 3.3 p. 5). | Explicitly separated from ConceptAttention's deeper-50% default. |

### 3. Verification Invariants
- **Unit Test (CPU)**:
  1. **Sequence-Order Equivalence**: Permuting `[img | txt]` ↔ `[txt | img]` (moving `Q_img, K_img, K_txt` and mask together) returns equivalent `img_to_txt` within `1e-6`.
  2. **CFG Cond Isolation**: With distinct `cond` and `uncond` batch tensors, extractor returns purely the `cond` map (`max_abs_diff < 1e-6` vs cond-only run).
  3. **GQA Mapping & Query-Chunking Equivalence**: With 12 distinct KV heads (`h_kv = h_q // 4`), query-chunked manual `probs @ V` matches full `F.scaled_dot_product_attention` `attn.ctx` within `1e-5` (fp32).
  4. **Gaussian Smoothing & Streaming Match**: Kernel sums to `1.0`, matches `ExploringMMDiT/utils/common.py:gaussian_smoothing` within `1e-6`, and streaming running mean matches batch-stacked `.mean()` within `1e-6`.
- **Sanity Reproduction (Colab GPU)**:
  - Run a 2×2 ablation (**All blocks vs Top-5** × **no-GS vs GS `k=7, sigma=1.0`**) on Anima, MOSS-Image, and Krea 2 (same seed/latent) so Top-5 selection and Gaussian smoothing effects are evaluated separately, and log peak VRAM & wall-clock time.

---

## Paper 3: `2025_NeurIPS_Seg4Diff.pdf` (NeurIPS 2025)

- **Title**: *Seg4Diff: Unveiling Open-Vocabulary Segmentation in Text-to-Image Diffusion Transformers* (Kim et al., KAIST CVLAB)
- **Verified Pages / Sections**: Sec 3.2–3.5 (pp. 4–8, Eq. 1–13), Appendix A.1/A.3 (p. 16), Appendix B.1–B.5 (pp. 17–19, Table 5, Fig. 15–17)
- **Official Repo Checked**: `reference_repo/paper_code/Seg4Diff` (`cvlab-kaist/Seg4Diff`, commit `bb500ca`)
  - `diffusers/src/diffusers/models/attention_processor.py` (L1033–1142)
  - `seg4diff/seg4diff_model_ovss.py` (L139–254), `seg4diff_model_unsup.py` (L209–238, L431–495), `seg4diff/modeling/backbone/sd3_backbone.py` (L139–144, L209–219, L330)

### 1. Exact Mechanism & Equations
- **Input / Hook Point**:
  1. **Attention probabilities (`img_to_txt`, called `I2T` in paper)**:
     - Post-QK-norm (no RoPE in SD3; post-RoPE in Flux/MOSS/Krea), **post-full-row-softmax** (`attention_processor.py:L1098-1111`, Eq. 4). In original SD3 experiments, T5 is disabled and only the first 77 CLIP tokens are sliced (`4096:4096+77`, App. A.1).
     - **OVSS vs Unsupervised Normalization (Critical Distinction)**:
       - **OVSS path** (Table 5, p. 18): Raw post-softmax probabilities **without** min-max normalization perform best (`VOC 89.0 / Object 61.8`). Prompt is constructed from ground-truth classnames present in the image, sub-tokens averaged (`seg4diff_model_ovss.py:L149-165, L219-221`), with `do_classifier_free_guidance=False` (`sd3_backbone.py:L219`).
       - **Unsupervised `<pad>` path** (`seg4diff_model_unsup.py:L209-238, L431-495`): Uses empty prompt, applies spatial min-max normalization + `output_power` ($p$), then spatial softmax (`softmax(flat, dim=-1)`) and greedy symmetric-KL merging $D_{\text{KL}}^{\text{sym}}(x, Y) = \frac{1}{2}\sum (x - Y)(\log x - \log Y)$.
  2. **Value-projection L2 norm (Candidate Expert Layer Profiler, Sec 3.3, Fig. 4, p. 5)**:
     - $\bar{v}_{\text{mod}}^{(L)} = \frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} \|V_i^{(L)}\|_2$ for image vs prompt tokens (excluding pad/template). Middle layers (e.g., 0-indexed Layer 9 in SD3, `ATTENTION_LAYERS: [9]`, `sd3_backbone.py:L139-144`) show a pronounced text V-norm peak.
  3. **Causal I2T Blur Perturbation & Guidance (Sec 3.3, Eq. 7; Appendix A.3, p. 16)**:
     - **Exact blur definition (App. A.3)**: Apply **1D Gaussian blur (`k = 5, sigma = 9`, reflective padding) along the TEXT-KEY axis** of post-softmax `img_to_txt` for each head (NOT 2D spatial blur!).
     - **Must feed perturbed $A' V$ back into the model**: In `attention_processor.py:L1095-1118`, `attn_cache` is observation-only while the forward pass calls separate `F.scaled_dot_product_attention`. For causal perturbation / Eq. (7) guidance ($\tilde{\epsilon}_\theta = \epsilon_\theta(A) + s(\epsilon_\theta(A) - \hat{\epsilon}_\theta(\tilde{A}))$), the perturbed probabilities $\tilde{A}$ **must** be multiplied with $V$ to produce the layer's actual output! Note: reflect padding alone does not guarantee exact row-sum conservation for arbitrary inputs; renormalize row sum if exact probability conservation is required.
- **Key Hyperparameters**:
  - **Noisy latent timestep**: `NOISE_STEPS: 8` uses `scheduler.timesteps[-8]` (8 steps from the **clean** end of SD3's 28-step shifted schedule — NOT `sigma = 8/28`!; `sd3_backbone.py:L209, L330`, `configs/eval_ovss.yaml:L27`). Sec 3.3 compares `t=8` and `t=28`.
  - **CFG Handling**: `cond`-only (`do_classifier_free_guidance=False` in OVSS).

### 2. Paper vs XAI_DiT Customizations
| Item | Original Paper | XAI_DiT Adaptation (Anima / MOSS-Image / Krea 2) | Reason |
|---|---|---|---|
| **V-Norm Profiler (Head & Token Scope)** | Joint-attn `V` in SD3 (text vs img inside same projection) | Measure **per-head or pre-repeat concatenated** L2 norm on explicit prompt span (excluding `</s>`, 0-pad, and chat-template tokens):<br>• **Anima** *(proxy)*: `cross_attn.v` (`[0 : Lt-1]`) vs `self_attn.v` (img) — note these come from two distinct projections.<br>• **MOSS-Image**: `attn.v` prompt span vs img span (`24` heads).<br>• **Krea 2**: `attn.v` on **12 KV heads pre-repeat** (repeating to 48 heads artificially doubles concatenated L2 norm!).<br>Treat V-norm peaks as **candidate** expert layers to be verified causally. | Fixes GQA norm-doubling bug and distinguishes Anima's two-projection proxy (`anima.md`, `krea2.md`). |
| **1D Text-Axis Blur Span** | Blurs along text-key axis (`k=5, sigma=9`) | Restrict 1D text-axis blur strictly to the **prompt-content span** (`[0:Lt-1]` on Anima; excluding chat-template / 5 tail tokens on MOSS-Image / Krea 2) so blur does not smear into padding or template sinks. | Prevents artifact leakage from pad/template tokens. |
| **Anima `<pad>` Behavior** | SD3 CLIP/T5 `<pad>` tokens form unsupervised masks | In Anima, `Lt..L-1` are 0-padded ($K=0 \Rightarrow \text{logit}=0$), so each pad slot has probability $1/Z(x)$ where $Z(x)$ **varies spatially** with image query $x$ (not spatially uniform; acts as an inverse text-attention-strength map, and total pad mass scales with $512 - L_t$). On MOSS-Image & Krea 2 (no padding), inspect **chat-template tokens** instead (links to Direction 04). | Exact math (`anima.md` L46, L76). |
| **Noisy Latent Schedule Mapping** | `scheduler.timesteps[-8]` on SD3 shifted schedule | Record exact equivalent noise level $\sigma$ (under `x_sigma = sigma * eps + (1 - sigma) * x0`) for each target model's schedule rather than assuming `8/28`. | Rectified-flow schedules have model-specific timestep shifts. |
| **MAGNET LoRA (Sec 3.5)** | Fine-tuning via `L_FM + lambda L_mask` | **Out of scope** (training-free `XAI_DiT`). | `docs/direction.md`. |

### 3. Verification Invariants
- **Unit Test (CPU)**:
  1. **Value L2-Norm Profiler**: Correctly excludes `</s>`, pad, and template indices; on Krea 2, verifies pre-repeat 12-KV-head norm does not double under GQA repeat.
  2. **Symmetric KL Safety**: `symmetric_kl(p, p) == 0.0`, `symmetric_kl(p, q) == symmetric_kl(q, p) >= 0` on strictly positive distributions (and clamps `min=eps` to prevent `NaN` on zeros).
  3. **Slice-Sum Completeness**: For joint attention, `img_to_img.sum(-1) + img_to_txt.sum(-1) == 1.0` (`max_abs_diff < 1e-6`), verifying exact slice boundaries (not just re-testing softmax).
  4. **Causal 1D Text-Axis Blur**: Identity kernel (`sigma -> 0`) or `s = 0` matches baseline within fp32 tolerance (`1e-5`); **non-identity kernel (`k=5, sigma=9`) strictly alters the intervened layer's output** (`max_abs_diff > 0`) while keeping pad/template columns untouched.
- **Sanity Reproduction (Colab GPU)**:
  - Profile V-norm ratio across all blocks of Anima, MOSS-Image, and Krea 2 (5–10 prompts), then apply causal 1D text-axis blur (`k=5, sigma=9`) at the peak candidate block vs an early/late non-peak block (same seed/latent) and report delta CLIP-T / LPIPS quantitatively.
