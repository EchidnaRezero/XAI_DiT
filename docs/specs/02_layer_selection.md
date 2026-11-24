# Direction 02: Layer Selection & Importance Profiling Specs

Verified method specifications extracted directly from `docs/reference/paper/02_layer_selection/2025_CVPR_StableFlow_VitalLayers.pdf` and `docs/reference/paper/2025_NeurIPS_LocalizingKnowledgeDiT.pdf`, cross-checked against `reference_repo/paper_code/{StableFlow,LocalizingKnowledgeDiT}`, `reference_repo/ComfyUI/comfy/ldm/`, and `docs/reference/models/{anima,moss_image,krea2}.md`, and **cross-reviewed by GPT 6.2 Astra & Claude Fable 5.2 (`high`)**.

---

## Paper 1: `2025_CVPR_StableFlow_VitalLayers.pdf` (CVPR 2025)

- **Title**: *Stable Flow: Vital Layers for Training-Free Image Editing* (Avrahami et al., Snap Research)
- **Verified Pages / Sections**: Sec 3.1–3.3 (pp. 3–5, Eq. 1–4, Fig. 3–7), Appendix A.1 (p. 14), Appendix B.2/B.4/B.5/B.6/B.7 (pp. 22–25)
- **Official Repo Checked**: `reference_repo/paper_code/StableFlow` (`snap-research/stable-flow`, commit `c99ac0e`, **Non-Commercial Research License — read-only reference, do NOT copy code**)
  - `run_stable_flow.py` (L9–118), `src/diffusers/pipelines/flux/pipeline_flux.py` (L708–744)
  - `src/diffusers/models/transformers/transformer_flux.py` (L109–247), `src/diffusers/models/attention_processor.py` (L1724–1770, L1797–1835)

### 1. Exact Mechanism & Equations
- **Input / Hook Point**:
  1. **True Block Bypass (for Vitality Measurement, Sec 3.1, `transformer_flux.py:L119, L205`)**:
     - Short-circuit **before** entering the block body (skipping both Attention and MLP; 0 sub-layer calls), returning `resid_in` unchanged across all denoising steps:
       $$M_{-\ell}:\quad x^{(\ell+1)} = x^{(\ell)}$$
  2. **Vital-Layer Attention K/V Injection (for Editing Verification, Sec 3.2, `attention_processor.py:L1748-1757`)**:
     - Original FLUX runs source (`batch=0`) and target (`batch=1:`) in parallel (valid on FLUX because T5 always pads to 512 tokens) and overwrites target **image** `K, V` immediately after `to_k`/`to_v` (**pre-QK-norm, pre-RoPE**), leaving text tokens untouched:
       $$K_{\text{img}}^{\text{tgt}} \leftarrow K_{\text{img}}^{\text{src}}, \quad V_{\text{img}}^{\text{tgt}} \leftarrow V_{\text{img}}^{\text{src}} \quad (\forall \ell \in \mathcal{V})$$
     - Note: Since QK-norm is per-token, injecting immediately post-QK-norm / pre-RoPE (e.g., ComfyUI `attn1_patch` in `krea2/model.py:84-97`) is mathematically equivalent for K/V.
  3. **Latent Nudging + Latent Caching (for Real-Image Inversion, Sec 3.3, Eq. 3–4, App. B.5; `run_stable_flow.py:L70-74`, `pipeline_flux.py:L743-744`)**:
     - Multiply encoded VAE latent $z_0$ by $\lambda = 1.15$ before 50-step inverse Euler ODE and cache inverted latents along the trajectory.
- **Formula / Algorithm (Vitality Score, Eq. 1–2, pp. 3–4)**:
  $$\text{vitality}(\ell) = 1 - \frac{1}{k} \sum_{(s,p)} d\!\left(M_{\text{full}}(s, p),\, M_{-\ell}(s, p)\right) \quad (\text{Eq. 1})$$
  $$\mathcal{V} = \{\ell \in \mathbb{L} \mid \text{vitality}(\ell) \ge \tau_{\text{vit}}\} \quad (\text{Eq. 2})$$
  - Paper uses **DINOv2 ViT-g/14** perceptual similarity (App. A.1, p. 14 L1380/L1442). Paper does not pin pooling or $\tau_{\text{vit}}$ numerically; in `XAI_DiT`, we pin **DINOv2 ViT-g/14 CLS-token cosine similarity** (theoretical vitality range `[0, 2]`) as the primary Eq. 1 decision metric, and select $\mathcal{V}$ via an explicit rule (top-$p\%$ or bootstrap-CI-separated elbow). LPIPS and latent L2 are logged as **auxiliary diagnostics only** (App. B.2 notes different perceptual metrics can swap layer ranks).
- **Key Hyperparameters (Original Paper)**:
  - Calibration: $k = 64$ prompts, Euler 15 steps, **distilled guidance scale `3.5`** (FLUX.1-dev has **no** separate `uncond` CFG pass; `pipeline_flux.py:L708-733`).
  - Reference vital sets (App. A.1, p. 14): FLUX.1-dev (57 blocks) = 10 layers `[0, 1, 2, 17, 18, 25, 28, 53, 54, 56]` (global indices; `run_stable_flow.py:L10-11` drops layer `2` to use 9 layers and subtracts `19` for single-stream indices); SD3 (24 blocks) = `[0, 7, 8, 9]` (note: final blocks are **not** vital in SD3).

### 2. Paper vs XAI_DiT Customizations
| Item | Original Paper | XAI_DiT Adaptation (Anima / MOSS-Image / Krea 2) | Reason |
|---|---|---|---|
| **True Block Bypass & CFG** | Modifies diffusers block `forward`; FLUX has no CFG `uncond` pass | Implement `ModelAdapter.bypass_blocks(indices)` as a **pre-body short-circuit** (0 attn/MLP calls; **XAI_DiT rule**: skip identically for both `cond` and `uncond` branches in CFG runs).<br>• **Anima** (28 blocks): skip Self-Attn + Cross-Attn + MLP.<br>• **MOSS-Image** (42 blocks): skip `[img \| txt]` block (do not bypass `refiner` in main sweep).<br>• **Krea 2** (28 blocks): skip `[txt \| img]` block (do not bypass `txtfusion`). | Clean adapter contract, zero Non-Commercial code copying, proper CFG handling. |
| **Variable-Length Prompts (Two-Pass K/V Cache)** | Batches source & target together (`key[1:, 512:] = key[:1, 512:]`) assuming 512 T5 pad | MOSS-Image and Krea 2 have **no padding** (source & target prompts have different lengths, and ComfyUI `CONDRegular` may run `cond`/`uncond` as separate calls; `model_base.py:L2845`). Use **two-pass cache-and-inject**: source pass caches **pre-RoPE** image K/V keyed by `(step, block, cond_or_uncond)`; target pass injects `source-cond → target-cond` and `source-uncond → target-uncond`. Pre-RoPE is **mandatory** on MOSS-Image because image RoPE `seq_pos = text_len` (`moss_image/model.py:L181`). | Prevents shape mismatch and RoPE position corruption on variable-length prompts. |
| **Colab L4 Compute & VRAM Budget** | $64 \times 57$ FLUX runs on high-VRAM GPU | Compute cost is $k \cdot (L+1) \cdot \text{steps} \cdot (2\text{ if CFG})$ forwards (e.g., MOSS $k=16$, 30 steps, CFG = `41,280` forwards!).<br>• **Stage 1 (screening)**: $k = 8 \to 16$ prompts (check Spearman rank stability between $k=8$ and $k=16$ + bootstrap CIs; App. A.1 variance is `0.026–0.222`).<br>• **VRAM plan (L4 24 GB)**: Anima (~2 GB BF16) fits natively; MOSS-Image (5B DiT ~10 GB + Qwen3.5-9B ~18 GB) requires **sequential encoder unload** before DiT; **Krea 2 base (12.8B DiT ~25.6 GB BF16 > L4 24 GB)** requires **FP8/GGUF quantization or block CPU-offload** on L4 (disclosing numerical diff per `AGENTS.md`) or **A100-80GB / H100**. | Realistic Colab L4+ execution without OOM. |
| **Granular Sub-Layer Bypass** | Full block only | Support `mode: Literal["full_block", "attn_only", "mlp_only"]` (default `"full_block"`), plus `"cross_attn_only"` on Anima. | Separates text-injection vs spatial-refinement blocks. |

### 3. Verification Invariants
- **Unit Test (CPU)**:
  1. **True Bypass & Call-Count Invariant**: Bypassing block $\ell$ yields `resid_out == resid_in` (`max_abs_diff == 0.0`), matches a model with block $\ell$ removed, and asserts **0 calls** to block $\ell$'s attn/MLP on both `cond` and `uncond` branches.
  2. **K/V Injection No-Op & Isolation**: With identical source & target prompt/seed, target output matches source (`max_abs_diff < 1e-6`), source branch is unchanged, and text slice K/V is untouched.
  3. **Vitality Score Bounds**: Identical embeddings $\Rightarrow$ `vitality == 0.0`; orthogonal $\Rightarrow$ `1.0`; opposite $\Rightarrow$ `2.0` (range `[0, 2]`).
- **Sanity Reproduction (Colab GPU)**:
  - Run `M_full` twice to establish non-determinism noise floor of DINOv2 $d$; run block-bypass sweep with $k=8$ and $k=16$ (log peak VRAM, wall-clock time, per-layer bootstrap CIs, and $k=8$ vs $k=16$ Spearman rank correlation; make no prescriptive claim that final blocks must be vital).

---

## Paper 2: `2025_NeurIPS_LocalizingKnowledgeDiT.pdf` (NeurIPS 2025)

- **Title**: *Localizing Knowledge in Diffusion Transformers* (Zarei et al., arXiv:2505.18832)
- **Verified Pages / Sections**: Sec 3.1–3.3 (pp. 3–6, Fig. 1, Fig. 3–5), Appendix B.2–B.3 (pp. 14–15)
- **Official Repo Checked**: `reference_repo/paper_code/LocalizingKnowledgeDiT` (`ArmanZarei/DiT-Knowledge-Localization`, commit `c7e99e7`)
  - `attention_processor.py` (L315–560), `localization_and_intervention/calc_attn_cont_varients.py` (L1–93), `localize_knowledge_and_intervene_flux.py` (L22–85), `pipelines/flux.py` (L332–367)

### 1. Exact Mechanism & Equations
- **Input / Hook Point**:
  - Post-QK-norm, post-RoPE (where applicable), **post-full-row-softmax** `img_to_txt` probabilities $A_{\text{img}\to\text{txt}}^{(t,\ell,h)}$ (full `[img + txt]` softmax denominator on joint-attn models, NO renormalization over text slice; `attention_processor.py:L424-426`), value vectors $V_{\text{txt}}^{(t,\ell,h)}$, and output projection $W_O^{(\ell)}$.
  - **CFG Branch Routing**: Official diffusers PixArt/SANA code hard-codes batch index `[1]` (`calc_attn_cont_varients.py:L10, L29`). In `XAI_DiT`, ALWAYS select the `cond` branch via adapter CFG metadata (`transformer_options["cond_or_uncond"]` in ComfyUI; `samplers.py:L289-323`).
- **Formula / Algorithm (Attention Contribution, Sec 3.2, pp. 4–5; `attention_processor.py:L330-355`)**:
  - For concept $\kappa$ spanning token indices $\mathcal{I}_\kappa$: **sum vector contributions over $\mathcal{I}_\kappa$ via `m @ v` BEFORE concatenating heads, applying $W_O$, and taking L2 norm in `float32`**, then average over all $N$ image tokens:
    $$o_i^{(t,\ell)}(\kappa) = W_O^{(\ell)} \Big[\text{concat}_{h=1}^H \Big(\sum_{j \in \mathcal{I}_\kappa} A_{i,j}^{(t,\ell,h)} v_j^{(t,\ell,h)}\Big)\Big] \in \mathbb{R}^{D}, \qquad \text{Cont}^{(t,\ell)}(\kappa) = \frac{1}{N} \sum_{i=1}^{N} \left\| o_i^{(t,\ell)}(\kappa) \right\|_2$$
  - *(Official FLUX code note: `localize_knowledge_and_intervene_flux.py:L22` iterates over BOTH `transformer_blocks` and `single_transformer_blocks`; in `FluxAttnContCalculatorProcessor:L337`, if `attn.to_out` is absent in a single-stream block, it falls back to pre-$W_O$ `M @ V` norm.)*
  - **Neutral-Prompt Verification (Sec 3.2, Fig. 3; `attention_processor.py:L477`, `pipelines/flux.py:L332-367`)**:
    - Run clean/neutral prompt $p^{\kappa\text{-neutral}}$ on its own trajectory, cache its **attention input text states** at Top-$K$ blocks $\mathcal{B}_K^\kappa$, and overwrite the original pass's attention input text states at $\mathcal{B}_K^\kappa$ (FLUX eval also uses `replace_pooled_prompt_embeds=True`).
- **Key Hyperparameters (Original Paper)**:
  - Top-$K$ evaluated at $K \in \{0\%, 10\%, 30\%, 50\%\}$ (Fig. 4), Fig. 1 uses $K = 40\%$, Fig. 5 uses $K \in [4, 12]$ blocks, repo default is 6 blocks; **ranking prompts are strictly separated from evaluation prompts** (p. 6 L393).

### 2. Paper vs XAI_DiT Customizations
| Item | Original Paper | XAI_DiT Adaptation (Anima / MOSS-Image / Krea 2) | Reason |
|---|---|---|---|
| **Exact Adapter Modules & Bias** | Diffusers `attn.to_out[0]` (FLUX has bias; use `bias=None` when reproducing FLUX) | All 3 target models' output projections are **bias-free**:<br>• **Anima**: `cross_attn.output_proj.weight` (`predict2.py:143`); keep full `L` (incl. 0-pad) in softmax, exclude pad from $\mathcal{I}_\kappa$.<br>• **MOSS-Image**: `layers[i].proj["o"].weight` (`moss_image/model.py:136, 150`); full `[img \| txt]` softmax then slice.<br>• **Krea 2**: `wo.weight` (`bias=False`, `krea2/model.py:75`); full `[txt \| img]` softmax then slice; GQA `repeat_interleave(4, dim=1)` (`h_kv = h_q // 4`, `krea2/model.py:94-97`). | Exact module paths verified against `reference_repo/ComfyUI`. |
| **Krea 2 Output Gate & AdaLN Gate** | Ungated MHA | `krea2/model.py:106` `attn.gate` module output is **pre-sigmoid** (`gate_val = sigmoid(attn.gate_logits)`). Expose both:<br>• `raw_mvo`: $\|W_O(\text{concat}_h(A_{\mathcal{I}_\kappa} V_{\mathcal{I}_\kappa}))\|_2$<br>• `gated_mvo`: $\|g_1 \odot W_O(\sigma(\text{gate\_logits}_i) \odot \text{concat}_h(A_{\mathcal{I}_\kappa} V_{\mathcal{I}_\kappa}))\|_2$ (exact residual delta).<br>Also report `Cont` for causal tail/EOS tokens separately. | Prevents double-sigmoid bug and captures Krea 2's true residual contribution. |
| **Variable-Length Neutral-Prompt Swap** | Assumes 512-padded FLUX text slice (`hidden_states[:, :512, :]`) | Because MOSS-Image & Krea 2 have variable text lengths:<br>• Require **token-length-matched neutral prompt pairs** (assert `len(tok(p)) == len(tok(p_neutral))`, e.g., replacing only `"car"` → `"box"` while keeping `"red sports"` or `"a red car"` → `"a red box"`), overwriting attention-input text states at $\mathcal{B}_K^\kappa$;<br>• On Krea 2 only (text RoPE = `(0,0,0)`), K/V-level swap is also valid; on MOSS-Image it is NOT valid without matching length. | Prevents shape mismatch and isolates single-concept causality. |
| **Memory-Safe Streaming (Colab L4+)** | Caches full `[T, H, N, L]` maps | **Ban materializing full `[H, N+L, N+L]` `probs` on GPU**: compute `A_{img->I_k} @ V_{I_k}` via **query-chunked softmax** (summing over all `img + txt` keys in denominator per query chunk, keeping only `I_k` columns in fp32) and accumulate per-block scalar `Cont` on-the-fly. | Eliminates 1.5–3.3 GB/block peak VRAM blowup on L4. |

### 3. Verification Invariants
- **Unit Test (CPU)**:
  1. **Zero-Contribution & Completeness**: When $A_{:,:,\mathcal{I}_\kappa}=0$ or $V_{:,\mathcal{I}_\kappa,:}=0$, `Cont == 0.0`. Summing pre-norm $o_i(j)$ over **all** keys (`img + txt` for joint, `0..L-1` for Anima) reproduces exact `attn_out` before $W_O$, after $W_O$, and on Krea 2 with `sigmoid(gate_logits)` and `g1` reproduces exact attention residual delta (`max_abs_diff < 1e-5`, fp32).
  2. **Non-Uniform Gate & GQA Mapping**: Test against Krea 2's actual `Attention.forward` on a tiny random config (8Q/2KV heads, non-uniform `gate_logits` and `g1`, channel-mixing `W_O`, distinct KV heads) to catch wrong `.repeat` vs `.repeat_interleave` (`h_kv = h_q // 4`) or swapped gate order (`max_abs_diff < 1e-5`).
  3. **Branch Routing**: Mixed `cond_or_uncond` batch order selects strictly the `cond` slice (`max_abs_diff < 1e-6`).
- **Sanity Reproduction (Colab GPU)**:
  - Compute per-block `Cont^(l)(k)` on a ranking prompt set, then evaluate **token-length-matched** neutral-prompt swaps on a **separate** held-out prompt set across multiple paired seeds (Top-20% vs Bottom-20% blocks), logging delta CLIP-T and peak VRAM.
