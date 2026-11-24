# Direction 04: Verification & Mechanistic Interpretability (MI) Specs

Verified method specifications extracted directly from `docs/reference/paper/04_verification_mi/*.pdf`, cross-checked against `reference_repo/paper_code/{AttentionSinksDiT,GatedAttention,DiffusionProbe,ProbeSelect}`, `reference_repo/ComfyUI/`, and `docs/reference/models/{anima,moss_image,krea2}.md`, and **cross-reviewed by GPT 6.2 Astra & Claude Fable 5.2 (`high`)**.

> **Common Execution & Test Policy**:
> - **Local vs Colab Testing**: Per `AGENTS.md`, PyTorch is NOT installed locally; CPU unit tests run either on a Colab CPU runtime (PyTorch) or via pure-NumPy reference math checks locally.
> - **Colab L4 (24 GB) Reality**: Anima (~2 GB BF16) fits natively; MOSS-Image (5B DiT ~10 GB + Qwen3.5-9B ~18 GB) requires sequential text-encoder unload before DiT; **Krea 2 base (12.8B DiT ~25.6 GB BF16 > L4 24 GB)** requires FP8/GGUF quantization or block CPU offload on L4 (logging precision because FP8 shifts `M-Act` and gate distributions!) or A100-80GB / H100. Full fp32 attention per block per CFG sample at 1024² is ~3.4 GB (Krea 2), ~1.7 GB (MOSS-Image), ~1.1 GB (Anima self-attn), ~0.13 GB (Anima cross-attn) — always use **chunked attention** and log peak VRAM.
> - **CFG & Time Convention**: Always select `cond` vs `uncond` via adapter metadata (`cond_or_uncond`). "20% steps" means fraction of generation step index counted from the noise end (`i < 0.2 S`), NOT $\sigma = 0.2$.

---

## Paper 1: `2026_ICML_AttentionSinksDiT.pdf` (ICML 2026)

- **Title**: *Attention Sinks in Diffusion Transformers: A Causal Analysis* (Wu et al.)
- **Verified Pages / Sections**: Sec 3.2–3.5 (pp. 4–7, Tables 2–4), Appendix A.5 (p. 13, L1319–1369)
- **Official Repo Checked**: `reference_repo/paper_code/AttentionSinksDiT` (`wfz666/ICML26-attention-sink`, commit `c021b56`, MIT)
  - `attention-sink-voodoo-release/src/dynamic_sink_processor.py` (L54–300)
  - `attention-sink-voodoo-release/src/sink_analysis.py` (L200–560)

### 1. Exact Mechanism & Equations
- **Two Distinct Mechanisms (Do NOT Conflate)**:
  1. **Dynamic Sink Pre-Softmax Hard Masking (Sec 3.3.1, `dynamic_sink_processor.py:L180-287`)**:
     - Scope: Incoming mass and the $-10^4$ mask apply **ONLY to image-query rows** (`Q_img`), summing over **ALL `[img + txt]` keys**, per sample and per head (`L180-205, L247-253`). Text-query rows are **untouched** (`L273-287`).
     - Exact sequence: fp32 baseline softmax → compute per-sample, per-head incoming mass $m_j^{(b,h)} = \frac{1}{N_{\text{img}}}\sum_{i \in \text{img}} A_{b,h,i,j}$ → select Top-$k$ dynamic sink indices $\mathcal{S}_k^{(b,h)}$ (specify **per-head top-$k$**, Sec 3.3.1, vs **union-budget top-$k$**, Sec 3.5.3) → subtract $10^4$ from pre-softmax logits at `(img_q, S_k)` → second softmax.
     - **Important metric bug in repo (`L255`)**: `top1_incoming_mass_after` picks the *new* maximum after masking. To verify Table 4's $>10^8\times$ reduction, re-read post-masking mass at the **pre-intervention indices $\mathcal{S}_k$**!
     - **Chunked execution (Colab L4)**: Must be **two-pass** over query chunks (Pass 1 accumulates incoming mass across all `Q_img` chunks to fix $\mathcal{S}_k$; Pass 2 recomputes chunk logits with $-10^4$ at $\mathcal{S}_k$). Intervened blocks bypass fused SDPA; disabled path hands off directly to native SDPA (`L125-128`).
  2. **Static-Index Score / Value Interventions (`sink_analysis.py:L255-284, L434-560`)**:
     - Applies to fixed `sink_token_indices` over all joint rows (SD3 key order `[txt, img]`).
     - **Paper vs Repo scaling difference**: Repo `_apply_prob_scale` (`L434-468`) scales post-softmax sink mass directly to $\eta \sum A_s$ while renormalizing rest; Paper App. A.5.1 (L1328–1331) adds pre-softmax logit bias $\log \eta$ (yielding $\frac{\eta \sum A_s}{1 - (1-\eta)\sum A_s}$). `XAI_DiT` exposes `mode: Literal["logit_bias", "prob_scale"]` (default `"logit_bias"` for paper match).
     - Value intervention (`L470-500`): replace sink `V` with `zero`, `mean` (mean over all joint tokens), `noise`, or `lerp(alpha)`.
     - `_record_metrics` (`L280-284, L502-560`): runs **after** intervention; code measures `hidden_states` at attention input (paper L1321 says attention block output) and clamps entropy at `1e-10` (paper L1319 says `1e-12`).
- **CFG Handling**: Official repo runs on the **full batch (both CFG halves)**, selecting per-sample top-$k$. Applying to `cond`-only is an **XAI_DiT adaptation** (explicit flag: `cfg_scope: Literal["both", "cond_only"]`).

### 2. Paper vs XAI_DiT Customizations
| Item | Original Paper | XAI_DiT Adaptation (Anima / MOSS-Image / Krea 2) | Reason |
|---|---|---|---|
| **Explicit Query/Key Ranges & Depth Mapping** | SD3 (24 blocks: L6/12/18) & SDXL (both self-attn & cross-attn, App. A.11 L1366–1369) | Map relative depths (~25%, 50%, 75%) onto Anima (28: L7/14/21), MOSS-Image (42: L10/21/31), Krea 2 (28: L7/14/21).<br>• **Anima**: support both `self_attn` (`Q_img × K_img`) and `cross_attn` (`Q_img × K_txt`, tracking `</s>` & 0-pad).<br>• **MOSS-Image**: `[img N \| txt L]` → `Q_img = [0:N]`, keys = `[0:N+L]`.<br>• **Krea 2**: `[txt T \| img N]` → `Q_img = [T:T+N]`, keys = `[0:T+N]`. Note GQA: any `V`-path edit affects all 4 Q heads in the KV group! Compare sink stats against **Paper 3 (`GatedAttention`)**. | Exact per-model slicing & GQA awareness. |

### 3. Verification Invariants
- **Unit Test (CPU)**:
  1. **Hard Mask & Text-Row Isolation**: Subtracting `1e4` at `(Q_img, S_k)` yields `probs[Q_img, S_k].max() < 1e-6`, `probs[Q_img].sum(-1) == 1.0`, and leaves `Q_txt` rows bitwise identical. Two-pass chunked dynamic masking matches full-matrix masking (`max_abs_diff < 1e-5`).
  2. **Prob-Scale / Logit-Bias Preconditions**: Require at least one non-sink key with positive residual mass; verify `eta=1.0` is no-op and `prob_scale` vs `logit_bias` match their respective analytical formulas.
  3. **Entropy Precision**: Use `torch.special.xlogy` (or small $M$ with explicit clamp tolerance) so one-hot entropy is `0.0` without clamp drift (`(M-1)*1e-10*|ln 1e-10| ≈ 9.7e-6` at $M=4200$).
- **Sanity Reproduction (Colab GPU)**:
  - Verify unhooked vs disabled-hook `v` diff is `0.0` (native SDPA handoff); measure pre- vs post-masking mass at $\mathcal{S}_k$, `max_activation`, and delta CLIP-T/LPIPS (defining explicit practical-equivalence threshold e.g. $|\Delta\text{CLIP-T}| < 0.005$) across all 3 models.

---

## Paper 2: `2026_arXiv_TemplateRegisters.pdf` (arXiv 2026)

- **Title**: *Text Template Tokens Are Implicit Semantic Registers in Diffusion Transformers* (Li et al., NJU / Alibaba / ZJU, arXiv:2607.19139)
- **Verified Pages / Sections**: Sec 3.2–3.3 (p. 4), Sec 4.1–4.5 (pp. 5–9, Fig. 5, Table 2 on Qwen-Image-2512), **Appendix E.2: Krea-2-Turbo** (p. 17)
- **Official Repo Checked**: `reference_repo/paper_code/TemplateRegisters` (`Met4physics/DiT-Interpretability`, commit `4716dec` — **README + teaser only; code not yet released**)

### 1. Exact Mechanism & Equations
- **Input / Hook Point**: Post-**full-key-softmax** `img_to_txt` attention mass decomposed into disjoint text spans (template/register $\mathcal{R}$, semantic prompt $\mathcal{S}$, plus padding $\mathcal{P}$ where present):
  $$m_{\mathcal{R}}^{(t,\ell,h)} = \frac{1}{N}\sum_{i \in \text{img}} \sum_{j \in \mathcal{R}} A_{i,j}^{(t,\ell,h)}, \qquad m_{\mathcal{S}}^{(t,\ell,h)} = \frac{1}{N}\sum_{i \in \text{img}} \sum_{j \in \mathcal{S}} A_{i,j}^{(t,\ell,h)}$$
  *(Note: On joint-attention models, $m_{\mathcal{R}} + m_{\mathcal{S}} < 1$ because image keys absorb the rest; on Anima cross-attention, $m_{\mathcal{R}} + m_{\mathcal{S}} + m_{\mathcal{P}} = 1$. Only **normalized I2T shares** $m_{\mathcal{R}} / (m_{\mathcal{R}} + m_{\mathcal{S}})$ are comparable across cross-attn vs joint-attn!)*
- **All 4 Causal Probes + Head Pruning (Sec 4.1–4.5, pp. 5–9)**:
  1. **Token-Level Attention Decomposition (Sec 4.1)**: Compare $\bar{m}_{\mathcal{R}}$ vs $\bar{m}_{\mathcal{S}}$ and per-token ratio. *Appendix E.2 Krea-2-Turbo anchor (8 steps, 553 GenEval prompts)*: $\bar{m}_{\mathcal{R}}=0.11$, $\bar{m}_{\mathcal{S}}=0.053$, 67% of I2T mass on $\mathcal{R}$, 3.1× per-token ratio, trailing `\n` after `assistant` dominant at 59% of sites.
  2. **Span-Level Conditioning Interventions (Sec 4.2)**: Swap or average encoder output of $\mathcal{R}$ across prompts pre-DiT (finding: $\mathcal{R}$ carries semantics *only weakly* at encoder output; averaging can cause drift mitigated by CFG, L560–567).
  3. **Cross-Trajectory Head Transplantation (Sec 4.3, L586–593)**: Copy donor's **per-head `(q, k, v)` projections** (NOT attention outputs!) into recipient (paper runs in 1-step distilled setting), comparing progressive transplantation ranked by **ascending vs descending $\bar{m}_{\mathcal{S}}$** (high-$\bar{m}_{\mathcal{S}}$ prompt-reading heads are causally inert).
  4. **Layer-Wise Causal Attention Masking (Sec 4.4, Fig. 5, L641–645)**: Cumulatively mask **pre-softmax** attention from register queries $\mathcal{R}$ to semantic keys ($\mathcal{R} \to \mathcal{S}$) vs image keys ($\mathcal{R} \to \mathcal{I}$) from early blocks onward to verify that registers acquire object identity from image latents, not directly from prompt tokens.
  5. **Training-Free Late-Step Head Pruning (Sec 4.5, Table 2, L686–695)**: Rank heads by descending $\bar{m}_{\mathcal{S}}$; skip Top-$K$ heads' QKV/O projections & attention during the **last $1-f=80\%$ of steps** ($f=0.2$, applied to both CFG passes). *(Note: FLOP reduction $(1-f)K/(LH)$ only holds when projections are actually skipped, not zeroed post-hoc, and does NOT hold linearly under Krea 2's GQA where 4 Q heads share 1 KV head!)*

### 2. Paper vs XAI_DiT Customizations
| Item | Original Paper | XAI_DiT Adaptation (Anima / MOSS-Image / Krea 2) | Reason |
|---|---|---|---|
| **Krea 2 Base (GQA + `txtfusion`)** | App. E.2 tested Krea-2-Turbo (8 steps, 553 prompts) | Use App. E.2 as reference benchmark only. Note Krea 2 base ALSO has **bidirectional token-axis refinement in `txtfusion`** (`krea2.md` L24)! Specify exact span-swap points: **`text_in`** (pre-`txtfusion`) vs **`txt_embed`** (post-`txtfusion`). Under GQA (48Q/12KV, 1,344 Q-heads total), per-head KV swaps cannot be isolated from the 4-head KV group; test Q-only vs KV-group swaps. | Fixes factual accuracy about `txtfusion` and GQA constraints. |
| **MOSS-Image (`refiner` + Prefix/Suffix)** | Qwen-Image / FLUX.2 / Krea-2-Turbo | Split $\mathcal{R}$ into **$\mathcal{R}_{\text{prefix}}$** (carries zero prompt content at causal `te.out`, `moss_image.md` L20) and **$\mathcal{R}_{\text{suffix}}$**. To isolate the 2-block bidirectional `refiner` effect, run paired swaps at **`te.out`** vs **`refiner.out`** on identical seeds/prompts (1,008 heads total). | Cleanly separates causal LLM vs bidirectional `refiner` mixing. |
| **Anima (3-Span Control)** | Chat-templated joint-attn DiTs | Define **3 disjoint half-open spans**: $\mathcal{S} = [0, L_t-1)$, $\mathcal{R} = [L_t-1, L_t)$ (`</s>`), and **$\mathcal{P} = [L_t, L)$ (0-pad)**. Because Anima's `ctx` is never updated inside DiT (`anima.md` L73), the register-reading mechanism ($\mathcal{R} \leftarrow \mathcal{I}$) is impossible by construction—making Anima a clean negative control (report `S/R/P` mass decomposition; mark register-query interventions structurally N/A). | Fixes 2-span partition & inclusive/half-open boundary bug; clarifies structural N/A for $\mathcal{R}\to\mathcal{I}$. |

### 3. Verification Invariants
- **Unit Test (CPU)**:
  1. **Span Partition Identity**: On MOSS-Image/Krea 2, $m_{\mathcal{R}_{\text{pre}}} + m_{\mathcal{R}_{\text{suf}}} + m_{\mathcal{S}} == \sum_{j \in \text{txt}} \bar{A}_{\text{img}, j}$; on Anima, $m_{\mathcal{S}} + m_{\mathcal{R}} + m_{\mathcal{P}} == 1.0$ (`max_abs_diff < 1e-6`).
  2. **Causal Masking Isolation**: Masking $\mathcal{R} \to \mathcal{I}$ pre-softmax zeroes post-softmax attention in $\mathcal{R} \to \mathcal{I}$ (`< 1e-6`), preserves **pre-softmax logits** outside $\mathcal{R} \to \mathcal{I}$ (exact), keeps affected $\mathcal{R}$ query rows normalized (`sum == 1.0`, so $\mathcal{R} \to \mathcal{S}$ probabilities renormalize upward), and leaves **other query rows** ($\mathcal{I} \to \mathcal{I}$, $\mathcal{I} \to \mathcal{S}$) post-softmax probabilities unchanged (`max_abs_diff < 1e-6`).
  3. **Head-Pruning No-Op**: $K = 0$ or $f = 1.0$ hands off to native path (`max_abs_diff == 0`).
- **Sanity Reproduction (Colab GPU)**:
  - Log $\bar{m}_{\mathcal{R}}$ (prefix/suffix), $\bar{m}_{\mathcal{S}}$, normalized I2T share, and dominant delimiter token across 10–20 prompts on Krea 2 base, MOSS-Image (`te.out` vs `refiner.out` swap), and Anima control.

---

## Paper 3: `2025_NeurIPS_GatedAttention.pdf` (NeurIPS 2025)

- **Title**: *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free* (Qiu et al., Qwen Team)
- **Verified Pages / Sections**: Sec 2.2 (p. 3), Sec 4.2–4.3 (pp. 7–8, Table 4, Fig. 2, L689–691), Appendix A.2–A.3 (pp. 13–14)
- **Official Repo Checked**: `reference_repo/paper_code/GatedAttention` (`qiuzh20/gated_attention`, commit `f4c2a5f`, MIT)
  - `modeling_qwen3.py` (L292–371, L309–318, L361–362)

### 1. Exact Mechanism & Equations
- **Input / Hook Point**: **G1 SDPA Output Gating** (`modeling_qwen3.py:L361-362`, Sec 2.2 p. 3):
  $$Y' = Y \odot \sigma(X W_\theta), \quad \sigma(z) = \frac{1}{1 + e^{-z}}$$
  applied to concatenated SDPA output $Y$ **before $W_O$ (`o_proj`)**, where $X$ is the post-pre-norm hidden state.
- **Diagnostics (Sec 4.2–4.3, Table 4)**:
  1. **Gate Sparsity**: Mean/quantiles of $g = \sigma(X W_\theta) \in [0, 1]$ (finite precision) per layer/head/token.
  2. **Massive Activations (`M-Act`)**: Max hidden-state activation output by each layer (L689–691; log per-layer profile, not only cross-layer mean).
  3. **Sink Attention**: In LLMs, `F-Attn` is first-token attention; in DiTs, define explicit **DiT sink metrics** (`template_span_mass`, `top1_dynamic_mass`).

### 2. Paper vs XAI_DiT Customizations
| Item | Original Paper | XAI_DiT Adaptation (Anima / MOSS-Image / Krea 2) | Reason |
|---|---|---|---|
| **Krea 2 Native G1 Gate & Causal Attribution** | Qwen3 LLM with G1 gating | **Krea 2 natively implements G1 elementwise SDPA output gating**: `ctx → × sigmoid(attn.gate_logits) → Wo → × g1` (`krea2/model.py:106`).<br>**Critical causal clarification**: Because `attn.gate` depends ONLY on the query token $i$ and sits **after** SDPA, opening `attn.gate` inside layer $\ell$ does **not** change layer $\ell$'s already-computed `attn.probs`! Instead:<br>(a) Define span $\mathcal{R}$'s post-gate contribution per `(query i, head h)` as $g_{i,h} \odot \sum_{j \in \mathcal{R}} A_{i,j}^h V_j^h$ (and full residual delta with `Wo` and `g1`);<br>(b) Test gate causality on **downstream blocks $\ell' > \ell$ and subsequent timesteps** within the **same Krea 2 model** by ablating `attn.gate <- 1.0` vs normal gate (comparing Krea 2 vs MOSS-Image alone cannot isolate gate effects due to architectural/encoder differences!). | Eliminates ill-posed same-layer gate causality claims. |
| **Anima & MOSS-Image** | Ungated baseline | Both lack SDPA output gates (Anima has 3 AdaLN gates: self-attn, cross-attn, MLP; MOSS-Image has 2 shared AdaLN gates: `gate1, gate2`). Log their per-layer `M-Act` and DiT sink profiles as cross-architecture reference baselines. | Accurate block anatomy (`anima.md` L32, `moss_image.md` L29). |

### 3. Verification Invariants
- **Unit Test (CPU)**:
  1. **Channel-Layout & G1 Formula Check**: Verify Krea 2's flat `[B, T+N, 6144]` gate maps channel `h*128 + d` strictly to head `h`, dim `d`, and matches `attn_ctx * sigmoid(gate_logits)` before `Wo` (`max_abs_diff < 1e-6`).
  2. **Gate Bounds & Zeroing**: Logged gate values lie in `[0, 1]`; setting gate to `0` zeroes `attn_out` (`wo` is bias-free in Krea 2).
- **Sanity Reproduction (Colab GPU)**:
  - On Krea 2 (disclosing weight precision since FP8 shifts `M-Act`/gate stats), profile `sigmoid(attn.gate_logits)` across 28 blocks (split by image / prompt / 5 tail tokens) and measure downstream `M-Act` and sink-mass shift when ablating early-block `attn.gate <- 1.0`.

---

## Paper 4: `2026_CVPR_DiTRelationCircuits.pdf` (CVPR 2026)

- **Title**: *Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers* (Wang, Fan, Pan)
- **Verified Pages / Sections**: Sec 4.1–4.4 (pp. 4–6), Sec 5 (pp. 7–8, Fig. 7A), Appendix B.5–B.7 (pp. 36–44)
- **Official Repo Checked**: `reference_repo/paper_code/DiTRelationCircuits` (`Animadversio/DiT-Relation-Circuits`, commit `f4a4832` — **README only; code not yet released**)

### 1. Exact Mechanism & Equations
- **Input / Hook Point**: Cross-attention / `img_to_txt` maps separated by **CFG branch (`cond` vs `uncond` analyzed separately**, App. B.5 L1900–1902, L1924).
- **Formula / Algorithm**:
  1. **Attention Map Synopsis (App. B.5, L1893+)**: Compute inner product between per-sample **object segmentation masks** and **semantic text-token masks** across attention maps, aggregated over timesteps separately for `cond` and `uncond` to identify spatial-relation vs object-generation heads.
  2. **Token-Specific Head Ablation (Sec 4.4, L408+)**: Inside a specific `(layer, head)`, ablate **ONLY the contribution of the target relation token or object token** (NOT the entire head!; e.g., ablating `L2H8` relation contribution drops relation accuracy `67% → 33%`; ablating `L4H3` object contribution drops shape accuracy `90% → 76%`).
  3. **T5/LLM Contextual Fusion (Sec 5, Fig. 7A)**: Pretrained contextual encoders fuse spatial layout into **object-2 (`shape2`) and EOS** tokens rather than explicit relation tokens, causing brittleness under minor prompt edits (e.g., inserting `"the"`).
  *(Note: App. B.7 weight-space screening relies on RoPE on both modalities and does not transfer directly to Anima cross-attn [no RoPE] or Krea 2 text [unrotated `(0,0,0)`].)*

### 2. Paper vs XAI_DiT Customizations
| Item | Original Paper | XAI_DiT Adaptation (Anima / MOSS-Image / Krea 2) | Reason |
|---|---|---|---|
| **Pretrained LLM-DiT Circuit Probe** | Trained-from-scratch PixArt DiTs (RTE vs T5) + App. A.10 | Run training-free **Attention Map Synopsis** (separate `cond`/`uncond`) + **token-specific single-head contribution ablation** on controlled two-object spatial prompts (`"A left of B"` + `"the"` perturbation), tracking relation token, object-1, **object-2**, and **EOS / `</s>` / tail tokens**. | Directly tests Sec 5's T5/LLM fusion hypothesis on Qwen-based DiTs. |

### 3. Verification Invariants
- **Unit Test (CPU)**:
  1. **Synopsis Inner-Product Exactness**: Synthetic mask × token attention yields exact expected region-token score, with `cond` and `uncond` kept separate.
  2. **Token-Specific Head Ablation Isolation**: Removing target token $j^*$ inside head $h^*$ zeroes only token $j^*$'s contribution in head $h^*$ while leaving other tokens $j \neq j^*$ in head $h^*$ AND all other heads $h \neq h^*$ bitwise unchanged.
- **Sanity Reproduction (Colab GPU)**:
  - Measure both **relation accuracy** and **object accuracy** before vs after token-specific ablation on top-synopsis heads (comparing relation token vs object-2 vs EOS/tail tokens) — do not rely on attention mass alone for causal claims.

---

## Paper 5: `2026_CVPR_DiffusionProbe.pdf` (CVPR 2026)

- **Title**: *Diffusion Probe: Generated Image Result Prediction Using CNN Probes* (Cui et al., Alibaba)
- **Verified Pages / Sections**: Sec 3.2–3.4 (pp. 4–6), Sec 4.1 (p. 6, L580–590)
- **Official Repo Checked**: `reference_repo/paper_code/DiffusionProbe` (`Alibaba-VELLDEPTH/DiffusionProbe`, commit `0e56916`, Apache-2.0)
  - `diffusion_probe/data_collection/flux_attention_hook.py` (L95–205), `app.py` (L88, L253), `training/dataset.py` (L85)

### 1. Exact Mechanism & Equations
- **Input / Hook Point (Critical Difference from TemplateRegisters!)**:
  - Official FLUX hook (`flux_attention_hook.py:L110-115, L181-194`) taps **`norm_q` / `norm_k` outputs BEFORE RoPE**, slices `(Q_img, K_txt)`, and applies **text-only softmax** (`softmax` over text keys only, so each row sums to `1.0` — distinct from TemplateRegisters' full-key joint softmax where row sum = text share!), then averages heads.
  - Paper Sec 4.1 (L580–590): extracts 10 consecutive **middle** blocks at **step 5 of 25** (repo `app.py:L88, L253` uses last 10 layers, steps 4–5); probe consumes `(attention, timestep)` to predict **ImageReward** (note: `dataset.py:L85` has `normalize_attention` commented out).
- **XAI_DiT Adaptation**: **Deferred / Reference-only for M1–M2** (probe training is out of scope). Expose a clearly named `img_to_txt_text_softmax` tap (distinct from `img_to_txt_joint_softmax`) in `ModelAdapter` for training-free early-step (20% steps) attention entropy/concentration vs final ImageReward/CLIP correlation checks.
- **Verification Invariants**: CPU test verifies tap does not alter denoiser `v` (`max_abs_diff == 0`) and distinguishes text-only softmax (row sum = `1.0`) from joint-key softmax (row sum $< 1.0$).

---

## Paper 6: `2026_CVPR_ProbeSelect.pdf` (CVPR 2026)

- **Title**: *Toward Early Quality Assessment of Text-to-Image Diffusion Models* (Guo et al.)
- **Verified Pages / Sections**: Sec 3.1–3.4 (pp. 4–5, Eq. 4–7), Supp. Sec 7 (p. 11, L1128–1133, L1210)
- **Official Repo Checked**: `reference_repo/paper_code/ProbeSelect` (`Guhuary/ProbeSelect`, commit `db75742`)
  - `src/data_gen/utils.py` (L140–169), `src/new_pipe/SD3.py` (L173), `src/models/loss/loss.py` (L42–89), `configs/extras/default.yaml` (L14)

### 1. Exact Mechanism & Equations
- **Input / Hook Point**: Early-step (`i ≈ 0.2 S` from noise end) intermediate DiT block residual activations $h_t$ (paper: SD3-M block 20, SD3-L block 28; `utils.py:L140-169` taps block 20, unpatchifies to spatial grid, resizes to `48×48`, applies per-sample 48-channel PCA).
  - **CFG Slice Warning**: `utils.py:L157` slices `output[1][0::2]`, which depends on batch interleaving and selects **negative** samples when `[neg, pos]` is concatenated (`SD3.py:L173`)! In `XAI_DiT`, ALWAYS record explicit `(cond_or_uncond, sample_id, img_token_range, grid_h, grid_w)`.
- **Loss (Paper vs Repo)**: Paper Eq. 5–7 combines margin listwise ranking $\mathcal{L}_{\text{list}}$ + InfoNCE $\mathcal{L}_{\text{Align}}$; repo default config (`default.yaml:L14`, `loss.py:L42-89`) adds a **correlation loss**, includes self in denominator, and increases InfoNCE temperature over epochs.
- **XAI_DiT Adaptation**: **Deferred / Reference-only for M1–M2** (requires per-backbone probe training). Expose `block[i].resid_out` feature tap at early step `i = round(0.2 * S)` (with explicit `cond` slice & grid metadata) for training-free early-vs-final structural PCA/cosine checks.
- **Verification Invariants**: CPU test verifies feature-tap no-op (`max_abs_diff == 0`), exact `cond` sample-ID matching under `[neg, pos]` vs interleaved CFG batches, and listwise loss ordering (requiring at least one sample pair exceeding margin $\alpha$).
