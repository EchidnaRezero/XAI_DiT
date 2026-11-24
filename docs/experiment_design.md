# 실험 및 검증 설계 (Experiment Design)

`docs/specs/01_heatmap.md` ~ `04_verification_mi.md` 및 `docs/architecture.md`를 바탕으로, **Anima → MOSS-Image(본 타깃) → Krea 2 Raw(본 모델)** 3종에 대한 핵심 실험·검증 설계를 정의한다. *(GPT 6.2 Astra & Claude Fable 5.2 `high` 교차 리뷰 반영 완료)*

> **세부 운영 규격 라우팅 (`experiments/design/` · `plan/`)**
> | 항목 | 세부 문서 |
> |---|---|
> | M1 범위 · 실행 백엔드(Headless ComfyUI) · 불변식 매트릭스 | [`plan/m1.md`](../plan/m1.md) |
> | 모델별 L4 기본 조합 · 추론 HP · 오프라인 TE/평가기 · Phase 4 보충 | [`experiments/design/runtime_and_hp.md`](../experiments/design/runtime_and_hp.md) |
> | 프롬프트 매니페스트 스키마 · 분할(`calib`/`rank`/`eval`) · 대조쌍 | [`experiments/design/prompts.md`](../experiments/design/prompts.md) |
> | 표준 출력 폴더 · Same-State 오버레이 · 원시 배열 · JSON 스키마 | [`experiments/design/output_spec.md`](../experiments/design/output_spec.md) |

---

## 1. 공통 통제 원칙

1. **기본 환경은 Colab L4 (24 GB)**이며, L4를 초과하는 정밀/대규모 옵션(Krea 2 Raw BF16, 대규모 스윕 `k=64`, `FlexAttention` 등)은 폐기하지 않고 **Phase 4 (A100 40/80GB · H100 일괄 보충 세션)**로 모아 검증한다.
2. **오프라인 BF16 텍스트 인코딩 캐시**: MOSS-Image(Qwen3.5-9B)·Krea 2(Qwen3-VL-4B)의 양자화 텍스트 인코더 교란을 제거하기 위해, **모든 프롬프트 매니페스트를 BF16으로 1회 사전 인코딩·해시 캐싱**한 뒤 DiT 단독 추론 세션에서 재사용한다. 평가기(DINOv2, LPIPS, CLIP-T, Grounded-SAM2, VQA)도 저장된 이미지 대상 **오프라인 패스**로 분리 실행한다.
3. **통일 샘플러 & $\sigma$ 정합**: 3개 모델 공통 XAI 기준선 샘플러는 결정론적 **`euler`**로 통일한다(Anima `er_sde`는 다단계·확률적 노이즈 주입으로 PCI prefix 캐시 및 DiT-ST를 교란하므로 보조 강건성 복제본으로 분리). 스텝은 호출 횟수 누적이 아닌 `transformer_options["sigmas"]`에서 역산한 `(i, σ_i)`를 병기한다.
4. **메모리 안전성 & 고정 CFG 배칭**: 전체 `[B, H, N+L, N+L]` GPU 적재를 금지하고 **query-chunked softmax + 스트리밍 집계**를 기본으로 한다. `cond`/`uncond`는 ComfyUI `cond_or_uncond` 청크를 샘플 행으로 명시 매핑하고, 러너에서 **고정 배칭 정책**을 강제해 훅 유무에 따른 배칭 전환을 차단한다.
5. **3단계 검증 경계 분리**:
   - **비활성 Exact No-Op (Pass/Fail)**: 훅 비활성 시 native fused SDPA로 위임하며 `max_abs_diff == 0`.
   - **경로 일치 기준선 (Path-Matched Identity)**: 관찰·개입 훅은 manual chunked SDPA 경로의 **항등 개입(identity) 기준선**과 비교한다(native fused SDPA 대비 BF16 커널 허용오차 `1e-3` 별도 기록).
   - **연구 관측치 & 통계 판정**: 비결정성 바닥선(다중 반복 실측) 대비 paired bootstrap 95% CI로 보고하며, 특정 층 우위·고정 시점·게이트 효과는 실측치로 기록한다.

---

## 2. 단계별 실험 설계 (E0 ~ E4)

### E0. 사전 검증 (CPU 단위 테스트 + GPU 1-Seed Sanity & 예산 실측)
- **CPU 단위 테스트** (Colab CPU PyTorch + 로컬 NumPy 수학 불변식): [`plan/m1.md`](../plan/m1.md)에 명시된 **M1 범위 불변식** 통과 확인 (시퀀스 순서 동등성, GQA `h_kv = h_q // 4`, `gate_logits` vs `gate_value` 분리, True block bypass 호출 수 `0`, `R→I` pre-softmax 마스킹 시 비선택 logits 불변·선택 확률 `<1e-6`·`R` 행 합 `=1`·다른 쿼리 행 불변, `cond_or_uncond` 분기 순서 반전/분리 호출/`B>1` 복원 동등성, 3-span 반열린 구간 `[start, end)` 파티션 등; M3 보류 `ProbeSelect` loss 및 범위 외 head pruning 제외).
- **GPU Sanity & 자원 예산 실측**: 모델별 1프롬프트 × 1시드(`42`) × 3회 반복으로 비결정성 바닥선, exact no-op(`0`), 경로 일치 identity 오차(`<=1e-3`), 그리고 **단계별(TE / DiT 기본 / 최중량 활성 훅 / VAE / 오프라인 평가기) GPU allocated·reserved, CPU RSS, `sec/forward`**를 측정해 스윕 분할(shard)·재개 계획을 확정한다.

### E1. 방향 01 — 히트맵 & Saliency ([`01_heatmap.md`](specs/01_heatmap.md))
| ID | 기법 | 대상 모델 | 핵심 설계 & 검증 |
|---|---|---|---|
| **E1-1** | **ExploringMMDiT** (`img_to_txt` + Top-5 + GS) | Anima, MOSS, Krea 2 | **`calib` 분할(Grounded-SAM2 마스크 기반 BCE·soft mIoU·MSE 평균 순위)**에서 Top-5 블록을 사전 선정·동결한 뒤, **분리된 `eval` 분할**에서 `{All vs Top-5}` × `{No-GS vs GS(k=7, σ=1.0)}` 2×2 절제(집계 스텝: 전체 스텝 평균 + 대표 6스텝; mIoU는 `localizable: yes` 타깃 한정). |
| **E1-2** | **ConceptAttention** ($o_c o_x^T$) | MOSS, Krea 2 *(단일 스트림 확장)* / Anima N/A | `prompt_output_space` vs `open_set_concept_stream`(**타깃 + 배경 개념 어휘** 포함), `per_layer` vs `after_mean`, **Krea 2 `attn_pre_proj` (`pre_gate`) vs `post_gate` (`ctx × gate_value`)** 비교(개념 잔차 갱신은 자체 게이트·`Wo`·`g1` 통과). |
| **E1-3** | **Seg4Diff** (V-norm & 1D 텍스트축 블러) | Anima *(proxy)*, MOSS, Krea 2 *(12 KV pre-repeat)* | `calib` 5–10프롬프트로 블록별 `||V_prompt||_2 / ||V_img||_2` 프로파일링 → 동일 시드에서 피크 vs 비피크 블록에 **프롬프트 구간 한정 1D 텍스트축 블러(`k=5, σ=9`, 스팬 질량 보존 정규화)**를 실제 $A'V$에 반영해 인과 확인. |

### E2. 방향 02 — 층 선별 & 중요도 프로파일링 ([`02_layer_selection.md`](specs/02_layer_selection.md))
| ID | 기법 | 대상 모델 | 핵심 설계 & 검증 |
|---|---|---|---|
| **E2-1** | **StableFlow** (True Block Bypass) | Anima (28), MOSS (42), Krea 2 (28) | 전 블록 1개씩 pre-body 우회(`cond`/`uncond` 양 분기). `k=16` 프롬프트를 **비중첩 8+8 반분할**로 Spearman $\rho$ 안정성 검증(nested 비교 금지), DINOv2 ViT-g/14 CLS 코사인 기반 Vitality + 프롬프트 단위 paired bootstrap 95% CI. Vital 층 선정 규칙: **elbow 우선, 미검출 시 상위 20% 반올림 고정**. *(E0 `sec/forward` 실측 후 L4 예산 초과 시 Krea 2 `k=16`은 Phase 4 승격)* |
| **E2-2** | **LocalizingKnowledgeDiT** (`Cont^(l)`) | Anima, MOSS, Krea 2 | **`rank` 분할**에서 전 스텝 평균 query-chunked `Cont^(l)(κ)` 산출(Krea 2: `raw_mvo` vs `gated_mvo`, 양자화 호환을 위해 `wo.weight` 직접 접근 대신 **`wo(x)` 모듈 호출** 사용) → **분리된 `eval` 분할**에서 **토큰 ID·길이 일치 중립 치환**(`Top-20%` vs `Bottom-20%` vs **`Random-20%` 대조**)으로 인과 검증. |

### E3. 방향 03 — 태그 영향도 & 시간·인과 개입 ([`03_tag_influence.md`](specs/03_tag_influence.md))
| ID | 기법 | 대상 모델 | 핵심 설계 & 검증 |
|---|---|---|---|
| **E3-0** | **Same-State `[태그 × 스텝]` 오버레이 & Rollout 브리지** | Anima, MOSS, Krea 2 | 고정 궤적 $x_{\sigma}$에서 **`direct_effect`(Logit Knockout)**와 **`total_effect`(길이·ID 일치 중립 치환)**를 분리 측정하고(Anima는 `value=0` 보조 대조), **CFG 결합 출력**(주) + `cond` 단독(진단)을 명시한다. **원시 $\|\Delta v\|$**와 **$\Delta\sigma_i \|\Delta v_i\|$ 가중 버전**, **동일 토큰 수 무작위 토큰 대조 행**을 함께 출력하며, **최종 $x_0$ 인과 주장은 동일 상태 분기 후 끝까지 생성한 Paired Full-Rollout 결과 및 브리지 지표($\sum_i \Delta\sigma_i \Delta v_i$ vs 실제 $\Delta x_0$)로만 판정**한다. |
| **E3-1** | **TextGenParamLocalization** (`K/V` 패칭) | Anima, MOSS, Krea 2 | 토큰 ID·길이 일치 대조쌍에 대해 `i >= start` 구간에서 텍스트 `K/V` 패칭(기본 `cfg_scope="both"`, `cond_only` 절제; Krea 2는 12 KV 그룹 단위). *(Anima 전 블록 E3-1 == E3-2 무료 교차검증 포함)* |
| **E3-2** | **PCI** (시간축 개념 삽입 곡선) | Anima, MOSS, Krea 2 | 결정론적 `euler` 기준선에서 **`{x_k}` Prefix 캐싱(~2× 연산 절감, 전체 $O(S^2)$ — 전환점 stride 명시)** 적용. 캐시 사용/미사용 최종 결과 일치 검증, `k=0`/`k=S` 경계·base 사전 발현 시드 처리 및 미교차(`NaN`) 규칙 고정, 16시드(L4 1차 8시드 잠정) 기준 $\tau := \sigma$ 축 $\tau_{50}, \tau_{70}, W$ 측정(비국소화 스타일·아티스트는 참조 정규화 점수 `(s(x)-s(P_b))/(s(P_c)-s(P_b))` 병행). |
| **E3-3** | **TACA** (전-softmax logit 스케일링) | MOSS, Krea 2 / Anima *(post-`k_norm` + additive)* | **규범 창: $\sigma \ge 0.97$** (모델별 실제 활성 `(i, σ_i, active)` 로깅; `first_n_steps`는 별도 절제). $\gamma_0 = 1.2$, `{both vs cond_only}` × `{템플릿 포함/제외}` 절제(**Krea 2 `neg=""` 시 `both × 템플릿 제외`는 `cond_only`와 동일하므로 N/A 표기**). L4에서는 **Listing 2 (`K_txt` 스케일링 + 텍스트 query 출력 원복)** / chunked SDPA 사용. |
| **E3-4** | **DiT-ST 구간 탐지 & SAP 스케줄링** | Anima, MOSS, Krea 2 | DiT-ST Eq. 5–9(`fp32`, $w=3, \tau=10^{-4}$ + elbow fallback)는 **$\hat{\Delta}_i / |\Delta\sigma_i|$ 정규화 또는 균일 $\sigma$ 보간 격자**에서 곡률을 산출하고 경계 $i \to \sigma_i$로 변환해 PCI $\tau_{50}$과 비교한다. SAP는 1단계 단일 프롬프트 대조 vs 2단계 $\sigma$-구간 프롬프트 번들(레이아웃 프록시 → 상세) 비교. |

### E4. 방향 04 — 검증 & 기계론적 해석(MI) ([`04_verification_mi.md`](specs/04_verification_mi.md))
| ID | 기법 | 대상 모델 | 핵심 설계 & 검증 |
|---|---|---|---|
| **E4-1** | **AttentionSinksDiT** (동적 싱크 마스킹) | Anima, MOSS, Krea 2 | 상대 깊이 25%/50%/75%에서 **2-pass chunked `Q_img` 행** 동적 Top-$k$ (`k=1, 4` per-head 기본 + `Random-k` 대조) `-1e4` 마스킹. 사전 선정 동일 인덱스에서 `mass_before`, `mass_after`, **`residual_ratio < 1e-6`** 확인. *(Anima는 0-패딩 로짓 동률 문제를 막기 위해 `P` 스팬을 단일 그룹으로 취급)* |
| **E4-2** | **TemplateRegisters** (템플릿 레지스터) | Krea 2, MOSS, Anima *(3-span 대조)* | 10–20프롬프트 정규화 I2T 비중 비교, MOSS `te.out` vs `refiner.out` · Krea 2 `text_in` vs `txt_embed` 스팬 교체, $\mathcal{R}\to\mathcal{I}$ vs $\mathcal{R}\to\mathcal{S}$ 경로 마스킹. **Anima는 `S/R/P` 질량 분해만 산출하고 $\mathcal{R}\to\mathcal{I}$ 개입은 구조적 N/A** 표기. Krea-2-Turbo App. E.2 수치는 외부 참조값으로만 기재. |
| **E4-3** | **GatedAttention (G1)** (Krea 2 출력 게이트) | **Krea 2** *(주 대상)* / Anima·MOSS *(M-Act 참조)* | Krea 2 28블록 `gate_value = sigmoid(gate_logits)` 희소성·`M-Act` 프로파일링 + 초기 블록(`[0..5]`)에 **`gate_value <- 1.0`, 평균 게이트 치환, 연속 스케일(dose-response)** 절제 시 후속 블록·스텝 변화 측정. *(L4 FP8 결과는 잠정 표기; Phase 4 BF16 또는 L4 BF16 block-offload 소규모 보정으로 정밀 확인)* |
| **E4-4** | **DiTRelationCircuits** (공간 관계 회로) | Anima, MOSS, Krea 2 | 좌우/상하 **역전쌍 포함 ~20개 2-객체 프롬프트**를 **헤드 선정(`rank`) vs 평가(`eval`) 분할**로 분리. `cond`/`uncond` 분리 Synopsis(Grounded-SAM2 객체 마스크 기반) → 상위 헤드 내 **post-softmax $A_{h,j} V_{h_{kv},j}$ 항 단독 차감**(GQA 공유 `V` 오염 차단) 후 **전체 성공률 및 객체 검출 조건부 관계 정확도** 분리 측정. |
| **E4-5** | **DiffusionProbe / ProbeSelect** *(단계 분리)* | Anima, MOSS, Krea 2 | **(M1)** `residual_out` 탭 계약·단위 검증 → **(M2)** 조기 스텝(`i ≈ 0.2S`) 무학습 탭 정합성·최종 품질 Spearman $\rho$ 점검 → **(M3+)** 프로브 학습. |

---

## 3. 실행 매트릭스 (`실험 ID × 모델`) 및 기록 경계

모든 실험의 실행 Phase·선행 조건·구조적 N/A를 아래 표로 확정한다(누락 실험 없음).

| 실험 ID | Anima (2B) | MOSS-Image (5B) | Krea 2 Raw (12.8B) | 비고 / 선행 조건 |
|---|---|---|---|---|
| **E0** | **Phase 1 (L4)** | **Phase 2 (L4)** | **Phase 3 (L4)** | 모든 본 실험 선행 필수 (예산·`sec/forward` 실측) |
| **E1-1** | **Phase 1 (L4)** | **Phase 2 (L4)** | **Phase 3 (L4)** | `calib` Top-5 동결 후 `eval` 2×2 |
| **E1-2** | 구조적 N/A | **Phase 2 (L4)** | **Phase 3 (L4)** | Anima는 텍스트 출력 공간 없음 |
| **E1-3** | **Phase 1 (L4)** *(proxy)* | **Phase 2 (L4)** | **Phase 3 (L4)** | Krea 2는 12 KV pre-repeat 기준 |
| **E2-1** | **Phase 1 (L4)** | **Phase 2 (L4)** | **Phase 3 (L4 `k=8`) / Phase 4 (`k=16`)** | E0 예산 실측 후 `k=16` 위치 확정; Phase 4 `k=64` |
| **E2-2** | **Phase 1 (L4)** | **Phase 2 (L4)** | **Phase 3 (L4 FP8 잠정) / Phase 4 (BF16)** | `rank` → `eval` 분리 필수 |
| **E3-0** | **Phase 1 (L4)** | **Phase 2 (L4)** | **Phase 3 (L4)** | `direct_effect` + `total_effect` + Rollout 브리지 |
| **E3-1** | **Phase 1 (L4)** | **Phase 2 (L4)** | **Phase 3 (L4)** | Anima 전 블록 E3-1 == E3-2 교차검증 |
| **E3-2** | **Phase 1 (L4)** | **Phase 2 (L4)** | **Phase 3 (L4)** | Phase 4에서 50–100시드 풀 스윕 보충 |
| **E3-3** | **Phase 1 (L4)** | **Phase 2 (L4)** | **Phase 3 (L4)** | Phase 4에서 `FlexAttention` 대조 |
| **E3-4** | **Phase 1 (L4)** | **Phase 2 (L4)** | **Phase 3 (L4)** | E3-2 완료 후 $\sigma$ 경계 대조 |
| **E4-1** | **Phase 1 (L4)** | **Phase 2 (L4)** | **Phase 3 (L4)** | Anima `P` 스팬 그룹 처리 |
| **E4-2** | **Phase 1 (L4)** *(S/R/P만, R→I N/A)* | **Phase 2 (L4)** | **Phase 3 (L4)** | App. E.2는 외부 참조값 |
| **E4-3** | **Phase 1 (L4)** *(M-Act 참조)* | **Phase 2 (L4)** *(M-Act 참조)* | **Phase 3 (L4 FP8 잠정) / Phase 4 (BF16)** | Krea 2 주 대상 (`gate_value` 절제) |
| **E4-4** | **Phase 1 (L4)** | **Phase 2 (L4)** | **Phase 3 (L4)** | `rank`/`eval` 분리, post-softmax $A_j V_j$ 차감 |
| **E4-5** | M1 탭 / M2 무학습 / M3+ 학습 | M1 탭 / M2 무학습 / M3+ 학습 | M1 탭 / M2 무학습 / M3+ 학습 | 프로브 학습은 M3 이후 |

- **기록 경계 ([`docs/workspace_guide.md`](workspace_guide.md))**: 계획 실험 결과는 **`experiments/`**, 예상치 못한 버그/OOM/수치 이상은 **`issues/`**(RAW 로그)로 분리 기록하고, 해결된 회귀 검사는 `tests/`로 승격한다.
