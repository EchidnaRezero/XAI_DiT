# 실험 및 적용 설계 (Experiment & Application Design)

이미 논문에서 통계적으로 입증된 2025–2026 DiT XAI/MI 기법(`docs/specs/01_heatmap.md` ~ `04_verification_mi.md`)을 **Anima · MOSS-Image(본 타깃) · Krea 2 Raw(본 모델)** 3종에 **실사용자·개발자 관점으로 적용해 직관적인 시각화·개입·구조 비교 결과를 얻는 것**을 목표로 한다.
*(논문 재검증용 수백 장 규모 통계 스윕은 기본 범위에서 제외하며, 모델당 대표 **1–2 프롬프트 × 1 시드** 기준 Colab L4 24 GB에서 수 분 내 실행 가능한 규모로 운용한다.)*

> **세부 규격 라우팅 (`experiments/design/` · `plan/`)**
> | 항목 | 세부 문서 |
> |---|---|
> | M1 코어 · 백엔드(Headless ComfyUI) · CPU 불변식 | [`plan/m1.md`](../plan/archive/completed_milestones.md) |
> | 모델별 L4 양자화 기본 조합 · 추론 HP · Phase 4 선택 대조 | [`experiments/design/runtime_and_hp.md`](../experiments/design/runtime_and_hp.md) |
> | 대표 프롬프트 & 관찰 타깃 · 대조쌍 | [`experiments/design/prompts.md`](../experiments/design/prompts.md) |
> | 표준 출력 폴더 · Same-State 오버레이 · PNG/JSON 규격 | [`experiments/design/output_spec.md`](../experiments/design/output_spec.md) |

---

## 1. 공통 실행 원칙

1. **기본 환경은 Colab L4 (24 GB) 양자화 조합**: Anima(BF16), MOSS-Image(`sft_int8_convrot` + `qwen3.5_9b_int8_convrot`), Krea 2 Raw(`fp8_scaled` + `qwen3vl_4b_fp8_scaled`). 원본 BF16 대조는 선택적 Phase 4(A100/H100)로 분리한다.
2. **오프라인 텍스트 인코딩 캐시**: L4 기본 텍스트 인코더로 프롬프트를 1회 인코딩·캐싱한 뒤 언로드하고 DiT 단독 추론에서 재사용한다.
3. **통일 샘플러 & $\sigma$ 정합**: 3개 모델 기준선 샘플러는 결정론적 **`euler`**로 통일하고, 스텝은 `transformer_options["sigmas"]`에서 역산한 `(i, σ_i)`를 병기한다.
4. **메모리 안전성 & 고정 CFG 배칭**: 전체 `[B, H, N+L, N+L]` GPU 적재를 금지하고 **query-chunked softmax + GPU 스트리밍 집계**를 기본으로 한다. `cond`/`uncond`는 `cond_or_uncond` 메타데이터로 라우팅하며 배칭 정책을 고정한다.
5. **최소 필수 무결성 검증**:
   - **CPU 단위 테스트 (`tests/`, 무료 Colab CPU)**: 훅·수학·GQA·스팬 불변식 통과 (`210/210` 완료).
   - **E0 1회 Sanity (모델당 1회)**: `exact_noop == 0.0`, 3회 비결정성 바닥선 `0.0`, L4 피크 VRAM·`sec/forward` 실측 (3모델 완료).

---

## 2. 실사용 적용 실험 목록 (E0 ~ E4, 개인 L4 규모)

> 각 실험은 **대표 1–2프롬프트 × 1시드(`42`)** 기준 수 분 내 결과를 확인하는 실용 규모로 수행한다.

### E0. 사전 검증 *(3모델 완료)*
- **목적**: 어댑터 부착·비활성 시 `exact_noop == 0.0`, 비결정성 바닥선 `0.0`, L4 VRAM·속도 확인.

### E1. 단어 히트맵 & 텍스트 집중 블록 ([`01_heatmap.md`](specs/01_heatmap.md))
| ID | 실사용 질문 / 기능 | 대상 모델 | L4 실용 규모 & 산출물 |
|---|---|---|---|
| **E1-1** | **단어 히트맵** ("내가 쓴 태그가 그림의 어디에 붙나?") | Anima, MOSS, Krea 2 | 대표 1프롬프트 전 스텝 단일 패스 → `overlay_<tag>.png`, `[4태그 × 6스텝]` `grid_attn_steps.png` *(3모델 완료)*; 선택적으로 상위 블록(Top-5) vs 전체 블록 + GS 비교 |
| **E1-2** | **선명한 개념 히트맵** (ConceptAttention $o_c o_x^T$) | MOSS, Krea 2 / Anima N/A | 대표 1프롬프트에서 원시 어텐션 vs 개념 출력 공간 히트맵 비교 (Krea 2 `pre_gate` vs `post_gate`) |
| **E1-3** | **텍스트 강도 블록 & 블러 확인** (Seg4Diff) | Anima *(proxy)*, MOSS, Krea 2 | 단일 패스 블록별 V-norm 프로파일(`e1_3_vnorm_profile.png`, *3모델 완료*) → 피크 vs 비피크 블록 1D 텍스트 블러 Before/After 2장 비교 |

### E2. 핵심 블록 찾기 ([`02_layer_selection.md`](specs/02_layer_selection.md))
| ID | 실사용 질문 / 기능 | 대상 모델 | L4 실용 규모 & 산출물 |
|---|---|---|---|
| **E2-2 (주력)** | **1회 추론으로 핵심 블록 순위** (`Cont^(l)`) | Anima, MOSS, Krea 2 | **단 1회 추론**으로 전 블록 기여도 순위 산출(Krea 2: `wo(x)` 모듈 호출, `raw_mvo` vs `gated_mvo`) → 상위 20% vs 하위 20% 블록 중립 치환 Before/After 2장 비교 *(LoRA/병합/SLG 근거)* |
| **E2-1 (선택)** | **블록 1개씩 꺼보기** (StableFlow True Block Bypass) | Anima / 선택적 MOSS·Krea 2 | 대표 **`k=1` 프롬프트 × 1시드** 기준 시각적 확인용으로만 축소 운용 (수백 장 `k=8~64` 스윕 제외) |

### E3. 스텝별 실제 변화량 & 프롬프트 제어 ([`03_tag_influence.md`](specs/03_tag_influence.md))
| ID | 실사용 질문 / 기능 | 대상 모델 | L4 실용 규모 & 산출물 |
|---|---|---|---|
| **E3-0 (핵심)** | **Same-State `[태그 × 스텝]` 오버레이** ("어느 스텝에서 어느 부위가 실제로 바뀌나?") | Anima, MOSS, Krea 2 | 고정 궤적 $x_{\sigma}$에서 **`direct_effect`(Logit Knockout)** + **`random_ctrl` 대조 행** + $\Delta\sigma$ 가중판 산출 → `grid_direct_effect_steps.png` & **`grid_attn_vs_delta_v.png`**(어텐션 착시 판별; 3모델 합계 ~6–7분). 필요 시 1개 태그에 대해 Full-Rollout 비교 |
| **E3-1** | **속성/소품 K·V 패칭** ("구도는 유지하고 단어 하나만 교체") | Anima, MOSS, Krea 2 | 토큰 길이·ID 일치 대조쌍 1–2개에 대해 초기/중기/후기 블록군별 Before/After 비교 |
| **E3-2 (축소)** | **개념 고정 타이밍 확인** (PCI) | Anima, MOSS, Krea 2 | 전역 스타일 1쌍 + 국소 소품 1쌍에 대해 **대표 6–8개 전환 스텝 × 1시드** 이미지 스트립으로 타이밍 직관 확인 *(다중 시드 통계 곡선 제외)* |
| **E3-3** | **초반 어텐션 강조** (TACA) | Anima, MOSS, Krea 2 | $\sigma \ge 0.97$ 구간 $\gamma_0 = 1.2$ 적용 시 Before/After 2장 비교 (L4 `Listing 2` / chunked SDPA) |
| **E3-4** | **구도→디테일 전환 탐지 & 2단계 프롬프트** (DiT-ST / SAP) | Anima, MOSS, Krea 2 | 단일 패스 어텐션 변화량으로 전환 스텝 확인 → 1단계 vs 2단계 프롬프트 Before/After 2장 비교 |

### E4. 3모델 내부 구조 해부 (MI · 포트폴리오 비교) ([`04_verification_mi.md`](specs/04_verification_mi.md))
| ID | 해부 질문 | 대상 모델 | L4 실용 규모 & 산출물 |
|---|---|---|---|
| **E4-1** | **어텐션 싱크 마스킹** | Anima, MOSS, Krea 2 | 대표 1프롬프트에서 동적 Top-$k$ 마스킹 Before/After 비교 (Anima `P` 스팬 그룹 처리) |
| **E4-2** | **3모델 템플릿 레지스터 비중 비교** | Anima · MOSS · Krea 2 | 단일 패스 `S/R_prefix/R_suffix/P` 정규화 비중 3모델 비교 차트(`e4_2_template_share.png`, *완료*) + 선택적 `R→I` 마스킹 Before/After 1프롬프트 확인 *(Anima `R→I` 구조적 N/A)* |
| **E4-3** | **Krea 2 출력 게이트 & 3모델 `M-Act`** | Krea 2 *(주)* / Anima·MOSS *(참조)* | 3모델 `M-Act` & Krea 2 28블록 `gate_value` 희소성 프로파일(*완료*) + Krea 2 초기 블록 `[0..5]` `gate_value <- 1.0` 절제 Before/After 1프롬프트 확인 |
| **E4-4** | **공간 관계 회로 확인** | Anima, MOSS, Krea 2 | 2-객체 좌/우 1–2프롬프트에서 상위 관계 헤드 내 관계어 항($A_j V_j$) 단독 제거 Before/After 확인 |
| **E4-5** | **조기 스텝 탭** *(M3+ 보류)* | — | `residual_out` 탭 계약만 유지; 프로브 학습은 M3 이후 |

---

## 3. 실행 우선순위 (실용 중심)

1. **1차 핵심 시각화 세트 (L4, 수 분 단위)**:
   - `E0` (3모델 완료 ✅) → `E1-1 + E1-3 + E4-2 + E4-3` (3모델 완료 ✅) → **`E3-0` Same-State $\Delta v$ & `attn vs Δv`** (다음 실행)
2. **2차 실사용 개입 Before/After 세트 (L4, 각 2–5분)**:
   - `E2-2` (`Cont^(l)` 핵심 블록) · `E3-1` (K/V 패칭) · `E3-3` (TACA) · `E4-3` (Krea 2 게이트 절제) · `E1-2` (ConceptAttention)
3. **3차 선택/보충 (필요 시)**:
   - `E3-2` (PCI 6–8스텝 스트립) · `E3-4` (SAP) · `E4-1` (싱크) · `E4-4` (공간 관계) · Phase 4 (A100/H100 BF16 정밀 대조)
4. **기록 경계 (`docs/workspace_guide.md`)**: 계획 실험 결과는 **`experiments/`**, 예상치 못한 버그/OOM은 **`issues/`**, 최종 사용자용 요약 대시보드는 **`showcase/`**에 정리한다.
