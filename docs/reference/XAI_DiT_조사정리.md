# XAI_DiT 조사 정리

대상 모델: Anima(cross-attn DiT), Krea 2(본 모델만, anime 변형 제외), MOSS-Image(joint-attn DiT)
대상 도구: 히트맵, 태그 영향도, V ablation

---

## 1. 요약

| 질문 | 답 |
|---|---|
| 1. 논문 기법이 실제로 많이 쓰이는가 | 주류는 두 가지다. 어텐션 지도 읽기, 그리고 블록·층 단위로 개입한 뒤 최종 이미지를 재는 것.<br>gradient·LRP는 DiT 해석에서 쓰이지 않는다.<br>후속 연구가 실제로 가져다 쓰는 DiT 도구는 Stable Flow(층 선택)와 ConceptAttention(지도) 정도다. 나머지는 인용은 돼도 실제 사용은 적다. |
| 2. V ablation을 무엇으로 바꿔야 표준에 가까운가 | 기본을 "반사실 프롬프트 패칭"으로 바꾼다. 태그를 중립어로 바꾼 실행의 텍스트 입력(또는 텍스트 K·V)으로 블록 단위 교체.<br>value 0은 기본값이 아니라 옵션으로 내린다. mean·반사실 값과 나란히 두고 대체값에 따른 차이를 함께 보여 준다.<br>같은 개수 무작위 토큰 ablation을 대조군으로 붙이고, 스텝 구간별로 나눠 보여 주고, 끝까지 생성한 최종 이미지 지표를 추가한다. |
| 3. 유효성을 어떻게 검증하는가 | 지도와 "태그를 빼고 끝까지 생성한 실제 차이"를 비교한 기준 충족 논문은 없다. 직접 설계해야 한다.<br>쓸 재료는 있다: 무작위 대조, no-op·시드 짝·bootstrap CI, VQA 판정기 여러 개, Grounded-SAM2 마스크와 위치 비교, 층 단위 대조, 무작위화 검사.<br>한 스텝 측정은 최종 결과를 약하게만 대변하므로(상관 약 0.5) 지도는 "그 스텝의 국소 민감도"로 표기한다. |

---

## 2. 근거 기준

### 기준

| 구분 | 조건 | 표시 |
|---|---|---|
| 근거 | 2025년 이후 + NeurIPS·ICML·ICLR·CVPR·ICCV·ECCV 본 트랙 + DiT 대상 | 연도·학회 |
| 예외(기반기술) | 기준 밖이지만 지금도 쓰는 기반기술 | "예외(기반기술)" |
| 경계 | AAAI·TMLR·Findings·워크숍 | "경계" |
| 근거로 안 씀 | 2025년 이전, UNet 전용, arXiv 전용 | 아래 "제외한 근거" 표 |
| 게재 미확인 | 공식 페이지로 게재를 확인 못 함 | "(공식 페이지 미확인)" |
| 본 트랙 추정 | 게재는 확인했으나 본 트랙인지 확정 못 함 | "(본 트랙 추정)" |

### 쓰는 근거

| 논문 | 연도·학회 | 구분 | 이 문서에서 쓰는 곳 |
|---|---|---|---|
| Stable Flow | CVPR 2025 | 근거 (보유 · `02_layer_selection`) | 질문 1, 3 |
| Exploring MM-DiT | ICCV 2025 | 근거 (보유 · `01_heatmap`) | 질문 1, 2, 3 |
| TACA | ICCV 2025 | 근거 (보유 · `03_tag_influence`) | 질문 2(스텝) |
| TextGen 파라미터 국소화 | ICLR 2025 | 근거 (보유 · `03_tag_influence`) | 질문 1, 2 |
| ConceptAttention | ICML 2025 | 근거 (보유 · `01_heatmap`) | 질문 1, 2 |
| Localizing Knowledge in DiTs | NeurIPS 2025 본 트랙 | 근거 (보유 · 루트) | 질문 1, 2, 3 |
| Seg4Diff | NeurIPS 2025 본 트랙 | 근거 (보유 · `01_heatmap`) | 질문 1, 3 |
| DiT-ST | NeurIPS 2025 | 근거 (보유 · `03_tag_influence`) | 질문 2(스텝) |
| DiT Relation Circuits | CVPR 2026 (공식 페이지 미확인) | 근거 (보유 · `04_verification_mi`) | 질문 1, 2 |
| SAP | CVPR 2026 | 근거 (보유 · `03_tag_influence`) | 질문 2(스텝) |
| Probe-Select | CVPR 2026 | 근거 (보유 · `04_verification_mi`) | 질문 3 |
| Diffusion Probe | CVPR 2026 | 근거 (보유 · `04_verification_mi`) | 질문 3 |
| PCI (Temporal Concept Dynamics) | ICLR 2026 | 근거 (보유 · `03_tag_influence`) | 질문 1, 2, 3 |
| Attention Sinks in DiT | ICML 2026 (PMLR v306) | 근거 (보유 · `04_verification_mi`) | 질문 1, 2, 3 |
| Template-Registers (`arXiv:2607.19139`) | 2026 arXiv | 실무 참고 (보유 · `04_verification_mi`) | 질문 2·6-3 (Qwen 계열 템플릿 레지스터) |
| Gated Attention (`arXiv:2505.06708`) | NeurIPS 2025 Best Paper | 구조 참고 (보유 · `04_verification_mi`) | 질문 6-3 (Krea 2 `sigmoid(Wg·h)` 게이트·sink-free) |
| FluxSpace · FreeFlux · CASteer | 2025~2026 본 트랙 | 조사 인용만 (PDF 미보유) | 질문 1 (스티어링·편집 계열, 4방향 코어 외) |
| Prompt-to-Prompt · ACDC · MIB · Sanity Checks | 2018~2025 | 예외(기반기술, PDF 미보유) | 질문 2·3 배경 근거만 인용 |

### 제외한 근거

| 자료 | 제외 이유 |
|---|---|
| DAAM | 2023, UNet |
| DiffQuickFix | ICLR 2024, UNet |
| LocoGen | ICML 2024, UNet 중심 |
| InfoDecomposition | ICLR 2024, UNet |
| EOS 논문 | NeurIPS 2024, UNet 중심 |
| Wen | ICLR 2024, UNet |
| HRV (Head Relevance Vectors) | ICLR 2025지만 UNet 전용(SD v1.4, SDXL) |
| MemorizationAnisotropy | ICLR 2026이지만 UNet 전용(SD v1.4, v2.0) |
| Template-Registers | arXiv 전용 (단, Qwen 계열 템플릿 실무 참고용으로 `04_verification_mi`에 별도 보유) |
| Uni-AdaVD | arXiv 전용 |
| Heimersheim & Nanda | 기준 밖(LLM 쪽 방법론 글) |
| Geva 2023 | 2023, LLM 대상 |
| ROAD | 기준 밖(분류기 XAI 평가용) |
| Lin 서베이 | arXiv 전용 |
| Lakhani 리뷰 | IEEE TAI, UNet 중심 |

- 기준을 충족하면서 DiT를 다룬 리뷰 논문은 찾지 못했다.

---

## 3. 질문 1: 논문 기법이 실제로 많이 쓰이는가

### 3-1. 기준 충족 논문 12편이 쓰는 기본 동작

대상 12편: FluxSpace, Stable Flow, Exploring MM-DiT, FreeFlux, TextGen 국소화, ConceptAttention, Localizing Knowledge, Seg4Diff, DiT Relation Circuits, CASteer, PCI, Attention Sinks

| 기본 동작 | 편수 | 쓰는 논문 |
|---|---|---|
| 어텐션 맵 읽기 | 9 | FluxSpace, Stable Flow, Exploring MM-DiT, FreeFlux, Localizing Knowledge, Seg4Diff, Relation Circuits, PCI, Attention Sinks (ConceptAttention은 기준선으로만) |
| 표현 분석(PCA 등) | 6 | Exploring MM-DiT, ConceptAttention, Seg4Diff, Relation Circuits, CASteer, FluxSpace |
| 프롬프트 교체·반사실 | 4 | TextGen, Localizing Knowledge, Relation Circuits, PCI |
| activation patching | 4 | TextGen, Localizing Knowledge, Relation Circuits, CASteer |
| steering | 3 | FluxSpace, Relation Circuits, CASteer |
| 어텐션 조작 | 3 | Exploring MM-DiT(q·k 교체), Seg4Diff(blur), Attention Sinks(logit) |
| 편집용 K·V 주입 | 3 | Stable Flow, FreeFlux, Exploring MM-DiT |
| knockout | 2 | Attention Sinks, Relation Circuits |
| 구성요소 ablation | 2 | Attention Sinks(value zero·mean·lerp), Relation Circuits(head) |
| norm 기반 기여 | 2 | Localizing Knowledge, Seg4Diff |
| 층 건너뛰기 | 1 | Stable Flow |
| RoPE 섭동 | 1 | FreeFlux |
| gradient 어트리뷰션 | 0 | |
| LRP | 0 | ConceptAttention이 CLIP ViT 비교 기준선으로만 씀 |
| SAE | 0 | |
| 선형 프로브 | 0 | |

### 3-2. 후속 연구에서 실제로 쓰이는가

- 후속 논문도 기준 학회 게재를 공식 페이지로 확인한 것만 셌다.
- "실제 사용" = 그 기법을 방법이나 실험에 가져다 쓴 것. 관련연구에서 언급만 한 것은 뺐다.

| 기법 | 실제 사용 | 쓴 논문 | 비고 |
|---|---|---|---|
| Stable Flow | 14 (+약한 사용 2) | SynPS(CVPR26), Steering Fields(NeurIPS26), TexTailor(ECCV26), Wavelet-Guided(ECCV26), PCI(ICLR26), OmniText(ICLR26), VMDiff(ICLR26), Generative Blocks World(ICLR26), EditInfinity(NeurIPS25), FSI-Edit(NeurIPS25), QK-Edit(ICCV25), Multi-Turn Consistent Editing(ICCV25), FreeFlux(ICCV25), FreeCus(ICCV25). 용도는 층 선택과 베이스라인 | 기준 학회 인용 56편 확인. 원저자 그룹 후속 없음 |
| ConceptAttention | 7 (+약한 사용 3) | IMAP(CVPR26), Attention Markov Chains(NeurIPS25), UVR(ICML26), MMDiff(ECCV26), LoTTS(NeurIPS26, 본 트랙 추정), Object-WIPER(CVPR26), S3OD(ICLR26) | 원저자 그룹 후속 없음. 경계 3편(RefAM TMLR 등) |
| CASteer | 3 | MidSteer(ICML26, 원저자 그룹), FairSteer/EquiSteer(ECCV26), Example-Based Spatial Guidance(NeurIPS26, 본 트랙 추정) | 경계 1편(TADA, ICLR 워크숍, 오디오) |
| FreeFlux | 2 | Layer-wise Instance Binding(CVPR26), SynPS(CVPR26) | 경계 1편(GRAG, CVPR Findings) |
| Localizing Knowledge | 1 | Z-Erase(ICML26) | |
| TextGen 국소화 | 0 | | 경계 1편(Beautiful Images, AAAI-26, 원저자 그룹) |

- 인용의 대부분은 관련연구 언급이다. 실제로 가져다 쓰는 경우는 소수다.

### 3-3. 라이브러리

| 도구 | 확산·DiT 지원 |
|---|---|
| nnsight | diffusers를 감싸는 방식으로 UNet과 DiT(FLUX, SD3)를 공식 지원(공식 문서·테스트). 원격 실행(NDIF)은 확산 모델 미배포. trace는 기본 1스텝 |
| TransformerLens | 언어모델 전용 |
| diffusers | 해석 기능 없음. 연구들은 Attention Processor를 갈아 끼우는 방식으로 씀 |
| diffusers-interpret | SD v1만 |
| DreamReader 툴킷 | UNet 계열만 |
| SDLens | SDXL Turbo. 코드상 FLUX SAE 데모 있음 |

### 3-4. 결론

- 주류: 어텐션 지도 읽기, 블록·층 단위 개입 후 최종 이미지 측정.
- 안 쓰임: gradient 어트리뷰션, LRP(DiT 해석 기준).
- 재사용되는 DiT 도구: Stable Flow(층 선택), ConceptAttention(지도).
- 확산 DiT용 범용 해석 라이브러리는 사실상 nnsight 하나. 나머지는 UNet 전용이거나 해석 기능이 없다.

---

## 4. 질문 2: V ablation을 무엇으로 바꿔야 하는가

### 4-1. 현행 V ablation이 하는 일

| 항목 | 내용 |
|---|---|
| 개입 위치 | Anima cross-attn 28개 전부 |
| 개입 내용 | 선택 토큰의 value만 0. key와 softmax는 그대로 |
| 적용 대상 | 긍정 조건에만. uncond는 원본 그대로 CFG로 합침 |
| 샘플링 | 원본 경로로 진행 |
| 측정 | 매 스텝 같은 x_t·σ에서 (예측 차이 / σ)의 채널 L2 지도 |
| 안 재는 것 | 최종 이미지 차이 |
| 수학적 정체 | Prompt-to-Prompt의 어텐션 재가중을 계수 0으로 둔 것과 같음(예외: 기반기술) |
| 구조적 약점 | Qwen이 단방향이고 어댑터가 Qwen hidden 전체를 읽어서, 태그 정보가 다른 슬롯에도 퍼진다. 한 슬롯 value만 끄면 이 누수를 놓친다 |

### 4-2. 기본을 반사실 프롬프트 패칭으로

| 근거 | 무엇을 바꾸나 | 무엇으로 | 범위 | 결과 |
|---|---|---|---|---|
| Localizing Knowledge (NeurIPS 2025) | 상위 K 블록의 텍스트 입력 전체 | 중립 프롬프트 실행 값("the Batman" → "a character") | 선택 블록 | PixArt K=6에서 제안법 CLIP 0.2061 대 전수탐색 0.2023 |
| TextGen 국소화 (ICLR 2025) | 텍스트 K·V 전체, 층별 | 목표 프롬프트 실행의 캐시 | SD3는 층 10 하나 | SD3에서 제안법 MSE 73.98·OCR 0.68 대 P2P 90.79·0.31 (표 정렬은 판독상 추정) |

- 문헌에 있는 형태는 토큰 단위가 아니라 텍스트 시퀀스 전체 교체다.
- 슬롯(토큰) 단위 패칭은 우리 쪽 확장이다. 도구와 문서에 그렇게 명시한다.
- Krea 2·MOSS-Image는 패딩이 없다. 토큰 수를 맞춘 반사실 프롬프트가 필수다.

### 4-3. 대체값(zero·mean·기타) 비교

| 근거 | 내용 | 한계 |
|---|---|---|
| Attention Sinks (ICML 2026) | SD3, sink 토큰 대상, 최종 이미지 지표, N=32.<br>ΔCLIP-T / ΔImageReward: zero −0.0012 / −0.049, mean −0.0010 / −0.022, lerp0.0 −0.0010 / −0.022 | 모든 CI가 0을 포함(유의하지 않음). zero가 IR을 mean의 약 2배 떨어뜨리며 저자의 "평탄" 기준(0.05) 경계에 걸림. mean 정의가 부록 두 곳에서 다름. sink 토큰 대상이라 내용 토큰 지도에 대해서는 간접 근거 |
| ConceptAttention (ICML 2025) Table 3 | 개념 지도를 value 공간에서 뽑으면 위치 정확도 최저.<br>Acc / mIoU / mAP: value 공간 45.93 / 29.81 / 65.79, output 공간 83.07 / 71.04 / 90.45 | 개입이 아니라 지도 추출 공간 비교(간접) |
| ACDC (NeurIPS 2023), MIB (ICML 2025) | zero는 학습 때 본 적 없는 분포 밖 값이라 비권장 | 예외(기반기술). LLM 쪽 방법론 |

- 결론: 기준 안 직접 근거는 Attention Sinks 하나이고 유의하지도 않다. 어느 대체값이 옳다고 단정할 근거가 없다.
- 그래서 대체값을 하나로 정하지 않고, 지도가 대체값에 얼마나 민감한지를 같이 보여 준다.

### 4-4. 무작위 대조

| 근거 | 내용 |
|---|---|
| Attention Sinks (ICML 2026) | sink 마스킹이 같은 개수 무작위 마스킹보다 LPIPS 약 6배(0.347 대 0.053, k=1). 그런데 CLIP-T는 변하지 않음 |

- 시사: "효과가 크다"는 무작위 대조 없이는 말할 수 없다. 같은 개수 무작위 토큰 ablation을 대조군으로 붙인다.
- 시사: 픽셀 변화(LPIPS)와 의미 변화(CLIP-T)는 따로 움직인다. 둘 다 잰다.

### 4-5. knockout (정보 흐름 검증용)

| 근거 | 방법 | 비고 |
|---|---|---|
| Attention Sinks (ICML 2026) | sink key의 logit에 log α를 더함. α=0이면 −10⁴ | |
| DiT Relation Circuits (CVPR 2026, 공식 페이지 미확인) | head × 토큰 기여 제거 | 관계 정확도 67% → 33% 등. 대체값과 적용 스텝은 논문·저장소 모두 미공개 |

- joint-attn 모델(Krea 2, MOSS-Image)에서 "이미지가 이 텍스트 토큰을 통해 정보를 받는가"를 확인하는 용도.

### 4-6. 어느 스텝에 개입하나

| 근거 | 내용 |
|---|---|
| TextGen 국소화 (ICLR 2025) | 패칭을 처음 2스텝 빼고 시작. SD3에서 28 → 26스텝으로 바꾸면 결과가 달라짐 |
| SAP (CVPR 2026, FLUX·SD3) | 초반 = 배치·색, 중반 = 형태, 후반 = 정체·재질. 구간 경계를 ±3 흔들어도 거의 같고 ±5면 하락 |
| TACA (ICCV 2025) | 초반 약 10% 스텝에만 조정 |
| PCI (ICLR 2026) | 전역 요인은 초반, 성별·나이는 중반, 액세서리는 중후반. rectified-flow(SD3.5, FLUX)는 더 일찍 고정 |
| DiT-ST (NeurIPS 2025) | 주입 시점을 바꾸면 CLIPScore 하락 |
| Exploring MM-DiT (ICCV 2025) | 4스텝 증류 모델은 한 스텝 주입만으로도 최종이 원본과 지나치게 비슷해짐. 강도를 블록 수로 조절 |

- 시사: 스텝 구간(초반·중반·후반)별로 나눠 보여 준다.
- 시사: 초반 1~2스텝 포함 여부를 옵션으로 둔다.
- 시사: 증류(소수 스텝) 모델은 따로 다룬다.

### 4-7. 측정 방식

- 한 스텝 지도만으로는 부족하다. 개입한 채 끝까지 생성한 최종 이미지 지표를 추가한다(근거는 5절).

---

## 5. 질문 3: 각 방법의 유효성을 어떻게 검증하는가

### 5-1. 문헌에 없는 것

| 없는 것 | 의미 |
|---|---|
| 지도와 "토큰을 빼고 끝까지 생성한 실제 차이"의 공간 상관을 잰 기준 충족 논문 | 핵심 검증은 직접 설계해야 한다 |
| 같은 개입을 한 스텝 예측 차이와 최종 차이로 나란히 잰 논문 | 현행 측정 방식의 타당성을 문헌으로 보장할 수 없다 |
| 국소화 정답 벤치마크 | Localizing Knowledge 저자도 한계로 적음 |

- 조사에서 읽은 기준 충족 논문은 개입 효과를 모두 최종 이미지로 쟀다. 한 스텝 측정은 ConceptAttention의 지도 추출뿐이고, 이것은 개입이 아니다.

### 5-2. 한 스텝 측정은 최종을 얼마나 대변하나

| 근거 | 내용 |
|---|---|
| Probe-Select (CVPR 2026, SD3.5·FLUX) | t=0.2의 x̂0를 디코딩해 바로 채점하면 최종과 Spearman 0.52(BLIP-ITM). 같은 시점 활성 프로브는 0.98–0.99 |
| Diffusion Probe (CVPR 2026, FLUX·Qwen-Image) | 초기 cross-attn 지도로 최종 ImageReward 예측 SRCC: 1스텝 0.52, 5스텝 0.76, 10스텝 0.79 |

- 시사: 한 스텝 측정은 최종을 약하게만 대변한다.
- 시사: 현행 지도는 "그 스텝의 국소 민감도"라고 표기한다. "태그가 최종 이미지에 준 영향"이라고 쓰지 않는다.

### 5-3. 검증 설계 재료

| 검증 | 방법 | 근거 |
|---|---|---|
| 지도-실제 차이 일치 | 같은 시드로 태그를 넣고 뺀 최종 이미지의 차이 지도와 공간 상관 | 직접 설계(선례 없음) |
| 무작위 대조 | 지도 상위 영역·토큰과 같은 개수 무작위에 개입해 효과 비교 | Attention Sinks (ICML 2026) |
| 실험 신뢰성 | no-op(개입 없이 돌려 차이 0 확인), 시드 짝 고정, bootstrap CI | Attention Sinks (ICML 2026) |
| 개념 출현 판정 | VQA 판정기 여러 개. CLIPScore·LPIPS는 개념이 없어진 것을 잘 못 잡아 PCI가 채택 안 함 | PCI (ICLR 2026). VLM 4종의 곡선이 일치 |
| 위치 일치 | 생성 이미지에 Grounded-SAM2를 돌린 마스크와 비교 | Exploring MM-DiT (ICCV 2025). 상위 블록: Flux-dev [18,17,12,14,11], SD3.5-L [18,21,20,24,16] |
| 층 단위 대조 | 핵심층 대 같은 수 비핵심층 비교 | Stable Flow (CVPR 2025): CLIP_txt / img / dir 0.23 / 0.92 / 0.14 대 0.25 / 0.72 / 0.09. 지표를 DINOv1·CLIP·LPIPS로 바꿔도 공통 핵심층 8개 유지 |
| 층 단위 대조 | 블록 수에 따른 곡선을 전수탐색과 비교 | Localizing Knowledge (NeurIPS 2025) |
| 층 단위 대조 | 층 분류를 편집 결과로 검증 | FreeFlux (ICCV 2025) |
| 지도와 섭동의 일치 | 분할이 가장 잘 되는 층과 어텐션을 흐렸을 때 효과가 큰 층이 같은지 | Seg4Diff (NeurIPS 2025): SD3·SD3.5 층 9, Flux 12·17에서 일치. 상관계수는 없음 |
| 어텐션 질량 ≠ 중요도 | 질량이 큰 토큰을 막아도 결과가 안 변할 수 있음을 확인 | Attention Sinks (ICML 2026): 질량 순위로 막아도 CLIP-T 불변 |
| 무작위화 검사 | 가중치를 섞으면 지도가 변하는지 | Sanity Checks (NeurIPS 2018, 예외: 기반기술) |

- 주의: 블록 건너뛰기는 "그 블록이 출력 구조에 중요한가"만 보여 준다. 특정 태그 정보를 담는다는 증거가 아니다.

### 5-4. 도구별 검증 묶음

| 도구 | 우선 검증 |
|---|---|
| 어텐션 히트맵 | 위치 일치(Grounded-SAM2) / 지도와 섭동의 일치 / 무작위화 검사 / 질량 ≠ 중요도 점검(특수·템플릿 토큰) |
| 태그 영향도 | 지도-실제 차이 일치 / 무작위 대조 / 시드 짝·bootstrap CI / 스텝 구간별 비교 |
| ablation·패칭 | no-op / 무작위 토큰·무작위 블록 대조 / 대체값별 민감도 / 개념 출현 판정(VQA 여러 개) / 최종 이미지 지표 |

---

## 6. 설계에 주는 시사

### 6-1. 공통

| 항목 | 권고 |
|---|---|
| 기본 개입 | 반사실 프롬프트 패칭. 태그를 중립어로 바꾼 실행의 텍스트 입력(또는 텍스트 K·V)으로 교체, 블록 단위 |
| 문헌 형태와 확장 구분 | 시퀀스 전체 교체 = 문헌 형태. 슬롯 단위 = 우리 확장이라고 표시 |
| 대체값 | zero는 옵션. mean·반사실 값도 옵션. 대체값에 따른 지도 민감도를 함께 출력 |
| 이름 | value 0을 남기면 "어텐션 재가중 계수 0"으로 부름 |
| 대조군 | 같은 개수 무작위 토큰 ablation |
| 스텝 | 구간별(초반·중반·후반)로 나눠 보여 줌. 초반 1~2스텝 포함 여부는 옵션. 증류 모델은 따로 |
| 측정 | 한 스텝 지도 + 끝까지 생성한 최종 이미지 지표 |
| 표기 | 한 스텝 지도는 "그 스텝의 국소 민감도" |
| 지표 | 픽셀 차이(LPIPS)와 의미 판정(VQA 판정기 여러 개)을 같이. CLIPScore만으로 개념 부재를 판정하지 않음 |
| 검증 | 5-4의 묶음을 도구에 내장(no-op, 시드 짝, bootstrap CI, 무작위 대조) |
| 구조 | ComfyUI와 독립된 코어 + ComfyUI 노드 래퍼. 모델별 어댑터, 기법은 어댑터에만 의존 |
| 모델 구분 질문 | 텍스트가 이미지와 같은 어텐션에 있나 / 텍스트 토큰이 블록마다 바뀌나 / 텍스트·이미지가 같은 가중치를 쓰나 / 블록 종류가 하나인가 |
| 어댑터 역할 | 어텐션 호출마다 "누가 누구를 보는가"와 개입 지점을 알려 줌 |
| MI의 위치 | XAI의 한계 확인과 검증 수단 |

### 6-2. Anima (cross-attn)

| 항목 | 권고 |
|---|---|
| 기본 개입 | 태그를 중립어로 바꾼 프롬프트로 반사실 실행. 그 실행의 텍스트 조건으로 선택 블록의 cross-attn 입력을 교체 |
| 왜 value 0으로는 부족한가 | Qwen 단방향 + 어댑터가 Qwen hidden 전체를 읽음 → 태그 정보가 다른 슬롯에 퍼짐. 반사실 프롬프트는 이 누수까지 함께 바꿈 |
| 대체어 고르기 | 슬롯 단위로 볼 때는 T5 슬롯 수가 같아지는 중립어를 고름. 시퀀스 길이는 512까지 0 패딩이라 맞춰짐 |
| 블록 국소화 | 같은 교체를 블록 하나 또는 블록 창 단위로 |
| 현행 value 0 | 옵션으로 남기고 "어텐션 재가중 계수 0"으로 표시. mean·반사실 값과 비교 출력 |
| 히트맵 백엔드 | cross-attn + norm 기반 기여도 사용. 텍스트 출력 공간이 없어 ConceptAttention은 그대로 적용 불가 |

### 6-3. Krea 2·MOSS-Image (joint-attn)

| 항목 | 권고 |
|---|---|
| 히트맵 백엔드 | joint-attn 출력 공간이 있어 ConceptAttention 적용 가능 |
| 반사실 프롬프트 | 패딩이 없으므로 토큰 수를 맞춘 반사실 프롬프트 필수 |
| 기본 개입 | 선택 블록의 텍스트 어텐션 입력(또는 텍스트 K·V) 시퀀스 전체를 같은 스텝의 반사실 실행 값으로 교체 |
| 정보 흐름 검증 | knockout(Attention Sinks의 logit 방식). 이미지 query → 해당 텍스트 key를 막음 |
| 텍스트 토큰이 블록마다 바뀜 | 두 모델 모두 텍스트 토큰이 매 블록 갱신되어 뒤쪽 블록에서는 이미지 정보가 섞임. 어느 블록 값을 패치했는지 표시 |
| 템플릿 토큰 | 히트맵·영향도에서 프롬프트 토큰과 구분 |
| Krea 2 템플릿 | 프롬프트 뒤 꼬리 5토큰이 단방향 인코더 특성상 프롬프트 전체를 본 상태를 담음 → 따로 개입·표시 |
| Krea 2 출력 게이트 | SDPA 뒤 토큰·채널별 출력 게이트 `sigmoid(Wg·h)`와 GQA(48Q/12KV)가 있음. raw `attn.probs`만 보면 왜곡될 수 있고, Gated Attention 특성상 attention sink 양상이 MOSS-Image·SD3와 다를 수 있음 |
| MOSS-Image 템플릿 | `te.out`(Qwen3.5 출력) 기준으로는 앞쪽 템플릿에 프롬프트 내용이 없으나, `refiner` 양방향 2블록을 거치면 앞쪽 템플릿도 프롬프트 정보를 흡수함 → `te.out`과 `refiner.out`을 구분해 다룸 |
| 어텐션 질량 | 질량이 큰 토큰(템플릿·특수 토큰)이 중요하다고 단정하지 않음. 무작위 대조로 확인 |

---

## 7. 남은 것

| 항목 | 상태 |
|---|---|
| Stable Flow 인용 논문 약 40편 | 학회 사이트 요청 제한(429)으로 게재 미확인. 확인되면 실제 사용 편수는 늘 수만 있음 |
| 그림 안 수치(Attention Sinks 그림 4의 lerp0.5 등) | PDF 렌더링 도구가 없어 미판독 |
| DiT Relation Circuits | CVPR 2026 공식 페이지 미확인 |
| *(확인 완료·제외)* `arXiv:2510.13232` · `FlowChef` | `2510.13232`는 VLM 부정어 객체 탐지 논문(비DiT), `FlowChef`는 Rectified Flow 제어·편집 기법으로 XAI/MI 조사 대상에서 제외 |
