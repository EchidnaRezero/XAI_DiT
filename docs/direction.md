# 프로젝트 방향 (Direction)

## 1. 목표
단일 GPU 환경(**Colab CLI · L4 기본**, 필요 시 상위 GPU 포함)에서 **이미지 생성 DiT 모델의 내부 동작을 관찰·개입·검증하는 범용 XAI / MI 도구**를 만든다.

- **출발점**: `Anima_HEATMAP` (Anima 단일 모델 cross-attention 히트맵 + ComfyUI 노드)
- **확장 목표**: **Anima → MOSS-Image(본 타깃) → Krea 2(본 모델)** 3종을 공용 어댑터 구조로 지원하고, 추후 새 DiT·영상 모델이 나와도 어댑터만 붙여 확장 가능한 기반을 만든다.
- **산출물 형태**:
  1. 프레임워크에 종속되지 않는 **순수 PyTorch 코어 (`src/`)**
  2. 실사용을 위해 코어를 감싸는 **ComfyUI 커스텀 노드 래퍼**
  3. Colab CLI(L4+)에서 재현 가능한 **배치 실험·검증 파이프라인**

---

## 2. 대상 모델 및 진행 순서

| 순서 | 모델 | 핵심 구조 | 참고 문서 · 코드 |
|---|---|---|---|
| **1차 (기준선)** | **Anima** | Cross-Attn DiT (28블록, ~2B, Qwen3-0.6B + LLMAdapter, 512 패딩) | `docs/reference/models/anima.md`<br>ComfyUI: `comfy/ldm/anima/model.py`, `cosmos/predict2.py`<br>DiffSynth: `diffsynth/models/anima_dit.py` |
| **2차 (본 타깃)** | **MOSS-Image** | Single-Stream Joint-Attn (42블록, ~5B, Qwen3.5-9B + 양방향 refiner 2블록, 4축 RoPE, 패딩 없음) | `docs/reference/models/moss_image.md`<br>ComfyUI: `comfy/ldm/moss_image/model.py`<br>DiffSynth: `diffsynth/models/moss_image_dit.py` |
| **3차 (후속)** | **Krea 2 (본 모델)** | Single-Stream Joint-Attn (28블록, ~12.8B, Qwen3-VL-4B 12층 + txtfusion, `sigmoid` 출력 게이트 · GQA 48Q/12KV, 꼬리 5토큰) | `docs/reference/models/krea2.md`<br>ComfyUI: `comfy/ldm/krea2/model.py`<br>DiffSynth: `diffsynth/models/krea2_dit.py` |

- **구현 기준**: ComfyUI와 DiffSynth-Studio 간 구현 차이가 있을 경우 **ComfyUI 구현을 우선 기준**으로 삼는다.

---

## 3. 4방향 코어 범위

| 방향 | 핵심 질문 | 적용 기법 | 근거 (`docs/reference/paper/`) |
|---|---|---|---|
| **1. 히트맵** | 어디를 보나? | 어텐션 확률 + Norm/Gate 기여도 (공통)<br>ConceptAttention (MOSS-Image · Krea 2) | `01_heatmap/` |
| **2. 층 선별** | 어느 블록이 진짜 일하나? | 블록별 중요도 프로파일링 · 핵심 층 선별 | `02_layer_selection/`, `LocalizingKnowledgeDiT` |
| **3. 태그 영향도** | 이 태그가 실제로 얼마나 바꿨나? | 반사실 프롬프트 패칭 + 스텝 구간 분해<br>*(기존 value 0 ablation은 비교 옵션으로 유지)* | `03_tag_influence/`, `LocalizingKnowledgeDiT` |
| **4. 검증 (MI)** | 방금 본 결과가 착시가 아닌가? | 무작위 대조 · no-op · logit knockout · 템플릿/싱크 토큰 분리 | `04_verification_mi/` |

---

## 4. 범위 밖 (Out of Scope)
- 대규모 사전학습 및 전 레이어 SAE 학습
- Krea 2 anime 변형 모델
- Gradient / LRP 기반 어트리뷰션 (DiT 해석에서 미사용 확인)
- 범용 모델 학습·파인튜닝 프레임워크화 (추론 시 관찰·개입·검증에 집중)
