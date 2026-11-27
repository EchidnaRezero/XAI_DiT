# XAI_DiT

[English](README.md) · **한국어**

이미지 생성 Diffusion Transformer(DiT)를 위한 XAI / 기계론적 해석 가능성(MI) 프레임워크 및 실증 연구 프로젝트입니다.

> [**인터랙티브 비주얼 쇼케이스 →**](https://echidnarezero.github.io/XAI_DiT/showcase/) · [**Anima 4단계 진단 프리뷰 (WIP) →**](https://echidnarezero.github.io/XAI_DiT/showcase/preview_anima_step2.html)

---

## 대상 모델

| 모델 | 백본 (DiT) | 텍스트 인코더 (TE) | VAE | 실전 XAI 프리셋 (L4) | 구조 분석 |
|---|---|---|---|---|---|
| **[Anima](https://huggingface.co/circlestone-labs/Anima)** | Cross-Attn DiT (28블록, ~2B) | Qwen3-0.6B + LLMAdapter (6블록) | Wan2.1 VAE (16ch, 8×) | `BF16` + `EasyCache(0.20)` + `Sel 6b` (`14.2s`, `r=0.97`) | [`anima.md`](docs/reference/models/anima.md) |
| **[MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)** ([SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)) | Single-Stream Joint-Attn DiT (42블록, ~5B) | Qwen3.5-9B + Text Refiner (2블록) | Flux2 VAE (128ch, 16×) | `int8_convrot` + `DMD 8s` + `Sel 6b` (`13.2s`, `r=0.93`) | [`moss_image.md`](docs/reference/models/moss_image.md) |
| **[Krea 2](https://huggingface.co/krea/Krea-2-Raw)** ([Turbo](https://huggingface.co/krea/Krea-2-Turbo) · [Krea2-Anime](https://civitai.com/models/2908999?modelVersionId=3435333)) | Gated Single-Stream Joint-Attn DiT (28블록, ~12.8B) | Qwen3-VL-4B (12층) + TextFusion | Qwen-Image VAE (16ch, 8×) | `Turbo FP8` + `Sel 5b` (`24.6s`) · `Anime INT4` (`38.9s`, `r=0.93`) | [`krea2.md`](docs/reference/models/krea2.md) |

- **실전 가속 프리셋 및 주의사항**: [`docs/acceleration.md`](docs/acceleration.md) · **가속 전체 보고서**: [`experiments/accel/REPORT.md`](experiments/accel/REPORT.md)

---

## 다음 우선순위 (진행 중)

1. **v2 실전 의미 단위 제어 (Semantic-Unit Control)**: 3개 모델 전반 비파괴적 제어(`Mid` 블록 프롬프트 라우팅, `S`-only TACA / 태그별 강도 제어, 초반 $\sigma$ SAP, StableFlow Stage 2).
2. **오프라인 의미 평가기 및 ComfyUI 노드 패키징**: GPU 의미 평가기(OpenCLIP/SigLIP, DINOv2, LPIPS) 도입 및 ComfyUI 커스텀 노드 패키징.
3. **Tier 2 하드웨어 검증 (Blackwell `sm_120`, ≥80 SMs)**: `SageAttention 3` (FP4), `NVFP4`, `MXFP8`, `FlexAttention`, `torch.compile(max-autotune)` + `cudagraphs`.

---

## 프로젝트 구조

| 경로 | 설명 |
|---|---|
| [`showcase/`](showcase/) | 인터랙티브 비주얼 가이드 및 프리뷰 |
| [`src/xai_dit/`](src/xai_dit/) | 핵심 XAI/MI 엔진, 3개 모델 어댑터, CLI 러너, ComfyUI 노드 |
| [`tests/`](tests/) | CPU 단위 테스트 (훅, 토큰 매핑, 불변식 검증 210개) |
| [`docs/`](docs/) | 프로젝트 설계 문서, 검증된 논문 스펙, 참고 문헌 |
| [`experiments/`](experiments/) | XAI/MI 실험 결과, 가속 전용 트랙 ([`accel/`](experiments/accel/REPORT.md)), 실전 사례 ([`case_studies/`](experiments/case_studies/README.md)) |
| [`issues/`](issues/) | 코드 레벨 붕괴 원인 규명 및 수정 기록 |

---

## 크레딧

- 2025–2026 DiT XAI/MI 핵심 논문 16편의 검증된 방법론을 기반으로 구축되었습니다 ([`docs/reference_guide.md`](docs/reference_guide.md), [`docs/specs/`](docs/specs/) 참고).
- 외부 참조 저장소, 커밋 및 라이선스는 [`reference_repo/reference_repo_guide.md`](reference_repo/reference_repo_guide.md)에 기록되어 있습니다.
