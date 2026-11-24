# 실행 환경 · 모델별 최적화 옵션 · 추론 하이퍼파라미터 · 평가기

[`docs/experiment_design.md`](../../docs/experiment_design.md)의 실행 세부 규격. 로컬 머신은 CPU 전용(PyTorch 미설치)이며, **기본 실행 환경은 Colab CLI · L4 (24 GB)**, **M1 실행 백엔드는 Headless ComfyUI (`comfy.*`)**다.

---

## 1. 모델별 Colab L4 (24 GB) 기본 조합 및 고정 추론 HP

> **명명 & 정밀도 기준**:
> - **Krea 2 Raw** = `AGENTS.md`의 **Krea 2 본 모델(base, anime 변형 제외)**을 가리키는 공식 배포명.
> - **MOSS-Image 체크포인트 고정**: 정밀도 대조 시 학습 체크포인트 혼동을 막기 위해 **SFT 계열(`sft_bf16` vs `sft_int8_convrot`)로 고정**한다(`base` 변형 혼용 금지).
> - **`fp32` stream**: DiT 잔차 스트림 및 누적 연산을 `float32`로 유지하는 모드(양자화 가중치와 분리 표기).

| 항목 | 1차: Anima (2B) | 2차: MOSS-Image (5B + 9B, 본 타깃) | 3차: Krea 2 Raw (12.8B + 4B, 본 모델) |
|---|---|---|---|
| **기준 DiT (L4 기본)** | `anima-base-v1.0.safetensors` (**BF16** + `fp32` stream, ~4 GB) | `moss_image_sft_bf16.safetensors` (**BF16**, ~10 GB) | `krea2_raw_fp8_scaled.safetensors` (**FP8 scaled**, 13.1 GB; 1D norm·bias·mod는 BF16/FP32 유지) |
| **L4 고속/대량 대안** | 양자화 불필요 | `moss_image_sft_int8_convrot.safetensors` (~5.4 GB) + `fp32` stream *(DiT 단독 추론·TE 오프로드 기준 피크 ~7.8 GiB; TE 동시 상주 시 합계 >13.8 GiB)* | `krea2_raw_int8_convrot.safetensors` (13.5 GB; Hadamard 회전으로 선형 기저 변화 주의 → E2-2/E4-3 비권장) |
| **텍스트 인코더 (사전 인코딩 기본)** | `qwen_3_06b_base.safetensors` (**BF16**, ~1.2 GB) | **오프라인 BF16 사전 인코딩 캐시 기본** (`qwen3.5_9b_bf16`, ~18 GB CPU/순차 오프로드; L4 대화형 보조 시 `int8_convrot` ~9.4 GB) | **오프라인 BF16 사전 인코딩 캐시 기본** (`qwen3vl_4b_bf16`, ~8 GB; L4 대화형 보조 시 `fp8_scaled` ~4 GB) |
| **VAE** | `qwen_image_vae.safetensors` *(Wan2.1 호환 디코더; repo·rev·sha256 로깅)* | `flux2-vae.safetensors` | `qwen_image_vae.safetensors` |
| **기본 해상도** | `1024×1024` (64×64 = 4,096 토큰) | `1024×1024` (64×64 = 4,096 토큰) | `1024×1024` (64×64 = 4,096 토큰) |
| **XAI 기준 샘플러 / 스케줄러** | **`euler` / `simple`** *(보조 강건성 복제: `er_sde`/`simple`, PCI·DiT-ST 기본 제외)* | **`euler` / `sgm_uniform`** | **`euler` / `simple`** |
| **스텝 수 ($S$)** | **30** | **28** *(고정밀 비교: 50)* | **32** *(원문 비교: 52)* |
| **CFG / Shift** | CFG **4.0** / `shift=3.0` | CFG **3.0** / `shift=5.0` *(SLG OFF 기본)* | CFG **4.0** / **고정 스케줄 명시**: ComfyUI `ModelType.FLUX` 기본값(`shift=1.15`, `supported_models.py:2033`) vs Krea 2 Raw 공식 동적 shift 적용 여부를 `config.json`의 **`sigma_table` 전체 배열로 고정** |
| **표준 네거티브 프롬프트** *(Single Source of Truth)* | `worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts` | `worst quality, bad anatomy, watermark, jpeg artifact` *(순수 2D 시 `, 3d, realistic` 추가를 매니페스트에 명시)* | `""` *(빈 문자열 고정; 공용 네거티브 비교 시 별도 run으로 분리)* |
| **Same-State 대표 6스텝** *(실제 `σ_i` 병기; 50/52스텝은 동일 `σ` 분위수 최근접 선택)* | `[1, 5, 10, 15, 22, 29]` | `[1, 5, 10, 15, 21, 27]` | `[1, 5, 10, 16, 23, 31]` |
| **고정 시드 세트** | Sanity: `[42]` / 기본(8): `[42, 101, 202, 303, 404, 505, 606, 707]` / PCI 확장(16): `+ [808, 909, 1010, 1111, 1212, 1313, 1414, 1515]` | 상동 | 상동 |
| **금지 / 주의 사항** | **FP16 강제 금지** (NaN/검은 이미지); `--highvram` 권장 | `w4a8`/GGUF는 L4에서 느림; **SLG(Layer 22)**는 통제 변수로 분리 | **SageAttention 전역 패치 금지**(NaN/검은 이미지); `nvfp4`는 Blackwell 전용이라 L4 비추천; FP8의 E2-2/E4-3 결과는 잠정 표기 |

---

## 2. 공통 L4 런타임 플래그 · 오프라인 TE 캐싱 · 비폐기 보류 옵션

### L4 공통 필수 설정
- **ComfyUI 플래그**: `--disable-dynamic-vram` (Colab High-RAM 50 GB+ 기준: MOSS-Image 가속 + Krea 2 `comfy-aimdo` 검은 이미지/NaN 방지).
- **고정 CFG 배칭 정책**: ComfyUI `samplers.py`의 여유 VRAM 기반 동적 배칭 전환을 막기 위해, 러너가 모델별 배칭 정책(Anima: `batched` 또는 `split` 고정, MOSS/Krea 2: `split` 고정)을 강제하고 `config.json`에 기록한다.
- **오프라인 BF16 텍스트 인코딩 캐시**: 실험 전 프롬프트 매니페스트 전체를 BF16 TE로 1회 인코딩해 `cache/cond/<model>_<manifest_sha256>.pt`에 저장하고 TE를 언로드한 뒤 DiT 단독 추론을 실행한다. *(양자화 TE 교란 제거 + L4 VRAM 여유 확보)*
- **메모리 파편화 방지**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **어텐션 백엔드**: 기본 **PyTorch SDPA** + 훅 대상 블록만 **query-chunked 수동 softmax** (`Anima_HEATMAP/compute.py` 계승, OOM 시 key-chunked `logsumexp` 자동 폴백).
  - `FlexAttention`은 L4에서 Triton shared-memory OOM(`131072 > 101376`)으로 실패하므로 L4 기본 경로에서 제외하고, TACA는 **Listing 2 (`K_txt` 스케일링 + 텍스트 query 출력 원복) / chunked SDPA** 사용.

### E0 자원 예산 & 스윕 분할(Shard) 규칙
- E0에서 **TE 인코딩 / DiT 기본 추론 / 최중량 활성 훅 / VAE 디코딩 / 오프라인 평가기**별 `gpu_allocated_gb`, `gpu_reserved_gb`, `cpu_rss_gb`, `sec_per_forward`를 실측한다.
- 대규모 스윕(예: E2-1 `k=16`은 Anima 27,840 / MOSS 38,528 / Krea 2 29,696 CFG 분기 평가)은 착수 전 **예상 월클록 시간과 샤드(프롬프트/블록 단위) 재개 매니페스트**를 선언한다. L4 세션 한도 초과 시 Krea 2 `k=16`은 Phase 4로 승격한다.

### XAI 기준선에서 OFF (폐기하지 않고 후속 비교용 보존)
- `TeaCache`, `Spectrum`, 4·8스텝 증류 LoRA(`DMD`/`HaDeS`/`Anima-Turbo`), `Krea-2-Turbo`, `SLG (Layer 22)`, `torch.compile`, `SageAttention`, `er_sde`(Anima 보조 복제)

---

## 3. 오프라인 평가기 규격 (Offline Evaluation Pass)

모든 평가기는 **DiT·VAE 언로드 후 저장된 PNG 대상 별도 오프라인 패스**로 실행해 L4 VRAM 경합을 완전히 제거한다. 모델·버전·해시는 `config.json`의 `evaluator_versions`에 기록한다.

| 지표 | 표준 모델 / 라이브러리 | 정밀도 | 적용 범위 및 규칙 |
|---|---|---|---|
| **Vitality / 구조 유사도** | **DINOv2 `vit_giant_patch14_dinov2`** (CLS 코사인) | FP16/BF16 | E0, E2-1, E4-1 등 전 이미지 공통 |
| **지각 거리** | **`lpips` (AlexNet 기본, VGG 보조)** | FP32 | E0, E1-3, E2-2, E3-0 Rollout 등 전 이미지 공통 |
| **텍스트-이미지 정합** | **OpenCLIP `ViT-H-14` (laion2b) 또는 `SigLIP-SO400M`** | FP16/BF16 | E1-3, E2-2, E3-3, E4-5 조기 상관. 비국소화 스타일·아티스트 PCI는 참조 정규화 `(s(x)-s(P_b))/(s(P_c)-s(P_b))` 병행 |
| **객체 마스크 (mIoU / Synopsis)** | **Grounded-SAM2** (`GroundingDINO-SwinT` + `SAM2.1-hiera-large`) | FP16/BF16 | **`localizable: yes` 타깃 한정** (E1-1 `calib`/`eval` mIoU, E4-4 2-객체 Synopsis 및 검출 성공 판정). 마스크 검출 실패 시 `mask_failed=true`로 분리 집계 |
| **개념 존재 / 관계 판정 (VQA)** | **Qwen2.5-VL-7B-Instruct** (또는 `7B-AWQ`/4bit, 단일 프롬프트 템플릿 고정) | BF16/4bit | E3-2 PCI CIS 판정, E4-4 객체·관계 정확도. 질문 템플릿은 프롬프트 매니페스트의 `vqa_question`으로 고정 |

---

## 4. 상위 GPU (A100 40/80GB · H100) 일괄 보충 목록 (Phase 4)
L4(24 GB) 한계를 넘거나 상위 하드웨어가 필요한 항목은 폐기하지 않고 **상위 GPU 세션에서 한 번에 몰아서 검증**한다.
1. **Krea 2 Raw 원본 BF16 (~25.6 GB DiT, A100 80 GB 권장; 40 GB는 오프라인 TE 캐시 필수)**: 동일 캐시 컨디셔닝을 재사용해 **DiT 정밀도(`fp8_scaled` vs `bf16`) 효과만 순수 분리**하여 `M-Act`·`gate_value`·`raw_mvo`/`gated_mvo` 대조(TE 정밀도 대조는 별도 쌍으로 분리).
2. **MOSS-Image 정밀도 대조 (`sft_bf16` vs `sft_int8_convrot`) & TE 정밀도 대조 (`bf16` vs `int8_convrot`)**: 동일 SFT 체크포인트 계열 내에서 DiT/TE 정밀도 효과를 각각 분리 검증.
3. **대규모 스윕 및 커널 대조**: StableFlow `k=16`(Krea 2 L4 초과 시)·`k=64` 풀 스윕, PCI 50–100시드 풀 스윕, 2MP+ 고해상도, H100/A100에서의 `FlexAttention` TACA 및 가속/증류 옵션 비교.
