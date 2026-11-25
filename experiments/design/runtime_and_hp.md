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
| **기준 DiT (L4 기본)** | `anima-base-v1.0.safetensors` (**BF16** + `fp32` stream, ~4 GB) | `moss_image_sft_int8_convrot.safetensors` (**INT8 convrot**, ~5.4 GB + `fp32` stream; DiT 단독 피크 ~7.8 GiB) | `krea2_raw_fp8_scaled.safetensors` (**FP8 scaled**, 12.2 GB; 1D norm·bias·mod는 BF16/FP32 유지) |
| **L4 정밀 대안 (Phase 4 BF16 전)** | 양자화 불필요 | `moss_image_sft_bf16.safetensors` (**BF16**, ~9.5 GB) | `krea2_raw_int8_convrot.safetensors` (13.5 GB; Hadamard 회전 주의 → E2-2/E4-3 비권장) |
| **텍스트 인코더 (L4 기본 · 사전 캐싱)** | `qwen_3_06b_base.safetensors` (**BF16**, ~1.2 GB) | `qwen3.5_9b_int8_convrot.safetensors` (**INT8 convrot**, ~9.4 GB; 1회 인코딩 후 언로드) | `qwen3vl_4b_fp8_scaled.safetensors` (**FP8 scaled**, ~4 GB; 1회 인코딩 후 언로드) |
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
- **오프라인 텍스트 인코딩 캐시**: 실험 전 프롬프트 매니페스트를 모델별 L4 기본 텍스트 인코더(`qwen_3_06b_base`, `qwen3.5_9b_int8_convrot`, `qwen3vl_4b_fp8_scaled`)로 1회 인코딩해 `cache/cond/<model>_<manifest_sha256>/`에 저장하고 TE를 언로드한 뒤 DiT 단독 추론을 실행한다. *(L4 VRAM 여유 확보 + 반복 인코딩 제거; 원본 BF16 TE 대조는 Phase 4)*
- **메모리 파편화 방지**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **어텐션 백엔드**: 기본 **PyTorch SDPA** + 훅 대상 블록만 **query-chunked 수동 softmax** (`Anima_HEATMAP/compute.py` 계승, OOM 시 key-chunked `logsumexp` 자동 폴백).
  - `FlexAttention`은 L4에서 Triton shared-memory OOM(`131072 > 101376`)으로 실패하므로 L4 기본 경로에서 제외하고, TACA는 **Listing 2 (`K_txt` 스케일링 + 텍스트 query 출력 원복) / chunked SDPA** 사용.

### E0 자원 예산 실측 기준
- E0에서 모델별 `gpu_allocated_gb`, `sec_per_forward`를 1회 실측해 L4 가용 범위를 확정한다(3모델 완료: Anima `0.36s/fwd`, MOSS `int8_convrot` `0.54s/fwd`, Krea 2 `fp8` `2.12s/fwd`).
- 본 실험은 **대표 1–2프롬프트 × 1시드(`42`)** 기준 수 분 내 끝나는 실용 규모로 실행하며, 수백 장 단위 대규모 스윕은 기본 범위에서 제외한다.

### XAI 기준선에서 OFF (폐기하지 않고 후속 비교용 보존)
- `TeaCache`, `Spectrum`, `EasyCache`/`LazyCache`, `TorchCompileModel`, `--use-ck-attention`, `--fast`, 4·8스텝 증류 LoRA(`DMD`/`eDMD`/`HaDeS`/`Anima-Turbo`), `Krea-2-Turbo`, `SLG (Layer 22)`, `SageAttention`, `er_sde`(Anima 보조 복제)
- 상세 옵션 비교(공통 vs 모델별) 및 XAI 훅 호환성 표는 [`docs/acceleration.md`](../../docs/acceleration.md), 실환경 가속 검증 계획은 [`plan/accel_xai_eval.md`](../../plan/accel_xai_eval.md) 참조.

---

## 3. 선택적 평가기 (필요 시 오프라인 패스)

시각적 Before/After와 내부 수치(V-norm, 템플릿 비중, `gate_value`, `M-Act`, $\Delta v$)가 주 산출물이며, 정량 지표가 필요한 경우에만 DiT 언로드 후 별도 오프라인 패스로 실행한다.

| 지표 | 표준 모델 / 라이브러리 | 용도 |
|---|---|---|
| **구조 유사도 / 지각 거리** | **DINOv2** / **`lpips`** | 개입 전후 이미지 변화량 정량화 (선택) |
| **텍스트-이미지 정합** | **OpenCLIP** 또는 **SigLIP** | TACA / K/V 패칭 전후 정합 변화 확인 (선택) |

---

## 4. 선택적 상위 GPU 대조 (Phase 4, A100/H100)
L4 양자화 기준선으로 모든 핵심 시각화를 완료한 뒤, 필요할 때만 상위 GPU에서 정밀 대조를 보충한다.
1. **Krea 2 Raw 원본 BF16 (~25.6 GB DiT)** vs L4 `fp8_scaled` 대조 (`gate_value`·`M-Act`·`Cont`).
2. **MOSS-Image `sft_bf16` + `qwen3.5_9b_bf16`** vs L4 `int8_convrot` 대조.
3. **H100/A100 `FlexAttention` TACA** 및 가속/증류 옵션(`TeaCache`/`Turbo`) 비교.
