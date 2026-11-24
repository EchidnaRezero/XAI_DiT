# 아키텍처 (Architecture)

## 1. 설계 원칙

1. **순수 PyTorch 코어 + 얇은 ComfyUI 래퍼**:
   - 모든 관찰·집계·개입·검증 로직은 ComfyUI에 종속되지 않는 **순수 PyTorch 코어(`src/xai_dit/`)**로 구현한다.
   - **M1 실행 백엔드**: `fp8_scaled`/`int8_convrot` 단일 파일 가중치와 검증된 샘플러·컨디셔닝을 일관되게 사용하기 위해 Colab CLI 러너(`runners/cli.py`)는 **ComfyUI를 headless 라이브러리(`comfy.*`)**로 호출하며, ComfyUI 커스텀 노드(`comfyui/nodes.py`)는 코어를 감싸는 얇은 UI 래퍼로만 둔다. ComfyUI 내부 변경이 생겨도 코어는 영향받지 않게 분리한다.
2. **코어 기준 신규 설계 (확장성 우선)**:
   - 새 DiT 모델이 추가돼도 **모델 어댑터 1개**만 구현하면 4방향 코어가 그대로 동작하도록 설계한다.
   - 기존 `Anima_HEATMAP` 코드를 그대로 늘리지 않고, 스트리밍 집계·프로파일러 등 재사용 가치가 있는 로직만 선별해 발전 이식한다.
3. **메모리·실행 분리**:
   - 로컬 CPU에서는 단위 테스트·구조 검증, Colab CLI(L4+)에서는 실제 모델 추론·배치 실험을 수행한다.
   - L4 VRAM 기준으로 대용량 어텐션 원본 전체 적재를 피하고, **스트리밍 집계**를 기본으로 하며 원본 보관은 선택 옵션으로 둔다.

---

## 2. 계층 구조

```text
[실사용 UI]     ComfyUI 커스텀 노드 (얇은 래퍼)         [배치 실험] Colab CLI / Python 스크립트
                        │                                          │
                        └──────────────┬───────────────────────────┘
                                       ▼
                     ┌───────────────────────────────────┐
                     │  4방향 XAI / MI 코어 (Pure PyTorch) │
                     │  - heatmap  - layer_selection     │
                     │  - patching - verification        │
                     └─────────────────┬─────────────────┘
                                       │ (공통 ModelAdapter 계약)
                                       ▼
                     ┌───────────────────────────────────┐
                     │        모델 어댑터 (Adapters)       │
                     │  Anima  |  MOSS-Image  |  Krea 2  │
                     └─────────────────┬─────────────────┘
                                       │ (PyTorch / DiffSynth / ComfyUI 백엔드 연결)
                                       ▼
                               실제 DiT 모델 실행
```

---

## 3. 모델 어댑터(ModelAdapter) 공통 계약

4방향 코어는 모델 내부 구현을 직접 알지 않고 어댑터가 제공하는 **메타정보 + 훅/개입 지점**만 사용한다.

### 3-1. 모델을 구분하는 메타정보 (Adapter Capability)
| 항목 | Anima | MOSS-Image | Krea 2 (본 모델) |
|---|---|---|---|
| 어텐션 형태 | `cross_attn` | `joint_attn` (`[img \| txt]`) | `joint_attn` (`[txt \| img]`) |
| 텍스트 블록별 갱신 | ✕ (고정 `ctx`) | ○ | ○ |
| ConceptAttention 지원 | ✕ (텍스트 출력 공간 없음) | ○ | ○ |
| 어텐션 출력 게이트 | 없음 | 없음 (타임스텝 `gate1`만) | **있음 (`sigmoid(Wg·h)`) + GQA** |
| 패딩 여부 | 512까지 0패딩 | 패딩 없음 | 패딩 없음 |
| 템플릿·특수 토큰 | 끝 `</s>`, 0패딩 구간 | 앞·뒤 채팅 템플릿 (`refiner` 양방향 2블록 통과) | 꼬리 5토큰 (접두부 제거됨) |

### 3-2. 어댑터가 제공해야 하는 공통 기능
1. **토큰 분류기 (`TokenMap` / `token_spans`)**:
   - 실제 `token_ids`, 출현 번호(`occurrence_idx`), 분기별(`cond`/`uncond`) 텍스트 길이, DiT 입력 슬롯 좌표, 대상 span을 저장한다.
   - **프롬프트(`S`) / 템플릿(`R`) / 패딩(`P`)**은 모두 반열린 구간(`[start, end)`)으로 정의한다(Anima: `S=[0, Lt-1)`, `R=[Lt-1, Lt)`, `P=[Lt, L)`).
2. **관찰 훅 (`capture`)** *(모델별 실제 텐서 시점 명시적 분리)*:
   - `q, k, v`, `probs` (query-chunked softmax 결과)
   - `attn_pre_proj`: 출력 사영(`Wo`) 및 게이트 적용 **전** 어텐션 컨텍스트(MOSS `block[i].attn.out`, Krea 2 `ctx`)
   - `attn_out`: 게이트(존재 시) 및 `Wo` 출력 사영 **후**, 타임스텝 변조 `g1` 적용 **전** 텐서
   - `gate_logits` / `gate_value` *(Krea 2 전용)*: `self.gate(x)`의 pre-sigmoid 원시 로짓(`gate_logits`)과 `sigmoid(gate_logits)` 승수(`gate_value`)를 별도 필드로 노출
   - `residual_out`: 블록 출력 잔차 탭(E4-5 프로브/정합성 점검용)
3. **개입 훅 (`intervene`)**:
   - **반사실 패칭 (`kv_patch`)**: 지정 블록·스텝 구간에서 텍스트 입력 또는 텍스트 `K/V`(Krea 2는 12 KV 그룹 단위)를 반사실 값으로 교체.
   - **Logit Knockout (`logit_knockout`)**: 이미지 query → 지정 텍스트 key의 pre-softmax logit을 $-\infty$로 마스킹.
   - **Logit 스케일링/바이어스 (`logit_scale`)**: TACA(E3-3)용 pre-softmax 스케일링(L4는 `K_txt` 스케일링 + 텍스트 query 출력 원복).
   - **어텐션 재주입 (`attn_override`)**: Seg4Diff(E1-3) 1D 텍스트축 블러 후 $A'V$ 실제 반영.
   - **동적 2-pass 싱크 마스킹 (`two_pass_sink_mask`)**: E4-1 `Q_img` 행 기준 동적 Top-$k$ `-1e4` 마스킹.
   - **스팬 쿼리 마스킹 & Pre-DiT 스팬 교체 (`span_query_mask`, `pre_dit_span_swap`)**: E4-2 $\mathcal{R}\to\mathcal{I}$ / $\mathcal{R}\to\mathcal{S}$ 마스킹 및 `te.out`/`refiner.out`/`text_in`/`txt_embed` 교체.
   - **출력 게이트 오버라이드 (`gate_override`)**: E4-3 `gate_value`를 `1.0`·평균 게이트·연속 스케일로 고정(항상 post-sigmoid 승수 기준).
   - **헤드·토큰 단위 기여 제거 (`per_head_token_ablation`)**: E4-4에서 GQA 공유 `V` 오염을 막기 위해 **post-softmax에서 지정 헤드의 $A_{h,j} V_{h_{kv}, j}$ 항만 차감**.
   - **개념 스트림 주입 (`concept_stream`)**: E1-2 오픈셋 개념 벡터 병렬 통과(자체 게이트·`Wo`·`g1` 통과).
   - **True Block Bypass (`true_block_bypass`)**: E2-1 전 블록 pre-body 우회(블록 내부 호출 수 `0`) 및 비교용 `value=0/mean` 대체.
4. **스텝·조건 메타데이터**:
   - 현재 스텝 인덱스 `i`와 $\sigma_i$는 호출 횟수 누적이 아니라 **`transformer_options["sigmas"]`에서 직접 역산**한다.
   - CFG `cond`/`uncond`는 ComfyUI `cond_or_uncond` 청크 메타데이터를 샘플 행 인덱스로 명시 매핑하고, 러너에서 **고정 배칭 정책**을 강제한다(메모리 변동에 따른 배칭 전환 방지).

---

## 4. 4방향 코어 모듈 구성 (`src/xai_dit/`)

```text
src/xai_dit/
├── adapters/
│   ├── base.py              # ModelAdapter 추상 인터페이스 · Capability · TokenMap
│   ├── anima.py             # 1차: Anima 어댑터
│   ├── moss_image.py        # 2차: MOSS-Image 어댑터
│   └── krea2.py             # 3차: Krea 2 어댑터
├── core/
│   ├── session.py           # 스트리밍 세션 · 메모리 관리 · 스텝/블록 필터
│   ├── heatmap.py           # 어텐션 확률 집계 · Norm/Gate 기여도 · ConceptAttention
│   ├── layer_selection.py   # 블록별 중요도 프로파일링 · 핵심 층 선별
│   ├── patching.py          # 반사실 프롬프트 패칭 · 스텝 구간 분해 · 대체값 비교(zero/mean)
│   ├── verification.py      # no-op · 무작위 대조 · logit knockout · 템플릿 분리 검증
│   └── profiling.py         # 단계별 시간·메모리 프로파일러 (Anima_HEATMAP에서 발전 이식)
├── runners/
│   └── cli.py               # Colab CLI / 순수 PyTorch 배치 실험 실행기
└── comfyui/
    └── nodes.py             # ComfyUI 커스텀 노드 얇은 래퍼

tests/
├── test_adapters.py         # 더미 모델 기반 어댑터 계약 · 토큰 분류 테스트 (CPU)
├── test_core_heatmap.py     # 집계 · Norm/Gate · ConceptAttention 단위 테스트
├── test_patching_noop.py    # 반사실 패칭 · knockout · no-op 오차 0 회귀 테스트
└── test_session.py          # 스트리밍 집계 메모리·정합성 테스트
```

---

## 5. `Anima_HEATMAP` 선별 이식 방침

| `Anima_HEATMAP` 기존 모듈 | 처리 방침 |
|---|---|
| `aggregate.py` (스트리밍 집계), `profiling.py` (프로파일러) | 핵심 아이디어를 `core/session.py`, `core/profiling.py`로 발전 이식 |
| `capture.py`, `comfyui_anima_heatmap/` (Anima·구 ComfyUI 전용 훅) | 이식하지 않고 새 `adapters/` + 얇은 `comfyui/` 래퍼로 재작성 |
| `origin/feature/tag-influence` (value 0 ablation) | 기본 방식은 반사실 패칭으로 대체, value 0은 `patching.py`의 비교 옵션으로만 유지 |
