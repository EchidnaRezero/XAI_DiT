# Anima

텍스트→이미지 DiT. XAI·기계론적 해석 기법을 붙일 때 모델의 전체상과 "어디를 봐야 하는지"를 잡기 위한 개요다.

## 전체 흐름

```
prompt ─┬─► Qwen 토크나이저 ─► Qwen3-0.6B ─► Qwen hidden [1, Lq, 1024] ──────┐ (key / value)
        │                                                                     ▼
        └─► T5 토크나이저 ─► T5 토큰 id ─► 임베딩 테이블 ─► LLMAdapter (query, 6블록)
                                                             │
                                          × 프롬프트 가중치 ─► 512까지 0 패딩
                                                             ▼
                                                   ctx [B, L, 1024]  (모든 블록의 cross-attn이 읽음)

x_t (latent) ─► 2×2 패치 임베딩 ─► 이미지 토큰 [B, N, 2048] ─► Block × 28 ─► FinalLayer ─► v
σ ─► 타임스텝 임베딩 ─► 모든 블록과 FinalLayer의 adaLN

v ─► x0 = x_t − σ·v ─► Wan2.1 VAE decode ─► 이미지
```

텍스트 인코더는 Qwen3-0.6B다. causal mask가 걸리는 단방향 LLM이라, 각 토큰의 hidden은 자기보다 앞에 있는 토큰만 보고 만들어진다. 마지막 층의 최종 RMSNorm 출력을 쓴다.

Qwen 출력은 DiT로 직접 가지 않는다. LLMAdapter는 같은 프롬프트를 T5 토크나이저로 자른 토큰 id의 임베딩을 query로 삼고, Qwen hidden을 key·value로 읽는다. T5 인코더는 없고 임베딩 테이블만 있다. 그 결과 DiT가 보는 텍스트 임베딩은 T5 토큰 위치마다 하나씩 생긴다.

DiT 출력은 rectified-flow의 velocity v다. 노이즈 섞인 latent는 x_t = σ·ε + (1−σ)·x0 이고, 깨끗한 latent의 추정은 x0 = x_t − σ·v 로 얻는다. 타임스텝 입력은 σ 자체(0~1)다.

규모는 DiT 본체 약 2B, 어댑터 약 0.13B다. VAE는 Wan2.1 VAE로, latent는 16채널이고 공간을 8배 줄인다.

## 블록 내부

```
x ─► LayerNorm ─► shift·scale ─► Self-Attn  (이미지 ↔ 이미지, q·k에 3D RoPE) ─► × gate ─► 잔차에 더함
x ─► LayerNorm ─► shift·scale ─► Cross-Attn (이미지 query → 텍스트 ctx)       ─► × gate ─► 잔차에 더함
x ─► LayerNorm ─► shift·scale ─► MLP                                          ─► × gate ─► 잔차에 더함
```

블록은 28개이고, hidden 2048, head 16개, head 차원 128이다. 잔차에는 self-attention, cross-attention, MLP 순으로 세 서브층의 출력이 더해진다.

타임스텝 조건은 adaLN으로만 들어간다. σ의 임베딩에서 서브층마다 shift·scale·gate 세 벡터를 만들고, 정규화된 입력을 shift·scale로 변조한 뒤 서브층 출력에 gate를 곱해 잔차에 더한다. 이 값은 모든 토큰에 공통이다.

어텐션의 q와 k는 head마다 RMSNorm을 거친다. 위치 정보는 3D RoPE(시간·세로·가로)로 self-attention의 q·k에만 걸린다. cross-attention과 잔차 스트림에는 위치 정보가 들어가지 않고, 절대 위치 임베딩도 없다.

## 개입 지점

N은 이미지 토큰 수, Lt는 T5 토큰 수, L은 max(512, Lt)인 텍스트 슬롯 수다.

| 지점 | 의미 | shape |
|---|---|---|
| `text.qwen_hidden` | Qwen3 최종 hidden, 어댑터의 key·value | [1, Lq, 1024] |
| `text.ctx` | 어댑터 출력에 가중치를 곱하고 0 패딩한 것, DiT가 보는 텍스트 | [B, L, 1024] |
| `input.tokens` | 패치 임베딩 직후의 이미지 토큰 | [B, N, 2048] |
| `block[i].mod` | 서브층별 shift·scale·gate | [B, 1, 2048] × 9 |
| `block[i].resid_in` / `resid_out` | 블록 입력과 출력 잔차 | [B, N, 2048] |
| `block[i].self_attn.q` / `k` | head별 RMSNorm과 RoPE를 거친 뒤 | [B, 16, N, 128] |
| `block[i].self_attn.v` | value 투영 직후 | [B, 16, N, 128] |
| `block[i].self_attn.probs` | 이미지 → 이미지 어텐션 확률 | [B, 16, N, N] |
| `block[i].self_attn.out` | self-attention 출력, gate 곱하기 전 | [B, N, 2048] |
| `block[i].cross_attn.q` | head별 RMSNorm 뒤, RoPE 없음 | [B, 16, N, 128] |
| `block[i].cross_attn.k` / `v` | 텍스트 슬롯의 key와 value | [B, 16, L, 128] |
| **`block[i].cross_attn.probs`** | 이미지 → 텍스트 어텐션 확률, 히트맵의 재료 | [B, 16, N, L] |
| `block[i].cross_attn.out` | cross-attention 출력, gate 곱하기 전 | [B, N, 2048] |
| `block[i].mlp.out` | MLP 출력, gate 곱하기 전 | [B, N, 2048] |
| `final.v` | 최종 출력 velocity | [B, 16, 1, h, w] |
| `final.x0` | x_t − σ·v 로 얻은 깨끗한 latent 추정 | [B, 16, 1, h, w] |

h, w는 latent의 세로·가로 크기다. 히트맵은 `cross_attn.probs`에서 앞의 Lt개 슬롯만 남기고, N을 (h/2, w/2) 그리드로 되돌려 만든다.

## 토큰 시퀀스

이미지: latent를 2×2 패치로 묶어 (h/2)×(w/2) 그리드를 만들고, 이를 행 우선으로 펼친 N개 토큰이 시퀀스가 된다. 1024×1024 픽셀이면 64×64 = 4096 토큰이다.

텍스트: 이미지와 한 시퀀스로 합쳐지지 않는 별도 context이고, cross-attention의 key·value로만 쓰인다. 모든 블록이 같은 ctx를 읽으므로 DiT 안에서 텍스트 토큰은 갱신되지 않는다. 텍스트 쪽 처리는 어댑터에서 끝난다. 따라서 블록마다 달라지는 것은 이미지 쪽 query뿐이고, 텍스트 슬롯 j는 어느 블록에서 보든 같은 대상을 가리킨다.

내용 없는 토큰:
- Lt번째부터 L−1번째까지는 0 패딩이다. 이 슬롯의 key·value는 0이라 logit이 0이 되고, softmax 분모에는 참여하지만 출력 값에는 기여하지 않는다. DiT cross-attention에 마스크가 없으므로 이 슬롯들은 key에 그대로 들어간다.
- T5 쪽 맨 끝에는 `</s>` 토큰 하나가 붙는다. 이 슬롯은 실제 임베딩을 가진다. 시작 토큰은 없다.

프롬프트 단어와 텍스트 슬롯의 대응: DiT의 텍스트 슬롯 j는 j번째 T5 토큰이다. 프롬프트는 가중치 괄호 구간 단위로 잘려 각각 토크나이즈되고, 이어붙인 뒤 맨 끝에 `</s>` 하나가 붙는다. 같은 방식으로 다시 토크나이즈하면 슬롯 j가 어느 단어에서 왔는지 알 수 있다. Qwen 토큰은 어댑터의 key·value로만 쓰이므로 DiT에서는 직접 보이지 않는다. `(word:1.2)` 같은 프롬프트 가중치는 해당 슬롯의 임베딩에 곱해진다.

## 코드에서 볼 곳

경로는 ComfyUI 저장소 기준이다. 이 문서는 커밋 `da8bd08e48deee74d9f6c3bc00aed365b00ec209`(2026-11-23, v0.46.0 이후 46번째 커밋)의 코드를 읽고 썼다.

- `comfy/ldm/cosmos/predict2.py`: DiT 본체, 즉 블록·어텐션·adaLN·패치 임베딩·RoPE·FinalLayer.
- `comfy/ldm/anima/model.py`: LLMAdapter와, DiT 본체에 어댑터를 붙인 Anima 모델.
- `comfy/text_encoders/anima.py`: Anima용 Qwen·T5 이중 토크나이즈와 텍스트 인코더 묶음.
- `comfy/text_encoders/llama.py`: Qwen3 텍스트 인코더 구현.
- `comfy/model_base.py`: 조건 준비, velocity에서 x0로의 변환 등 샘플러와 DiT 사이의 연결.
