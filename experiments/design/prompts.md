# 표준 벤치마크 프롬프트 매니페스트 & 관찰 타깃 규격

[`docs/experiment_design.md`](../../docs/experiment_design.md)의 입력 프롬프트 세부 규격.
실제 커뮤니티 문법(**Anima·MOSS-Image는 2D `@artist` 태그·하이브리드 중심, Krea 2 Raw는 실사 물리 카메라/광원 자연어 중심**)에 맞추되, **재현 가능한 매니페스트 스키마 · `calib`/`rank`/`eval` 비중첩 분할 · 중립 치환자 · VQA 질문**을 사전 고정한다.

---

## 0. 프롬프트 매니페스트 스키마 및 분할 원칙

M1 실행 시 모든 프롬프트는 `experiments/design/manifest_<model>.json`으로 동결하며, 각 항목은 아래 필드를 필수로 포함한다.

```json
{
  "prompt_id": "P-A1",
  "family_id": "anima_2d_kanao",
  "model": "anima",
  "split": "eval",
  "negative_prompt": "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts",
  "prompt": "...",
  "targets": [
    {
      "text": "sketchbook",
      "role": "prop",
      "occurrence_idx": 0,
      "localizable": true,
      "neutral_sub": "notebook",
      "vqa_question": "Is the person holding a sketchbook? Answer yes or no."
    }
  ],
  "seed_set": [42, 101, 202, 303, 404, 505, 606, 707]
}
```

- **분할 격리 (`family_id` 기준)**: 동일 원형(`family_id`)의 변형은 반드시 같은 `split`에만 배정한다.
  - `calib` (보정 세트): E1-1 Top-5 블록 선정(Grounded-SAM2 마스크 기반) 및 E1-3 V-norm 피크 블록 탐색용 (`5–10` 프롬프트/모델).
  - `rank` (순위 산출 세트): E2-2 `Cont^(l)` 블록 순위 및 E4-4 공간 관계 헤드 선정용.
  - `eval` (독립 평가 세트): E1-1 2×2 절제, E2-2 중립 치환 인과 검증, E4-4 헤드 제거 검증용(`calib`/`rank`와 완전 비중첩).
- **스윕 규모 확보 (`k >= 16`/모델)**: 아래 2D 아티스트(5) × 캐릭터(6) 풀 및 실사 장면 템플릿을 조합해 모델별 최소 `16`개(`E2-1` 비중첩 8+8 반분할 + `E4-2` 10–20프롬프트 충족)와 공통 관계 프롬프트 `20`개(역전쌍 포함)를 M1 착수 시 매니페스트 JSON으로 동결한다.
- **국소화 플래그 (`localizable`)**: `true`(의상·소품·인물·객체)만 E1-1 mIoU 및 E4-4 마스크 집계에 포함하고, `false`(아티스트·렌즈·필름톤)는 mIoU에서 제외하며 PCI 평가 시 참조 정규화 점수 `(s(x)-s(P_b))/(s(P_c)-s(P_b))`를 사용한다.

### 2D 공통 아티스트 · 캐릭터 풀 (Anima · MOSS-Image 공용)
*(Anima/MOSS-Image는 언더스코어 대신 공백 표기; ComfyUI 강조 문법 충돌 방지를 위해 괄호는 반드시 `\(`, `\)` 이스케이프)*
- **아티스트 풀 (5명)**: `@ogipote`, `@mika pikazo`, `@ciloranko`, `@mochizuki kei`, `@yoneyama mai`
- **캐릭터 풀 (6명)**: `tsuyuri kanao` (`kimetsu no yaiba`), `emilia \(re:zero\)` (`re:zero kara hajimeru isekai seikatsu`), `arona \(blue archive\)` (`blue archive`), `frieren` (`sousou no frieren`), `hoshino ai` (`oshi no ko`), `raiden shogun` (`genshin impact`)

---

## 1. Anima 대표 프롬프트 (2D 태그 + `@artist` + 자연어 혼합)

> **표준 네거티브 ([`runtime_and_hp.md`](runtime_and_hp.md) 단일 출처 일치)**:
> `worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts`

### P-A1. `Anima_HEATMAP` v1 계승 (`@ogipote` + `tsuyuri kanao`)
```text
masterpiece, best quality, amazing quality, very aesthetic, absurdres, wallpaper, @ogipote, 1girl, upper body, solo, expressionless, looking at viewer, tsuyuri kanao, kimetsu no yaiba, demon slayer uniform, sitting, holding sketchbook, from above, sunlight, soft lighting, futuristic city, glass building, nostalgic, sunset, depth of field
```
- **관찰 타깃 (4행)**:
  - `@ogipote` *(style, `localizable: false`, 중립/대조: 동일 토큰 길이 아티스트 또는 무스타일 중립어)*
  - `tsuyuri kanao` *(character, `localizable: true`, 주의: `kimetsu no yaiba`·`demon slayer uniform`과 단서 중복 → 단독 격리 실험 시 비중복 변형 `P-A1b` 병행)*
  - `demon slayer uniform` *(outfit, `localizable: true`, 중립 치환: `casual school jacket` — T5 토큰 길이 검증)*
  - `sketchbook` *(prop, `localizable: true`, 중립 치환: `notebook` 또는 `newspaper`)*

### P-A2. 풀 조합 확장 (`@yoneyama mai` + `frieren`)
```text
masterpiece, best quality, score_7, safe, 1girl, solo, @yoneyama mai, frieren, sousou no frieren, full body, white mage coat, striped scarf, holding wooden staff, standing beside a suitcase, hillside flower field, stone ruins, clear morning sky, drifting petals, dramatic rim lighting
```
- **관찰 타깃 (4행)**: `@yoneyama mai` *(style, `localizable: false`)* · `frieren` *(character, `localizable: true`)* · `white mage coat` *(outfit, `localizable: true`, 치환: `dark wool coat`)* · `wooden staff` *(prop, `localizable: true`, 치환: `metal cane`)*

### P-A3. 다중 캐릭터 + 공간 배치 (`@ciloranko` + `emilia` & `arona`)
```text
masterpiece, best quality, score_7, safe, 2girls, @ciloranko, sitting on bench, train station platform, late afternoon. On the left, emilia \(re:zero\) wears a beige trench coat and holds a closed red umbrella. On the right, arona \(blue archive\) wears a sailor uniform and holds a blue sketchbook.
```
- **관찰 타깃 (4행)**: `@ciloranko` *(style, `localizable: false`)* · `emilia \(re:zero\)` *(left char, `localizable: true`)* · `red umbrella` *(left prop, `localizable: true`, 치환: `gray parcel`)* · `arona \(blue archive\)` *(right char, `localizable: true`)*

---

## 2. MOSS-Image 대표 프롬프트 (2D 태그형 · 2D 자연어 · 실사/텍스트)

> **표준 네거티브 ([`runtime_and_hp.md`](runtime_and_hp.md) 단일 출처 일치)**:
> `worst quality, bad anatomy, watermark, jpeg artifact` *(순수 2D 항목 `P-M1`/`P-M2`에 `, 3d, realistic` 추가 시 `manifest`의 `negative_prompt`에 명시)*

### P-M1. 2D 캐릭터 태그형 (`@mika pikazo` + `hoshino ai`)
```text
best quality, highres, hoshino ai, oshi no ko, @mika pikazo, 1girl, solo, star-shaped pupils, rabbit hair ornament, idol dress, holding a vintage microphone, standing on a sunlit wooden rooftop overlooking a coastal city, floating confetti, bright afternoon sky, vivid color palette
```
- **관찰 타깃 (4행)**: `@mika pikazo` *(style, `localizable: false`)* · `hoshino ai` *(character, `localizable: true`)* · `idol dress` *(outfit, `localizable: true`, 치환: `plain dress`)* · `vintage microphone` *(prop, `localizable: true`, 치환: `wooden lantern`)*

### P-M2. 2D 자연어 일러스트 (`@mochizuki kei` + `raiden shogun`, 중복 출현 제거)
```text
best quality, highres, @mochizuki kei, raiden shogun, genshin impact, vertical portrait, a medium shot of her with braided violet hair and hairpins, wearing a dark purple kimono with gold embroidery and holding a wooden folding fan near a temple veranda. Golden sunset light slants in from the right, drifting sakura petals, cool indigo shadows, crisp 2D anime illustration.
```
- **관찰 타깃 (4행)**: `@mochizuki kei` *(style, `localizable: false`)* · `raiden shogun` *(character, `localizable: true`, `occurrence_idx: 0` 유일)* · `dark purple kimono` *(outfit, `localizable: true`, 치환: `plain gray robe`)* · `wooden folding fan` *(prop, `localizable: true`, 치환: `ceramic tea cup`)*

### P-M3. 실사 야경 + 텍스트 렌더링 (E3-1 `TextGenParamLocalization` 연계)
```text
A cinematic medium shot of a street vendor in a rainy downtown Tokyo alley at night, glowing neon signs reflecting in puddles on wet asphalt, realistic skin texture and weathered waterproof jacket. Moody cinematic lighting, light steam rising from the stall, shot on 35mm film, Kodak Portra 400 aesthetic, fine film grain, shallow depth of field. A glowing neon sign on the wall reads "NOODLES".
```
- **관찰 타깃 (4행)**: `Kodak Portra 400` *(film tone, `localizable: false`)* · `street vendor` *(person, `localizable: true`)* · `waterproof jacket` *(outfit, `localizable: true`, 치환: `cotton sweater`)* · `"NOODLES"` *(rendered text, `localizable: true`, 치환: `"COFFEE!"` 또는 동일 토큰 길이 문자열)*

---

## 3. Krea 2 Raw 대표 프롬프트 (본 모델 — 실사 물리 카메라 · 광원 중심)

> **표준 네거티브 ([`runtime_and_hp.md`](runtime_and_hp.md) 단일 출처 일치)**: `""` (빈 문자열 고정)

### P-K1. 35mm 스트리트 다큐멘터리
```text
A candid street documentary photograph of a commuter wearing a navy trench coat and carrying a brown leather briefcase walking briskly through a rainy intersection in downtown Tokyo at twilight. Shot on a 35mm lens, natural available light, neon reflections glistening on wet asphalt, subtle motion blur in passing crowds, authentic film grain, unmanipulated natural colors.
```
- **관찰 타깃 (4행)**: `35mm lens` *(camera, `localizable: false`)* · `navy trench coat` *(outfit, `localizable: true`, 치환: `gray wool coat`)* · `brown leather briefcase` *(prop, `localizable: true`, 치환: `black canvas backpack`)* · `wet asphalt` *(surface, `localizable: true`, 치환: `dry stone`)*

### P-K2. 85mm f/1.4 인물 포트레이트
```text
A close-up editorial portrait of an elderly woodworker wearing a dark canvas apron in a rustic sunlit workshop, holding a wooden chisel, natural side-lighting from a dusty window casting deep soft shadows across his face, visible skin texture, pores, and weathered hands. Shot on an 85mm f/1.4 lens, shallow depth of field, creamy background bokeh, rich tonal density.
```
- **관찰 타깃 (4행)**: `85mm f/1.4` *(lens, `localizable: false`)* · `elderly woodworker` *(person, `localizable: true`)* · `dark canvas apron` *(outfit, `localizable: true`, 치환: `plain linen shirt`)* · `wooden chisel` *(prop, `localizable: true`, 치환: `metal hammer`)*

---

## 4. 공통 특수 검증 프롬프트 (3개 모델 공통)

### 4-1. 공간 관계 회로 프로브 (E4-4 `DiTRelationCircuits` · E3-3 `TACA`)
객체와 방향의 결부를 제거하기 위해 **모든 객체쌍에 대해 좌/우 · 상/하 역전쌍**을 구성한다(총 ~20프롬프트: `rank` 10 + `eval` 10).
```text
# P-C1a / P-C1b (좌우 역전쌍) + "the" 변형
A red wooden chair to the left of a blue ceramic vase on a white table
A red wooden chair to the right of a blue ceramic vase on a white table
A blue ceramic vase to the left of a red wooden chair on a white table
The red wooden chair to the left of the blue ceramic vase on a white table
```
```text
# P-C2a / P-C2b (상하 역전쌍)
A yellow cat sitting on top of a black suitcase in front of a brick wall
A yellow cat sitting underneath a black suitcase in front of a brick wall
A black suitcase resting on top of a wooden crate in front of a brick wall
```
- **관찰 타깃**: 관계어(`left`/`right`/`on top of`/`underneath`), 객체1, 객체2, `EOS / 템플릿 꼬리 토큰`

### 4-2. E1-2 ConceptAttention 개념 어휘 (MOSS · Krea 2)
`open_set_concept_stream`의 개념축 softmax 분모를 안정화하기 위해 매 프롬프트마다 **`[타깃 개념 2~4개] + [배경 개념: "background", "sky", "wall", "floor"]`**를 함께 입력한다.

### 4-3. E3-4 SAP 2단계 프롬프트 번들 (레이아웃 프록시 → 상세)
```text
# Stage 1 (Layout Proxy, σ >= σ_switch):
Two objects on a wooden table: a tall ceramic vase on the left, a small brass lamp on the right, clean studio framing
# Stage 2 (Detailed Target, σ < σ_switch):
A tall blue-and-white porcelain vase with floral patterns on the left and a vintage tarnished brass desk lamp on the right on a weathered oak table, soft afternoon window light
```

### 4-4. 반사실 대조쌍 (E2-2 · E3-0 · E3-1 · E3-2)
- **엄격한 토큰 정렬 검증 (`assert_strict_token_alignment`)**:
  1. 대상 토크나이저(**Anima는 T5 토크나이저 기준**; MOSS는 Qwen3.5; Krea 2는 Qwen3-VL)에서 `len(tok(p_src)) == len(tok(p_tgt))` 일치 확인.
  2. **치환 스팬 `[s, e)` 바깥의 모든 토큰 ID가 100% 동일**함을 `assert` (스팬 경계 바깥 BPE 병합 변화 차단).
  3. **아티스트 치환 폴백**: 두 `@artist`의 토큰 길이가 다를 경우, (a) 동일 토큰 길이 아티스트 쌍을 우선 매칭하고, 불가 시 (b) E3-0 `direct_effect`(Logit Knockout) 단독 또는 동일 길이 일반 스타일 태그로 대체하며 `pair_role`에 명시한다.
- **대조쌍 역할 구분 (`pair_role`)**:
  - `attribute_swap` (전이쌍, E3-1 K/V 패칭): `red scarf` ↔ `blue scarf`, `yellow sweater` ↔ `purple sweater`
  - `concept_neutralize` (제거 중립 대조, E2-2 / E3-0 `total_effect`): `holding a sketchbook` ↔ `holding a plain box`, `wooden chisel` ↔ `smooth stone`, `red scarf` ↔ `plain cloth` *(scarf가 양쪽에 모두 남는 속성 전이쌍을 개념 제거 중립 대조로 혼용하지 않음)*
- **PCI 시간축 삽입쌍 ($P_b \to P_c$, E3-2)**:
  - 전역 스타일(2D): `masterpiece, best quality, 1girl, solo, standing by a window, afternoon light` → `+ @ogipote`
  - 전역 스타일(실사/공통): `A candid photograph of a quiet city street at sunset` → `An oil painting of a quiet city street at sunset`
  - 국소 소품: `A close-up portrait of a young woman in a cafe, natural window light` → `+ wearing a red beret`

---

## 5. 모델별 토큰 스팬 분리 규칙 (반열린 구간 `[start, end)`)
- **Anima (T5 슬롯 기준)**: `S = [0, Lt-1)` (프롬프트), `R = [Lt-1, Lt)` (`</s>`), `P = [Lt, L)` (0-패딩).
- **MOSS-Image (템플릿 포함 BPE 위치)**: `R_prefix`, `S`, `R_suffix`를 실제 템플릿 특수 토큰 ID로 검증·분리.
- **Krea 2 (접두부 제거 후 위치)**: `S = [0, L-5)`, `R = [L-5, L)` (`<|im_end|>`, `\n`, `<|im_start|>`, `assistant`, `\n` 실제 토큰 ID 검증).
