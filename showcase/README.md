# XAI_DiT Showcase

**Anima · MOSS-Image · Krea 2 Raw** 3종 Diffusion Transformer의 어텐션 히트맵 원리·모델 구조·실제 결과·제어를 AI 개론 수준에서도 직관적으로 볼 수 있게 정리한 시각 쇼케이스입니다.

---

## 페이지 안내 (브라우저로 바로 열기)

| 페이지 | 내용 |
|---|---|
| **[메인 비주얼 쇼케이스 (`index.html`)](index.html)** | **4탭 구성**: ① 개요 · CAM vs DiT (구조적 차이) · ② 모델 3종 3단 구조 (전체 → 1-블록 회로 → 어텐션 세부/훅) · ③ 실제 코드 동작 5단계 애니메이션 (`Anima / MOSS / Krea 2` 전환) · ④ 실제 결과 & 중반 블록 소품 교체 제어 |
| **[Anima 4단계 진단 프리뷰 (`preview_anima_step2.html`)](preview_anima_step2.html)** | Anima(BF16) 기준 공간 매칭 → 결정 블록 층 → 고정 스텝 → 좌/우 공간 관계 헤드 4축 진단 상세 |
| **[이전 버전 아카이브 (`archive/`)](archive/)** | 구버전 v1(`v1_index.html`, `01~03_*.html`) 및 v2(`v2_01_heatmap.html`, `v2_02_control_wip.html`) 보관 |
