# 표준 출력 및 시각화 산출물 규격

[`docs/experiment_design.md`](../../docs/experiment_design.md)의 출력·시각화·스키마 세부 규격. L4와 Phase 4(BF16) 결과가 충돌하지 않도록 실행 단위를 고유화하고, 모든 표·그래프·통계를 **원시 배열과 시드별 원시 수치로부터 100% 재계산·재생성**할 수 있게 구성한다.

---

## 1. 공통 실험 폴더 구조

```text
experiments/
├── <exp_id>_<model>_<precision>_<run_id>/   # 예: e3_0_krea2_fp8_20261125_01
│   ├── config.json          # 재현 메타데이터 전체 (아래 필수 스키마)
│   ├── manifest.json        # 스윕 샤드 진행·재개(resume) 상태 및 실패/미교차 로그
│   ├── metrics.json         # 시드별 원시 수치 + 집계 95% CI + pass/fail 판정
│   ├── summary.md           # 실험 요약 (experiments/_template.md 기준)
│   ├── raw/                 # 원시 배열 (.npz/.pt: 히트맵, Δv, V-norm, gate, latent diff)
│   └── visuals/             # 표준 PNG 산출물
└── cross_model/<exp_id>/    # 3개 모델 비교 통합 차트 (예: e4_2_template_share.png)
```

### 필수 `config.json` / `metrics.json` 스키마
- **`config.json` 필수 필드**:
  - `schema_version` (`"1.0"`), `run_id`, `baseline_run_id` (경로 일치 identity 기준선 참조)
  - `model`: `name`, `variant` (`sft_bf16`/`fp8_scaled` 등), `repo_revision`, `weights_sha256`, `vae_sha256`, `te_cache_sha256`
  - `runtime`: `gpu_name`, `comfyui_commit`, `attention_path` (`fused_sdpa` | `manual_chunked_sdpa`), `cfg_batching_policy` (`batched` | `split`), `sampler`, `scheduler`, `cfg`, `shift`, **`sigma_table`** (전체 스텝 $\sigma_0 \dots \sigma_S$ 배열)
  - `prompt_manifest_sha256`, `token_map` (실제 `token_ids`, `occurrence_idx`, `S/R/P` 반열린 구간, `cond`/`uncond` 길이)
  - `evaluator_versions` (DINOv2, LPIPS, OpenCLIP, Grounded-SAM2, Qwen2.5-VL 모델명·버전·정밀도)
- **`metrics.json` 필수 구조**:
  - `pass_fail`: `exact_noop_max_abs_diff` (`== 0`), `path_matched_identity_max_abs_diff` (`<= 1e-3`), `non_determinism_floor`
  - `resources`: 단계별 `gpu_allocated_gb`, `gpu_reserved_gb`, `cpu_rss_gb`, `sec_per_forward`, `wall_clock_sec`
  - `per_sample`: `prompt_id × seed × condition` 단위의 원시 지표 및 `status` (`ok` | `mask_failed` | `non_crossing` | `base_already_present`)
  - `summary`: 집계 평균, paired bootstrap 95% CI, 통계량(Spearman $\rho$ 등)

---

## 2. 핵심 공용 시각화 포맷 (Same-State Δv & Rollout 브리지)

> **인과 경계 명시**: Same-State $\Delta v(x_{\sigma})$는 기준 궤적 $x_{\sigma}$에서의 **1스텝 직교 미분 효과**이며 최종 이미지 $x_0$에 대한 인과 효과가 아니다. 기준 최종 이미지는 시각적 위치 참조용 배경일 뿐이며, **최종 $x_0$ 인과 판정은 Paired Full-Rollout 결과 및 브리지 지표**로만 수행한다.

1. **Same-State `[태그 × 스텝]` 오버레이 그리드**:
   - **개입 유형 분리 저장** *(절대 혼용 금지)*:
     - `grid_direct_effect_steps.png`: **Logit Knockout** (직접 `img→txt` 어텐션 경로만 차단)
     - `grid_total_effect_steps.png`: **토큰 ID·길이 일치 중립 치환** (인코더·refiner·텍스트 간 어텐션 포함 총효과)
     - *(Anima 보조 대조: `grid_value0_steps.png`)*
   - **스케일 & 대조 행**:
     - 원시 $\|\Delta v_i\|_2$ 버전과 스텝 크기 보정 **$\Delta\sigma_i \|\Delta v_i\|_2$ 가중 버전**을 모두 저장(`raw/` + PNG).
     - 각 그리드 최하단에 **동일 토큰 수 무작위 토큰 대조 행(`random_ctrl`)** 추가.
     - 출력 종류(`cfg_combined` 주 / `cond_only` 진단), 단위, 공통 컬러 스케일(동일 지표 24칸 통합, 상위 0.5% 클리핑 전 min/max), 각 칸 상단 `tag | i (σ_i) | mean`, 공간 축 복원·보간 방식(`bilinear` + 잠재 격자 `64×64`)을 메타데이터 및 제목에 명시.
2. **어텐션 vs Same-State Δv 나란히 비교 (`grid_attn_vs_delta_v.png`)**:
   - 어텐션(분모 차이 명시: Anima 텍스트 슬롯 vs MOSS/Krea 2 전체 키)과 `direct_effect` / `total_effect`를 각 지표별 독립 공통 컬러 스케일로 나란히 배치.
3. **Full-Rollout 브리지 차트 (`e3_0_rollout_bridge.png`)**:
   - 동일 상태에서 분기해 동일 난수로 끝까지 생성한 Full-Rollout의 실제 $\Delta x_0$(및 LPIPS/DINOv2)와 누적 $\sum_i \Delta\sigma_i \Delta v_i$의 공간 상관(`bridge_corr`)을 나란히 출력.

---

## 3. 실험별 산출물 목록

| 실험 ID | 주요 시각화 PNG (`visuals/`) | 원시 배열 (`raw/`) 및 핵심 수치 (`metrics.json`) |
|---|---|---|
| **E0 (Sanity)** | `baseline_run1_vs_run2.png` | 3회 반복 비결정성 바닥선(DINOv2/LPIPS/pixel diff), exact no-op(`0`), 경로 일치 identity 오차(`<=1e-3`), 단계별 VRAM·RSS·`sec/forward` |
| **E1 (히트맵)** | • `generated.png`, `heatmap_<tag>.png`, `overlay_<tag>.png`<br>• `e1_1_ablation_2x2.png`<br>• `e1_2_concept_vs_raw.png` (Krea 2 `pre_gate` vs `post_gate`)<br>• `e1_3_vnorm_profile.png` | 원시 히트맵 `.npz`, `localizable: yes` 한정 mIoU·BCE·MSE, `calib` 확정 Top-5 리스트, 블록별 V-norm 비율, 1D 블러 전/후 $\Delta\text{CLIP-T}$·LPIPS |
| **E2 (층 선별)** | • `e2_1_vitality_curve.png` (전 블록 Vitality + 95% CI)<br>• `e2_1_bypass_sheet.png`<br>• `e2_2_cont_ranking.png` (`Top/Bottom/Random-20%` 비교) | 블록별 Vitality·`Cont^(l)` `.npz`, 비중첩 8+8 Spearman $\rho$, **확정 Vital / Top-20% 층 리스트**, $\Delta\text{CLIP-T}$·LPIPS |
| **E3 (태그 영향도)** | • `grid_direct_effect_steps.png`, `grid_total_effect_steps.png`, `grid_attn_vs_delta_v.png`, `e3_0_rollout_bridge.png`<br>• `e3_1_kv_patch_grid.png` + `e3_tag_block_cont.png`<br>• `e3_2_pci_curves.png` ($\tau=\sigma$ 축 CIS + 삽입 스트립)<br>• `e3_3_taca_compare.png`<br>• `e3_4_ditst_boundary.png` | $\Delta v$, $\Delta\sigma \Delta v$, `bridge_corr`, 태그별 `Cont^(l,t)` `.npz`, $\tau_{50}, \tau_{70}, W$(+ `non_crossing` 로그), DiT-ST 정규화 $\hat{\Delta}_i$ 및 $\sigma$ 경계, VQA·참조 정규화 점수·LPIPS |
| **E4 (검증 · MI)** | • `e4_1_sink_ablation.png` (`Top-k` vs `Random-k`)<br>• `cross_model/e4_2/e4_2_template_share.png` + `e4_2_masking.png`<br>• `e4_3_krea2_gate.png` (`gate_value<-1.0`, 평균 게이트, dose-response)<br>• `e4_4_relation_ablation.png`<br>• `e4_5_early_tap_corr.png` *(M2)* | 사전 선정 인덱스 기준 `mass_before`, `mass_after`, **`residual_ratio < 1e-6`**(`k` 및 per-head/union 명시), `entropy`, `M-Act`, 정규화 `S/R/P` 비중, 전체·조건부 관계 정확도, E4-5 조기 탭 Spearman $\rho$ |
