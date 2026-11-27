# Project Status

Last updated: 2026-11-27  
Management System: [`docs/project_management.md`](../docs/project_management.md) · Completed Milestones: [`plan/archive/completed_milestones.md`](archive/completed_milestones.md)

---

## 1. Active Plans (`plan/active/`)
- [ ] **v2 Practical Semantic-Unit Control** ([`active/v2_practical_control_replan.md`](active/v2_practical_control_replan.md)): 3-model non-destructive control (`kv_patch`, StableFlow Stage 2, `S`-only TACA, early-$\sigma$ SAP), offline semantic evaluators, ComfyUI node packaging
- [ ] **Docs Reconciliation** ([`active/design_updates_pending.md`](active/design_updates_pending.md)): Propagate approved spec/workflow updates into `docs/` and `experiments/design/` upon user approval
- [ ] **Git Commit & Push**: Commit and push updated structure, reports, and docs via `GPT-6.1-Sol` (`Echidna` account) upon user approval

## 2. Active Interrupts (`plan/interrupts/`)
- *(None currently active — unplanned event-driven detours are logged here via `python -m scripts.tracker.cli plan new-interrupt <slug>` and archived on completion.)*

## 3. Future Roadmap (`plan/roadmap.md`)
- Tier 2 Blackwell (`sm_120`, ≥80 SMs) acceleration verification (`SageAttention 3`, `NVFP4`, `MXFP8`, `FlexAttention`, `max-autotune+cudagraphs`)
- Curated 2D public XAI LoRA diagnosis showcase (`experiments/case_studies/showcase/`)

---

## 4. Completed Milestones (`plan/archive/completed_milestones.md`)
- [x] **M0 (2026-11-24)**: 3-model architecture analysis, 16-paper survey & verified specs (`docs/specs/01~04`), dual cross-review, Git init
- [x] **M1 (2026-11-25)**: Core engine (`src/xai_dit/`), 3 adapters, CLI runner, thin ComfyUI nodes, 4 frozen manifests, **210 CPU unit tests**, E0–E4 L4 suite
- [x] **M1.5 (2026-11-25)**: Heatmap display fix (`issues/001`), Step 0–2 Anima suite & prop-transplant PoC (`+0.483`), 4-tab Visual Showcase v3 (`showcase/index.html`)
- [x] **Accel Phase 1–3 Tier 1 on L4 (2026-11-26–27)**: T2I & accelerated XAI benchmarks (`14–38s`, `r=0.93–0.99`), `Krea2-Anime v3.0 INT4 W4A4` (`32s`, `8.38 GB`), ComfyUI fixes (`issues/003–007`), consolidated into [`docs/acceleration.md`](../docs/acceleration.md) & [`experiments/accel/REPORT.md`](../experiments/accel/REPORT.md)
- [x] **Tracking Architecture & Repo Hygiene (2026-11-27)**: `experiments/accel/` & gitignored `experiments/case_studies/` tracks, `plan/` 4-bucket layout, `scripts/tracker/` 3-module CLI, 3-Scribe automation, bilingual `README.md` / `README_KR.md`
