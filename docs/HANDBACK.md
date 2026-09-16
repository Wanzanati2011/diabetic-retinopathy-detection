# Fundus Console v2 — Handback Document

**Last updated:** Session completing Phase 3 (UI shell)

## Completed Work

### Phase 0: Verify (V1-V5) — 5 commits
- V1: Verified app loads correct EMA weights (not raw)
- V2: Reproduced Part 1.4 numbers from calibration_reject.json
- V3: Recorded exact JSON/CSV structures needed by Phase 4
- V4: Recorded environment versions, generated requirements-lock.txt
- V5: Record baseline test suite (130 passed, 3 pre-existing failures)

### Phase 1: Report parity (A1-A7 + D1) — 7 commits
- A1-A3: Calibration module, referral rule, four-state outcome
- A4: Per-prediction context card data + wiring
- A6: Both Eyes DEC-2 patient outcome + identical-image guard
- A7: Session log entry fields + PDF header/footer
- D1: Full UI wiring — app now makes validated decisions (temperature scaling, rejection gate, referral rule)

### Phase 2: Robustness (B1-B8) — 8 commits
- B1: Image quality checks wired into both tabs
- B2: METRICS loader — single source of truth, T-9 passes
- B3: Module split — model/inference/gradcam/CSS moved to app/core/
- B4: Curated demo samples from resized test 15
- B5: Runtime hardening — queue, inference_mode, warm-up, Grad-CAM timeout, cache cleanup
- B6: Decision-level Acceptance Test 12.1
- B7: Integrity and status panel
- B8: Repo hygiene — gitignore, cleanup

### Phase 3: UI shell (U1-U6) — 5 commits
- U1: Single-page scaffold — replaced gr.Tabs() with gr.Column sections, anchor-linked sticky nav, hidden Gradio footer
- U2: Design tokens v2 — dark palette (#0C0C0B ground), Fraunces/Inter/Plex Mono, no text <12px, 16px card radius
- U3: Hero section — headline, 3 stat tiles from live results JSONs (QWK, referral sens/spec, model count), disclaimer chip
- U4: Console v2 single-eye — unified result card (outcome badge, grade + expected, calibrated confidence meter, 5-class probability bars replacing gr.Label, per-grade context, SHA+timing strip), side-by-side image comparison HTML replacing gr.Image outputs, 4-output predict() generator
- U5: Console v2 both-eyes — unified result card (pooled grade + per-eye detail block + fusion caveat), side-by-side preprocessed images HTML, 4-output predict_both_eyes() generator

## Test Baseline

**130 passed, 3 pre-existing failures** (never touched these — they existed before Phase 3):
1. `test_app.py::test_app_pipeline_matches_eval_pipeline_on_real_test_images` — 2/20 images show 1-grade mismatch (aptos decode-level noise, not a code bug)
2. `test_dedup.py::test_every_dup_group_member_pairwise_within_threshold` — TypeError: int() can't convert non-string with explicit base
3. `test_splits.py::test_splits_deterministic` — p1.json differs between runs (split not deterministic)

## Current State

- Branch: `app-v2` (never committed to master)
- Commit count: 25 from master
- Working directory: `C:\Users\visha\Premier Pro\EDITING\OneDrive\Desktop\mini`
- Gradio version: 6.27.0

## What's Next: Phase 4 (Evidence Sections C1-C9)

Per AGENT_EXECUTION_PLAN.md Part 4, these charts are inline SVG generated from METRICS in `app/render/charts.py`:

| ID | Section | Description | Key Source Files |
|----|---------|-------------|------------------|
| C1 | `results` | 10×10 waffle (TP/FN/TN/FP), ROC curve, "with uncertainty gate" toggle | test CSV, grade1_diagnosis.json |
| C2 | `results` (collapsed) | Reliability diagrams before/after, risk-coverage explorer, τ marker | calibration_reject.json, curves.max_softmax.test |
| C3 | `results` | Literature dot plot (Gulshan 0.991, Voets 0.951, this model) | literature.json |
| C4 | `integrity` | Split diagram P1/P2, shortcut ceiling, training-size sweep | excluded_images.json, duplicates.json |
| C5 | `evidence` (tabs) | Design levers, Two eyes waterfall, Mild disease, Transfer, Explanations, Quantum | Multiple JSONs |
| C6 | `record` | Prediction scorecard table (8+ rows from results JSON) | results/*.json expected-range fields |
| C7 | `built` | Codebase stats, 5-layer arch SVG, acceptance tests, data-cleaning story | codebase_stats.json, acceptance_12_1.json |
| C8 | `built` (collapsed) | Limitations table (single seed, grade-1 ceiling, etc.) | Self-documented |
| C9 | `about` | Author card, future work, disclaimer, dataset credits, citations | H4 values, TODO_AUTHOR_LINKS |

### Key Design Decisions to Preserve

1. **All numbers from METRICS** — zero hardcoded values. Every chart must read from results/*.json at runtime.
2. **"Data unavailable" degradation** — when a source file is missing, show graceful message, not crash.
3. **T-9 passes** — grep for metric literals must find zero matches (except literature.json which carries citations).
4. **No report link** (R5) — never embed PDF/DOCX links.
5. **Honesty labels** (R4) — "research prototype", "single seed", "not separately measured", etc.

### Files That Will Need Touching in Phase 4

- `app/render/charts.py` — new file, SVG chart builders from METRICS
- `app/data/metrics.py` — may need new loaders for C1-C9 source JSONs
- `app/app.py` — section builders for C1-C9, inserted into build_demo()
- `app/render/tokens.css` — chart-specific CSS (waffle grid, ROC curve, dot plot)

### Critical Verification Points

- C6 requires ≥8 prediction-record rows from results JSONs (R9-Stop condition)
- C5 Quantum tab references qml_pqc.json and qcnn_no_cnn_pixels.json — verify both exist
- Fusion effect value for both-eyes card: already loaded from context, verify source

## Files to Review for Phase 4 Source Data

Before building charts, verify these files exist and have expected structure:
```
results/calibration_reject.json       # C2 reliability, referral counts
results/inter_eye_correlation.json     # C3 QWK baseline
results/grade1_diagnosis.json          # C1 confusion, grade-1 stats
results/finetune_app_converged_p2_*.json # C1, C7 training stats
results/claim3_decomposed_*.json       # C5 waterfall
results/claim2_protocols.json          # C5 transfer
results/qml_pqc.json                   # C5 quantum
results/qcnn_no_cnn_pixels.json        # C5 quantum
results/acceptance_12_1.json           # C7, T-7
results/excluded_images.json           # C7 data-cleaning
results/duplicates.json               # C7 dedup
results/dedup_threshold_sweep.json    # C7 sweep
results/app/release/thresholds.json    # T-9 values, calibration params
app/data/literature.json              # C3 literature values
```
