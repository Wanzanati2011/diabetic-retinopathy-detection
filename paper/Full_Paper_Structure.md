# Two Eyes, One Patient — Full Paper Structure
### Detailed heading outline, all phases (1–9). [DONE] = data exists now · [RUNNING] = Phase 5b in progress · [PENDING] = Phases 6b–9, not started

---

## Title
**Two Eyes, One Patient: Inter-Eye Correlation in Diabetic Retinopathy and Its Consequences for Model Evaluation**

## Abstract
One paragraph. Lead with QWK 0.855 / 0.838 lookup ceiling, state the Claim 2 null plainly, lead the results claim with whichever Phase 5b verdict (a/b/c) survives, note Phase 6b's re-test of the null under full convergence.

---

## 1. Introduction
- 1.1 Motivation and clinical context (screening bottleneck, ICDR scale, Table 1)
- 1.2 The image-level vs. patient-level splitting problem (the leakage mechanism)
- 1.3 Contributions and paper roadmap
- 1.4 Summary of findings, stated up front (what's confirmed / refined / null / pending)

## 2. Related Work
- 2.1 Automated DR grading — prior CNN approaches on EyePACS/APTOS
- 2.2 Evaluation methodology and shortcut learning in medical imaging
- 2.3 Patient-level vs. image-level splitting practice in the literature (motivates §9's survey)

## 3. Data
- 3.1 EyePACS and APTOS 2019 — scale, structure, patient/eye identifiers
- 3.2 Manifest construction and partner-eye linkage (`partner_id`)
- 3.3 Deduplication — 256-bit pHash, threshold validation, 148 near-duplicates (APTOS only) [DONE]
- 3.4 Image exclusion — 8 flat-black EyePACS captures, pre-split, 0/8 in P2 test [DONE]
- 3.5 Split protocols — P1 (image-level, leaky), P2 (patient-level, honest), P3 (cross-dataset, both directions) [DONE]
- 3.6 Frozen test-set declaration (commit hash, date, primary/secondary endpoints) [DONE]

## 4. Methods
- 4.1 Preprocessing pipeline (retinal-circle crop, resize, pad; shared with the app)
- 4.2 Frozen feature extraction — EfficientNet-B0 @224/@384, ResNet-50 @224 [DONE]
- 4.3 Classifier heads — multinomial logistic regression vs. ordinal (ridge + optimized thresholds)
- 4.4 Fine-tuning configuration
  - 4.4.1 Phase 6 initial config (15 epochs) [DONE]
  - 4.4.2 Phase 6b converged config (40 epochs, lr_backbone 1e-4, sampler ablation) [PENDING — script ready, awaiting your GPU runs]
- 4.5 Statistical procedure — patient-level bootstrap CIs, paired permutation tests, permuted-patient null

## 5. Results
- 5.1 Claim 1 — Inter-eye correlation (QWK 0.855, lookup ceiling 0.838) [DONE]
- 5.2 Claim 1b — Stratified robustness (per-grade breakdown, grade-1 collapse) [DONE]
- 5.3 Claim 2 — Protocol comparison, frozen features (P1 vs P2 vs P3, null on P1>P2) [DONE]
- 5.4 Claim 2b — Partner-eye ablation (null result, both heads) [DONE]
- 5.5 Claim 3, revised — Both-eyes grading, decomposed [RUNNING]
  - 5.5.1 The original comparison and its confound (head + fusion + aggregation entangled)
  - 5.5.2 Isolating head effect, fusion effect, aggregation effect (2×2 + max/mean/min sweep)
  - 5.5.3 Verdict: (a) fusion stands / (b) it's the ordinal head / (c) max-aggregation bias — reported per the data, all three backbones
- 5.6 Phase 6 — Initial fine-tuning (headline 0.555, app 0.613; P1≈P2 under fine-tuning) [DONE]
- 5.7 Phase 6b — Converged fine-tuning and the Claim 2 retest [PENDING]
  - 5.7.1 Training/validation QWK curves, convergence check at 40 epochs
  - 5.7.2 Sampler ablation — class_balanced vs. none vs. sqrt_inverse, per-class recall
  - 5.7.3 Does the P1-vs-P2 null survive full convergence? (direct test of §5.3's finding)
  - 5.7.4 Re-run Claim 2b ablation on the converged P1 model
- 5.8 Phase 7 — Calibration and reject option [PENDING]
  - 5.8.1 Temperature scaling, ECE/MCE before vs. after (P2 test + both P3 tests)
  - 5.8.2 Cross-dataset calibration transfer (expected to degrade — reportable if so)
  - 5.8.3 Confidence (max softmax) vs. predictive entropy as reject signals
  - 5.8.4 Risk–coverage curve, pre-registered τ rule (val rDR sensitivity ≥ 0.90)
  - 5.8.5 Grad-CAM qualitative panel, successes and failures
- 5.9 GitHub split-practice survey (20 repos, image- vs. patient-level fraction) [PENDING]

## 6. The Application [PENDING]
- 6.1 Release artifacts — `best_model.pt`, `config.yaml`, `thresholds.json`, `MODEL_CARD.md`
- 6.2 Gradio interface — single-eye mode, bilateral mode (Phase 5b's winning method)
- 6.3 On-screen disclaimer and referral logic
- 6.4 Acceptance Test 12.1 — app vs. eval-pipeline prediction match (100 images, exact match required)

## 7. Discussion
- 7.1 What the data actually show, vs. the original plan
- 7.2 Reframing Claim 3 per the Phase 5b verdict
- 7.3 Reframing Claim 2 in light of Phase 6b's converged re-test
- 7.4 Practical implications for DR screening pipelines (bilateral grading, reject option)
- 7.5 Practical implications for the wider DR literature (the survey's framing: "less harmful than assumed," not "everyone is wrong")

## 8. Limitations
- Labels noisy/single-grader · APTOS unpaired · single screening population · frozen-feature underestimation · Claim 2 null may be regime-specific (cite Phase 6b curves as evidence for/against) · app-config fine-tune single-run · no prospective validation · single-field imaging · research prototype, not a medical device

## 9. Conclusion

## References
[1] EyePACS (Kaggle, 2015) · [2] APTOS 2019 (Kaggle) · [3] timm · [4] PyTorch · (add Grad-CAM, calibration citations once Phase 7 methods are finalized)

## Appendix A — Full results tables (script-generated from `results/*.json`, never hand-copied)
## Appendix B — GitHub survey aggregate table (no individual repos named)

---

### Figure list (final numbering, pending Phase 7–9 additions)
1. Stratified inter-eye correlation (Claim 1b)
2. Protocol comparison, P1/P2/P3 (Claim 2)
3. Partner-eye ablation (Claim 2b)
4. Claim 3 decomposed — full head × fusion × aggregation matrix
5. Claim 3 decomposition — paired effect sizes (head vs. fusion vs. aggregation)
6. Headline fine-tune summary (Phase 6)
7. Headline vs. app config (Phase 6)
8. Phase 6b convergence curves (train/val QWK vs. epoch, both samplers)
9. Risk–coverage curve (Phase 7)
10. Reliability diagrams, before/after temperature scaling (Phase 7)
11. Grad-CAM panel, including failures (Phase 7)
12. App screenshot (Phase 8)
