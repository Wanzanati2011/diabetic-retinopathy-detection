Two Eyes, One Patient — Inter-Eye Correlation in Diabetic Retinopathy and Its Consequences for Model Evaluation
Slide-by-slide outline. Status noted per slide: [DONE] real data now / [RUNNING] Phase 5b in progress / [PENDING] awaiting Phase 6b-9.

Slide 1 — Title
Two Eyes, One Patient
Inter-Eye Correlation in Diabetic Retinopathy and Its Consequences for Model Evaluation
Y. L. Vishal — August 2026

Slide 2 — The Problem
Diabetic retinopathy is silent until damage is irreversible; one annual photo catches it early.
Bottleneck: grading needs scarce specialist graders.
Published automated-grading results are often implausibly high, sometimes exceeding human-grader agreement.
Suspected cause: every patient has two eyes sharing one disease process — split by image, not patient, and one eye's answer leaks into training for its twin.

Slide 3 — The ICDR Grading Scale
0 No DR, 1 Mild NPDR (microaneurysms only, ~10px), 2 Moderate, 3 Severe, 4 Proliferative.
Referable DR = grade >= 2 — the decision that drives real clinical action.

Slide 4 — Three Core Claims
Claim 1 (Measurement): inter-eye correlation is real and large.
Claim 2 (Consequence): image-level splitting inflates scores vs. patient-level.
Claim 3 (Fix): grade patients using both eyes jointly.
Result up front: 1/1b confirmed and sharpened. 2/2b: genuine null result, re-tested in Phase 6b. 3: under active re-examination (Phase 5b) to isolate what's actually driving the gain.

Slide 5 — Data and Splits
EyePACS: 35,126 images, 17,563 patients, both eyes each. APTOS: 3,662 images, unpaired.
8 flat-black images excluded pre-split; 0/8 touched the primary test set.
148 near-duplicates found (dedup), all within APTOS.
P1 (image-level, leaky, never optimized) / P2 (patient-level, honest, primary) / P3 (cross-dataset, both directions).

Slide 6 — Claim 1: Inter-Eye Correlation
QWK 0.855 (95% CI 0.846-0.863), n=17,555 patients.
Label-only lookup (zero pixels): QWK 0.838 — the shortcut ceiling.
Published models: 0.85-0.92 — uncomfortably close to a zero-pixel baseline.
Verdict: STRONG.

Slide 7 — Claim 1b: Survives Stratification
Objection pre-empted: not just "healthy-healthy pairs" (73% grade 0).
QWK holds at every severity level: 0.656 (>=1), 0.538 (referable, >=2). Permuted-patient null: p ~ 0.
Sharper than predicted: only grade 1 collapses (45.8% lookup accuracy); grades 0, 2, 3, 4 all well predicted (72-94%).

Slide 8 — Claim 2: Does Leakage Inflate Scores?
Frozen features, 5 seeds per protocol.
P1 (leaky): 0.514 / 0.595 (mult/ordinal). P2 (honest): 0.524 / 0.621.
Paired permutation P1 vs P2: p = 0.125 — not significant. P1 is not higher.

Slide 9 — Claim 2b: The Partner-Eye Ablation
Direct mechanism test on one P1-trained model: partner-present (n=3,647) vs partner-absent (n=1,641).
Multinomial: +0.051, p=0.185 (n.s.). Ordinal: +0.061, p=0.0505 (borderline, still n.s.).
Both CIs on the difference include zero. Pre-planned as an acceptable outcome.

Slide 10 — Claim 3 Under the Microscope [RUNNING]
Original comparison (Arm A multinomial vs Arm C ordinal, +0.075) confounds THREE things at once: classifier head, feature fusion, and aggregation rule (max).
Ordinal head alone is worth ~+0.10 elsewhere in this project — is the +0.075 really about both eyes, or just the head?
Phase 5b isolates head effect / fusion effect / aggregation effect (max vs mean vs min) across all 3 backbones. Verdict reported honestly, whichever the data shows.

Slide 11 — Phase 6: Real Fine-Tuning
Headline (224px, 4 runs): QWK 0.548-0.561, mean 0.555. App (384px, 1 run): 0.613.
Same P1~P2 null pattern holds under fine-tuning as under frozen features (diff +0.0004).
Both configs land in the "undertrained" acceptance band (0.40-0.70) — a resource-constrained budget (15 epochs, laptop GPU), not a bug.

Slide 12 — Phase 6b: Does the Null Survive Real Convergence? [PENDING]
Best epochs were 12-14 of 15 — models were still improving. Re-running at 40 epochs, patience 8, higher backbone LR, sampler ablation (class_balanced vs none vs sqrt_inverse), 8 runs total.
This is a direct test of Claim 2's null: if a converged model shows a P1/P2 gap, the paper's headline reverses back toward Claim 2.
Will report train/val QWK curves and whether the null holds.

Slide 13 — Phase 7: Calibration and Reject Option [PENDING]
Temperature scaling on validation only; ECE/MCE before/after, in-domain and cross-dataset.
Reject option: confidence vs. predictive entropy; risk-coverage curve; tau chosen by a pre-stated rule (val rDR sensitivity >= 0.90).
Grad-CAM panel including failure cases.

Slide 14 — Phase 8: The Application [PENDING]
Gradio app, imports the exact same preprocess() used in training.
Bilateral mode demonstrates the paper's own both-eyes finding.
On-screen disclaimer; confidence or explicit UNCERTAIN; Acceptance Test 12.1 (100 images, app must match eval pipeline exactly).

Slide 15 — Phase 9: The Survey and the Real Headline [PENDING]
GitHub survey: 20 DR-grading repos, image-level vs patient-level split practice, aggregate only.
Framing: image-level splitting is widespread AND (per Claim 2's null) less harmful than assumed here — a more defensible contribution than "everyone is wrong."
Paper restructured around what the data actually showed, not the original plan.

Slide 16 — Limitations
Noisy single-grader labels · APTOS unpaired · one screening population · Claim 2 null may be regime-specific (Phase 6b tests this directly) · single fine-tune run for the app config · no prospective validation · research prototype only.

Slide 17 — Summary
Claim 1/1b: strong, refined correlation. Claim 2/2b: honest null, re-tested at full convergence. Claim 3: reframed per rigorous decomposition, not defended as originally stated. Phases 7-9 complete the study: calibration, the app, the survey, and a paper restructured around Claim 3 + the Claim 2 null as the actual headline.
