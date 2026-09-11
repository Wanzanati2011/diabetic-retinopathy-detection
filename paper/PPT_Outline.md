# Two Eyes, One Patient — Slide Outline
### Inter-Eye Correlation in Diabetic Retinopathy and Its Consequences for Model Evaluation
**Status: Phases 1–6 complete. Text outline for slide-building — not a built deck.**

---

## Slide 1 — Title
- Two Eyes, One Patient: Inter-Eye Correlation in Diabetic Retinopathy and Its Consequences for Model Evaluation
- Y. L. Vishal — August 2026
- Subtitle: Project status report, Phases 1–6

---

## Slide 2 — The Problem
- Diabetic retinopathy is silent until damage is irreversible; one annual retinal photo catches it early
- Bottleneck: grading requires trained specialists, who are scarce (especially rural areas)
- Automated grading is well-studied — but published results are often implausibly high, sometimes exceeding human-grader agreement
- **Suspected cause:** every patient contributes two eyes sharing the same disease process. Split by image, not by patient, and one eye's answer can leak into training for its twin.

---

## Slide 3 — The ICDR Grading Scale
- 5 levels: No DR (0) → Mild (1) → Moderate (2) → Severe (3) → Proliferative (4)
- Referable DR = grade ≥ 2 — the binary decision that drives real clinical action
- Grade 1 (Mild) is defined by microaneurysms only, ~10px lesions — this becomes important later

---

## Slide 4 — Three Core Claims
- **Claim 1 (Measurement):** inter-eye grade correlation is real and large
- **Claim 2 (Consequence):** image-level splitting inflates reported performance vs. an honest patient-level split
- **Claim 3 (Fix):** grade patients using both eyes jointly, since screening already photographs both
- Two supporting studies: Claim 1b (stratification check) and Claim 2b (a direct ablation of the leakage mechanism)
- **Spoiler, stated up front:** 1 and 1b confirmed and sharpened · 3 confirmed, robustly · 2 and 2b: genuine null result (not a failure — planned for)

---

## Slide 5 — Data
- EyePACS (2015): 35,126 images, 17,563 patients, every patient has both eyes
- APTOS 2019: 3,662 images, no patient/eye IDs — treated as unpaired
- 8 flat-black EyePACS captures excluded before any split was touched (0/8 hit the primary test set)
- Dedup: 148 near-duplicates found, all within APTOS, zero cross-dataset — threshold empirically validated, not just computed
- 3 frozen split protocols: **P1** (image-level, deliberately leaky, never optimized) · **P2** (patient-level, the honest primary protocol) · **P3** (cross-dataset, both directions)

---

## Slide 6 — Claim 1: Inter-Eye Correlation
- QWK between left/right eye grade: **0.855** (95% CI [0.846, 0.863]), n=17,555 patients
- Exact agreement 87.3%, within-1-grade 95.7%
- **Label-only lookup (zero pixels): QWK 0.838** — the shortcut ceiling
- Published image-level-split models commonly report 0.85–0.92 — uncomfortably close to a zero-pixel baseline
- Verdict: STRONG (well above the 0.60 bar, above the pre-registered 0.65–0.80 expectation)

---

## Slide 7 — Claim 1b: Does It Survive Stratification?
- Objection to pre-empt: "it's just healthy-healthy pairs" (73% of eyes are grade 0)
- QWK holds at every severity level: 0.656 (left eye ≥1), 0.538 (referable, ≥2)
- Permuted-patient null: p ≈ 0 at every level — correlation is real, not chance
- **Sharper finding than predicted:** per-grade lookup accuracy is 94/46/73/75/72% (grades 0–4) — only grade 1 truly collapses, not grades 1/3/4 as originally expected
- [Insert Figure 1 — stratified correlation + per-grade heatmap]

---

## Slide 8 — Claim 2: Does Leakage Inflate Scores? (Frozen Features)
- Same classifier heads, trained under P1/P2/P3, 5 seeds each
- P1 (leaky): 0.514 multinomial / 0.595 ordinal
- P2 (honest): 0.524 multinomial / 0.621 ordinal
- **P1 vs P2 paired permutation test: p = 0.125 — not significant.** P1 is not higher.
- Both sit below their own pre-registered targets; only EyePACS→APTOS met its target
- [Insert Figure 2 — protocol comparison bar chart]

---

## Slide 9 — Claim 2b: The Partner-Eye Ablation
- Direct mechanism test: one P1-trained model, same weights, split its own test set by whether the partner eye was in training
- 3,647 partner-present vs. 1,641 partner-absent images
- Multinomial: +0.051 raw diff, p = 0.185 (n.s.) · Ordinal: +0.061 raw diff, p = 0.0505 (borderline, still n.s.)
- Both CIs on the difference include zero
- **This was pre-planned as a possible outcome:** "roughly equal → model isn't exploiting the shortcut → report honestly"
- [Insert Figure 3 — partner ablation]

---

## Slide 10 — Reconciling 2 & 2b: A Genuine Null Result
- Two framings were live: (a) real negative result under frozen features, or (b) wait for fine-tuning, where memorization could reveal leakage
- **Phase 6 settles it:** fine-tuned headline model, same P1 vs P2 test — 0.555 vs 0.555 (diff +0.0004)
- Same null pattern under full fine-tuning as under frozen features
- **Takeaway:** no measurable leakage-driven inflation found, in this data/model setup, under either regime — a real, informative negative result, not a failed experiment

---

## Slide 11 — Claim 3: Both-Eyes Grading (The Fix)
- 4 arms compared on the honest P2 split, patients with both eyes (n=2,629 test patients)
- A: per-eye then worse (standard) = 0.494
- B: concatenated features = worse than A
- **C: mean-pooled features + ordinal head = 0.569 — significantly BETTER than A (+0.075, CI [0.045, 0.105])**
- Effect is ordinal-head-specific — plain softmax shows no benefit
- [Insert Figure 4 — four-arm comparison]

---

## Slide 12 — Claim 3 Is Robust
- Repeated across 3 independent backbone/resolution configs
- effnetb0@224: +0.075 · effnetb0@384: +0.072 · resnet50@224: +0.063
- **Every single CI excludes zero.** Not a fluke of one feature space.
- Strongest, most actionable result in the project — adoptable today by any pipeline that already photographs both eyes
- [Insert Figure 5 — robustness across backbones]

---

## Slide 13 — Phase 6: Real Fine-Tuning
- Headline config (224px): 4 runs (P1×2 seeds, P2×2 seeds) — QWK 0.548–0.561, mean 0.555 ± 0.007
- App config (384px, best config): 1 run — QWK 0.613 (+0.058 over headline, single run, not yet seed-replicated)
- [Insert Figure 6/7 — headline summary + headline-vs-app]
- Both configs land in the "undertrained" acceptance-test band (0.40–0.70), below the 0.70–0.85 target

---

## Slide 14 — Why Is Absolute QWK Below Target?
- Not isolated to fine-tuning — frozen-feature Claim 2 results miss their targets too (project-wide pattern)
- Arguments against a bug: QWK scales predictably with better config (224→0.555, 384→0.613); grade-1 weakness matches a known, expected cause (microaneurysms destroyed by downsampling)
- **Most likely cause:** resource-constrained training — single laptop GPU, 15 epochs, VRAM-limited batch size, vs. pre-registered targets that assumed more headroom
- Doesn't undermine any of the comparative claims (1, 1b, 2, 2b, 3) — those hold regardless of absolute performance

---

## Slide 15 — Limitations
- EyePACS labels noisy, single-grader
- APTOS unpaired — excluded from both-eyes analyses
- One screening population (US teleretinal); may not generalize
- App-config fine-tune is a single run (n=1)
- No prospective clinical validation
- Calibration, the deployable app, and a GitHub split-practice survey (Phases 7–9) not yet done

---

## Slide 16 — What's Next
- Phase 7: temperature-scaled calibration, reject option, Grad-CAM interpretability
- Phase 8: deployable Gradio screening app (with disclaimer, confidence, referral logic)
- Phase 9: survey of public DR-grading repos' split practice + full write-up

---

## Slide 17 — Summary
- **Claim 1/1b:** strong inter-eye correlation (0.855), refined to a single well-characterized failure mode (grade 1)
- **Claim 2/2b:** no significant leakage-driven inflation found, frozen or fine-tuned — an honest, planned-for null result
- **Claim 3:** robust, significant, backbone-independent fix — mean-pool both eyes + ordinal head beats per-eye grading
- **Phase 6:** two real fine-tunes reproduce the null pattern; absolute QWK below target for explainable, non-bug reasons
- Data foundation, frozen-feature claims, and both fine-tunes are done — calibration, the app, and the survey remain
