# How to use this file

Copy everything below the line into Claude Design as one prompt. It's a complete 15-slide brief — every number, every result, every caveat is already filled in from your actual `results/*.json` files (nothing is placeholder). Claude Design will still make its own layout/visual choices; where useful I've suggested a chart type so it has something concrete to draw.

If you want a different title, swap slide 1's title line before pasting — everything else stands on its own regardless of title.

---

Create a 15-slide academic presentation deck titled **"Diabetic Retinopathy Detection and Evaluation using Deep Learning: Inter-Eye Correlation, Data Leakage, and Bilateral Fusion"**, subtitle "Y. L. Vishal — August 2026". Style: clean academic/scientific poster style, dark or navy accent on white/off-white background, one clear idea per slide, real numbers always shown with their 95% confidence interval, never a bare point estimate presented as if it were certain. Use a monospace or slab font for numbers/tables to make them easy to read from the back of a room. This is a research project being presented to a professor, so favor clarity and rigor over marketing polish — no hype language, no exclamation marks, every claim should be immediately followed by the number that backs it.

## Slide 1 — Title
Title: "Diabetic Retinopathy Detection and Evaluation using Deep Learning"
Subtitle: "Inter-Eye Correlation, Data Leakage, and Bilateral Fusion in Automated DR Screening"
Author line: Y. L. Vishal, August 2026
Visual: a single stylized retinal fundus image or eye icon, minimal.

## Slide 2 — The Problem
Header: "Why this matters"
Content:
- Diabetic retinopathy (DR) is a leading cause of preventable blindness. It is asymptomatic until damage is advanced, so annual retinal photo screening is the main tool for catching it early.
- The bottleneck: grading each photo needs a trained ophthalmologist. There are far more screening photos taken than specialists available to grade them — this is exactly the gap automated deep learning models are built to fill.
- The red flag motivating this project: published automated DR-grading models often report accuracy figures that seem implausibly high — in some cases exceeding the agreement rate between two human expert graders looking at the same image.
- The suspected cause: almost every patient contributes two images (left eye, right eye), and both eyes share the same disease process, genetics, and blood sugar history. If a dataset is split by IMAGE instead of by PATIENT, one eye's answer can leak into training data that predicts its twin — inflating the model's apparent accuracy without it actually being smarter.
Visual: simple two-panel diagram — left panel "naive image-level split" showing a patient's two eye photos landing in different (train/test) buckets with a leak arrow between them; right panel "patient-level split" showing both eyes correctly kept together in one bucket.

## Slide 3 — The Grading Scale and the Dataset
Header: "ICDR severity scale and data used"
Content — grading scale table:
- Grade 0: No DR
- Grade 1: Mild NPDR (microaneurysms only, ~10 pixels in a downsampled image — easy to lose)
- Grade 2: Moderate NPDR
- Grade 3: Severe NPDR
- Grade 4: Proliferative DR
- "Referable DR" = grade ≥ 2, the threshold that actually triggers a clinical referral — this is the decision that matters most in practice, more than getting the exact 0–4 grade right.
Dataset facts:
- EyePACS: 35,126 images from 17,563 patients, both eyes photographed per patient.
- APTOS 2019: 3,662 images, unpaired (no reliable eye-pairing metadata).
- Data cleaning performed before any modeling: 8 completely flat-black/corrupted EyePACS images excluded pre-split (confirmed zero of the 8 touched the primary test set); 148 near-duplicate images found via perceptual hashing (all within APTOS) and deduplicated.
Visual: small bar showing dataset sizes (EyePACS 35,126 vs APTOS 3,662) plus the 5-grade severity scale as a horizontal color gradient (green→red) with icons/thumbnails if available.

## Slide 4 — Methodology: Three Splits, One Statistical Standard
Header: "How this study tests for leakage rigorously"
Content:
- Three evaluation protocols, same test images, different training-set construction:
  - **P1 (image-level, "leaky")** — images split randomly, ignoring which patient/eye they belong to. This is the split practice most published DR papers actually use.
  - **P2 (patient-level, "honest")** — every image from a given patient (both eyes) is forced into the same fold. This is the primary, most trustworthy protocol in this study.
  - **P3 (cross-dataset)** — train on one dataset, test on the other (EyePACS→APTOS and APTOS→EyePACS), to check whether findings generalize beyond one source.
- Statistical standard used everywhere in this project: never trust a single point estimate. Every reported number is a **patient-level bootstrap 95% confidence interval** (resampling patients, not images) — and every comparison between two conditions is a **paired bootstrap CI on the difference**, not two separate CIs eyeballed side by side. A gap only counts as "real" if the CI on the difference excludes zero.
- Primary metric: Quadratic Weighted Kappa (QWK) — the standard agreement metric for ordinal grading tasks like this one (rewards being "close", penalizes being far off, more than plain accuracy does).
Visual: simple 3-box diagram of P1 / P2 / P3 showing how patients are assigned to folds in each.

## Slide 5 — Claim 1: Inter-Eye Correlation Is Real and Large
Header: "Claim 1 — how correlated are a patient's two eyes?"
Content:
- Directly measuring how well one eye's grade predicts its partner eye's grade: **QWK = 0.855 (95% CI: 0.846–0.863)**, n = 17,555 patients.
- Even a trivial "label-only lookup" — no image, no pixels, just copying one eye's known grade as the prediction for the other eye — achieves **QWK = 0.838**. This is the shortcut ceiling: any model that reaches numbers near this without clearly beating it may just be exploiting eye-pair correlation rather than reading the image.
- Context: several published DR models in the literature report QWK in the 0.85–0.92 range — uncomfortably close to this zero-pixel baseline.
- Verdict: **STRONG** — this correlation is large enough to be a real methodological concern, not a minor curiosity.
Visual: a simple bar comparing "Real inter-eye QWK: 0.855" vs "Zero-pixel lookup ceiling: 0.838" side by side, plus a scatter or confusion-style heatmap of left-eye grade vs. right-eye grade if available (Figure available: existing project figures on inter-eye correlation).

## Slide 6 — Claim 1b: The Correlation Survives Scrutiny
Header: "Claim 1b — is this just healthy-healthy pairs?"
Content:
- Objection pre-empted: 73% of patients are grade 0 in both eyes, so a naive critic might say the correlation is just "healthy eyes look like healthy eyes."
- Rebuttal: stratifying by severity, the correlation holds even when excluding the trivial healthy-healthy majority — QWK = 0.656 among patients with at least one eye graded ≥1, and QWK = 0.538 restricted to the referable-DR threshold (≥2).
- A permutation test (randomly shuffling which eye belongs to which patient) gives p ≈ 0, confirming the correlation is not a statistical artifact.
- Sharper, unexpected finding: correlation is NOT uniform across severity. Grade 1 (mild) is the one grade that essentially fails to transfer between eyes — only 45.8% lookup accuracy — while grades 0, 2, 3, and 4 are all well predicted from the partner eye (72–94% accuracy).
Visual: bar chart of per-grade lookup accuracy: Grade 0 ~high, Grade 1 ~45.8% (call out as the outlier, different color), Grades 2–4 ~72–94%.

## Slide 7 — Claim 2: Does the Leakage Actually Inflate Model Scores?
Header: "Claim 2 — testing the practical consequence"
Content:
- Given Claim 1's strong correlation exists, the natural prediction is: models trained/evaluated under the leaky P1 protocol should score higher than under the honest P2 protocol.
- Tested with frozen CNN features (EfficientNet-B0) + two classifier heads (multinomial logistic regression, and an ordinal regression head), 5 random seeds per protocol:
  - P1 (leaky): 0.514 QWK (multinomial) / 0.595 QWK (ordinal)
  - P2 (honest): 0.524 QWK (multinomial) / 0.621 QWK (ordinal)
- Paired permutation test, P1 vs. P2: **p = 0.125 — not statistically significant**. P1 is not measurably higher than P2; if anything the point estimates run slightly the other way.
- This is a genuine, honestly-reported **null result**: despite Claim 1's strong correlation, splitting by image instead of by patient did not measurably inflate scores in this setup.
Visual: grouped bar chart, P1 vs P2, multinomial vs ordinal, four bars with the two P1/P2 pairs adjacent for easy visual comparison; annotate "n.s., p=0.125" between the P1/P2 pair.

## Slide 8 — Claim 2b: A Direct Mechanism Test
Header: "Claim 2b — the partner-eye ablation"
Content:
- A more targeted test of the leakage mechanism itself: on a single P1-trained model, compare patients whose partner eye WAS present somewhere in the training set (n = 3,647) against patients whose partner eye was ABSENT from training (n = 1,641). If leakage were real and helping, the "partner present" group should score noticeably higher.
- Multinomial head: +0.051 QWK difference, p = 0.185 (not significant).
- Ordinal head: +0.061 QWK difference, p = 0.0505 (borderline, still technically not significant at the 0.05 threshold).
- Both confidence intervals on the difference include zero.
- This was a pre-registered, acceptable possible outcome — not a failed experiment. Combined with Claim 2, this builds a consistent picture: the theoretically-expected leakage effect is small enough that it does not clearly show up under this training regime.
Visual: two side-by-side box/violin-style comparisons (partner-present vs partner-absent) for multinomial and ordinal heads, with the p-values labeled directly on the chart.

## Slide 9 — Claim 3 (Original): Can Grading Both Eyes Together Help?
Header: "Claim 3 — using both eyes as a diagnostic advantage instead of a leak"
Content:
- Reframe: instead of treating inter-eye correlation only as a threat, can it be used deliberately as a feature — grading a patient using both eye images jointly, the way a real ophthalmologist would consider both eyes for one patient?
- Original comparison: "Arm A" (grade each eye separately with a multinomial head, then take the worse/max grade as the patient's label) scored **0.494 QWK**, vs. "Arm C" (average the two eyes' feature vectors together before classifying, using an ordinal head) which scored **0.569 QWK** — an apparent **+0.075 QWK gain** from using both eyes.
- The problem, caught during review: this comparison changes THREE things at once — the classifier head (multinomial → ordinal), the fusion method (per-eye → feature-averaging), AND the aggregation rule (max → implicit single joint prediction) — all in one comparison. Since the ordinal head alone is known (from Claim 2) to be worth roughly +0.10 QWK on its own, the "+0.075 both-eyes gain" could be entirely due to the head switch, not to actually using both eyes.
Visual: a "confound" diagram — one arrow from Arm A to Arm C labeled with three overlapping tags (HEAD, FUSION, AGGREGATION) to visually show three variables changing in one step.

## Slide 10 — Phase 5b: Decomposing the Confound
Header: "Isolating what's actually driving the +0.075 gain"
Content:
- A full decomposition was run, holding every variable fixed except one at a time, with a paired bootstrap 95% CI on every comparison, repeated across three backbone configurations (EfficientNet-B0 @224px, EfficientNet-B0 @384px, ResNet-50 @224px):
- **Head effect** (ordinal vs. multinomial, same per-eye-max method): **+0.050 (b0@224), +0.048 (b0@384), +0.091 (resnet50@224)** — QWK gain, all three statistically significant. This is the single largest, most consistent effect found.
- **Fusion effect** (mean-pooled features vs. per-eye-max, both ordinal): **+0.025 (b0@224, significant), +0.024 (b0@384, significant), −0.028 (resnet50@224, NOT significant)**.
- **Concatenated-feature fusion**: never helps in any of the three backbones — flat or significantly worse every time.
- **Aggregation rule** (mean vs. max, min vs. max): min-aggregation is significantly WORSE than max in every backbone; mean is never significantly different from max in any backbone.
Visual: a table or forest plot with 3 columns (one per backbone) and rows for head effect / fusion effect / concat effect / mean-agg effect / min-agg effect, each cell showing the diff and whether the CI crosses zero (color-code significant green, not-significant gray).

## Slide 11 — Phase 5b Verdict: The Honest, Undefended Conclusion
Header: "What the data actually says — not what the original plan assumed"
Content:
- The ordinal classifier head is the dominant, fully consistent driver of the original "+0.075" result — present and significant in all three backbone configurations tested.
- Feature-level fusion (averaging both eyes' features before classifying) adds a further small, real, statistically significant benefit **only for EfficientNet-B0** (+0.024 to +0.025 QWK on top of the head effect) — but this fusion benefit does **not replicate for ResNet-50** (numerically negative, not significant).
- Stated plainly, without defending the original framing: **this is not a clean "bilateral fusion wins" story.** The correct, honest framing is that switching to an ordinal head accounts for most of the original claimed gain everywhere, and a smaller additional both-eyes benefit exists but is backbone-dependent, not universal.
- This reframes Claim 3 from "both eyes beat one eye" to a more precise and more defensible claim: "an ordinal classifier head is the primary driver of grading improvement; joint bilateral feature fusion provides a smaller, architecture-dependent additional benefit."
Visual: a simple annotated diagram splitting the original +0.075 gain into two stacked segments — a large "head effect" segment and a small, dashed/uncertain "fusion effect (backbone-dependent)" segment on top.

## Slide 12 — Phase 6: Real End-to-End Fine-Tuning
Header: "Moving from frozen features to a real trained model"
Content:
- All results so far used frozen, pretrained CNN features with a lightweight classifier on top (cheap, fast to iterate). Phase 6 tested real end-to-end fine-tuning of the whole network.
- Headline configuration (EfficientNet-B0 @224px, 15 epochs, 4 runs across P1/P2 × 2 seeds): QWK range 0.548–0.561, mean **0.555**.
- App-config run (EfficientNet-B0 @384px, 1 run): **0.613**.
- The same P1 ≈ P2 pattern seen with frozen features holds under fine-tuning too (difference of only +0.0004 — essentially zero).
- An honest caveat flagged during review, not hidden: these fine-tuning runs are under-trained, not resource-broken. The best-performing epoch out of 15 landed at epochs 11–14 for four of the five runs — meaning the model was still improving when training was stopped on a limited compute budget (a single laptop GPU). This falls in the expected "undertrained" QWK band (0.40–0.70) per this project's own pre-registered acceptance-test thresholds, and is the direct motivation for the next phase.
Visual: simple bar of the 5 fine-tune runs' QWK values, with a horizontal dashed line marking "0.621" (the frozen ordinal linear-probe result from Claim 2, for comparison) to visually show the fine-tuned model currently trailing the much cheaper frozen baseline.

## Slide 13 — Current Status and What's Next
Header: "Where this project stands right now"
Content — status table, three columns (Phase / Status / What it tests):
- Phases 1–4 (data, splits, inter-eye correlation measurement) — **DONE**
- Claim 1 / 1b (correlation is real and robust) — **DONE, STRONG**
- Claim 2 / 2b (does leakage inflate scores?) — **DONE, genuine null result under current (undertrained) training regime**
- Claim 3, decomposed (Phase 5b) — **DONE this week** — reframed as an ordinal-head effect with a smaller, backbone-dependent fusion benefit
- Phase 6 (initial fine-tuning) — **DONE** — undertrained, motivating a re-test
- **Phase 6b (in progress)** — re-running fine-tuning properly converged (40 epochs vs. 15, higher backbone learning rate, sampler ablation) specifically to re-test whether Claim 2's null result survives full convergence — if a real patient-level gap appears once the model is properly trained, the paper's headline finding reverses back toward "leakage matters."
- **Phase 7 (planned)** — model calibration (temperature scaling), a confidence-based reject option for uncertain cases, and Grad-CAM visual explanations.
- **Phase 8 (planned)** — a deployable screening application (Gradio interface) with bilateral (both-eyes) mode, on-screen medical disclaimer, and an exact-match acceptance test between the app and the evaluation pipeline.
- **Phase 9 (planned)** — a survey of public GitHub DR-grading repositories to measure how common image-level (leaky) splitting actually is in the wider field, and a full paper rewrite centered on what the data actually showed.
Visual: a horizontal roadmap/timeline bar with 9 segments, color-coded done (green) / in progress (yellow) / planned (gray).

## Slide 14 — Limitations
Header: "What this study does not claim"
Content:
- Labels come from single-grader annotations, not multi-grader consensus — some inherent label noise is expected.
- APTOS 2019 lacks reliable eye-pairing metadata, so cross-dataset (P3) analysis cannot use bilateral methods there.
- Data comes from a single screening population/context; generalization to other populations or camera equipment is untested.
- Frozen-feature experiments likely understate what a fully fine-tuned, converged model can do — this is exactly what Phase 6b is designed to test directly.
- The Claim 2 null result may be specific to this project's current (undertrained) training regime, not a universal finding — Phase 6b's convergence curves will provide direct evidence for or against that possibility.
- The 384px "app config" fine-tune is currently a single run, not yet repeated across seeds.
- No prospective clinical validation has been performed; this is a research prototype, not a validated medical device.
Visual: none needed, or a simple muted icon list — this slide should read as measured and careful, not defensive.

## Slide 15 — Summary and Key Takeaways
Header: "What we've learned so far"
Content:
- Inter-eye correlation in diabetic retinopathy is real, large, and robust to stratification (Claims 1/1b) — QWK 0.855, holding at 0.538–0.656 even excluding trivial healthy-healthy pairs.
- Despite that correlation, image-level ("leaky") data splitting did not measurably inflate this project's model scores under the current training regime (Claims 2/2b) — a genuine, carefully-tested null result, currently being re-verified under full model convergence (Phase 6b).
- The apparent benefit of grading both eyes together is mostly explained by a better classifier head, not by bilateral fusion itself — with a smaller, real, but architecture-dependent fusion benefit found only in one of three backbones tested (Phase 5b decomposition).
- Every one of these conclusions was reached by requiring a statistically significant confidence interval before calling any effect "real" — this project treats "no significant difference" as a legitimate, reportable finding, not a failure.
- Remaining work (Phase 6b–9) will determine whether the null result survives full convergence, add calibration and a reject option for clinical safety, ship a working bilateral screening app, and situate these findings against how the wider DR research community actually handles this problem.
Visual: three-icon summary row (correlation / leakage-null / bilateral-reframe), clean closing slide, contact/thank-you line at the bottom.
