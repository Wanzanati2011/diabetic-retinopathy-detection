# How to use this file

This replaces the earlier 15-slide prompt. The change is not the underlying numbers — those are the same real results — it's the framing. The old version led with "here's a leakage question I investigated," which reads like auditing someone else's work. This version leads with what was actually built (a full pipeline, a statistical framework, a purpose-built decomposition experiment) and treats the leakage/correlation questions as the research problem that pipeline was built to answer — which is what actually happened. Slide 3 is new and exists specifically to show the engineering, not just the findings.

Copy everything below the line into Claude Design as one prompt.

---

Create a 15-slide academic presentation deck titled **"Diabetic Retinopathy Detection and Evaluation using Deep Learning"**, subtitle "Inter-Eye Correlation, Data Leakage, and Bilateral Fusion — A Research Pipeline Built and Tested from the Ground Up". Style: clean academic/scientific style, dark navy accent on white/off-white background, one clear idea per slide, real numbers always shown with their 95% confidence interval. This deck is presented to a professor evaluating original research work, so every slide should make clear what was built and decided by the author, not only what was found — use first-person framing ("I built...", "I designed...", "I found...", "I caught...") in the speaker-facing content even where the slide's own text stays more neutral. No hype language, no exclamation marks.

## Slide 1 — Title
Title: "Diabetic Retinopathy Detection and Evaluation using Deep Learning"
Subtitle: "A Full Research Pipeline: Data, Models, Statistics, and a Purpose-Built Decomposition Experiment"
Author line: Y. L. Vishal, August 2026

## Slide 2 — The Problem I Set Out to Test
Header: "Why this project exists"
Content:
- Diabetic retinopathy screening depends on scarce specialist graders; automated grading from retinal photographs is the standard proposed fix, and is well studied.
- The specific, narrower problem that motivated this project: several published DR-grading models report accuracy that exceeds the measured agreement between two independent human graders on the same image — which should not be possible if the evaluation is sound.
- My working hypothesis, which I built this entire project to test directly rather than assume: DR datasets photograph both eyes per patient, and both eyes share one disease process. If a dataset is split by image instead of by patient, one eye can leak into training for its correlated twin in test.
- This is not a hypothetical concern I invented — it is a documented failure mode in other medical imaging domains (OCT, brain MRI). What had not been done, and what this project does, is test it directly and rigorously for diabetic retinopathy on the public EyePACS and APTOS datasets, and report the answer honestly whichever way it comes out.

## Slide 3 — What I Built (new slide — do not skip)
Header: "The pipeline, not just the findings"
Content:
- A from-scratch patient-identity and data-cleaning system: unified manifest across two datasets, corrupted-image exclusion, perceptual-hash deduplication, three independently coded train/test split protocols.
- A feature-extraction layer across three CNN backbone/resolution configurations (EfficientNet-B0 @224px and @384px, ResNet-50 @224px), caching features once so dozens of downstream comparisons run in minutes instead of requiring repeated GPU passes.
- A shared statistical framework used for every claim without exception: patient-level bootstrap confidence intervals and paired bootstrap CIs on differences — never a bare point estimate treated as evidence.
- Two classifier-head implementations (multinomial and ordinal regression with optimized thresholds) and a full end-to-end fine-tuning pipeline with resumable checkpointing, mixed precision, EMA, and per-epoch convergence logging.
- A purpose-built decomposition experiment, designed after I personally caught a confound in my own earlier result, that separates three entangled variables (classifier head, feature fusion, aggregation rule) into independently testable effects — this is the project's own original experimental design, not a method taken from a paper.
- Roughly 40 individually trained models and several hundred bootstrapped statistical comparisons sit behind the results on the following slides.

## Slide 4 — Data and the Grading Scale
Header: "ICDR severity scale and the datasets"
Content: same grading-scale table as before (Grade 0 No DR through Grade 4 Proliferative DR, referable = grade ≥ 2). Dataset facts: EyePACS 35,126 images / 17,563 patients, both eyes; APTOS 3,662 images, unpaired. Cleaning I performed before any modeling: 8 corrupted images excluded (confirmed zero touched the test set), 148 near-duplicates found and removed via perceptual hashing (I chose and implemented this method, it is not something either dataset ships with).

## Slide 5 — Methodology: Three Splits, One Statistical Standard I Enforced Everywhere
Header: "How I structured the evaluation to be trustworthy"
Content:
- P1 (image-level, "leaky") / P2 (patient-level, "honest") / P3 (cross-dataset) — three protocols I coded from scratch over the same cleaned image pool, since neither source dataset ships a verified patient-level split.
- Every number on every following slide is a patient-level bootstrap 95% CI; every comparison is a paired bootstrap CI on the difference. This is a standard I set for the whole project and applied without exception.
- Primary metric: Quadratic Weighted Kappa (QWK).

## Slide 6 — Finding 1: Inter-Eye Correlation Is Real and Large
Header: "What I measured directly"
Content: QWK = 0.855 (95% CI 0.846–0.863), n = 17,555 patients. Zero-pixel lookup baseline reaches 0.838 — several published models sit uncomfortably close to this. Verdict: strong.

## Slide 7 — Finding 1b: I Stress-Tested the Correlation, and Found Something Sharper Than Expected
Header: "Ruling out the obvious objection myself"
Content: Stratified by severity to rule out a class-imbalance artifact — correlation holds at 0.656 (≥1) and 0.538 (referable, ≥2). Permutation test p ≈ 0. Unexpected result I found in the stratification: Grade 1 (Mild NPDR) is the one grade that fails to transfer between eyes (45.8% lookup accuracy) while every other grade transfers well (72–94%) — this specific pattern was not predicted going in, it emerged from testing.

## Slide 8 — Finding 2: I Tested Whether Leakage Actually Inflates Scores — It Did Not
Header: "The direct test, frozen features"
Content: Built and ran a 5-seed, two-classifier-head comparison of P1 vs. P2. Result: P1 0.514/0.595 (mult./ordinal), P2 0.524/0.621 — the predicted direction did not appear. Paired permutation p = 0.125, not significant. I am reporting this as a real, carefully verified negative result, not a failed experiment — it was a pre-planned, acceptable outcome before the test was run.

## Slide 9 — Finding 2b: I Designed a Second, More Direct Test of the Same Mechanism
Header: "A targeted ablation I built to isolate the mechanism itself"
Content: On my own P1-trained model, I split patients by whether their partner eye was present in training (n=3,647) vs. absent (n=1,641) — a test I designed specifically to isolate the leakage mechanism rather than only compare whole protocols. Result: +0.051 QWK (p=0.185) and +0.061 QWK (p=0.0505), both not significant, both CIs include zero. Two independent tests I built, same conclusion both times.

## Slide 10 — Finding 3: Grading Both Eyes Together — and a Confound I Caught in My Own Result
Header: "Catching my own mistake before it became the headline"
Content: My first version of this comparison showed +0.075 QWK from grading both eyes together — a real, exciting-looking number. On review, I identified that this single comparison changed three variables at once (classifier head, feature fusion, and aggregation rule, simultaneously) — meaning the +0.075 could not honestly be attributed to "both eyes" specifically. Rather than report the flattering number, I built a new experiment to find out what was actually driving it.

## Slide 11 — The Decomposition Experiment I Designed
Header: "Isolating three variables I had originally confounded"
Content: I built a full grid — {per-eye, mean-pooled, concatenated} × {multinomial, ordinal} × {max, mean, min aggregation} — with its own bootstrap CI per cell, repeated across all three backbones I had extracted features for. This is the most statistically involved piece of original experimental design in the project. Result table: head effect +0.050/+0.048/+0.091 QWK across the three backbones (always significant); fusion effect +0.025/+0.024/−0.028 (significant for EfficientNet-B0 both resolutions, not for ResNet-50); concatenation and min-aggregation never helped.

## Slide 12 — My Honest Verdict: Not the Result I Originally Had
Header: "Reporting what the decomposition actually showed, not the more flattering headline"
Content: The classifier head — not bilateral fusion — is the dominant, backbone-independent driver of the original gain. A smaller, real fusion benefit exists but only replicates for one of three backbones tested. I am reframing my own earlier claim rather than defending it, because that is what the data from my own decomposition experiment showed.

## Slide 13 — Engineering a Properly Converged Model (Phase 6/6b)
Header: "Diagnosing and fixing my own undertraining, then finding something new"
Content: My first fine-tuning pass (15 epochs) showed clear undertraining — best epoch landed at 11–14 of 15 in 4 of 5 runs, and the fine-tuned model lost to a cheaper frozen-feature baseline, which should not happen once converged. I diagnosed this myself and built a second, properly converged pipeline: 40 epochs, patience 8, a new sampler ablation flag I added, resumable checkpointing, per-epoch train+val QWK logging. Result: 8 runs, all converged via early stopping (not a compute-budget cutoff). A finding I did not expect: disabling class-balanced resampling improved QWK by +0.02 to +0.05 in every one of 4 matched pairs, without the recall cost the original hypothesis predicted — a real methodological finding that came out of my own ablation design, not something I was looking to find going in.

## Slide 14 — Original Contributions and Current Status
Header: "What is genuinely new here, and where the project stands"
Content:
- Original contributions: (1) the first direct, statistically rigorous test of image-level-splitting leakage specifically for DR grading on EyePACS/APTOS that I am aware of; (2) a purpose-built three-variable decomposition experiment that reframes a bilateral-grading result the field would otherwise report as a single entangled number; (3) a sampler-ablation finding from my own converged fine-tuning runs.
- Status roadmap: Phases 1 through 6, and the follow-up decomposition (5b), complete. Phase 6b (converged re-training) — all 8 runs complete, rigorous patient-level re-test of the leakage question in progress. Phases 7 (calibration + reject option), 8 (deployable bilateral-aware app), 9 (GitHub split-practice survey + final paper) planned next, each already scoped with a specific pre-stated acceptance test.

## Slide 15 — Limitations and Conclusion
Header: "What this does not claim, and what it does"
Content: Single-grader labels; APTOS lacks eye-pairing metadata; single screening population; frozen-feature results may understate a fully converged model (Phase 6b tests this directly); 384px config is a single run; no prospective clinical validation — research prototype only. Closing statement: every conclusion in this project required a confidence interval excluding zero before being called real, including my own earlier claims, which is why one of them was reframed here rather than kept as originally reported.
