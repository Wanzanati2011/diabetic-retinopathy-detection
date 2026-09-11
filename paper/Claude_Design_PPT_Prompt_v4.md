# How to use this file

This replaces `Claude_Design_PPT_Prompt_v3.md`. The numbers are the same real results, plus one
set that did not exist when v3 was written (the referable-DR operating point). What changed is the
centre of gravity.

v3 still led with the leakage question, so the deck read as an audit of other people's evaluation
practices with a model attached. v4 leads with the thing that was actually built — a diabetic
retinopathy screening model — reports how well it works at a clinically meaningful operating point,
then explains which design choices got it there. The leakage and correlation work moves to slide 14,
where it belongs: due diligence proving the earlier numbers are trustworthy, plus one genuine
negative finding.

Practical differences from v3:
- Slides 6, 7 and 8 are new: the converged model, its clinical operating point, and how it compares
  to the published DR literature. Slide 7 carries the project's declared primary endpoint, which had
  never been computed before this version.
- Slides 9 and 10 are new: the ordinal head and the sampler ablation, presented as engineering
  findings rather than as confounds inside a leakage decomposition.
- The old slides 6–9 (correlation, leakage, ablation) compress into one slide, 14.
- Slide 15 closes on the pre-registered prediction record, which is more distinctive than a generic
  limitations slide and covers the same ground.

Copy everything below the line into Claude Design as one prompt.

---

Create a 15-slide academic presentation deck titled **"A Deep Learning Pipeline for Diabetic
Retinopathy Screening"**, subtitle "Model Design, Bilateral Grading, and Trustworthy Evaluation on
EyePACS and APTOS". Style: clean academic/scientific, dark navy accent on a white or off-white
background, one clear idea per slide, generous whitespace. Every performance number must appear with
its 95% confidence interval where one exists. No hype language, no exclamation marks, no stock
photography.

This deck is presented to a professor evaluating original engineering and research work, so each
slide should make clear what the author built, chose, or discovered — not only what was observed.
Use first-person framing ("I built", "I measured", "I assumed wrong", "I caught") in the
speaker-facing content. Where a finding contradicts what the author predicted, say so on the slide
rather than hiding it; that honesty is a deliberate feature of this project.

Deck arc, so the design reinforces it: slides 2–5 set up the clinical problem and the system;
slides 6–8 are the model and its performance (this is the centre of the deck and should feel like
the payoff); slides 9–12 are what makes the model work; slides 13–14 are honest limits and
evaluation integrity; slide 15 closes on the prediction record.

## Slide 1 — Title
Title: "A Deep Learning Pipeline for Diabetic Retinopathy Screening"
Subtitle: "Model Design, Bilateral Grading, and Trustworthy Evaluation on EyePACS and APTOS"
Author line: Y. L. Vishal, August 2026

## Slide 2 — The Clinical Problem
Header: "Why automated DR grading, and what it has to get right"
Content:
- Diabetic retinopathy is a leading cause of preventable blindness in working-age adults. It
  progresses silently; by the time a patient notices vision loss the damage is usually permanent.
- The defence is annual retinal photography, graded by a specialist on the 5-level ICDR scale.
  Grade 2 and above is *referable* — the patient is sent to a specialist.
- The number of screening photographs taken each year far exceeds the number of specialists
  available to grade them, and the gap widens as diabetes prevalence rises. That gap is what
  automated grading is for.
- The error costs are wildly asymmetric: a false alarm wastes a clinic appointment, a missed
  referable case can end in irreversible sight loss. This is why the referral decision, not the
  exact 0–4 grade, is what a screening model must be judged on.
- One complication that shaped the whole project: the published DR literature is not a reliable
  yardstick to build against, because several models report accuracy exceeding the measured
  agreement between two human graders on the same photograph. So I built my own evaluation rather
  than inheriting one.

## Slide 3 — What I Built
Header: "The system behind the results"
Content:
- 7,179 lines of Python across 28 modules and 8 test files, organised as numbered phases, each
  writing a versioned JSON results file with its own confidence interval and verdict field.
- A from-scratch patient identity system: EyePACS encodes patient and eye only in filenames, so I
  parsed and validated that into an explicit manifest carrying `patient_id` and `partner_id` on all
  38,788 rows before any patient-level or bilateral work was possible.
- Three train/test split protocols coded from scratch and guarded by 20 acceptance tests (7 on the
  splits, 8 on the manifest, 5 on deduplication).
- A feature-caching layer across three backbone/resolution configurations — EfficientNet-B0 @224px
  and @384px, ResNet-50 @224px — so dozens of controlled comparisons re-fit a light head in seconds
  instead of needing repeated GPU passes.
- Two classifier heads (multinomial logistic regression; ordinal ridge regression with four
  Nelder–Mead-optimised thresholds) and a full end-to-end fine-tuning pipeline with resumable
  checkpointing, mixed precision, EMA weights, and per-epoch train/validation logging.
- Roughly 40 individually trained models sit behind the following slides. Everything ran on a single
  consumer laptop GPU; the longest single training job took five and a half hours.

## Slide 4 — Data and the Grading Scale
Header: "EyePACS, APTOS, and what I had to fix first"
Content:
- Show the ICDR scale as a compact table: Grade 0 No DR / 1 Mild NPDR (microaneurysms only, ~10px
  wide after downsampling) / 2 Moderate NPDR / 3 Severe NPDR / 4 Proliferative DR. Mark grades 2–4
  as referable.
- EyePACS: 35,126 images, 17,563 patients, every patient photographed in both eyes — the primary
  dataset and the only one with reliable eye pairing. APTOS 2019: 3,662 images, unpaired, used only
  where pairing is not needed.
- Severe class imbalance: grade counts 25,810 / 2,443 / 5,292 / 873 / 708. Grade 1 — the class that
  decides whether early disease is caught — has 2,443 examples.
- Cleaning I performed before any modelling: 8 EyePACS images found to be almost entirely flat black
  (failed captures) and excluded, with zero of them touching the primary test set; 148 near-duplicate
  pairs found inside APTOS by perceptual hashing across 131 groups and 270 images.
- Worth flagging: 3 of those 8 black frames carry a human grade of 1 — a grader assigned
  "microaneurysms only" to a near-uniformly black image. This becomes relevant again on slide 13.

## Slide 5 — How I Structured the Evaluation
Header: "Three protocols, a frozen test set, one statistical rule"
Content:
- P2 (patient-level) — both of a patient's eyes always land on the same side of the split. This is
  the honest protocol and every headline number in this deck is a P2 number.
- P1 (image-level) — ignores patient identity. A deliberately flawed protocol I built to measure the
  cost of that mistake, never to optimise against.
- P3 (cross-dataset) — train on one dataset, test on the other, both directions.
- Test sets were declared in the repository at a named git commit before any model scored on
  anything, re-declared at a second commit after the image exclusion with the reason logged, and
  scored once. Primary endpoint fixed in advance: referable-DR sensitivity at a fixed operating point
  under P2. Secondary: five-class QWK.
- One statistical rule, applied without exception: every metric carries a 95% patient-level bootstrap
  confidence interval (resampling patients, so a patient's two eyes always move together), and every
  comparison is a *paired* bootstrap interval on the difference. A gap counts as real only if that
  interval excludes zero.

## Slide 6 — The Model
Header: "Two passes to a converged grader"
Content:
- Pass 1 (15-epoch budget): 0.5550 ± 0.0071 QWK. Clearly undertrained rather than converged — the
  best epoch landed at 11 or later in four of five runs, and the fine-tuned model actually *lost* to
  a much cheaper frozen-feature linear probe at 0.621. A model should not lose to a lighter version
  of itself. My per-epoch logging is what made this visible.
- Pass 2: 40 epochs, early-stopping patience 8, backbone learning rate raised to 1e-4 from 3e-5, plus
  a sampler ablation I added. Eight jobs; all eight stopped on early stopping between epochs 18 and
  35, well short of the ceiling — genuine convergence, not a budget cutoff.
- Present this results table (P2 is the honest protocol and carries the headline):

  | Split | Sampler | Seed 42 QWK | Seed 43 QWK | Referable AUROC |
  |---|---|---|---|---|
  | P1 (image-level) | class-balanced | 0.6509 | 0.6461 | 0.879 / 0.881 |
  | P1 (image-level) | none | 0.6749 | 0.6808 | 0.902 / 0.899 |
  | P2 (patient-level) | class-balanced | 0.6302 | 0.6163 | 0.871 / 0.859 |
  | **P2 (patient-level)** | **none** | **0.6709** | **0.6672** | **0.886 / 0.887** |

- The anomaly resolved: at 0.671 QWK the converged model now clearly beats the frozen probe's 0.621.

## Slide 7 — The Clinical Operating Point
Header: "QWK is the grading metric. This is the clinical one."
Content:
- A screening model is never deployed at its own argmax. It is deployed at a threshold chosen to hit
  a required sensitivity, because missing a referable case costs far more than a false alarm.
- I reduced the five-class softmax to a referable score by summing P(grade ≥ 2), swept the threshold,
  and read off the operating point. This is the project's declared primary endpoint.
- **Referable-DR AUROC 0.887, 95% CI [0.873, 0.900]** on the frozen patient-level test set.
- **Held at 90% sensitivity: 61.3% specificity.** The model catches 9 of every 10 referable patients,
  missing 124 of 1,246, while correctly clearing about three in five of those who do not need
  referral.
- For contrast, at its own argmax the same model sits at 62.3% sensitivity and 95.4% specificity — far
  too conservative for screening. This is exactly why the operating point has to be set deliberately
  rather than inherited from the loss function.
- Be direct about the weakness: 61.3% specificity means roughly two in five healthy patients are
  referred unnecessarily. This is the number I would fix next, and slide 15 says how.
- Suggested visual: an ROC curve with a horizontal line marked at 90% sensitivity and the operating
  point highlighted where it crosses.

## Slide 8 — How This Compares to Published DR Models
Header: "A public-data result, benchmarked against public-data results"
Content:
- Present this comparison table:

  | Study | Training data | Test set | AUC | Sens / Spec |
  |---|---|---|---|---|
  | Gulshan et al. (2016) | Private, ~128k adjudicated | EyePACS-1 (9,963 img) | 0.991 | 97.5 / 93.4 |
  | Gulshan et al. (2016) | Private, ~128k adjudicated | Messidor-2 (1,748 img) | 0.990 | 96.1 / 93.9 |
  | Gargeya & Leng (2017) | Not stated | 5-fold cross-validation | 0.97 | 94 / 98 |
  | Voets et al. (2019) | Public (Kaggle EyePACS) | Kaggle EyePACS test | 0.951 | 90.6 / 84.7 |
  | Voets et al. (2019) | Public (Kaggle EyePACS) | Messidor-2 | 0.853 | 81.8 / 71.2 |
  | **This work** | **Public EyePACS, patient-level** | **Frozen P2 test (5,814 img)** | **0.887** | **90.0 / 61.3** |

- The gap between Gulshan's 0.99 and everything below it is a *data* gap, not a modelling gap: that
  result rests on ~128,000 images graded by multiple ophthalmologists with adjudication. Voets et al.
  — the only independent reproduction on public data — could not approach it with public data alone.
- My 0.887 sits between Voets' two numbers, which is where a public-data result should sit.
- My specificity at matched sensitivity is clearly worse than Voets', and I would rather explain that
  than bury it. Three plausible causes: they trained a purpose-built *binary* referable classifier
  while mine is a five-class grader whose referable score is derived by summing the tail; they used
  higher input resolution (my own 224→384 comparison suggests resolution alone is worth several
  points); and my test set is patient-level, a stricter evaluation than the image-level Kaggle split.
  The first two are fixable with compute I did not have.

## Slide 9 — What Actually Drives Performance: The Classifier Head
Header: "The largest and cheapest lever I found"
Content:
- Four levers were available — classifier head, input resolution, backbone, class balancing — and
  they are not equally important. The ranking surprised me.
- Switching from a multinomial head to an ordinal one, holding everything else fixed, is worth
  **+0.050 QWK on EfficientNet-B0 @224, +0.048 at 384px, and +0.091 on ResNet-50 @224**. All three
  paired bootstrap intervals exclude zero. It is the only single design choice that is significant on
  every backbone I tested.
- It costs essentially nothing: a ridge regression on the numeric grade plus four decision thresholds
  optimised by Nelder–Mead on the training set only, instead of a softmax.
- Why it works: DR grades are *ordered*. Predicting grade 4 when the truth is grade 0 should be
  penalised far more than predicting grade 1, and a multinomial head treating the five grades as
  unrelated categories throws that structure away.
- Resolution is the second lever: 224px → 384px was worth +0.058 QWK, consistent with the
  microaneurysm argument on slide 13. Single run, so reported as an observation rather than an
  established gain — but it is the cheapest improvement still available to this project.
- The backbone mattered less than the head. Choosing a better head did more for this pipeline than
  choosing a better backbone.

## Slide 10 — What Actually Drives Performance: The Sampler I Was Wrong About
Header: "A standard technique that made my model consistently worse"
Content:
- EyePACS is dominated by healthy eyes, so class-balanced resampling during training is the textbook
  response. I applied it without questioning it.
- Making it an *ablation* rather than a fixed setting was the best decision I made in the training
  pipeline, because it turned out to be wrong.
- Disabling class-balanced resampling improved test QWK in **every one of the four matched pairs**, by
  +0.024, +0.035, +0.041 and +0.051. It improved referable AUROC too, 0.871 → 0.886 on P2.
- My stated hypothesis was that class balancing trades a little overall QWK for better recall on the
  rare Mild NPDR class. The per-class recall data does not support that trade at all: grade-1 recall
  sits between 8.7% and 14.0% under *both* settings. I was paying a real cost for a benefit that was
  never arriving.
- The generalisable lesson, and the one I would put to the audience: make your defaults into
  ablations. I would never have found this if I had left the sampler on as a setting.

## Slide 11 — Grading Both Eyes, and a Confound I Caught in My Own Result
Header: "The flattering number I did not report"
Content:
- Screening photographs both eyes anyway, so the information is free, and a patient's two eyes are
  strongly correlated in grade (QWK 0.855 — slide 14). Joint grading is the obvious thing to try.
- My first attempt looked like a clean win: mean-pooling the two eyes' feature vectors before an
  ordinal classifier reached 0.569 QWK against 0.494 for the conventional per-eye-then-worst-grade
  approach with a multinomial head. **+0.075 QWK, interval [+0.045, +0.105], comfortably excluding
  zero.** I was about to write it up as evidence that using both eyes helps.
- Then I looked at what I had actually changed between those two conditions. Three things at once:
  the classifier head, the fusion method, *and* the aggregation rule. The comparison was real, but it
  was not a test of "does using both eyes help" — which is what I had been about to call it.
- Nobody flagged this for me. Rather than report the flattering number, I built a new experiment to
  find out what was really driving it.

## Slide 12 — The Decomposition
Header: "Two-thirds of the gain was the classifier head, not the second eye"
Content:
- I built a full grid — {per-eye, mean-pooled, concatenated} × {multinomial, ordinal} × {max, mean,
  min aggregation} — with its own patient-level bootstrap interval per cell, on the same 2,629 test
  patients, repeated across all three backbone configurations.
- **Classifier head effect:** +0.050 / +0.048 / +0.091 QWK across the three backbones. Significant
  everywhere.
- **Bilateral fusion effect** (mean-pooling vs per-eye-max, ordinal head held fixed):
  +0.025 [+0.003, +0.047] on EfficientNet-B0 @224 and +0.024 [+0.004, +0.042] at 384px — both
  significant, both small. On ResNet-50: **−0.028 [−0.053, −0.002]**. That interval excludes zero on
  the *wrong side* — pooling the two eyes was significantly *worse* on that backbone.
- Aggregation rule: taking the minimum was significantly worse everywhere, by 0.080 to 0.114 QWK,
  which is what you would hope — "best eye wins" is exactly the wrong rule for screening, and it is
  reassuring that the data agrees with the clinical convention. Concatenating features never helped.
- The honest recommendation is narrower than the one I started with: **use an ordinal head
  unconditionally; add bilateral fusion only after verifying it helps on your specific backbone.**
- Note for the audience: had I run one backbone instead of three, I would have reported the fusion
  benefit as architecture-independent. It is not.

## Slide 13 — Where the Model Breaks Down
Header: "Two limits worth naming"
Content:
- **Grade 1 (Mild NPDR).** Every converged model recalls it at 8.7%–14.0%, far below the 20–45% band I
  predicted in advance, and no sampler setting moved it. On its own that reads as a weak class — but
  two other results point at the same place:
  - It is the only grade that fails to transfer between a patient's two eyes: 45.8% partner-lookup
    accuracy, near chance, while grades 0, 2, 3 and 4 all sit between 72.1% and 94.3%.
  - Three of the eight corrupted images found in cleaning carry a human grade of 1 on a near-black
    frame (slide 4).
- One explanation covers all three: Mild NPDR is defined by microaneurysms roughly ten pixels wide in
  the source image. They do not survive downsampling to 224px, and human graders are themselves
  unreliable on them. So grade 1 may not be a class my pipeline fails to learn — it may be a class the
  data cannot express at this resolution. That distinction matters because the two call for opposite
  responses: more model capacity versus more pixels, or an honest decision to collapse the class.
- **Cross-dataset transfer, sharply asymmetric.** EyePACS → APTOS reaches 0.660 QWK (multinomial) and
  0.754 (ordinal) — the only place in the project a model clears 0.75. The reverse direction collapses
  to 0.287. Training on 3,111 APTOS images and testing on 35,118 EyePACS images is a far harder ask,
  so this is mostly a data-volume effect. What it establishes: a grader trained on one screening
  population should not be assumed to transfer to another without being re-measured.

## Slide 14 — Was My Evaluation Honest?
Header: "The due diligence behind every previous slide — and one negative result"
Content:
- Everything so far is only worth reporting if the evaluation underneath it is sound. Three checks:
- **The correlation that makes patient-level splitting necessary.** One eye's grade predicts its
  partner's at QWK 0.855 [0.846, 0.863] over 17,555 patients; 87.3% exact agreement. Stratifying
  rules out class imbalance as the explanation: 0.656 [0.634, 0.675] restricted to patients with at
  least one eye at grade ≥ 1, and 0.538 [0.510, 0.566] at the referable threshold. Permutation test
  p ≈ 0.
- **A shortcut ceiling worth adopting.** Copy a patient's known grade from one eye onto the other —
  no image, no model, no pixels — and you reach **QWK 0.838**. Any DR model scoring near 0.838 without
  clearly beating it may be riding the inter-eye correlation rather than reading the retina. It costs
  one pass over the labels. I would argue it belongs next to the headline metric in any paper on a
  dataset where subjects contribute more than one correlated sample. Every model in this project sits
  well below it.
- **Does image-level splitting inflate scores?** The leakage literature in retinal OCT and brain MRI
  says it should, and I predicted in writing that it would. It did not. P1 reaches 0.536 / 0.595
  (multinomial / ordinal), P2 reaches 0.559 / 0.621 — if anything the point estimates run the *wrong*
  way, and a paired permutation test gives p = 0.125. A targeted ablation splitting the test set by
  whether a patient's partner eye was in training (n = 3,647 vs 1,641) gives +0.051 (p = 0.185) and
  +0.061 (p = 0.0505), both intervals including zero; grade-matched, +0.018 and +0.036.
- Three separate tests, all null, all pre-registered with an acceptable null outcome stated before
  running. This disagrees with Tampu et al., who measured 0.07–0.43 MCC inflation from the same
  failure mode in retinal OCT. I report it as a real finding, not a failed experiment — and the most
  likely explanation, which I have designed the experiment to test, is that leakage inflation is a
  small-data phenomenon that washes out at 27,000 training images.

## Slide 15 — What I Got Wrong, and What Comes Next
Header: "The prediction record, and the remaining work"
Content:
- Every experiment had its expected outcome written into the plan before it ran, and the scripts carry
  those expectations as fields in their output, so a miss is flagged by the script that produced it.
  Present a compact version of this record:

  | Prediction, written in advance | Expected | Actual |
  |---|---|---|
  | Class balancing buys grade-1 recall | Recall improves | No change; QWK worse |
  | Both-eyes fusion beats single-eye grading | Fusion helps | Mostly the head, not fusion |
  | Image-level splitting inflates scores | P1 higher | P2 higher, p = 0.125 |
  | Grade-1 recall, fine-tuned models | 0.20 – 0.45 | 0.087 – 0.140 |
  | Fine-tuned QWK clears acceptance bar | ≥ 0.70 | 0.555 → 0.681 |

- All three structural predictions were reversed by experiments I designed to be capable of reversing
  them, which is the only reason I know they were wrong.
- **Limitations:** single-grader labels; APTOS ships no eye-pairing metadata; one screening
  population; the 384px result is a single run; 61.3% specificity at clinical sensitivity is not yet
  deployable; no prospective clinical validation — this is a research prototype.
- **Next, in priority order:** calibration and a confidence-based reject option (the highest-value
  remaining work, since the model must be run far from its argmax and a threshold is only as
  trustworthy as the probabilities under it); a grade-1 merge test to settle whether Mild NPDR is a
  measurement ceiling; a training-set-size sweep to test the small-data explanation for the leakage
  null; and a deployable bilateral-aware screening application with Grad-CAM overlays.
- Closing line: every conclusion in this project needed a confidence interval excluding zero before
  being called real — including three of my own predictions, which is why they appear on this slide
  instead of being quietly dropped.
