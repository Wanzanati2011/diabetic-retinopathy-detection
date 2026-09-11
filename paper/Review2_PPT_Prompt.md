# Review 2 PPT — Claude Design Prompt (25 slides)

## READ THIS FIRST (do not paste this part)

**Status of the content:**

1. **Both guides are on slide 1** — Dr. B V A N S S Prabhakar Rao and Dr. Sridevi S. Venue AB1-302.
2. **All 22 works are verified** against Crossref metadata — author lists, journals, volumes, years,
   pages, DOIs. Ten appear in the detailed table on slides 6-7; the other twelve are synthesised on
   slides 8-9. Every one is in the reference list on slide 25.
3. **All results are from your own JSON output files** and are accurate as of your latest run.

**Two things you must be able to defend:**

- **The app is not built.** The model layer inside it is built and measured. Slide 15 says this
  explicitly. Say "the engine is done and measured, the interface is the final phase." You are at
  6 of 9 phases, well past the 30% required.
- **The grade-1 contrast on slide 22.** Yan et al. (2026) report pooled Grade-1 sensitivity of
  72.1%; yours is 8.7-14%. Your answer is on the slide: only 12.2% of the studies they pooled used
  external validation, and your number comes from a frozen patient-level test set scored once.
  Rehearse this one - it is the most likely place a panel member pushes.

**If your slot is short**, present 1-7, 10, 12, 13, 17, 18, 19, 20, 24, 25 and flip to the rest on
questions. Do not delete slides - panels mark on completeness.

---

Copy everything below the line into Claude Design as one prompt.

---

Create a 25-slide academic project review presentation. Style: clean, formal, university review
panel. Dark navy (#1B2A4A) accent on white. Generous whitespace, one idea per slide, readable from
the back of a room - minimum 18pt body text. Every performance figure must carry its 95% confidence
interval where one exists. No hype words, no exclamation marks, no stock photos. Where I give table
data, render a real table - do not convert it to bullets.

Narrative the design must reinforce, in order:
**Problem -> Evidence -> Existing Solutions -> Research Gap -> Proposed Solution -> Implementation
-> Results -> Novel Contribution.** Slide 4 is the evidence, slide 10 is the gap, slide 25 is the
contribution. Those should feel like the structural pillars.

---

## Slide 1 — Title
- Project Title: **"AI-Assisted Diabetic Retinopathy Referral Screening for Low-Resource Settings"**
- Subtitle: "A patient-level deep learning pipeline and referral decision engine for non-specialist screening"
- Register Number: **22MIA1073**
- Student Name: **Y. L. Vishal**
- Guides: **Dr. B V A N S S Prabhakar Rao** and **Dr. Sridevi S**
- Footer: Mini Project — Review 2 — September 2026

## Slide 2 — Outline
Numbered contents, two columns:
Introduction · Problem and Evidence · Problem Statement and Scope · Literature Review (22 works) ·
Research Gap · Research Challenges · Research Objectives · Proposed Architecture · Methodology ·
Implementation Progress · Results and Discussion · Comparison with Existing Works · Novel
Contribution · Conclusion · Limitations and Future Work · References

## Slide 3 — Introduction: The Disease and Where Screening Breaks
Top half — the ICDR scale as a compact table:

| Grade | Name | Defining lesions | Action |
|---|---|---|---|
| 0 | No DR | None visible | Routine recall |
| 1 | Mild NPDR | Microaneurysms only (~10 px wide after downsampling) | Routine recall |
| 2 | Moderate NPDR | Haemorrhages, exudates | **Refer** |
| 3 | Severe NPDR | Extensive lesions, high progression risk | **Refer** |
| 4 | Proliferative DR | Abnormal new vessel growth, sight-threatening | **Refer urgent** |

Bottom half — a 4-stage horizontal flow with the third stage highlighted in red:
**Patient with diabetes → Fundus photograph captured by a technician → GRADED BY A TRAINED
SPECIALIST → Referred / recalled**
- DR is asymptomatic early. Vision loss appears only after damage is largely irreversible, so
  patients do not self-present in time. Annual fundus photography is the defence.
- Stages 1, 2 and 4 scale cheaply — cameras are portable and a technician learns one in days.
  **Stage 3 does not scale.** It needs an ophthalmologist or certified grader. That is the step this
  project targets.
- Grade ≥ 2 is *referable DR*. The referral decision, not the exact grade, is what changes care —
  and the errors are asymmetric: a false referral wastes an appointment, a missed referable case can
  end in blindness. The system must be built sensitivity-first.

## Slide 4 — Establishing the Problem: The Evidence
Header: "The burden is growing; grading capacity is not, and it is in the wrong places"
Three large statistics across the top, each with its citation beneath:
- **103.12 million** adults had DR in 2020 (global prevalence **22.27%** among people with
  diabetes), projected to **160.50 million** by 2045 *(Teo et al., Ophthalmology, 2021)*
- Vision-threatening DR: **28.54M → 44.82M** over the same period *(Teo et al., 2021)*
- **1 of 82** — validation studies of regulator-approved DR AI systems drawn from a low-income
  setting *(Wang et al., npj Digital Medicine, 2026)*

Then three supporting points:
- That 2026 meta-analysis covered **82 studies, 887,244 examinations, 25 approved devices, 28
  countries**. Roughly **two-thirds of validation came from high-income economies** and only **44%
  used a fully external test set**.
- India's SMART India population study found DR prevalence stratified sharply by urban–rural
  location and socioeconomic index, including a large pool of *undiagnosed* diabetes carrying
  retinopathy — people not in any screening programme *(Raman et al., Lancet Global Health, 2022)*.
- Where Indian public-health deployment has been evaluated *(Duggal et al., JMIR Medical
  Informatics, 2025)*, the analysis reports aggregate accuracy and says little about behaviour on
  low-confidence or ungradable images — exactly the behaviour a technician without a grader to fall
  back on depends on.
- Key line: **the tools exist, but they have been validated almost entirely where the specialists
  already are.**

## Slide 5 — Problem Statement and Scope
Boxed problem statement at the top:

> **In rural and low-resource screening settings, fundus photographs can be captured but cannot be
> graded, because trained graders are concentrated in urban tertiary centres. The screening
> programme fails at the interpretation step, not the imaging step.**

- **Why this problem:** the imaging problem is solved — portable cameras are cheap and a technician
  learns one quickly. The interpretation problem is not.
- **Why it matters:** DR is asymptomatic until too late, so a patient who is photographed but never
  graded gains nothing from having been screened.
- **What I am building:** a **referral decision engine** a non-specialist can operate — load the
  photograph, get a referable / not-referable decision with a confidence score, and an explicit
  *"uncertain — send to a human"* flag when the model should not be trusted. The abstention is
  central, not an add-on: a tool used by someone who cannot check its answer must be able to say when
  it does not know.
- **A second reason I built my own evaluation** rather than benchmarking against published numbers:
  reported accuracies in this field reach **99.36% for four-stage DR grading** *(Akhtar et al.,
  Scientific Reports, 2025)*, above the level at which two human graders agree on the same
  photograph. I am not claiming any paper is wrong — I am saying published accuracy alone is not a
  safe design target, so I built a frozen, patient-level evaluation and report every number with a
  confidence interval.
- **In scope:** referable-DR detection (grade ≥ 2) primary, 5-class ICDR grading secondary; public
  datasets only; patient-level evaluation; calibration and a reject option; technician interface.
- **Out of scope:** lesion segmentation, OCT, prospective trial, treatment advice, any claim of
  diagnostic or regulatory validity. Research prototype.
- **Constraint that shaped every decision:** all training on a single consumer laptop GPU —
  deliberate, because a method needing a compute cluster is not one a low-resource programme can
  retrain.

## Slide 6 — Literature Review (1/2): Core Works
Render as a five-column table, small but legible font. Do not convert to bullets.

| S.No. | Author(s), Year | Method / Approach | Key Contribution | Gap Identified |
|---|---|---|---|---|
| 1 | Gulshan et al., 2016, *JAMA* 316(22) | Inception-v3, 128k private images, multi-grader adjudicated labels | Referable-DR AUC 0.991 (EyePACS-1), 0.990 (Messidor-2); 97.5% sens / 93.4% spec | Depends on large private adjudicated data unavailable to public-data projects; no uncertainty output |
| 2 | Gargeya & Leng, 2017, *Ophthalmology* 124(7) | CNN feature extractor + gradient-boosted classifier | AUC 0.97, 94% sens / 98% spec | Headline metric from 5-fold cross-validation, not a held-out frozen test set |
| 3 | Voets et al., 2019, *PLOS ONE* 14(6) | Independent reproduction of Gulshan using public data only | AUC 0.951 (Kaggle EyePACS), 0.853 (Messidor-2) — could not reach 0.99 | Quantifies the public-data ceiling but does not address deployment or uncertainty |
| 4 | Tymchenko et al., 2020, *arXiv*:2003.02261 | EfficientNet-B4/B5 + SE-ResNeXt50 ensemble, three heads including ordinal regression | Competition QWK ≈ 0.925 on APTOS | Ensemble too heavy for low-resource deployment; split integrity not independently verifiable |
| 5 | Ali et al., 2025, *The Visual Computer* 41(8) | Binocular network grading one eye using features from its fellow | 87.3% accuracy private, 86.5% Kaggle-DR | Reports binocular fusion helps but never decomposes *why*; head effect not separated from fusion effect |

## Slide 7 — Literature Review (2/2): Deployment and Evaluation Integrity

| S.No. | Author(s), Year | Method / Approach | Key Contribution | Gap Identified |
|---|---|---|---|---|
| 6 | Wang et al., 2026, *npj Digital Medicine* 9:110 | Systematic review + meta-analysis of 25 regulator-approved DR AI devices, 82 studies, 887,244 exams | Pooled 0.93 sens / 0.90 spec per patient; most comprehensive evaluation of deployed DR AI | Only **1 dataset from a low-income setting**; two-thirds of studies from high-income economies; only 44% fully external validation |
| 7 | Yan et al., 2026, *Frontiers in Endocrinology* 17:1853785 | Systematic review + meta-analysis of DL diagnostic accuracy for DR grading, 41 studies | Pooled per-stage sensitivity: 95.2% (G0), **72.1% (G1)**, 84.3% (G2), 75.8% (G3), 78.8% (G4) | Only **12.2% of studies used external validation**; inadequate reporting of class-imbalance handling; heterogeneous preprocessing |
| 8 | Duggal et al., 2025, *JMIR Medical Informatics* 13:e67529 | Real-world validation and implementation of AI DR screening in **Indian public health settings** | Validates AI screening outside controlled research conditions, in this project's exact target context | Reports aggregate accuracy; limited analysis of behaviour on low-confidence or ungradable images |
| 9 | Tampu et al., 2022, *Scientific Data* 9:580 | Quantified image-level vs patient-level splitting on 3 public retinal OCT datasets | Splitting by image inflated Matthews correlation coefficient by **0.07–0.43** | Demonstrated in OCT only; never tested for fundus DR grading on EyePACS/APTOS |
| 10 | Raman et al. (SMART India), 2022, *Lancet Global Health* 10:e1764–e1773 | Population-based cross-sectional DR screening across India, stratified urban/rural and by socioeconomic index | Establishes DR burden in India including among **undiagnosed** diabetes | Epidemiological only — quantifies the need but proposes no scalable interpretation mechanism |

## Slide 8 — What I Took from the Wider Literature (1/2): Deployment and Access
Header: "Six further works on where DR AI actually reaches patients"
Write as short paragraphs, not a table. Each should read as an understanding, not a summary.
- **Teo et al. (2021), *Ophthalmology* 128(11)** gave me the scale of the problem and, more usefully,
  its shape: the growth is in people needing screening, not in people able to grade. That framing is
  why my project targets interpretation rather than image capture.
- **Nderitu & Keane (2025), *Clin & Exp Ophthalmology* 53(7)** survey which DR AI systems have
  actually reached clinical deployment. My reading: adoption is concentrated in well-resourced
  programmes, and the unsolved barriers are integration, workflow and clinician trust — not raw
  accuracy. That pushed me toward an interpretable, abstaining system rather than a higher AUC.
- **Li et al. (2025), *Aust. J. Rural Health* 33(2)** implement mobile AI-assisted DR screening in
  remote Western Australia — proof the pathway works. But the model is used as a black box and no
  abstention behaviour is reported, which is the gap my reject option addresses.
- **DeLuca et al. (2025), *Current Ophthalmology Reports* 13:6** review AI screening for low-income
  immigrant populations and conclude deployment needs algorithmic-bias and workforce-support work
  alongside the technology. I took this as confirmation that a model reporting honest uncertainty is
  a deployment requirement, not a research luxury.
- **Ashrafzadeh et al. (2026), *Ophthalmology Science* 6:101191** show that prospectively curated
  local data substantially raises AI performance in a resource-limited setting. This is the strongest
  counter-argument to my approach — and its own limitation is my justification: it requires
  prospective local collection and expert curation, precisely what a low-resource programme lacks.
- **Bhoyar & Patel (2026), *Arch. Comput. Methods Eng.* 33:4359–4380** review the field end to end
  and flag dataset imbalance, interpretability and clinical implementation as the unresolved issues.
  All three are objectives in my project.

## Slide 9 — What I Took from the Wider Literature (2/2): Evaluation Integrity and Methods
Header: "Six further works that shaped how I measure, not what I build"
- **Akhtar et al. (2025), *Scientific Reports* 15:3763** report 99.36% accuracy for four-stage DR
  grading using a custom CNN on Messidor-1. I use this as a calibration point for my own scepticism:
  a number above human inter-grader agreement, from a single small dataset with image-level
  evaluation, is what convinced me to freeze my test set before scoring and report intervals rather
  than point estimates.
- **Yagis et al. (2021), *Scientific Reports* 11:22544** independently confirm the leakage failure
  mode in brain MRI — slice-level splitting inflated accuracy, and models reached near-perfect scores
  on randomly labelled data. This gave me the design for my strongest test: the partner-eye ablation,
  which targets the mechanism directly rather than comparing whole protocols.
- **Rouzrokh et al. (2022), *Radiology: AI* 4(5):e210290** list patient-level splitting as required
  practice in medical-imaging ML. My contribution is to convert guidance into measurement: I built a
  deliberately flawed image-level protocol specifically to quantify what ignoring the advice costs.
- **Guo et al. (2017), *ICML* PMLR 70** introduced temperature scaling and showed modern networks are
  systematically overconfident even when accurate. This determines my calibration design directly:
  one parameter, fitted on validation, never on test. It matters more here than usual because my
  model must run far from its argmax to reach clinical sensitivity.
- **Selvaraju et al. (2017/2020), *ICCV* / *IJCV* 128(2)** — Grad-CAM. What I took is a constraint on
  use rather than a method: the explanation panel must include misclassified images, because an
  explanation method applied only to successes cannot tell me anything I do not already believe.
- **EyePACS (2015) and APTOS 2019 (Kaggle)** — the data foundation of this field. What I found in
  them was a gap rather than a resource: neither ships a verified patient-level split, EyePACS hides
  patient identity in filenames, and APTOS exposes no eye-pairing metadata at all. That gap is the
  direct cause of the data work on slide 14.

## Slide 10 — Research Gap
Header: "The gap this project addresses"
Four numbered gaps, each phrased *"X did Y, but did not address Z"*:
1. **Deployment gap.** Wang et al. (2026) evidence 25 approved DR AI devices across 887,244 exams,
   but only one dataset from a low-income setting. Existing systems are not evidenced where the
   grading shortage is worst.
2. **Operating-point gap.** Published work reports AUC. A technician does not use an AUC — they use a
   threshold. Almost no public-data study reports the specificity actually achieved at a fixed
   clinical sensitivity, which is what determines whether a clinic is flooded.
3. **Uncertainty gap.** A tool used by a non-specialist who cannot check its answer must be able to
   abstain. Deployment studies (Nderitu & Keane 2025; Li et al. 2025; Duggal et al. 2025) report
   aggregate accuracy and do not characterise low-confidence behaviour.
4. **Attribution gap.** Ali et al. (2025) report that bilateral fusion improves grading, but change
   the classifier design at the same time, so the improvement cannot be attributed. No published work
   separates the two.
Closing line: gaps 1–3 define what I am building; gap 4 is what my decomposition experiment closes.

## Slide 11 — Research Challenges
Six challenges, one line of problem and one of consequence each:
1. **Severe class imbalance** — 73% of EyePACS images are grade 0; a model can score well by always
   predicting "healthy".
2. **Grade 1 may be below the sensor floor** — Mild NPDR is defined by ~10-pixel microaneurysms that
   do not survive downsampling to 224px.
3. **No patient identity in the public data** — neither dataset ships patient IDs; patient-level
   evaluation is impossible until they are reconstructed.
4. **Silent data corruption** — corrupted captures and near-duplicate images that appear in no loss
   curve but invalidate results.
5. **Single consumer GPU** — no cluster; every experiment had to be affordable, and one training pass
   ran 5.5 hours.
6. **Trusting my own numbers** — with cached features making comparisons nearly free, the risk of
   unconsciously selecting a favourable result is real.

## Slide 12 — Research Objectives
Five objectives, each with its success criterion:
1. Train a referable-DR model to genuine convergence and report its performance at a fixed clinical
   operating point, on a test set frozen before any model scored on it.
2. Establish which design choices actually drive performance — classifier head, input resolution,
   backbone, class balancing — rather than inheriting defaults.
3. Determine whether bilateral (two-eye) grading helps, and decompose any gain into its causes.
4. Characterise the failure modes: which grades cannot be learned, and how performance transfers
   across screening populations.
5. Build an evaluation that can be trusted, and quantify how much the choice of splitting protocol
   changes the reported score.

## Slide 13 — Proposed Architecture
Top-to-bottom block diagram, five labelled layers with arrows. Green tick on layers 1–4, amber
"in progress" marker on layer 5.
- **Layer 1 — Data foundation.** Unified manifest (38,788 rows) · patient identity reconstruction ·
  corrupted-image exclusion · perceptual-hash deduplication · P1/P2/P3 protocols · 20 acceptance tests
- **Layer 2 — Feature extraction.** EfficientNet-B0 @224 · EfficientNet-B0 @384 · ResNet-50 @224 ·
  cached once
- **Layer 3 — Classification.** Multinomial head vs **Ordinal head** (ridge + 4 Nelder–Mead
  thresholds) · optional bilateral mean-pooled fusion
- **Layer 4 — Referral decision engine.** Softmax → P(grade ≥ 2) → threshold set for 90% sensitivity
  → **REFER / NO REFER**
- **Layer 5 — Deployment wrapper (final phase).** Temperature calibration → confidence-based reject
  option → technician interface with Grad-CAM overlay
Side annotation: every layer validated by patient-level bootstrap confidence intervals.

## Slide 14 — Methodology: Data Pipeline and Evaluation Design
**Data.** EyePACS 35,126 images / 17,563 patients (both eyes); APTOS 2019 3,662 images, unpaired.
Combined manifest 38,788 rows. Grade distribution **25,810 / 2,443 / 5,292 / 873 / 708** — verified
against the published EyePACS totals as a correctness check on my parser.
- Patient identity reconstructed from EyePACS filenames into explicit `patient_id` and `partner_id`
  columns on all 38,788 rows.
- **8 corrupted images** (near-uniform black frames) found by a mean-pixel check and excluded before
  splitting; **0 reached the primary test set**. 3 of the 8 carried a human grade of 1 — evidence of
  label noise in the source data.
- **148 near-duplicate pairs** found in APTOS by 256-bit perceptual hashing with LSH banding, across
  131 groups / 270 images; zero in EyePACS, zero cross-dataset. Threshold independently audited
  afterwards across all **616,900,375** EyePACS pairs; the 790 closest candidates were 786 different
  patients with 62.7% grade agreement against a 56.8% chance baseline — coincidental lookalikes, not
  duplicates.

**Evaluation design.** **P2 (patient-level)** — both of a patient's eyes on the same side; the honest
protocol, and every headline number is a P2 number. **P1 (image-level)** — deliberately flawed, built
to measure the cost. **P3 (cross-dataset)** — train on one dataset, test on the other.
- Test sets declared at a named git commit **before any model scored**, re-declared after image
  exclusion with the reason logged, and scored once.
- **One statistical rule, no exceptions:** every metric carries a 95% patient-level bootstrap CI
  (resampling patients so a patient's two eyes move together); every comparison is a *paired*
  interval on the difference; a difference is real only if that interval excludes zero.

## Slide 15 — Methodology: Model Design and the Referral Decision Engine
**Models.** Two families. *Frozen features* — backbone run once, features cached, light head trained
in seconds; this is what made dozens of controlled comparisons affordable on one GPU. *Full
fine-tuning* — end-to-end training producing the deployed model. Three backbone/resolution
configurations so findings can be tested for architecture dependence.
- Two heads: **multinomial** (five grades as unrelated classes) and **ordinal** (ridge regression on
  the numeric grade with four thresholds jointly optimised by Nelder–Mead on training data only).
- Fine-tuning: two-stage LR (head 3e-4, backbone 1e-4, backbone frozen 2 epochs), EMA weights, mixed
  precision, early stopping on validation QWK, resumable checkpointing, per-epoch logging.

**Referral decision engine.** The five probabilities reduce to one referable score,
**P(grade ≥ 2) = p₂ + p₃ + p₄** — using the whole distribution, not just the argmax.
- **A screening model is never deployed at its argmax.** The threshold is chosen to hit a required
  sensitivity. At argmax my model sits at 62.3% sensitivity, far too conservative for screening. Set
  for 90% sensitivity it becomes usable. A deliberate design decision, not a tuning artefact.
- Planned reject option: where calibrated confidence falls below a threshold fixed in advance on
  validation data, the case is flagged **"uncertain — refer to human grader"** rather than decided.
- **Honest status, put this on the slide:** *the decision engine and its operating point are built and
  measured. The technician interface and calibration layer are the final phase, in progress.*

## Slide 16 — Implementation Progress
Header: "Progress: 6 of 9 phases complete"
Horizontal phase tracker, phases 1–6b filled, 7–9 outlined:
Phase 1 Data foundation ✅ · Phase 2 Preprocessing ✅ · Phase 3 Feature extraction ✅ ·
Phase 4 Protocol experiments ✅ · Phase 5 + 5b Bilateral grading and decomposition ✅ ·
Phase 6 + 6b Fine-tuning and converged re-run ✅ · Phase 7 Calibration + reject option ◻ ·
Phase 8 Application ◻ · Phase 9 Practice survey + paper ◻
Supporting figures: **7,179 lines of Python across 28 modules and 8 test files · 20 acceptance tests ·
~40 individually trained models · every result written to a versioned JSON with its own confidence
interval.**

## Slide 17 — Results: The Converged Model
- First pass (15 epochs) reached QWK 0.5550 ± 0.0071 — and was **undertrained, not converged**.
  Detected because the cheap frozen-feature probe (0.621) beat the fine-tuned model (0.555), which
  cannot happen once converged. Best epoch fell at 11 or later in 4 of 5 runs.
- Second pass: 40 epochs, early-stopping patience 8, backbone LR raised to 1e-4. All 8 runs stopped by
  early stopping between epochs 18 and 35 — genuine convergence, not a budget cutoff.

| Split | Sampler | Seed 42 QWK | Seed 43 QWK | Referable AUROC |
|---|---|---|---|---|
| P1 (image-level) | class-balanced | 0.6509 | 0.6461 | 0.879 / 0.881 |
| P1 (image-level) | none | 0.6749 | 0.6808 | 0.902 / 0.899 |
| P2 (patient-level) | class-balanced | 0.6302 | 0.6163 | 0.871 / 0.859 |
| **P2 (patient-level)** | **none** | **0.6709** | **0.6672** | **0.886 / 0.887** |

- Anomaly resolved: at 0.671 the converged model now clearly beats the 0.621 probe.

## Slide 18 — Results: The Clinical Operating Point
Header: "The number that decides whether this is usable"
- Draw an ROC curve, x = 1 − specificity, y = sensitivity, rising steeply then flattening, AUC 0.887.
  Dashed horizontal line at y = 0.90 labelled "clinical requirement"; mark the crossing at x = 0.387.
- **Referable-DR AUROC 0.887, 95% CI [0.873, 0.900]** on the frozen patient-level test set.
- **Held at 90% sensitivity: 61.3% specificity.** Catches 9 of every 10 referable patients, missing
  **124 of 1,246**; correctly clears about 3 in 5 who do not need referral.
- At argmax: 62.3% sensitivity / 95.4% specificity — the wrong trade-off for screening.
- State the weakness plainly: 61.3% specificity means roughly 2 in 5 healthy patients would be
  referred unnecessarily. This is the primary target of the calibration phase.

## Slide 19 — Comparison with Existing Works
Table, then a bar chart of the AUC column with private-data rows grey and public-data rows in accent:

| Study | Training data | Test set | AUC | Sens / Spec |
|---|---|---|---|---|
| Gulshan et al. (2016) | Private, ~128k adjudicated | EyePACS-1 (9,963 img) | 0.991 | 97.5 / 93.4 |
| Gulshan et al. (2016) | Private, ~128k adjudicated | Messidor-2 (1,748 img) | 0.990 | 96.1 / 93.9 |
| Gargeya & Leng (2017) | Not stated | 5-fold cross-validation | 0.97 | 94 / 98 |
| Wang et al. (2026), pooled | 25 approved devices, 82 studies | 887,244 exams, 28 countries | — | 93 / 90 |
| Voets et al. (2019) | Public (Kaggle EyePACS) | Kaggle EyePACS test | 0.951 | 90.6 / 84.7 |
| **This work** | **Public EyePACS, patient-level** | **Frozen P2 test (5,814 img)** | **0.887** | **90.0 / 61.3** |
| Voets et al. (2019) | Public (Kaggle EyePACS) | Messidor-2 | 0.853 | 81.8 / 71.2 |

- The gap between 0.99 and everything below is a **data gap, not a modelling gap** — Gulshan used
  ~128,000 multi-grader adjudicated images. Voets, the only independent public-data reproduction,
  reached 0.951 and 0.853.
- My 0.887 sits between Voets' two results, which is where a public-data result should sit.
- My specificity at matched sensitivity is below Voets', with three identifiable causes: they trained
  a purpose-built *binary* classifier while mine derives the referable score from a 5-class head; they
  used higher input resolution (my own 224→384 test gained +0.058 QWK); and my test set is
  patient-level, a stricter evaluation. The first two are compute-bound, not method-bound.

## Slide 20 — Results: What Actually Drives Performance
Header: "Four levers, ranked — and two of them surprised me"
- **The classifier head is the largest and cheapest lever.** Multinomial → ordinal, everything else
  fixed: **+0.050** (EffNet-B0 @224), **+0.048** (@384), **+0.091** (ResNet-50 @224). All three paired
  bootstrap intervals exclude zero — the only single design choice significant on *every* backbone. It
  works because DR grades are ordered: predicting 4 when the truth is 0 should cost far more than
  predicting 1, and a multinomial head discards that structure. Cost: a ridge regression and four
  thresholds.
- **Resolution is second.** 224 → 384px gained **+0.058 QWK**, consistent with the microaneurysm
  argument. Single run, so reported as an observation.
- **The backbone mattered less than the head.**
- **Class balancing made it worse** — the surprise. Removing class-balanced resampling improved test
  QWK in **all four matched pairs** (+0.024, +0.035, +0.041, +0.051) and improved referable AUROC from
  0.871 to 0.886 on P2. My stated hypothesis was that it trades QWK for grade-1 recall; the data does
  not support it — grade-1 recall stayed between **8.7% and 14.0% under both settings**.
- Include a grouped bar chart of the four paired runs, y axis 0.60–0.70, delta labelled above each
  pair.
- Transferable lesson to say aloud: **turn your defaults into ablations.**

## Slide 21 — Results: Bilateral Grading and a Confound I Caught
Header: "The flattering number I did not report"
- Patients are photographed in both eyes anyway, so the second eye is free information, and the two
  eyes are strongly correlated (QWK 0.855).
- First comparison: mean-pooled features + ordinal head reached 0.569 vs 0.494 for per-eye-max +
  multinomial. **+0.075, CI [+0.045, +0.105]** — comfortably significant.
- On review I found I had changed **three variables simultaneously** — the head, the fusion method and
  the aggregation rule. The comparison was valid but it was not a test of "does using both eyes help."
  Nobody flagged this; I found it myself.
- So I built a 10-cell grid varying one factor at a time, each cell with its own bootstrap interval,
  repeated on all three backbones. Include the three-panel decomposition forest plot.
- **Result: roughly two-thirds of the gain was the classifier head, not the second eye.** Genuine
  fusion benefit is +0.025 [+0.003, +0.047] at 224px and +0.024 [+0.004, +0.042] at 384px — but
  **−0.028 [−0.053, −0.002] on ResNet-50**, significantly *worse* on that backbone.
- Honest recommendation: use the ordinal head unconditionally; add bilateral fusion only after
  verifying it on your specific backbone. This closes gap 4 from slide 10.

## Slide 22 — Results: Failure Modes and Evaluation Integrity
**Where the model breaks down.** Grade 1 recall is 8.7–14.0% across all 8 converged runs. Three
independent findings converge on one cause: it is also the only grade that fails to transfer between a
patient's two eyes (**45.8%** partner-lookup accuracy vs 72.1–94.3% for grades 0, 2, 3, 4), and **3 of
the 8 corrupted black frames carried a human grade of 1**. ~10-pixel microaneurysms do not survive
downsampling to 224px, and human graders are unreliable on them — so this may be a **measurement
ceiling, not a modelling failure**.
- **Be ready for this.** Yan et al. (2026) report pooled Grade-1 sensitivity of **72.1%** across 41
  studies. I do not think this contradicts my result — I think it reflects evaluation protocol. Only
  **12.2%** of those studies used external validation; mine comes from a frozen patient-level test set
  scored once. The grade-1 merge test on slide 25 is designed to settle it.
- **Cross-dataset transfer is sharply asymmetric.** EyePACS→APTOS: 0.660 multinomial / **0.754**
  ordinal. APTOS→EyePACS: **0.287**. Largely a data-volume effect (3,111 train vs 35,118 test).
  Implication: a model trained on one screening population must be **re-measured**, not assumed to
  transfer.

**Verifying my own evaluation.** Inter-eye correlation is QWK **0.855 [0.846, 0.863]** over 17,555
patients; stratification rules out class imbalance (0.656 at grade ≥ 1, 0.538 at referable).
- **A shortcut ceiling worth adopting:** copy one eye's known grade onto the other — no image, no model
  — and reach **QWK 0.838**. Any DR model near 0.838 without clearly beating it may be exploiting
  inter-eye correlation rather than reading the retina. Every model here sits well below it.
- **Does image-level splitting inflate scores?** Tampu et al. measured 0.07–0.43 MCC inflation in OCT;
  I predicted the same. **It did not appear.** P1 reaches 0.536/0.595, P2 reaches 0.559/0.621 — point
  estimates run the *wrong* way, paired permutation p = 0.125. A targeted partner-eye ablation
  (n = 3,647 vs 1,641) gave +0.051 (p = 0.185) and +0.061 (p = 0.0505), both intervals including zero.
  Three tests, all null, all pre-registered with an acceptable null stated before running.

## Slide 23 — Discussion: What I Predicted vs What Happened
Header: "The prediction record"
Every experiment had its expected outcome written into the plan before it ran, and the scripts carry
those expectations as fields in their output, so a miss is flagged by the script that produced it.

| Prediction, written in advance | Expected | Actual |
|---|---|---|
| Class balancing buys grade-1 recall | Recall improves | No change; QWK worse |
| Both-eyes fusion beats single-eye grading | Fusion helps | Mostly the head, not fusion |
| Image-level splitting inflates scores | P1 higher | P2 higher, p = 0.125 |
| Grade-1 recall, fine-tuned models | 0.20 – 0.45 | 0.087 – 0.140 |
| Fine-tuned QWK clears acceptance bar | ≥ 0.70 | 0.555 → 0.681 |

- All three structural predictions were reversed by experiments designed to be capable of reversing
  them — which is the only reason I know they were wrong.
- The pattern I would defend as this project's real contribution: every interesting result came from
  refusing to accept a single simple comparison — making the sampler an ablation instead of a setting,
  decomposing a three-variable confound, running three backbones instead of one, auditing a
  deduplication threshold I had already shipped, and computing the clinical operating point instead of
  quoting the argmax.

## Slide 24 — Conclusion
- Built a complete referable-DR screening pipeline on public data, targeted at settings where
  photographs can be captured but not graded.
- **Referable-DR AUROC 0.887 on a frozen patient-level test set; 61.3% specificity at the clinically
  required 90% sensitivity.** This sits inside the range public data supports — between the two
  test-set results of the only published independent public-data reproduction — and well short of what
  private multi-grader data buys. That gap is a data gap, and naming it as such is more useful than
  pretending otherwise.
- Three design findings, two of which reversed my own assumptions: the ordinal head is the largest and
  cheapest lever (+0.05 to +0.09 on every backbone); class-balanced resampling made the model
  consistently worse; bilateral fusion helps far less than it first appeared.
- The evaluation was verified three separate ways, including a zero-pixel shortcut ceiling of 0.838
  that any similar project could adopt.
- 6 of 9 phases complete, ~40 models trained, every number carrying a patient-level confidence
  interval.

## Slide 25 — Novel Contribution, Limitations, Future Work and References
Header: "Contribution, honest limits, and what comes next"

**Novel contribution** — each tied to a gap from slide 10:
1. **A decomposition method for bilateral grading claims** (gap 4) — the 3-factor × 3-backbone grid
   separating classifier-head effect from fusion effect from aggregation effect. Reverses the
   interpretation of a result the field reports as a single entangled number.
2. **The label-only shortcut ceiling as a reusable diagnostic** — a zero-cost baseline any
   correlated-sample medical dataset can report to detect whether a model is exploiting subject
   correlation rather than image content.
3. **The first direct measurement, to my knowledge, of image-level splitting cost for fundus DR
   grading on EyePACS/APTOS** — the gap left by Tampu et al., who measured it only in OCT.
4. **A clinically-anchored public-data baseline** (gaps 2 and 3) — referable performance reported at a
   fixed operating point with confidence intervals rather than as an AUC, which is the form a
   deployment decision actually needs.

**Limitations:** single-grader public labels; APTOS ships no eye-pairing metadata; one screening
population; the 384px result is a single run; 61.3% specificity is not yet deployable; the technician
interface is not built; no prospective clinical validation — research prototype only.

**Future work, in priority order:** (1) temperature calibration; (2) the confidence-based reject
option — the "uncertain, refer to human" flag that makes the tool safe for non-specialist use, and the
most likely fix for the specificity; (3) a grade-1 merge test to settle whether Mild NPDR is a
measurement ceiling; (4) a training-set-size sweep to explain the disagreement with Tampu et al.;
(5) Grad-CAM including failure cases; (6) the technician-facing application with a 100-image
consistency test. Code for four of these is already written.

**References** (small font, bottom band; spill to a 26th slide only if it will not fit):
1. Gulshan V, Peng L, Coram M, et al. *JAMA*. 2016;316(22):2402–2410.
2. Gargeya R, Leng T. *Ophthalmology*. 2017;124(7):962–969.
3. Voets M, Møllersen K, Bongo LA. *PLOS ONE*. 2019;14(6):e0217541.
4. Tymchenko B, Marchenko P, Spodarets D. *arXiv*:2003.02261. 2020.
5. Ali SG, Wang X, Bi L, Jung Y, Chen T, Zhang H. *The Visual Computer*. 2025;41(8):5675–5688.
6. Wang T-W, Luo W-T, Tu Y-K, Chou Y-B, Wu Y-T. *npj Digital Medicine*. 2026;9:110.
7. Yan X, Lei S, Hu L, Qin M, Wu N. *Frontiers in Endocrinology*. 2026;17:1853785.
8. Duggal M, Chauhan A, Gupta V, Kankaria A, et al. *JMIR Medical Informatics*. 2025;13:e67529.
9. Tampu IE, Eklund A, Haj-Hosseini N. *Scientific Data*. 2022;9:580.
10. Raman R, Vasconcelos JC, Rajalakshmi R, et al. *Lancet Global Health*. 2022;10:e1764–e1773.
11. Teo ZL, Tham YC, Yu M, et al. *Ophthalmology*. 2021;128(11):1580–1591.
12. Nderitu P, Keane PA. *Clin Exp Ophthalmol*. 2025;53(7):741–743.
13. Li Q, Drinkwater JJ, Woods K, Douglas E, Ramirez A, Turner AW. *Aust J Rural Health*. 2025;33(2):e70031.
14. DeLuca NJ, Wertheimer B, Ansari Z. *Curr Ophthalmol Rep*. 2025;13:6.
15. Ashrafzadeh CM, Bahi M, Mostafa A, et al. *Ophthalmology Science*. 2026;6:101191.
16. Bhoyar V, Patel M. *Arch Comput Methods Eng*. 2026;33:4359–4380.
17. Akhtar S, Aftab S, Ali O, Ahmad M, Khan MA, Abbas S, Ghazal TM. *Scientific Reports*. 2025;15:3763.
18. Yagis E, Atnafu SW, García Seco de Herrera A, et al. *Scientific Reports*. 2021;11:22544.
19. Rouzrokh P, Khosravi B, Faghani S, et al. *Radiology: AI*. 2022;4(5):e210290.
20. Guo C, Pleiss G, Sun Y, Weinberger KQ. *Proc. ICML*. 2017;PMLR 70:1321–1330.
21. Selvaraju RR, Cogswell M, Das A, et al. *Proc. ICCV*. 2017:618–626; *IJCV*. 2020;128(2):336–359.
22. EyePACS. *Diabetic Retinopathy Detection*, Kaggle, 2015; APTOS. *2019 Blindness Detection*, Kaggle, 2019.
