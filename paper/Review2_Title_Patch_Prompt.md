# Patch prompt — retitling the deck

Your fixed title is **"Diabetic Retinopathy Detection and Grading using Deep Learning on Retinal
Fundus Images"**. That title names two tasks — *detection* (referable / not referable) and *grading*
(the 5-class ICDR scale). Both are already in your work, so nothing in the methodology or results has
to change. What changes is the framing: the low-resource access story stops being the headline and
becomes the **motivation** for why the detection threshold and the evaluation standard matter.

**10 slides change. 15 stay exactly as they are.**

Unchanged: 4 (Evidence), 6–7 (Literature tables), 8–9 (Wider literature), 11 (Challenges),
14 (Data & evaluation design), 16 (Progress), 17 (Converged model), 18 (Operating point),
19 (Comparison), 20 (What drives performance), 21 (Bilateral), 22 (Failure modes & integrity),
23 (Prediction record).

Copy everything below the line into Claude Design.

---

Update only the slides listed below in the existing deck. Leave every other slide exactly as it is —
do not restyle, reorder, or regenerate them. Keep the same visual theme, fonts and colours
throughout.

## Slide 1 — Title (replace)
- Project Title: **"Diabetic Retinopathy Detection and Grading using Deep Learning on Retinal Fundus Images"**
- Subtitle: "A patient-level pipeline for referable-DR detection and 5-class ICDR grading, evaluated
  at a clinical operating point"
- Register Number: **22MIA1073**
- Student Name: **Y. L. Vishal**
- Guides: **Dr. B V A N S S Prabhakar Rao** and **Dr. Sridevi S**
- Footer: Mini Project — Review 2 — September 2026

## Slide 2 — Outline (small edit)
Keep the same contents list, but make the first two entries read "Introduction: Detection and
Grading" and "Motivation and Evidence". Everything else in the list stays.

## Slide 3 — Introduction (edit the closing bullets only)
Keep the ICDR table and the 4-stage workflow diagram exactly as they are. Replace the closing
bullets with:
- This project addresses **two linked tasks on the same fundus photograph**. **Grading** assigns the
  full 0–4 ICDR severity score. **Detection** reduces that to the binary clinical decision —
  referable (grade ≥ 2) or not.
- Grading is the harder, finer-grained task and is measured with Quadratic Weighted Kappa, which
  penalises a prediction in proportion to how far off it is. Detection is the task that changes
  patient care and is measured with sensitivity and specificity at a fixed operating point.
- The errors are asymmetric: a false referral wastes an appointment; a missed referable case can end
  in irreversible blindness. Any system here must be built sensitivity-first, which is why the
  detection threshold is set deliberately rather than taken from the model's default output.
- Stages 1, 2 and 4 of the workflow scale cheaply. **Stage 3 — interpretation — does not.** That is
  the step deep learning is being asked to fill, and it is why this project exists.

## Slide 5 — Problem Statement and Scope (replace the boxed statement and first three bullets)
Replace the boxed statement with:

> **Automated detection and grading of diabetic retinopathy from retinal fundus photographs, trained
> and validated on public data alone, at a standard of evidence that supports the referral decision
> in real screening — not just a benchmark score.**

Then these bullets, replacing the "why this problem / why it matters / what I am building" block:
- **The clinical need.** Fundus photographs can be captured cheaply and by a technician, but
  interpreting them requires an ophthalmologist or certified grader. Screening programmes therefore
  fail at the interpretation step, not the imaging step — and that failure is concentrated where
  specialists are scarce (slide 4).
- **What that demands of the model.** It is not enough to report a high benchmark score. A model
  that supports a referral decision must be evaluated at a threshold chosen for clinical
  sensitivity, must report honest uncertainty, and must be validated in a way that cannot be
  inflated by how the data was split.
- **A second reason I built my own evaluation** rather than benchmarking against published numbers:
  reported accuracies in this field reach **99.36% for four-stage DR grading** *(Akhtar et al.,
  Scientific Reports, 2025)*, above the level at which two human graders agree on the same
  photograph. I am not claiming any paper is wrong — I am saying published accuracy alone is not a
  safe design target, so I built a frozen, patient-level evaluation and report every number with a
  confidence interval.

Keep the existing In scope / Out of scope / Constraint bullets, but change the first In-scope line to:
- **In scope:** 5-class ICDR grading and referable-DR detection (grade ≥ 2) as the two reported
  tasks; public datasets only; patient-level evaluation; calibration and a reject option; a
  technician-facing interface.

## Slide 10 — Research Gap (replace, reordered)
Header stays "The gap this project addresses". Replace the four gaps with these, in this order — the
methodological gaps now lead and the deployment gap becomes the context:
1. **Evaluation-protocol gap.** Tampu et al. (2022) measured that image-level rather than
   patient-level splitting inflated results by 0.07–0.43 MCC in retinal OCT, and Rouzrokh et al.
   (2022) list patient-level splitting as required practice — but the cost of violating it has never
   been measured for fundus DR grading on EyePACS/APTOS. Guidance exists; the measurement does not.
2. **Operating-point gap.** Published work reports AUC. A referral decision needs a threshold.
   Almost no public-data study reports the specificity actually achieved at a fixed clinical
   sensitivity, which is the number that determines whether a clinic is flooded.
3. **Attribution gap.** Ali et al. (2025) report that bilateral (two-eye) fusion improves grading,
   but change the classifier design at the same time, so the improvement cannot be attributed to the
   second eye. No published work separates the two.
4. **Uncertainty and deployment gap.** A model supporting a non-specialist must be able to abstain.
   Wang et al. (2026) evidence 25 approved DR AI devices across 887,244 examinations, but only one
   dataset from a low-income setting; the deployment studies (Nderitu & Keane 2025; Li et al. 2025;
   Duggal et al. 2025) report aggregate accuracy and do not characterise low-confidence behaviour.
Closing line: gaps 1–3 are what this project measures directly; gap 4 is what the remaining phases
address.

## Slide 12 — Research Objectives (replace objective 1, keep 2–5)
Replace objective 1 with:
1. Train a deep learning model for 5-class ICDR grading and referable-DR detection to genuine
   convergence, and report **both** — QWK for grading and sensitivity/specificity at a fixed
   operating point for detection — on a test set frozen before any model scored on it.
Objectives 2, 3, 4 and 5 stay exactly as they are.

## Slide 13 — Proposed Architecture (rename two layers only)
Keep the diagram and all layer contents. Change two labels:
- Layer 3 label becomes **"Layer 3 — Grading head"** (contents unchanged)
- Layer 4 label becomes **"Layer 4 — Detection: from grade distribution to referral decision"**
  (contents unchanged)
Keep the green ticks on layers 1–4 and the amber marker on layer 5.

## Slide 15 — Methodology: Model Design and the Detection Decision (edit second half)
Keep the entire "Models" block unchanged. Replace the "Referral decision engine" heading and its
first bullet with:

**From grading to detection.** The grading head outputs five probabilities. Detection reduces them
to one referable score, **P(grade ≥ 2) = p₂ + p₃ + p₄** — using the whole distribution rather than
only the argmax, which is the correct way to collapse a 5-class head into a binary decision.
- **A detection model is never deployed at its argmax.** The threshold is chosen to hit a required
  sensitivity. At argmax my model sits at 62.3% sensitivity, far too conservative for screening. Set
  for 90% sensitivity it becomes usable. A deliberate design decision, not a tuning artefact.

Keep the reject-option bullet and the honest-status line exactly as they are.

## Slide 24 — Conclusion (replace first two bullets, keep the rest)
- Built and evaluated a deep learning pipeline for **detection and grading of diabetic retinopathy
  from retinal fundus images**, trained on public data alone and validated at the patient level.
- **Grading: QWK 0.671. Detection: referable-DR AUROC 0.887, 95% CI [0.873, 0.900], with 61.3%
  specificity at the clinically required 90% sensitivity.** The detection result sits inside the
  range public data supports — between the two test-set results of the only published independent
  public-data reproduction — and well short of what private multi-grader data buys. That gap is a
  data gap, and naming it as such is more useful than pretending otherwise.
Keep the remaining bullets (the three design findings, the evaluation verification, the 6-of-9
phases line) exactly as they are.

## Slide 25 — Novel Contribution (replace contribution 4 and the opening line only)
Change the opening line to "Novel contribution — each tied to a gap from slide 10:" and replace
contribution 4 with:
4. **A clinically-anchored public-data baseline for both tasks** (gap 2) — grading reported as QWK
   and detection reported at a fixed operating point with confidence intervals, rather than as a
   single AUC. This is the form a deployment decision actually needs, and it is what makes the
   61.3% specificity visible instead of hidden behind 0.887.
Contributions 1, 2 and 3 stay as they are, as do the Limitations, Future work and References blocks.
