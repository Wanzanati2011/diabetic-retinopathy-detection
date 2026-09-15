# Model Card -- Diabetic Retinopathy Screening Assistant (Research Prototype)

**This is a research prototype, not a medical device. It has not been validated
for clinical use and must not be used to make or influence a real diagnostic or
treatment decision. It is not a substitute for examination by a qualified
ophthalmologist.**

## What it is

A 5-class diabetic retinopathy grader (ICDR scale, grades 0-4), fine-tuned
end-to-end from an ImageNet-pretrained `tf_efficientnet_b0` backbone, on
retinal fundus photographs from two public datasets (EyePACS and APTOS 2019).

| | |
|---|---|
| Architecture | `tf_efficientnet_b0` (timm), 5-way softmax head |
| Input | 384x384 RGB, cropped/resized/padded by `src/data/preprocess.py` |
| Training split | **P2** -- patient-level (both eyes of a patient always in the same fold); the project's primary, honest split |
| Training data | 27,123 images (train), 5,843 (val) |
| Seed | 42 |
| Checkpoint | `checkpoints/finetune_app_converged_p2_class_balanced_seed42/best.pt` (epoch 32 of 40, selected by best validation QWK) |

**This checkpoint supersedes an earlier one** (`finetune_app_p2_seed42`, 15
epochs, test QWK 0.613, "undertrained" band) trained with the original
`src/train/finetune.py` recipe. This is the converged retrain
(`src/train/finetune_converged.py`, `configs/finetune_app_converged.yaml`:
40 epochs, early-stop patience 8, lr_backbone 1e-4) done specifically to
fix that undertraining -- see "Honest limitations" #1 below for what
actually changed.

## Measured performance (P2 test set, n=5,814, evaluated once)

| Metric | Value | Prior checkpoint |
|---|---|---|
| 5-class quadratic-weighted kappa (QWK) | **0.716** | 0.613 |
| Referable-DR (grade ≥ 2) AUROC | **0.912** | 0.872 |

Per-class recall:

| Grade | n | Recall | Prior checkpoint |
|---|---|---|---|
| 0 -- No DR | 4,121 | 0.916 | 0.687 |
| 1 -- Mild NPDR | 447 | 0.161 | 0.461 |
| 2 -- Moderate NPDR | 945 | 0.570 | 0.354 |
| 3 -- Severe NPDR | 157 | 0.287 | 0.452 |
| 4 -- Proliferative DR | 144 | 0.542 | 0.549 |

**Note the grade-1 recall drop (0.461 -> 0.161).** This was not expected
going in and is flagged, not smoothed over: the training script's own
`grade1_recall_note` field for this run reads *"grade-1 recall outside the
MASTER_PLAN.md 0.20-0.45 expected band."* Grade 0 recall rose sharply
(0.687 -> 0.916) alongside it, which is consistent with a more-converged
model settling into a stronger "healthy vs. not" boundary at grade 1's
expense on this class-imbalanced task, but that is a plausible
explanation, not a verified mechanism -- it has not been separately
investigated (e.g. via the confusion matrix row for class 1, which shows
most grade-1 errors landing at grade 0). Anyone using this checkpoint for
grade-1-sensitive screening should treat this as an open, real caveat, not
a footnote.

## Honest limitations -- read before using this for anything

1. **Now inside the "correct, proceed" band -- with a real tradeoff.**
   MASTER_PLAN.md Part 10 / Acceptance Test 10.1 defines a 0.70-0.85 test
   QWK band as "correct, proceed" (this is the actual band the training
   script's own verdict logic uses -- see its printed output below). This
   checkpoint's 0.716 lands there, up from the prior checkpoint's 0.613
   ("undertrained" band). That is a genuine improvement in overall ordinal
   agreement, achieved by retraining with this project's converged recipe
   (`src/train/finetune_converged.py` -- 40 epochs, patience-8 early
   stopping) instead of the original 15-epoch one. It did **not** come for
   free: grade-1 recall dropped substantially (see the note above). The
   training script's own printed verdict for this run was:
   `ACCEPTANCE TEST 10.1 VERDICT: test_qwk=0.7160 -> correct -- proceed`.
2. **Confidence is not calibrated.** The app shows the model's raw softmax
   max-probability. MASTER_PLAN.md Part 7 (temperature scaling on the
   validation set, plus a reject threshold picked by a pre-stated rule) has
   not been run for this checkpoint. There is no "UNCERTAIN -- refer" gate in
   the current app; do not read the shown confidence number as a calibrated
   probability.
3. **Grad-CAM implemented (Phase 2), not independently verified.** Shows
   attention from the backbone's last conv block for the predicted class,
   alongside the preprocessed photo. Requires `pip install grad-cam` and
   fails gracefully (blank panel, grading unaffected) rather than crashing
   if that library or this specific model/version combination misbehaves --
   check that the panel actually renders something plausible before relying
   on it in a demo.
4. **Single seed, single split.** This is one training run (seed 42, P2
   split). It has not been checked for seed-to-seed variance the way this
   project's Claim 2 experiments were (5 seeds on frozen features).
5. **Known, expected weak spot:** grade-1 (Mild NPDR) recall is limited
   because grade 1 is defined by microaneurysms only, ~10px lesions that are
   physically destroyed by downsampling to 384px. This is a documented,
   expected limitation of this resolution, not a bug -- see MASTER_PLAN.md
   Part 10.
6. **Training data limitations** (inherited from the whole project, not
   specific to this model): EyePACS labels are single-grader and noisy;
   APTOS has no patient IDs, so its patient-level splitting is approximate;
   both datasets are from specific screening populations and may not
   generalize elsewhere; single-field, macula-centred imaging misses
   peripheral lesions.

## What Acceptance Test 12.1 found, before and after the retrain

**Status: re-run and confirmed.** Acceptance Test 12.1 was re-run against
this checkpoint (test QWK 0.716) with `$env:N_APP_TEST_IMAGES=100; pytest
tests\test_app.py -v`. Result: **8/100 mismatches**, down from 18/100 on
the prior, undertrained checkpoint (test QWK 0.613) -- roughly half. The
mismatches that remain look like the same phenomenon at reduced scale, not
a different problem: 0/8 carry a non-default EXIF tag, and all 8 have
pixel MAE between 0.8-1.2 (out of 255) between the app's freshly-processed
image and the cached evaluation image -- the same "near-identical pixels,
model flips anyway" signature as before. So the retrain's wider decision
margins did roughly what the theory below predicted, but did not
eliminate the effect: this checkpoint is measurably more robust to
ordinary JPEG-decoder noise, not perfectly robust to it. **Verified
apples-to-apples**: `_load_p2_test_sample()` in `tests/test_app.py` walks
the P2 test split in a fixed order with no shuffling or seeding, so
`N_APP_TEST_IMAGES=100` always selects the identical 100 images run to
run -- confirmed by re-running the test twice on this checkpoint and
getting the exact same 8 image IDs both times. The 18-vs-8 comparison is
the same 100 images scored by two different checkpoints, not two
different samples.

The original investigation (on the prior checkpoint) that established this
was a model-robustness finding, not a preprocessing bug, follows below for
reference.

Acceptance Test 12.1 originally found that ~18% of a real P2
test sample got a *different predicted grade* through the app's code path
than through the evaluation pipeline, on that prior checkpoint. Two
hypotheses were checked in order, per this project's own
investigate-before-adjusting standard -- not assumed:

1. **EXIF orientation.** `cv2.imread()` (used by `build_cache.py` to build
   the training/eval cache) auto-applies EXIF rotation for JPEG sources;
   PIL's `Image.open()` does not, unless `ImageOps.exif_transpose()` is
   called explicitly. Fixed in `to_model_input()` -- worth keeping, since a
   real user's phone photo can carry EXIF rotation -- but **ruled out** as
   the cause of these particular mismatches: 0/18 flipped images carry a
   non-default EXIF orientation tag.
2. **Pixel-level mean absolute difference**, computed between the app's
   freshly-decoded-and-processed image and the cached training/eval image,
   for every mismatch: all 18 are 0.8-1.2 (out of 255) -- consistent with
   ordinary JPEG-decoder differences (cv2/libjpeg-turbo vs. PIL/libjpeg
   decoding the same source file to very slightly different RGB values),
   not a different crop region.

**Conclusion (on the prior checkpoint): this was not a preprocessing bug.**
The app's pipeline and the evaluation pipeline produced near-identical
pixels for these images; the model's predicted grade flipped anyway on
~18% of the test sample. That was consistent with that checkpoint's
documented status (test QWK 0.613, "undertrained" band): a less-converged
model has a narrower decision margin between classes, so sub-1/255 input
noise alone was enough to tip a borderline case to the neighboring grade.
Supporting this, sampled flips landed between adjacent or near grades
(e.g. 1->0, 2->1, 2->3, 4->2) rather than jumping randomly across the
5-way scale -- the signature of low-margin, borderline predictions, not
corrupted input.

This was tracked as a **model-robustness finding**, not an app bug, and
retraining with this project's converged recipe
(`src/train/finetune_converged.py` -- 40 epochs, patience-8 early
stopping) was the expected real fix, on the reasoning that a higher-QWK
model should have wider decision margins and far fewer flips under this
level of ordinary decoding noise.

**This checkpoint is that retrain (test QWK 0.716), and the prediction
held on re-run: 8/100 mismatches, down from 18/100** -- see the confirmed
result at the top of this section.

## Intended use

A research/teaching demonstration of an end-to-end DR-screening pipeline,
and of this project's broader methodological point about patient-level
evaluation. Not intended for, and not validated for, real screening,
diagnosis, or triage of real patients.
