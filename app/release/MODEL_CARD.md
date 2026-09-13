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
| Checkpoint | `checkpoints/finetune_app_p2_seed42/best.pt` (epoch 11 of 15, selected by best validation QWK) |

## Measured performance (P2 test set, n=5,814, evaluated once)

| Metric | Value |
|---|---|
| 5-class quadratic-weighted kappa (QWK) | **0.613** |
| Referable-DR (grade ≥ 2) AUROC | **0.872** |

Per-class recall:

| Grade | n | Recall |
|---|---|---|
| 0 -- No DR | 4,121 | 0.687 |
| 1 -- Mild NPDR | 447 | 0.461 |
| 2 -- Moderate NPDR | 945 | 0.354 |
| 3 -- Severe NPDR | 157 | 0.452 |
| 4 -- Proliferative DR | 144 | 0.549 |

## Honest limitations -- read before using this for anything

1. **Undertrained by this project's own standard.** MASTER_PLAN.md Part 10
   defines a 0.75-0.85 test QWK band as "correct, proceed." This checkpoint's
   0.613 falls in the 0.40-0.70 "undertrained or preprocessing bug" band. It
   was trained with the original 15-epoch recipe (`configs/finetune_app.yaml`,
   `src/train/finetune.py`), not the later 40-epoch, early-stop-patience-8
   "converged" recipe (`src/train/finetune_converged.py`) that this same
   project used to get a properly-converged result for its headline claims.
   A retrain of the app model with that recipe is a natural, cheap next step
   (~30-90 min of GPU time at 384px) and would likely raise this number.
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

## What Acceptance Test 12.1 actually found

Acceptance Test 12.1 (`tests/test_app.py`) found that ~18% of a real P2
test sample get a *different predicted grade* through the app's code path
than through the evaluation pipeline. Two hypotheses were checked in order,
per this project's own investigate-before-adjusting standard -- not assumed:

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

**Conclusion: this is not a preprocessing bug.** The app's pipeline and the
evaluation pipeline produce near-identical pixels for these images; the
model's predicted grade flips anyway on ~18% of the test sample. That is
consistent with this checkpoint's own documented status above (test QWK
0.613, "undertrained" 0.40-0.70 band): a less-converged model has a
narrower decision margin between classes, so sub-1/255 input noise alone is
enough to tip a borderline case to the neighboring grade. Supporting this,
sampled flips land between adjacent or near grades (e.g. 1->0, 2->1, 2->3,
4->2) rather than jumping randomly across the 5-way scale -- the signature
of low-margin, borderline predictions, not corrupted input.

This is tracked here as a **model-robustness finding**, not an app bug.
Retraining the app checkpoint with this project's converged recipe
(`src/train/finetune_converged.py`, limitation #1 above -- 40 epochs,
patience-8 early stopping) is the expected real fix: a higher-QWK model
should have wider decision margins and far fewer flips under this level of
ordinary decoding noise.

## Intended use

A research/teaching demonstration of an end-to-end DR-screening pipeline,
and of this project's broader methodological point about patient-level
evaluation. Not intended for, and not validated for, real screening,
diagnosis, or triage of real patients.
