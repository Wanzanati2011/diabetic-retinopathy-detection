# V3: Data structures

## `calibration_reject.json → reject_option.curves.max_softmax.test`

List of 200 dicts (one per swept τ quantile), each:
```
{tau, coverage, n_kept, selective_qwk, selective_sensitivity, selective_specificity}
```
Same shape for `.validation`. Row 0 (smallest tau ≈ 0.288) has `coverage:
1.0`, i.e. full coverage — useful as the "no rejection" baseline for the
risk-coverage explorer (C2).

## `calibration.p2_test.before.bins` (also `.after.bins`)

List of 15 dicts (equal-width confidence bins, see V2 for the binning rule):
```
{bin_lower, bin_upper, count, accuracy, confidence, gap}
```
Empty bins have `count: 0` and `accuracy`/`confidence`/`gap` all `null`
(JSON `None`) — the reliability-diagram renderer (C2) must skip those, not
plot a zero.

## `grade1_diagnosis.json → ordinal_read_vs_argmax`

```
{comparison, observed_diff, paired_patient_bootstrap_ci95: [lo, hi],
 excludes_zero, reading}
```
`observed_diff = +0.011949` QWK, CI `[0.00553, 0.01854]`, excludes zero.

**Important for A4/A5/C5:** `expected_grade` here is computed from the CSV's
`prob_0..4` columns directly (`grade1_diagnosis.py:180`,
`exp_grade = (probs * np.arange(5)).sum(axis=1)`), and those CSV columns are
**raw, pre-temperature** probabilities (`dump_app_predictions.py:125`,
`probs = torch.softmax(torch.from_numpy(logits), dim=1)` — no `/T`). So this
`+0.0119` finding describes the *raw* expected-grade marker, not the
*calibrated* one DEC-1/A5 asks the app to display. If C5's "Mild disease"
tab or A5's dial copy cites this number, it must say "raw probabilities" or
avoid implying it validates the calibrated dial. See V2's correction.

Top-level `grade1_diagnosis.json` keys (for C4/C5 sourcing):
`run_name, n_test, n_test_patients, n_grade1, expected_written_in_advance,
confusion_row_grade1, confusion_row_grade1_fractions, confusion_matrix_full,
decision_rules, ordinal_read_vs_argmax, prior_checkpoint_recall,
best_non_argmax_rule, best_non_argmax_grade1_recall, partial_recovery_note,
clinical_note, merge_experiment_link, verdict, verdict_text`.

## The training-curve JSON (`results/finetune_app_converged_..._curve.json`)

A flat list of 40 dicts (one per epoch), each:
```
{epoch, train_loss, train_qwk, val_qwk, epoch_seconds}
```
No nesting — directly pluck `epoch`/`val_qwk` (and `train_qwk` for a second
line) for the C7 training curve chart.

## `claim3_decomposed*.json` cells

Top-level keys: `backbone, size, n_train_patients, n_test_patients,
n_fold_mismatches_dropped, patient_label_definition, cells,
arm_D_label_only_lookup_cited, decomposition, verdict, verdict_text`.

`cells` is a dict keyed by 10 arm names:
`per_eye_max_multinomial, per_eye_max_ordinal, per_eye_mean_multinomial,
per_eye_mean_ordinal, per_eye_min_multinomial, per_eye_min_ordinal,
concat_multinomial, concat_ordinal, pool_multinomial, pool_ordinal`.
Each cell:
```
{label, qwk, patient_bootstrap_ci95: [lo, hi], referable_sens_spec_near_90pct_sens: {...}}
```

`decomposition` is a dict keyed by 8 named comparisons, each:
```
{mean_diff, ci95: [lo, hi], significantly_better}
```
Keys: `1_head_effect_ordinal_minus_multinomial_at_max`,
`2_fusion_effect_pool_minus_per_eye_max_ordinal` (**this is `{fusion_384}`
for A6's restored caveat sentence and the "fusion effect by backbone" in
C5**, `mean_diff = 0.024408`, CI `[0.00419, 0.04193]`, significant),
`3_fusion_effect_concat_minus_per_eye_max_ordinal` (not significant, CI
crosses zero: `[-0.0504, 0.00096]`),
`4_aggregation_effect_mean_minus_max_ordinal`,
`5_aggregation_effect_min_minus_max_ordinal`,
`6_aggregation_effect_mean_minus_max_multinomial`,
`7_aggregation_effect_min_minus_max_multinomial`,
`8_original_total_gain_pool_ordinal_minus_per_eye_max_multinomial`.

For C5's "Two eyes" waterfall (0.4945 → 0.5447 → 0.5693), the three
QWK values are `per_eye_max_multinomial.qwk`, `per_eye_max_ordinal.qwk`
(head effect step), and `pool_ordinal.qwk` (fusion effect step) — confirm
exact match against the 384px file when building C5; `per_eye_max_ordinal`
in this file is `0.59206`, not `0.5447`, so **the three waterfall values
in the master-plan copy come from a different backbone/size file than
`tf_efficientnet_b0_384`** (there are 3 `claim3_decomposed*.json` files:
default, `resnet50_224`, `tf_efficientnet_b0_384`) — verify which file
actually produces 0.4945/0.5447/0.5693 before building C5; don't assume
it's the 384px one just because that's the deployed model's size.

There's also a bare `results/claim3_decomposed.json` (no backbone/size
suffix) with the same schema — check its `backbone`/`size` fields to see
which run it is before using it.

## `claim2d_sample_size_sweep.json → rows`

Top-level: `backbone, size, sizes, replicates, scope, n_common_test_images,
n_common_test_patients, n_partner_present, n_partner_absent, rows,
paired_ci_endpoints, gap_at_smallest_N, gap_at_largest_N,
partner_gap_at_smallest_N, partner_gap_at_largest_N,
gap_shrinkage_small_minus_large, verdict, verdict_text`.

`rows` is a flat list, one per `(n_train, replicate)` pair:
```
{n_train, replicate, seed,
 multinomial: {qwk_p1, qwk_p2, gap_p1_minus_p2, qwk_partner_present, qwk_partner_absent, partner_gap},
 ordinal: {same 6 fields}}
```
Good directly for the C4 training-size sweep chart (x = `n_train`, y =
`gap_p1_minus_p2`, averaged over replicates or with error bars per `n_train`).

## `qml_pqc.json` and `qcnn_no_cnn_pixels.json`

`qml_pqc.json` top-level keys: `backbone, size, chosen_config,
n_train_sample, n_val_sample, n_train_full_p2, n_val_full_p2, n_test,
diff_method, epochs_max, batch_size, lr, patience, seed,
matched_subsample_heads, paired_diffs, full_training_set_reference_cited,
verdict, verdict_text, literature, selection_method, val_qwk_at_selection,
sweep`. `sweep` is presumably the heatmap source (qubits × conv-reps grid)
for C5's Quantum tab — inspect `sweep`'s row shape before building the
heatmap (not dumped here for brevity; do so in Phase 4 when C5 is built).

`qcnn_no_cnn_pixels.json` top-level keys: `experiment, chosen_config,
n_train_sample, n_val_sample, n_train_full_p2, n_val_full_p2, n_test,
proc_size, diff_method, epochs_max, batch_size, lr, patience, seed, qcnn,
matched_classical_baseline, paired_diff_qcnn_minus_baseline,
full_training_set_reference_cited, verdict, verdict_text, literature,
selection_method, val_qwk_at_selection, sweep`. `qcnn` and
`matched_classical_baseline` are dicts (used directly in `app.py`'s
existing `qwk_meter_html` calls, confirmed by reading the current
`app.py` Quantum Lab tab code, which already reads these two paths).

## Dark-frame cutoff (for B1, and Part 7 STOP condition #3)

Found. `results/excluded_images.json.exclusion_criterion`:
> "original-source image mean pixel value < 5 (flat-black/underexposed
> capture), flagged by test_no_low_mean_pixel_crops (Acceptance Test 6.1)
> after the full local Phase 2 preprocessing run"

**Caveat for B1:** this cutoff (mean < 5) was computed on the
**original-source** image, not on `preprocess()`'s cropped/resized output.
B1's spec asks for "mean of the `preprocess()` output below the cutoff used
to exclude the 8 corrupted frames" — the *cutoff value* (5) transfers, but
whether it's still the right cutoff *on the processed output's pixel scale*
(uint8 0–255, post-crop-and-resize) needs a sanity check: `preprocess()`
crops to the detected bright/retinal region, which could raise the mean of
an otherwise-dark frame if a small bright region gets zoomed in, or could
leave it near-zero if the whole frame is genuinely blank. `low_mean_diagnostic.json`
has each flagged image's `original_stats.mean` (range ~1.7–9.3 in the two
samples inspected) — all comfortably under a `< 5`-or-similar cutoff on the
0–255 scale either way. No STOP: use cutoff `5` on the `preprocess()`
output's mean (0–255 uint8 scale, matching `original_stats.mean`'s scale),
and note in B1's code comment that it was originally computed on the
unprocessed source.
