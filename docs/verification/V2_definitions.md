# V2: Reproduce the reference numbers offline

## Finding

All Part 1.4 numbers reproduce exactly from `results/calibration_reject.json`
(produced by `src/experiments/calibrate.py`, the reference implementation).
No STOP condition.

## Cross-check

| Quantity | Part 1.4 | `calibration_reject.json` |
|---|---|---|
| Referral TP/FN/TN/FP | 1116 / 130 / 3186 / 1382 | 1116 / 130 / 3186 / 1382 (exact) |
| Referral sens/spec | 0.8957 / 0.6975 | 0.895666 / 0.697461 (matches to 4dp) |
| Reject kept/rejected | 4,682 / 1,132 | 4682 / 1132 (exact) |
| Reject coverage | 0.8053 | 0.805298 (matches to 4dp) |
| Selective QWK at τ | 0.77407 [0.7496, 0.7983] | 0.774067 [0.749609, 0.798336] (matches) |
| Selective referable | TP 714 · FN 111 · TN 2972 · FP 885 | 714 / 111 / 2972 / 885 (exact) |
| Test ECE before→after | 0.1628 → 0.0285 | 0.162844 → 0.028542 (matches to 4dp) |
| Test QWK (argmax) | 0.71595 | 0.715951 (matches) |

Also: `T = 3.367438` (matches `thresholds.json`'s `temperature: 3.3674`),
`referral_threshold.value = 0.134856` (matches `0.13486`), `reject_tau =
0.592744` (matches `0.59274`).

## Exact definitions (for `app/core/decision.py` — Task A1–A3)

- **Softmax input for calibration:** `softmax(logits / T)`, with `T` fitted
  by minimizing NLL on the validation fold only (`calibrate.py:124-142`,
  `softmax_T`/`fit_temperature`). Numerically stable: subtract row max before
  `exp`.

- **Referral score:** `referral_score = p2 + p3 + p4` (calibrated probs, the
  whole tail from grade 2 up, not argmax). **Inequality direction: `>=`.**
  `REFER` iff `referral_score >= referral_threshold` (`calibrate.py:186`,
  `referable_stats`: `pred = (scores >= threshold).astype(int)`).

- **Referral threshold itself** was chosen on validation as the *smallest*
  threshold with sensitivity `>= 0.90`
  (`referral_threshold_for_sensitivity`, `calibrate.py:179-194`), scanning
  ascending unique score values and keeping the last one that still clears
  target sensitivity before it drops below. The app does not need to
  reproduce this fitting step — it just reads the fitted value
  `0.13486` from `thresholds.json` and applies the `>=` rule above.

- **Reject/uncertainty gate:** confidence = **max calibrated softmax
  probability** (`primary = "max_softmax"`, i.e. `p_cal.max(axis=1)`).
  **Inequality direction for "kept" (not rejected): `>=`.**
  `keep = conf_test >= tau` (`calibrate.py:450`). So the UNCERTAIN state is
  the complement: `max_calibrated_prob < tau`.

- **τ itself (0.59274)** is *not* from the pre-registered sensitivity rule
  (that rule turned out degenerate — see `rule_degeneracy` in the JSON). It
  is shipped from the pre-registered ~80%-coverage point on validation
  instead (`tau_basis = "acceptance_test_11_1_coverage"`). Again, the app
  only needs to read `reject_tau` from `thresholds.json` and apply `>=`.

- **ECE binning:** 15 equal-width bins over `[0, 1]` on **confidence = max
  probability** (post- or pre-scaling as appropriate)
  (`calibration_error`, `calibrate.py:145-170`). Bin membership: bin 0 is
  `conf >= lo & conf <= hi` (closed both ends); bins 1..14 are
  `conf > lo & conf <= hi` (open-low, closed-high) — i.e. every bin's
  right edge is inclusive, and only the very first bin's left edge (0.0)
  is also inclusive. `gap = |accuracy - avg_confidence|` per bin;
  `ece = Σ (bin_count/n) * gap` over non-empty bins; `mce = max(gap)`.
  Empty bins contribute 0 to ECE/MCE and are recorded with `count: 0`,
  `accuracy: null`, `confidence: null`, `gap: null`.

- **Expected-grade marker (DEC-1 / A5):** `Σ k · pₖ`. **Correction after
  reading `grade1_diagnosis.py` line-by-line (see V3):** that script computes
  `exp_grade` from the CSV's `prob_0..4` columns directly, which are **raw**
  (pre-temperature) probabilities — `dump_app_predictions.py` writes
  `probs = softmax(logits)` with no temperature division. So the
  `ordinal_read_vs_argmax` finding (`+0.0119 QWK`, CI excludes zero) was
  computed on **raw**, not calibrated, probabilities. DEC-1 in the execution
  plan asks the app's expected-grade marker to use *calibrated* probs. These
  are different quantities: temperature scaling is monotone per-row (same T
  for every class) but not a no-op on `Σk·pₖ` for a specific row — it
  flattens the distribution before the sum, so calibrated expected-grade can
  differ from raw expected-grade image-by-image, even though neither changes
  the argmax. **Task A4's "most common true grade when wrong" and A5's dial
  should therefore not cite the `+0.0119` ordinal-QWK figure as if it were
  about the calibrated marker** — if the app's copy references that number,
  it must say "raw probabilities" explicitly, or the number is being
  misapplied to a different quantity than the one it was measured on.

## Script

Verified in-session via a direct read of the already-computed
`results/calibration_reject.json` (produced by `calibrate.py`, which is the
reference implementation itself — re-deriving it in a second script would
just re-run the same code) plus line-by-line reading of `calibrate.py`'s
numeric functions to confirm inequality directions and binning, which the
JSON alone doesn't reveal.

## Conclusion

No STOP condition. Proceed to V3.
