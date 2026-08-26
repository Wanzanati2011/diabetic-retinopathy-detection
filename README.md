# Two Eyes, One Patient — Diabetic Retinopathy Study

Project state: **Phase 1 complete** (data foundation). See `MASTER_PLAN.md` for the full execution plan and `STEP0_CORRELATION.md` for the Step 0 inter-eye correlation analysis that motivated this project.

## FROZEN TEST SETS — declared 2026-08-26, commit b03cf393050018467dac16f2987b702fe9662fac

```
P2 test: patient-level held-out split, seed 42          (data/splits/p2.json)
P3 test: APTOS (full) and EyePACS (full), per direction (data/splits/p3_eyepacs_to_aptos.json,
                                                           data/splits/p3_aptos_to_eyepacs.json)

PRIMARY endpoint:   referable-DR sensitivity at fixed 90% operating point, Protocol 2
SECONDARY endpoint: 5-class QWK

Evaluated ONCE, at the end of Phase 7.
Any earlier evaluation must be logged here with a reason.
```

P1 (`data/splits/p1.json`) is the deliberately flawed image-level baseline used only to
*measure* the leakage effect (Claims 2 and 2b) — it is never a target to optimize or "fix".

## Phase 1 summary

- **Manifest** (`data/manifests/manifest.csv`): 38,788 images = 35,126 EyePACS (17,563 patients,
  every one with both eyes, `partner_id` populated) + 3,662 APTOS (no patient IDs, `eye=unknown`).
  Pooled EyePACS grade counts match the published totals exactly (25,810/2,443/5,292/873/708).
- **Deduplication** (`results/duplicates.json`): pHash (hash_size=16, 256-bit), Hamming
  threshold ≤5, found via LSH banding (6 bands) + exact verification. **148 confirmed
  near-duplicate pairs, all within APTOS** (131 groups, 270 images involved) — zero
  cross-dataset duplicates, zero within EyePACS. Hash pipeline self-test (byte-identical
  copy → distance 0) passed.
- **Splits** (`data/splits/`): P1/P2/P3 built respecting `dup_group` integrity throughout,
  and `patient_id` integrity for P2/P3. All 7 required acceptance tests pass (see
  `tests/test_splits.py`), plus 8 manifest tests and 5 dedup tests (20 total).

Run `PYTHONPATH=/tmp/pylibs python3 -m pytest tests/ -v` from the project root to reproduce.

Environment note: pip installs on this machine go to `/tmp/pylibs` (the default target ran
out of disk) — prefix Python commands with `PYTHONPATH=/tmp/pylibs`.

## Dedup threshold review (post-Phase-2, pre-Phase-3)

Reviewed whether hash_size=16 (256-bit phash) + Hamming<=5 was too tight —
"5" is the common convention for the *64-bit* phash default, so as an
absolute bit count on a 256-bit hash it's ~4x stricter than the convention
it was borrowed from. Checked empirically (`src/data/dedup_threshold_sweep.py`,
`results/dedup_threshold_sweep.json`, `figures/dedup_threshold_sweep.png`)
rather than trusting the scaling arithmetic alone:

- **Calibration**: a real near-duplicate (resize +/-2px, re-encode JPEG q90)
  scores Hamming distance **0** — nowhere near the threshold=5 cutoff.
- **EyePACS threshold sweep**: 0 pairs at <=5, 1 at <=10, 123 at <=20, 8,876 at
  <=30 — no plateau, meaning there's no hidden cluster of real duplicates
  sitting just past 5. It's a smooth climb straight into noise.
- **Diagnostic on the <=25 candidates** (790 pairs, `results/eyepacs_near_dup_candidates.csv`):
  786/790 are *different* patients, and grade agreement is 62.7% — barely
  above the 56.8% chance baseline from EyePACS's grade distribution. A true
  duplicate photo must carry the same grade (same image -> same diagnosis);
  chance-level agreement means these are coincidental structural matches
  (similar illumination/framing, common to the photography protocol), not
  duplicated images. Visually confirmed on the closest pair (distance 10,
  eyepacs_17153_right vs eyepacs_19840_right, grades 0 vs 2) — different
  vasculature, not the same photo.
- **APTOS**, by contrast, shows a real plateau (148 pairs at both <=5 and
  <=10), consistent with the genuine duplicates already found in Phase 1.

**Decision: threshold unchanged, splits NOT rebuilt.** `tests/test_dedup_threshold.py`
locks this in — it fails loudly if a future data change makes it stop
holding. The math behind the original concern was correct (the threshold
*is* numerically stricter than the 64-bit convention); it just turned out
not to matter for this dataset, because genuine duplicates sit at distance
~0, far inside the margin regardless of which convention is used.

## Next: Phase 3 — frozen feature extraction (see MASTER_PLAN.md Part 7, needs GPU/local machine)
