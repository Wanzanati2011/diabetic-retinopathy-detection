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

## Next: Phase 2 — preprocessing cache (see MASTER_PLAN.md Part 6)
