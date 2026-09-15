# V5: Baseline

## `pytest tests/ -v`

**71 passed, 3 failed**, 162.95s wall time. Existing failures, recorded per
R7/V5 ("existing failures are recorded, not fixed, unless a task says so")
— not touched in Phase 0:

1. **`tests/test_app.py::test_app_pipeline_matches_eval_pipeline_on_real_test_images`**
   (the pre-existing Acceptance Test 12.1). 2/20 images get a different
   predicted grade from the app pipeline vs. the eval pipeline
   (`aptos_0243404e8a00`: eval=3, app=2; `aptos_059bc89df7f4`: eval=3,
   app=2). Neither carries a non-default EXIF tag, and pixel MAE between the
   app's freshly-processed image and the cached one is small (0.9, 1.0 out
   of 255) — so per the test's own diagnostic framing, this reads as "model
   not robust to small decoder-level pixel noise on these 2 images," not a
   crop/preprocessing bug. **This is exactly the number D1 asks to update**
   ("re-run: 18/100 → 8/100 on the same 100 images" — this run is a
   different, smaller 20-image check, not the 100-image one D1 references;
   B6 extends this test and should reconcile which count the app surfaces).
   Task B6 must not weaken this test; it adds a second, decision-level
   assertion alongside it.

2. **`tests/test_dedup.py::test_every_dup_group_member_pairwise_within_threshold`**
   — `TypeError: int() can't convert non-string with explicit base`. Looks
   like a pandas dtype issue reading `_phash_cache.csv` (a phash value isn't
   coming back as a hex string in some rows, possibly `NaN` for an
   unhashed/missing entry). Pre-existing, unrelated to any app-v2 task;
   R2 forbids touching `src/data/dedup.py`. Not fixed.

3. **`tests/test_splits.py::test_splits_deterministic`** — re-running
   `splits.py` with the same seed does not reproduce byte-identical
   `p1.json`. Pre-existing; `src/data/` splits code is outside R2's touch
   list for this project (not one of the named protected files, but also
   not something any task here asks to fix). Not fixed.

None of these three touch any file this plan's tasks will modify (R2-listed
files: `to_model_input()`, `preprocess.py`, `src/train/`, `src/experiments/`,
`src/features/`, `results/`, `data/`, `labels/`, `checkpoints/`,
`best_model.pt`) in a way Phase 1+ tasks change, so they're independent
baselines to watch: **if any of these three start passing or a fourth test
starts failing after a later task, that's a signal, not a red herring.**

## App startup

`python app\app.py` started cleanly, no exceptions, model loaded, and the
Gradio server answered `GET /` with **HTTP 200** within ~30s of process
start (exact startup-to-serving time not more precisely isolated because
stdout was buffered in the background run; well under a minute either way).
Process was killed manually after confirming the 200 response — no crash,
no hang.
