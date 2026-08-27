# Progress Log — "Two Eyes, One Patient" (Inter-Eye Correlation in DR)

Last updated: 2026-08-27. This file summarizes everything done so far, in plain bullet points, organized by phase. See `MASTER_PLAN.md` for the full spec and `README.md` for the frozen-split declarations.

## Step 0 — Setup & correlation sanity check
- Confirmed we have EyePACS (`trainLabels15.csv`, 35,126 images) and APTOS (`trainLabels19.csv`, 3,662 images) on disk, with grade labels for both.
- Confirmed EyePACS filenames encode patient ID + eye (`<patient>_<left|right>.jpg`), which is what makes the inter-eye analysis possible; APTOS has no eye/patient info, so it's treated as unpaired.

## Phase 1 — Data foundation (manifest, dedup, splits)
- Built `data/manifests/manifest.csv`: one row per image with `image_id, filepath, patient_id, eye, partner_id, grade, dataset, width, height, sha256`.
- Computed `partner_id` for every EyePACS image (the fellow eye of the same patient), null for APTOS.
- Ran near-duplicate detection (perceptual hashing) across the dataset — found and logged duplicate/near-duplicate candidates.
- Built three frozen train/test split protocols and saved them to `data/splits/`:
  - **P1** — image-level split (deliberately leaky: same patient's two eyes can land on both sides). Used as a baseline to show what happens *without* patient-level control.
  - **P2** — patient-level split (both eyes of a patient always stay together). This is the **primary, honest** protocol.
  - **P3** — cross-dataset split (train on EyePACS → test on APTOS, and vice versa), both directions.
- P1's split file uniquely records `partner_in_train` per test image — this is what makes the Claim 2b ablation possible later.
- Wrote 20 acceptance tests for Phase 1 — all passing.
- Committed with git, seed = 42 throughout.

## Phase 2 — Image preprocessing
- Built a shared crop/resize/pad pipeline (`preprocess.py`) so every image is processed identically regardless of source dataset.
- Built a disk cache of preprocessed images (`build_cache.py`).
- Ran a full local pass over all images; caught and investigated a low-mean-pixel diagnostic (some images unexpectedly dark).
- Found and excluded **8 flat-black EyePACS captures** (corrupted/blank images) — rebuilt P1/P2/P3 splits after exclusion (still seed 42), documented the exact 8 IDs and the exclusion criterion in the README.
- Fixed a stray-file bug in the dedup script.
- Used a **soft-exclude** pattern throughout: excluded images are flagged in the manifest (`excluded`, `excluded_reason` columns), never deleted — every downstream script filters on this flag via a shared `load_active_manifest` helper.

## Phase 3 — Frozen feature extraction
- Wrote `src/features/extract.py`: loads a frozen (non-fine-tuned) pretrained backbone from `timm`, strips the classifier head, and extracts one feature vector per image, cached to `.npz` (features + aligned image IDs, reordered to a canonical order on finalize).
- Built in resumability: checkpoints every N images, atomic writes, safe to interrupt and restart.
- Hit and fixed a real bug: on Windows, `multiprocessing` uses "spawn" instead of "fork", which requires the dataset class to be defined at the top of the file instead of nested inside a function — otherwise it can't be pickled to worker processes. Fixed and verified.
- Extracted features for all active images (38,780) with three backbone configs, run for real on your machine:
  - EfficientNet-B0 @ 224px — 84 seconds, 460 images/sec
  - EfficientNet-B0 @ 384px — 205 seconds, 189 images/sec
  - ResNet-50 @ 224px — 81 seconds, 476 images/sec
- Wrote 13 acceptance tests (row counts, ID alignment, no-NaNs, and a "sanity" test that a simple logistic regression on the frozen features gets reasonable QWK on the P2 split).
- One test initially failed (ResNet-50 QWK slightly below threshold). Investigated properly instead of just changing the number:
  - Confirmed all three configs use byte-identical P2 train/test ID sets (rules out a wiring bug).
  - Confirmed ResNet-50 features are fine: an ordinal (ridge regression + optimized thresholds) head scores well above the original threshold on the same features.
  - Confirmed ResNet-50 still has good discriminative signal (AUROC ≈ 0.80 for referable DR).
  - Conclusion: the multinomial classifier head, not the features, was the weak link (expected — plain softmax discards the ordinal structure that QWK rewards).
  - Lowered the sanity threshold from 0.5 to 0.4 with the evidence documented directly in the test file.
- All 13 tests now pass on your machine.

## Phase 4 — Core paper claims (in progress)

### Claim 1b — Stratified inter-eye correlation
- Question: does the strong left-eye/right-eye grade correlation hold up once you control for the left eye's grade (i.e., is it a real clinical signal or just "healthy patients have two healthy eyes")?
- Ran for real (17,555 patients with both eyes, post-exclusion):
  - Overall QWK between left and right eye grades: **0.855**
  - Restricted to patients with left eye ≥ Mild NPDR (grade ≥ 1): QWK **0.656** (n=4,689)
  - Restricted to referable DR (grade ≥ 2): QWK **0.538** (n=3,480)
  - A permutation test confirms this correlation is real, not chance (p ≈ 0).
- Also checked a "lookup shortcut" (predict right eye = left eye's grade) at every left-eye grade level. Found the real, honest pattern: it's very accurate for most grades (72–94%) but collapses specifically at Mild NPDR (grade 1: only 45.8% accuracy) — patients often start diverging between eyes right at this early stage.
- Caught two of my own mistakes before finalizing: (1) an early per-grade QWK metric was mathematically meaningless and was removed in favor of accuracy, which is what the plan actually asked for; (2) a figure caption contradicted its own chart — fixed to state the real finding (shortcut survives in aggregate, but genuinely fails at grade 1).

### Claim 2 — Does patient-level leakage inflate scores? (P1 vs P2 vs P3)
- Question: if you don't control for patients (P1), do you get inflated/unrealistic performance vs. the honest patient-level split (P2)?
- Trained simple classifiers (multinomial and ordinal) on the frozen EfficientNet-B0 @ 224 features for all four protocols, real run on your machine:
  - P1 (leaky): multinomial QWK 0.51–0.54, ordinal QWK 0.60
  - P2 (honest): multinomial QWK 0.52–0.56, ordinal QWK 0.62
  - P3 EyePACS→APTOS: multinomial QWK 0.64–0.66, ordinal QWK 0.75
  - P3 APTOS→EyePACS: multinomial QWK 0.29–0.30, ordinal QWK 0.29 (small APTOS-only training set)
- Included patient-level bootstrap confidence intervals and an exact paired permutation test comparing P1 vs P2.
- **Notable finding**: P1 does NOT score higher than P2 here — if anything it's very slightly lower. This is the opposite of what you'd expect if patient-level leakage were inflating frozen-feature results. (Likely explanation: with frozen, non-fine-tuned features, the model can't actually "memorize" a specific eye's exact appearance — leakage effects like this typically show up more with fine-tuned models that can memorize image-specific detail.)
- Added the ordinal head as an independent second check specifically to rule out "is this just because of a weak classifier head" — it wasn't; both heads agree.
- Fixed a plotting bug along the way (a figure crashed because the bar height and its error bar came from two different model fits that could disagree).

### Claim 2b — Partner-eye ablation (does having the fellow eye in training leak into test performance?)
- Question: for P1 test images, does it matter whether their partner eye was in the training set?
- Real run on your machine (P1 test set, EfficientNet-B0 @ 224 features): 3,647 test images had their partner eye in training, 1,641 did not (531 APTOS images have no partner concept and were excluded from this specific analysis).
- Multinomial head: QWK with partner in training = 0.490, without = 0.438 (+0.051 difference). After matching the grade distributions between groups, the difference shrinks to +0.018. Bootstrap CI includes zero. Permutation test p = 0.185 → **not statistically significant**.
- Ordinal head (robustness check): same direction, slightly larger effect (+0.061 raw, +0.036 grade-matched), permutation p = 0.0505 — right at the edge of conventional significance, but still not under it.
- Both heads agree on direction and on "not significant," so this isn't an artifact of which classifier head was used.
- One loose thread noted but not yet investigated: the ordinal head gets 0% accuracy on the rarest class (Proliferative DR, grade 4) in both groups — likely a quirk of the ridge+threshold method on very small classes, not a leakage finding.

## Bugs caught and fixed along the way (full list)
- Windows multiprocessing pickling crash in feature extraction (fixed, verified).
- `.gitignore` was silently excluding the entire feature-extraction source folder (fixed).
- Learned the sandbox environment used for development can't run true background processes across tool calls, and can't fit realistic-sized model training in its time/CPU budget — adjusted workflow so heavy compute always runs on your machine, validated first with synthetic-data tests.
- Fixed a mathematically-degenerate metric in the Claim 1b script before it was reported.
- Fixed a self-contradictory figure caption in Claim 1b before it was reported.
- Fixed a crash in the Claim 2 figure (mismatched bar height vs. error bar source).
- ResNet-50 sanity-test threshold was investigated with real evidence (not just lowered blindly) before being adjusted.

## Part 8.4 — three-way comparison figure (done)
- Built `src/experiments/summary_comparison.py`: one figure putting side by side (a) the label-only lookup shortcut, (b) the P1 model, (c) the P2 model, (d) partner-present vs partner-absent. Pulls every number from the already-written result JSONs — computes nothing new.
- Ran for real (EfficientNet-B0 @ 224): label-only lookup QWK 0.838, P1 multinomial 0.536 / ordinal 0.595, P2 multinomial 0.559 / ordinal 0.621, partner-present 0.490, partner-absent 0.438.
- `results/summary_comparison.json`, `figures/figure1_summary_comparison.png` — committed.
- Added `run.ps1` commands to rerun Claim 2 / Claim 2b for the other two feature configs (EfficientNet-B0 @ 384, ResNet-50 @ 224), writing to separate result/figure files so they don't overwrite the EfficientNet-B0 @ 224 run. Not yet run — up to you whether/when to run these.

## Open decision — parked for later (per your instruction 2026-08-27)
**Not resolved yet, revisit when ready:** how to frame Claims 2 and 2b in the paper, given that the honest result is a genuine negative finding under frozen-feature evaluation:
- P1 (leaky split) does not outperform P2 (honest split) on frozen features — 0.536 vs 0.559, difference not significant (p=0.125). If anything P1 is a touch lower.
- The partner-eye ablation found no statistically significant leakage effect either (multinomial p=0.185, ordinal p=0.05 — borderline but not under the 0.05 cutoff).
- Two candidate framings to weigh later: (a) report this as a genuine, informative negative result — "leakage does not manifest under frozen-feature evaluation, which may explain why prior work using frozen/linear-probe evaluation missed this effect" — or (b) hold off on any claim about leakage until Phase 6 (fine-tuning), where a model can actually memorize image-specific detail and a leakage effect (if real) would be expected to show up.
- This is the MASTER_PLAN.md §8.4 checkpoint ("🛑 Report all Phase 4 results to the user before continuing — this is where the paper's contribution is either confirmed or not"). Nothing downstream depends on this decision yet, so it's safe to leave parked.

## Phase 5 — Claim 3, the both-eyes model (done — real, confirmed positive result)
- Built `src/experiments/claim3_both_eyes.py` per MASTER_PLAN.md Part 9: compares four ways to get a PATIENT-level grade from frozen features — (A) grade each eye separately then take the worse one (the standard clinical approach), (B) glue both eyes' feature vectors together and train on that, (C) average both eyes' feature vectors and train on that, (D) the label-only lookup shortcut, cited for reference only. Patient label is defined as the worse of the two eyes, per the plan.
- Scope: EyePACS patients with both eyes only (APTOS has no eye-pairing info), evaluated under the honest P2 split.
- Real run (EfficientNet-B0 @ 224, 2,629 test patients):
  - Arm A (per-eye, then max): QWK 0.494
  - Arm B (concat features): multinomial 0.414 (significantly *worse* than A), ordinal 0.483 (no significant difference)
  - Arm C (mean-pooled features): multinomial 0.457 (significantly *worse* than A), **ordinal 0.569 — significantly *better* than A** (+0.075, 95% CI [0.045, 0.105], confirmed by a paired bootstrap check, not just a raw comparison)
  - Arm D (label-only lookup, cited for reference): 0.838
- **This is a real, confirmed positive result**: averaging both eyes' frozen features together and using the ordinal head beats grading each eye separately and taking the worse one. It's specifically an ordinal-head effect — the plain multinomial head does not show this benefit (consistent with the pattern seen throughout this project: QWK rewards the ordinal structure a softmax head throws away).
- Along the way, caught and fixed two things before trusting the "usable method found" verdict: (1) the script originally only compared point estimates, which isn't evidence of a real effect on its own — added a proper paired bootstrap CI on the arm-vs-arm difference before calling anything confirmed; (2) a real test bug surfaced by your run (pandas on your machine normalizes a bare `None` mixed with strings in a column to `NaN`, differently from the sandbox's pandas version) — fixed the test to check the actual invariant (`pd.isna`) instead of an implementation detail.
- 7 unit tests (`tests/test_claim3_both_eyes.py`) plus an end-to-end synthetic-fixture validation in the sandbox, all passing.
- **Checked for robustness across all three feature configs, not just one** — reran on EfficientNet-B0 @ 384 and ResNet-50 @ 224:
  - effnetb0@224: diff +0.075, 95% CI [0.045, 0.105] — significant
  - effnetb0@384: diff +0.072, 95% CI [0.046, 0.098] — significant
  - resnet50@224: diff +0.063, 95% CI [0.029, 0.097] — significant
  - Same effect, same direction, every time. This is a backbone-independent result, not a fluke of one feature space — as solid a positive finding as this project has produced so far.
- `results/claim3_both_eyes*.json` (3 configs), `results/claim3_robustness.json`, `figures/figure4_claim3_both_eyes*.png` (3 configs), `figures/figure4b_claim3_robustness.png` — all committed.

## Where things stand right now
- Phases 1–3 are fully complete, tested, and committed.
- Phase 4's three claims (1b, 2, 2b) plus the Part 8.4 summary figure have all been run for real on the EfficientNet-B0 @ 224 features and results are committed to git.
- The leakage-framing decision above is intentionally parked, not blocking further work. Next up: Phase 5 (Claim 3, both-eyes model) — see "Next Steps".
