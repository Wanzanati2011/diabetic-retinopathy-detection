# MASTER_PLAN.md
### Complete execution brief — "Two Eyes, One Patient" diabetic retinopathy study

> **You are an AI coding agent (Claude Code / Cowork) executing this project end to end.**
> This document is self-contained and covers everything from the current state to a submitted paper.
> Read all of it before writing code. Work through the phases in order.

**Project state:** Step 0 complete. Verdict STRONG. Ready to begin Phase 1.
**Your next action:** Phase 1, §5.1.

---

# PART 0 — HOW TO OPERATE

## 0.1 Standing rules

| Rule | Why |
|---|---|
| **Work phases in order.** Each phase has an ACCEPTANCE TEST. Report pass/fail explicitly before moving on. | Later phases depend on invariants created earlier. |
| **If a model result looks too good, treat it as a bug.** | See Part 14. This project has one dominant failure mode. |
| **Every number goes to JSON in `results/`.** | These become the paper's tables. Console output is not a record. |
| **Never silently modify a frozen split.** | Ask the user first and explain why. |
| **Seed everything; log the git commit hash with every run.** | Reproducibility is a claim this paper makes. |
| **Don't expand scope.** | See §3.5. Default answer to new features is no. |
| **Explain reasoning, keep it brief.** | The user is learning and this is likely their first paper. |
| **Pause at the checkpoints marked 🛑.** | Those are decision points for the user, not for you. |

## 0.2 Environment notes from Step 0

- pip target `/sessions` was full; packages were installed to `/tmp/pylibs`. **Reuse that target.** Prefix commands with `PYTHONPATH=/tmp/pylibs` or install with `--target /tmp/pylibs`.
- Project root is the `mini` folder. Paths already discovered and written to `mini/configs/paths.yaml`.
- Check disk space before the preprocessing cache step — it needs ~25 GB.

---

# PART 1 — WHAT THIS PROJECT IS

## 1.1 The domain

Diabetes damages the retina and causes preventable blindness. It is silent until the damage is permanent, but one annual retinal photograph catches it early. The bottleneck is that only trained ophthalmologists can read those photographs, and there are far too few — especially in rural India.

Grades follow the 5-level ICDR scale:

| Grade | Name | Action |
|---|---|---|
| 0 | No DR | Annual re-screen |
| 1 | Mild NPDR — **microaneurysms only**, ~10px lesions | Annual re-screen |
| 2 | Moderate NPDR | **Refer** |
| 3 | Severe NPDR | Refer urgently |
| 4 | Proliferative DR | Refer urgently, sight-threatening |

**Referable DR (rDR) = grade ≥ 2.** This binary drives real clinical action and matters more than the 5-class number.

## 1.2 The problem with the literature

Thousands of papers train CNNs on this exact public data. Reported results are implausibly high — many claim agreement with graders that exceeds the agreement between two human graders.

**The suspected mechanism:** every patient contributes a left and a right eye, sharing the same systemic disease, duration, and glycaemic control. Split randomly **by image** and one patient's left eye lands in training while their right eye lands in testing. The model has effectively seen the answer.

Nearly every public implementation splits by image.

---

# PART 2 — WHAT WE ALREADY FOUND (Step 0, complete)

## 2.1 Results

Analysis of `labels/trainLabels15.csv`, 35,126 rows → 17,563 patients, every one with both eyes. `rows_unparsed = 0`. Pooled grade counts match published EyePACS totals exactly, so the parse is trustworthy.

| Measurement | Value |
|---|---|
| **Inter-eye quadratic weighted kappa** | **0.8545** (95% CI 0.846–0.863) |
| Exact agreement between a patient's two eyes | 87.2% |
| Agreement within one grade | 95.7% |
| **Label-only lookup QWK** (predict one eye from the other's *grade*, no pixels) | **0.838** |
| Referable-DR agreement (grade ≥ 2) | 94.0% |
| Referable-DR kappa | 0.808 |

**Verdict: STRONG.** Well above the 0.60 threshold, and above the 0.65–0.80 that was expected.

## 2.2 Why 0.838 is the important number

A five-row lookup table, using **no image data whatsoever**, achieves QWK 0.838. Published models trained on image-level splits typically report 0.85–0.92.

This does **not** yet prove those models are exploiting the shortcut. It establishes the **ceiling** of what the shortcut is worth. Phase 4 tests whether models actually take it.

State this carefully in the paper. Overclaiming here is the fastest way to get rejected.

## 2.3 Two consequences for the plan

**A refinement that is now mandatory (§3.3):** ~73% of eyes are grade 0, so a reviewer will immediately ask whether the correlation is just "healthy patients have two healthy eyes." QWK is chance-corrected, so 0.8545 is genuinely meaningful — but you must demonstrate this explicitly with a stratified analysis. Do not skip it.

**A new experiment this makes possible (§3.4):** because correlation is this strong, a controlled partner-eye ablation becomes decisive rather than suggestive. This is now the paper's best figure.

---

# PART 3 — THE PAPER

## 3.1 Title

> *Two Eyes, One Patient: Inter-Eye Correlation in Diabetic Retinopathy and Its Consequences for Model Evaluation*

## 3.2 The three core claims

**Claim 1 — Measurement.** ✅ *Done.* Inter-eye QWK 0.8545 across 17,563 patients. The largest clean measurement of this we are aware of.

**Claim 2 — Consequence.** Image-level random splitting leaks patient identity across the train/test boundary and inflates results. Quantified by evaluating identical models under three protocols.

**Claim 3 — Fix.** Since screening photographs both eyes anyway, grade at the **patient level** using both eyes jointly. Show it helps.

## 3.3 Claim 1b — Stratified correlation (MANDATORY, new)

**The reviewer objection to pre-empt:** *"Your correlation is an artifact of class imbalance — 73% grade 0."*

**Run and report all of these:**

| Analysis | What it answers |
|---|---|
| Inter-eye QWK on patients where **left eye ≥ 1** | Does correlation survive when healthy-healthy pairs are removed? |
| Inter-eye QWK on patients where **left eye ≥ 2** (referable) | Does it hold where it clinically matters? |
| Lookup-table QWK on each of the above subsets | Is the shortcut still available for diseased patients? |
| Per-grade conditional distribution: P(right = g′ \| left = g) for all 25 cells | The full picture, not a summary statistic |
| Lookup accuracy **per left-eye grade** | Almost certainly collapses for grades 1, 3, 4 |
| Comparison against a **permuted-patient null**: shuffle patient IDs, recompute | Establishes the chance floor empirically |

**Expected honest finding:** the lookup table probably predicts grade 0→0 and 2→2 well, but fails on grades 1, 3, and 4. Report this plainly. It **strengthens** the paper — it shows exactly which part of published performance is shortcut-driven and which is real.

Write to `results/claim1b_stratified.json`. Produces **Figure 2**.

## 3.4 Claim 2b — The partner-eye ablation (NEW, the decisive experiment)

This is the strongest experiment available and it is cheap. It proves the mechanism rather than merely correlating with it.

**Design:**

1. Train one model on an **image-level (P1) split**.
2. Partition the P1 **test set** into two groups:
   - **Partner-present:** the image's fellow eye was in the training set
   - **Partner-absent:** the fellow eye was not
3. Evaluate the *same model, same weights* on both groups separately.

**Interpretation:**

| Outcome | Meaning |
|---|---|
| Partner-present ≫ partner-absent | **Direct proof of leakage.** Same model, same weights, only difference is whether the partner eye was seen. |
| Roughly equal | The model isn't exploiting the shortcut. Report honestly — still a valuable negative result. |

**Critical detail:** the two groups must be matched on grade distribution, or a difference could reflect case difficulty rather than leakage. Either stratify the comparison by grade, or resample to match, and say which. Also report bootstrap CIs on the difference and a permutation test.

Write to `results/claim2b_partner_ablation.json`. Produces **Figure 3 — the paper's headline figure.**

## 3.5 Scope limits

**IN:** Kaggle-only data (already downloaded), standard `timm` backbones, Gradio web app.

**OUT — do not propose:** novel architectures, GANs, datasets needing forms or credentialing (mBRSET, official Messidor-2, FGADR), hospital data collection, IRB, mobile deployment, ensembles beyond 2 models.

## 3.6 Target venues

IEEE ISBI · MICCAI workshop · IEEE Access · Scientific Reports · IEEE JBHI. Post the arXiv preprint the day results lock.

---

# PART 4 — HARDWARE AND SPEED STRATEGY

## 4.1 Machine

| Component | Spec | Consequence |
|---|---|---|
| GPU | **RTX 4050 Laptop, 6 GB VRAM** | Hard limit. 512px + batch 16 = OOM. |
| CPU | i7-13700H, 14C/20T | Strong. Data loading won't bottleneck. |
| RAM | 16 GB | Cap DataLoader workers at 6. |
| OS | Windows | Native PyTorch CUDA, not WSL2. Wrap training in `if __name__ == "__main__":`. |

**Before any training run:** plug in the charger (on battery the GPU is heavily power-limited), set Windows power to Best Performance, NVIDIA Control Panel → Prefer Maximum Performance, disable sleep, and elevate the laptop for airflow. Expect 20–35% thermal throttling over long runs regardless.

## 4.2 The speed principle — frozen feature caching

**This is what makes the project fit in days rather than weeks.**

1. Take a pretrained backbone. **Freeze it entirely.**
2. Run every image through it **once**. Save the feature vector, not the image. ~18k images at 224px ≈ **15 minutes**.
3. Every subsequent experiment is a linear head on an 18k × 1280 matrix — **5–20 seconds on CPU**.

All of Claims 1b, 2, 2b, and 3 are answered on cached features. Full end-to-end fine-tuning happens **only twice**: once for headline numbers, once for the app.

## 4.3 Fine-tune config (when you do it)

```yaml
model: tf_efficientnet_b0
image_size: 224            # 384 for the final model only
batch_size: 32             # 8 at 384
grad_accum_steps: 1        # 4 at 384
epochs: 15
amp: fp16
optimizer: adamw
lr_head: 3.0e-4
lr_backbone: 3.0e-5
scheduler: cosine
warmup_pct: 0.05
ema_decay: 0.999
freeze_backbone_epochs: 2
sampler: class_balanced
early_stop_metric: val_qwk
early_stop_patience: 4
num_workers: 6             # NOT 14 — each worker copies batches into RAM
pin_memory: true
persistent_workers: true
```

**VRAM guide (fp16, forward+backward):**

| Model | 224px | 384px | 512px |
|---|---|---|---|
| effnetb0 | bs 32 ✅ | bs 16 ✅ | bs 8 ✅ |
| resnet50 | bs 32 ✅ | bs 8 ✅ | bs 4 ⚠️ |
| effnetb3 | bs 16 ✅ | bs 8 ⚠️ | ❌ OOM |

On OOM: halve batch → raise grad_accum → enable gradient checkpointing (−30% speed, −40% VRAM) → drop resolution.

Skip `torch.compile` on Windows.

---

# PART 5 — PHASE 1: DATA FOUNDATION (CPU only, ~half a day)

> **Write no model code in this phase.** These are the invariants everything depends on.

## 5.1 Build the manifest

`src/data/manifest.py` → `data/manifests/manifest.csv`:

| Column | Notes |
|---|---|
| `image_id` | Unique across datasets, e.g. `eyepacs_4521_left` |
| `filepath` | Relative to project root |
| `patient_id` | `eyepacs_4521` / `aptos_<id_code>` |
| `eye` | `left` / `right` / `unknown` |
| `partner_id` | **NEW — required for Claim 2b.** The `image_id` of the fellow eye, or null |
| `grade` | 0–4 |
| `dataset` | `eyepacs` / `aptos` |
| `width`, `height` | Original dimensions |
| `sha256` | File hash |

**APTOS has no patient IDs.** Each APTOS image becomes its own patient, `eye = unknown`, `partner_id = null`. Log a warning; this goes in Limitations.

**ACCEPTANCE TEST 5.1**
- Row count == image files found on disk
- Zero nulls in `patient_id`, `grade`
- Grades are exactly {0,1,2,3,4}
- EyePACS: 17,563 unique patients, every one with exactly 2 images
- Every EyePACS row has a non-null `partner_id`
- Print the `dataset` × `grade` cross-tab; EyePACS should total 25,810 / 2,443 / 5,292 / 873 / 708

## 5.2 Deduplication

`src/data/dedup.py`: pHash (`imagehash.phash`, `hash_size=16`) over the union of datasets. Flag pairs at Hamming distance ≤ 5. Add a `dup_group` column; duplicates share a group ID. **Splitters must keep a whole `dup_group` on one side of any split.**

Write `results/duplicates.json`.

**ACCEPTANCE TEST 5.2** — Report the count including cross-dataset pairs. **Zero found means the hash is broken** — verify by hashing a deliberately copied file.

## 5.3 The split protocols

`src/data/splits.py` writes JSON mapping `image_id → {train|val|test}`.

**P1 — image-level random.** 70/15/15 over images, stratified by grade. **Deliberately flawed.** Comment it as such. Additionally record, for each test image, whether its partner is in train — this drives Claim 2b.

**P2 — patient-level.** Group by `patient_id`; stratification key = the patient's **max** grade; stratified 70/15/15 **over patients**; all of a patient's images go to their fold; enforce `dup_group` integrity.

**P3 — cross-dataset.** Train+val on dataset A (patient-level within A), test on all of B. Both directions.

**ACCEPTANCE TEST 5.3 — permanent pytest suite, all must pass:**
```python
test_p2_no_patient_overlap()          # patient_id sets disjoint across folds
test_p2_no_dupgroup_overlap()
test_p3_no_dataset_overlap()
test_splits_deterministic()           # same seed -> byte-identical
test_all_images_assigned()            # each row in exactly one fold
test_grade_distribution_similar()     # folds within 3 percentage points
test_p1_partner_flag_present()        # every P1 test image has partner_in_train recorded
```

🛑 **Do not proceed until all seven pass. Report the results to the user.**

## 5.4 Frozen test declaration

Commit to `README.md`:

```
FROZEN TEST SETS — declared <date>, commit <hash>

P2 test: patient-level held-out split, seed 42
P3 test: APTOS (full) and EyePACS (full), per direction

PRIMARY endpoint:   referable-DR sensitivity at fixed 90% operating point, Protocol 2
SECONDARY endpoint: 5-class QWK

Evaluated ONCE, at the end of Phase 7.
Any earlier evaluation must be logged here with a reason.
```

---

# PART 6 — PHASE 2: PREPROCESSING CACHE (~1–2 hours)

`src/data/preprocess.py`

> ⚠️ **This module is imported by BOTH training and the app. Single source of truth. Never reimplement it elsewhere.** Preprocessing mismatch is the most common reason a working model produces garbage in deployment.

```python
def preprocess(image_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """
    1. Detect the retinal circle:
         gray = channel mean
         mask = gray > 0.1 * gray.max()
         bounding box of the largest connected component
    2. Crop to it (removes black borders — typically 30-40% of pixels)
    3. Resize longer side to `size`, preserving aspect ratio
    4. Pad to square (size, size) with black
    5. Return uint8 RGB
    """
```

Also implement, as a config-controlled ablation (not the default):
```python
def ben_graham(img, sigma_ratio=30):
    """2015 Kaggle winner's local contrast normalisation."""
    blur = cv2.GaussianBlur(img, (0, 0), img.shape[1] / sigma_ratio)
    return cv2.addWeighted(img, 4, blur, -4, 128)
```

**Cache to `data/processed/{size}/{image_id}.png` at 224 and 384.** Never preprocess inside the DataLoader.

**ACCEPTANCE TEST 6.1**
- Contact sheet of 20 random processed images per dataset → `figures/sanity_crops.png`
- 🛑 **Tell the user to look at it personally.** Checking that the crop didn't slice off half the retina is the one step that can't be automated.
- Assert no processed image has mean pixel value < 5
- Report cache size and median processing time

---

# PART 7 — PHASE 3: FROZEN FEATURE EXTRACTION (~15 min)

`src/features/extract.py`

```python
backbone = timm.create_model('tf_efficientnet_b0', pretrained=True, num_classes=0)
backbone.eval().cuda()
# forward every cached image once, no grad, fp16
# save features (N, 1280) + aligned image_id array to features/effnetb0_224.npz
```

Extract for: `effnetb0 @224`, `effnetb0 @384`, `resnet50 @224`. Three files, ~45 minutes total.

**ACCEPTANCE TEST 7.1** — Feature array row count matches the manifest; `image_id` order is recorded and aligned; no NaNs; a quick logistic regression on P2 gives QWK > 0.5 (sanity — frozen features are weaker than fine-tuning, this just confirms the pipeline works).

---

# PART 8 — PHASE 4: CLAIMS 1b, 2, 2b ON CACHED FEATURES (~2–3 hours, mostly CPU)

## 8.1 Claim 1b — stratified correlation

Pure label analysis, no features needed. Implement everything in §3.3. → `results/claim1b_stratified.json`, **Figure 2**.

## 8.2 Claim 2 — protocol comparison

Train the same linear head (logistic regression or a small MLP on cached features) under P1, P2, P3. Three seeds each — with cached features, seeds are nearly free, so use 5.

| Protocol | Expected QWK (frozen features) | Expected QWK (fine-tuned) |
|---|---|---|
| P1 image-level | 0.80–0.88 | 0.88–0.93 |
| P2 patient-level | 0.70–0.80 | 0.78–0.85 |
| P3 cross-dataset | 0.50–0.65 | 0.60–0.72 |

Frozen-feature numbers sit below fine-tuned ones. That's expected — **the gaps are what the paper is about**, and gaps survive.

Paired permutation test between P1 and P2; report p-value and bootstrap CIs (resample **patients**, not images).

→ `results/claim2_protocols.json`, **Figure 1**.

## 8.3 Claim 2b — partner-eye ablation

Implement §3.4 on the P1-trained head. Report:
- QWK on partner-present vs partner-absent test images
- The same, stratified by grade
- Grade-matched resampled comparison
- Bootstrap CI on the *difference*
- Permutation test

→ `results/claim2b_partner_ablation.json`, **Figure 3 (headline)**.

## 8.4 The three-way comparison plot

One figure putting side by side:
- Label-only lookup (0.838, no pixels)
- P1 image-level model
- P2 patient-level model
- Partner-present vs partner-absent

This single panel tells the whole story. Consider making it Figure 1.

🛑 **Report all Phase 4 results to the user before continuing.** This is where the paper's contribution is either confirmed or not.

---

# PART 9 — PHASE 5: CLAIM 3, THE BOTH-EYES MODEL (~1 hour)

On cached features, concatenate the two eyes' feature vectors per patient (1280 × 2 = 2560) and predict a **patient-level** label.

**Define the patient label explicitly and justify it:** use `max(left, right)`, because clinical management follows the worse eye. State this in the paper.

**Comparison arms:**

| Arm | Description |
|---|---|
| A | Per-eye grading, then take max — the standard approach |
| B | Concatenated both-eye features → patient grade |
| C | Attention/mean pooling over the two eyes |
| D | **Label-only lookup** (0.838) — the shortcut ceiling, for reference |

Evaluate under **P2 only** (patient-level; P1 is meaningless here). Report 5-class QWK and referable-DR sensitivity/specificity at a fixed 90% sensitivity operating point.

**If B or C beats A:** you have a usable method. **If not:** report it honestly — a negative result on Claim 3 doesn't hurt a paper whose headline is Claims 1 and 2.

→ `results/claim3_both_eyes.json`, **Figure 4**.

---

# PART 10 — PHASE 6: TWO REAL FINE-TUNES (~overnight)

Frozen features prove the claims; fine-tuned models give the headline numbers reviewers expect.

**Run 1 — headline.** `effnetb0 @224`, fast config, under P1 and P2 (2 runs, 2 seeds each = 4 runs, ~40 min each ≈ 3 hours). Rerun the Claim 2b partner ablation on the fine-tuned P1 model — **this is the version that goes in the paper.**

**Run 2 — the app model.** `effnetb0 @384`, P2 split, best config, 1 run (~2 hours).

**ACCEPTANCE TEST 10.1**

| P2 QWK | Verdict |
|---|---|
| 0.00–0.40 | Broken. Check label alignment, LR, normalization. |
| 0.40–0.70 | Undertrained or preprocessing bug. |
| **0.75–0.85** | ✅ Correct. Proceed. |
| 0.90–0.95 | Suspicious. Run Part 14. |
| > 0.95 | 🚨 Definitely a leak. Stop. |

**Also expected and correct: grade-1 recall 0.20–0.45.** Grade 1 is microaneurysms only, physically destroyed by downsampling. **Do not try to fix this.** Report it — it supports a secondary finding that resolution matters more than architecture.

**P1 is supposed to look inflated.** Never "improve" it.

---

# PART 11 — PHASE 7: CALIBRATION AND REJECT OPTION (~2 hours)

`src/calibrate.py`

**Temperature scaling:** fit a single scalar T on the **validation** set by minimising NLL of `softmax(logits / T)`. Never fit on test. Report ECE and reliability diagrams before/after, separately for P2 test and each P3 test set.

**Expect and report:** in-domain calibration transfers poorly across datasets. This is a known effect and reproducing it strengthens the case for abstention.

**Reject option:** confidence = max softmax probability post-temperature. Sweep threshold τ. At each τ record coverage, selective QWK, selective rDR sensitivity/specificity.

**Choose τ by a pre-stated rule, never by looking at test:** *the smallest τ such that selective rDR sensitivity on validation ≥ 0.90.* Write τ, T, and the rule to `app/release/thresholds.json`.

**ACCEPTANCE TEST 11.1** — Rejecting the least-confident ~20% should lift selective QWK by ~+0.05–0.10. If it doesn't, the confidence score is uninformative — check calibration, then report honestly.

**Grad-CAM** via `pytorch-grad-cam` for the app and a qualitative figure. **Include failure cases** — a panel showing only successes reads as dishonest.

→ **Figures 5 and 6**.

---

# PART 12 — PHASE 8: THE APPLICATION (~half a day)

## 12.1 Release artifacts

```
app/release/
├── best_model.pt      # state_dict + architecture NAME — never a pickled nn.Module
├── config.yaml        # exact training config
├── thresholds.json    # {"temperature": T, "reject_tau": τ, "rule": "...", "fitted_on": "<val hash>"}
└── MODEL_CARD.md      # training data, metrics with CIs, intended use, limitations
```

## 12.2 Gradio app

```python
# app/app.py
import gradio as gr, torch, numpy as np
from src.data.preprocess import preprocess   # ← THE SAME FUNCTION USED IN TRAINING

GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]

def predict(image):
    x = to_tensor(preprocess(np.array(image), size=CFG.image_size))
    with torch.no_grad():
        logits = model(x.unsqueeze(0))
    probs = torch.softmax(logits / TEMPERATURE, dim=-1)[0]
    conf, grade = probs.max(0); grade = int(grade)

    if conf < REJECT_TAU:
        verdict  = "⚠️ UNCERTAIN — refer to an ophthalmologist"
        referral = "REFER (low model confidence)"
    else:
        verdict  = f"Grade {grade} — {GRADE_NAMES[grade]}"
        referral = "REFER (referable DR)" if grade >= 2 else "Routine annual follow-up"

    return verdict, referral, float(conf), bar_chart(probs), gradcam_overlay(x)

DISCLAIMER = ("Research prototype trained on public datasets. "
              "NOT a medical device. NOT validated for clinical use. "
              "Not a substitute for examination by a qualified ophthalmologist.")

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload retinal fundus photograph"),
    outputs=[gr.Text(label="Assessment"), gr.Text(label="Recommendation"),
             gr.Number(label="Confidence"), gr.Plot(label="Grade probabilities"),
             gr.Image(label="Model attention (Grad-CAM)")],
    title="Diabetic Retinopathy Screening Assistant (Research Prototype)",
    description=DISCLAIMER, article=DISCLAIMER,
).launch(share=True)
```

**Optional, and a nice touch given the paper's thesis:** allow uploading *both* eyes and run the Claim 3 patient-level model. The app then demonstrates the paper's own finding.

**Mandatory UI elements:** grade in clinical language (not a bare integer); referable yes/no; confidence or explicit UNCERTAIN; probability bars; Grad-CAM; **disclaimer visible on screen**, not buried in a README.

## 12.3 ACCEPTANCE TEST 12.1 — the critical one

Run 100 test images through **both** the app's prediction function and the evaluation pipeline. **Predictions must match exactly.** Write it as a pytest.

If they differ: preprocessing mismatch. Fix by importing `preprocess()` rather than reimplementing it.

---

# PART 13 — PHASE 9: FIGURES AND PAPER

## 13.1 Figures

| # | Content | Source |
|---|---|---|
| 1 | Three-way comparison: label-only lookup vs P1 vs P2, with CIs | §8.4 |
| 2 | Stratified correlation + per-grade conditional distributions | §3.3 |
| 3 | **Partner-present vs partner-absent ablation** — headline | §3.4 |
| 4 | Both-eyes vs per-eye grading | §9 |
| 5 | Risk–coverage curve | §11 |
| 6 | Reliability diagrams before/after temperature scaling | §11 |
| 7 | Grad-CAM panel **including failures** | §11 |
| 8 | App screenshot | §12 |
| T1 | Inter-eye contingency table (5×5) | Step 0 |
| T2 | Protocol comparison, all metrics with CIs | §8.2 |
| T3 | Per-class recall + confusion matrix | §10 |
| T4 | rDR sensitivity/specificity at fixed 90% sensitivity | All |

**Generate every table from `results/*.json` with a script.** Never hand-copy numbers.

## 13.2 Paper skeleton

```
1. Introduction     — lead with the 0.838 lookup number. It's arresting.
2. Related Work     — DR grading; evaluation methodology; shortcut learning
3. Methods          — datasets, protocols, models, the partner ablation design
4. Experiments      — implementation, metrics, statistical procedure
5. Results          — Claims 1, 1b, 2, 2b, 3
6. Discussion       — what this means for the DR literature
7. Limitations      — honest and at length (§13.3)
8. Conclusion
```

## 13.3 Limitations to state explicitly

- EyePACS labels are noisy and largely single-grader
- APTOS lacks patient IDs; its patient-level splitting is approximate
- Correlation is measured on one population (US teleretinal screening); it may differ elsewhere
- Frozen-feature experiments underestimate absolute performance, though the *gaps* are the claim
- No prospective clinical validation
- Single-field, macula-centred imaging misses peripheral lesions
- The app is a research prototype, not a regulated medical device
- **We show the shortcut is available and that a P1-trained model exploits it; we do not claim any specific published paper did so**

## 13.4 Reviewer objections, pre-answered

| Objection | Answer |
|---|---|
| "Correlation is just class imbalance" | §3.3 stratified analysis + permuted-patient null |
| "Everyone already knows to split by patient" | Then show how many public implementations don't. Survey 15 GitHub repos and report the fraction. Cheap and damning. |
| "Frozen features aren't representative" | Phase 6 fine-tuned runs reproduce the gaps |
| "The partner difference could be case difficulty" | Grade-matched resampling + stratified comparison |
| "This is negative/critical work" | Claim 3 provides the constructive fix |

**Do the GitHub survey.** Twenty repos, one column: does it split by patient or by image? It takes an afternoon and it's the most persuasive table in the paper.

---

# PART 14 — TROUBLESHOOTING AND LEAKAGE

**If P2 QWK > 0.90, check in this order:**
1. Patient overlap — `set(train.patient_id) & set(test.patient_id)` must be empty. **#1 cause.**
2. Duplicate-group overlap across splits
3. Randomness in validation/test transforms (eval must be deterministic)
4. Oversampling applied before splitting instead of within the training fold
5. Normalization statistics computed over the whole dataset instead of train only
6. Any threshold or hyperparameter tuned on test
7. Stale preprocessing cache built before dedup
8. Off-by-one join accidentally aligning train and test labels

**Other issues:**

| Symptom | Cause | Fix |
|---|---|---|
| GPU utilisation < 60% | CPU/disk-bound | Confirm cache is being used; raise `num_workers` to 8; keep cache on SSD |
| Epoch time creeping up | Thermal throttling | Cooling pad, elevate laptop, accept ~20% |
| CUDA OOM | 6 GB limit | §4.3 escalation order |
| App ≠ eval predictions | Preprocessing mismatch | §12.3 |
| Grade-1 recall poor | **Expected** | Report it, don't chase it |
| P1 looks inflated | **Intended** | Never fix it |

---

# PART 15 — TIMELINE

| Day | Work | GPU |
|---|---|---|
| 1 | Phase 1 — manifest, dedup, splits, 7 tests passing | None |
| 1 (pm) | Phase 2 — preprocessing cache at 224 and 384 | Light |
| 2 (am) | Phase 3 — frozen feature extraction | ~45 min |
| 2 | Phase 4 — Claims 1b, 2, 2b. **The paper's core.** | Minimal |
| 3 | Phase 5 — Claim 3 both-eyes model | Minimal |
| 3 (night) | Phase 6 — fine-tunes | ~5 h overnight |
| 4 | Phase 7 — calibration, reject option, Grad-CAM | Light |
| 5 | Phase 8 — the app + Acceptance Test 12.1 | None |
| 6–7 | GitHub survey; figures; write Methods and Results | None |
| 8–10 | Introduction, Discussion, Limitations; revise | None |
| 11 | Buffer, then arXiv | — |

**Realistic: core results in 3 days, complete paper in about 2 weeks.**

---

# PART 16 — QUICK REFERENCE

| Situation | Action |
|---|---|
| **Starting now** | Phase 1, §5.1 |
| Split tests fail | Do not proceed. Fix and report. |
| P2 QWK > 0.90 | Part 14 checklist. Investigate, don't celebrate. |
| Grade-1 recall bad | Expected. Report, don't chase. |
| P1 inflated | Intended. Don't fix. |
| App ≠ eval | §12.3 preprocessing mismatch |
| Claim 3 shows no gain | Report honestly. Headline is Claims 1 and 2. |
| User wants new scope | §3.5. Default no. |
| User unsure what's next | Find today in Part 15, give one deliverable |

**Checkpoints where you must stop and report:** after §5.3 (split tests), §6.1 (user inspects crops personally), §8.4 (core results).

**Tone:** the user is learning and moving fast. Explain reasoning briefly. Treat failures as expected and debuggable.

**Your next action: Phase 1, §5.1 — build the manifest, including the new `partner_id` column.**
