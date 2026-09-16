# Development Log — Fundus Console / "Two Eyes, One Patient"

**Purpose of this file:** a complete, honest record of what was investigated, built,
broken, fixed, and shipped for this project, covering the diagnostic work on
Acceptance Test 12.1, the QML/QCNN quantum experiments, the app-checkpoint
retrain, the Fundus Console app's Phase 3/4 build-out, and the GitHub +
Hugging Face deployment. Written so that a future reader (including you,
months from now) can reconstruct not just *what* the final numbers are, but
*why* each decision was made and what was tried and rejected along the way.

This is a log of engineering work, not a polished pitch — it includes the
mistakes (mine and the debugging false starts) alongside the fixes, because
that's the same standard the rest of this project holds itself to
(MASTER_PLAN.md's "no claim ships without a check that could have failed
it").

> **Note on the two GitHub/Hugging Face links below:** these are filled in
> from what you told me directly — double check them against your actual
> repo/Space before treating this as the canonical record.

- **GitHub repo:** https://github.com/vishalyl/diabetic-retinopathy-detection
- **Hugging Face Space (live demo):** https://huggingface.co/spaces/vishal1829/fundus-console

---

## Table of contents

1. [Starting point: what existed before this session](#1-starting-point)
2. [Acceptance Test 12.1 — root-cause investigation and correcting an overclaim](#2-acceptance-test-121)
3. [QML, explained in plain language](#3-qml-explained)
4. [The QCNN performance crisis and three real bugs](#4-qcnn-performance-crisis)
5. [QCNN final results and what they mean](#5-qcnn-final-results)
6. [QML (PQC) track results, for reference](#6-qml-pqc-results)
7. [Retraining the app's checkpoint (the converged recipe)](#7-retraining-the-app-checkpoint)
8. [The WinError 1455 crash and its fix](#8-winerror-1455)
9. [Fundus Console Phase 3: Both Eyes mode + Instrument Card](#9-phase-3)
10. [Fundus Console Phase 4: Quantum Lab tab + Session Log + PDF export](#10-phase-4)
11. [Packaging for GitHub + Hugging Face: README, LICENSE, requirements, runbook](#11-packaging)
12. [Swapping in the converged checkpoint + fixing a real bug in my own Instrument Card](#12-checkpoint-swap)
13. [Re-running Acceptance Test 12.1 against the new checkpoint — confirmed result](#13-acceptance-121-rerun)
14. [Live deployment](#14-live-deployment)
15. [Complete file manifest — everything touched this session](#15-file-manifest)
16. [Open items / honest state of the project as of now](#16-open-items)

---

<a name="1-starting-point"></a>
## 1. Starting point: what existed before this session

Before this stretch of work began, the project already had (from earlier
sessions, not covered blow-by-blow here):

- The full leakage study: Claims 1/1b/2/2b, showing inter-eye QWK 0.8545
  and a label-only lookup table scoring QWK 0.838 — the headline finding
  that patient-level splitting (P2) is mandatory, not optional.
- Claim 3 (both-eyes ensembling): mean-pooled features beat per-eye-then-max
  grading, QWK 0.617 vs 0.543 (paired bootstrap CI [+0.046, +0.098]).
- A deployed Gradio app (`app/app.py`) — the "Fundus Console" — wrapping a
  `tf_efficientnet_b0` checkpoint trained with the original 15-epoch
  `src/train/finetune.py` recipe (`configs/finetune_app.yaml`), scoring
  **test QWK 0.613** on the P2 test set — inside this project's own
  "undertrained or preprocessing bug" acceptance band (0.40–0.70).
- Acceptance Test 12.1 (`tests/test_app.py`), a pytest check comparing the
  app's live inference pipeline against the offline evaluation pipeline on
  real P2 test images, which had found an ~18% grade-mismatch rate — flagged
  as unresolved.
- Grad-CAM overlays already implemented in the Single Eye tab.

---

<a name="2-acceptance-test-121"></a>
## 2. Acceptance Test 12.1 — root-cause investigation and correcting an overclaim

**The question:** why do ~18% of test images get a different predicted grade
through the app's code path than through the offline evaluation pipeline?

Two hypotheses were tested, in order, rather than assumed:

1. **EXIF orientation.** `cv2.imread()` (used when building the training/eval
   image cache) auto-applies EXIF rotation; PIL's `Image.open()` (used in the
   app) does not, unless `ImageOps.exif_transpose()` is called explicitly.
   This was a real, fixable gap — `to_model_input()` in `app/app.py` was
   patched to call `ImageOps.exif_transpose()` before preprocessing, which is
   worth keeping for real phone-camera uploads. But checked against the
   actual 18 mismatched images: **0/18 carried a non-default EXIF orientation
   tag.** Ruled out as the cause of *these* mismatches.
2. **Pixel-level mean absolute difference (MAE)** between the app's
   freshly-decoded image and the cached training/eval image, computed for
   every mismatch: all 18 were **0.8–1.2 out of 255** — consistent with
   ordinary JPEG-decoder differences (cv2/libjpeg-turbo vs. PIL/libjpeg
   decoding the same source file to very slightly different RGB values), not
   a different crop region.

**Conclusion:** this was not a preprocessing bug. The two pipelines produce
near-identical pixels; the model's predicted grade flips anyway on a
meaningful fraction of borderline cases. That's a **model-robustness
finding** — a less-converged model (test QWK 0.613, "undertrained" band) has
narrow decision margins, so sub-1/255 noise alone is enough to tip a
borderline prediction to the neighboring grade. Supporting this, sampled
flips landed between *adjacent* grades (1→0, 2→1, 2→3, 4→2), not randomly
across the 5-way scale — the signature of low-margin predictions, not
corrupted input.

**The overclaiming mistake that got corrected:** an earlier pass at
documenting this (in `app/release/MODEL_CARD.md` and `app/release/config.yaml`)
had framed the EXIF fix as "a real bug this app already caught and fixed" —
which conflated *fixing a real, separate EXIF-handling gap* with *having
found and fixed the cause of the 18% mismatch rate*. It hadn't: EXIF was
ruled out, not confirmed, as the mismatch cause. This was corrected in both
files to honestly walk through both hypotheses tested and land on the actual
conclusion (model fragility, not a preprocessing bug) — see
`app/release/MODEL_CARD.md`'s "What Acceptance Test 12.1 found" section for
the corrected version, which is now further updated per §13 below with the
post-retrain re-run.

---

<a name="3-qml-explained"></a>
## 3. QML, explained in plain language

You asked, essentially: *what is QML, how is it different from
EfficientNet/ResNet, what did your teacher likely mean by suggesting it, what
have we actually done about it, and are we on the wrong track?* Full answer,
condensed for this log:

**What QML is.** Quantum Machine Learning replaces some part of a normal
neural network with a *parameterized quantum circuit* (PQC) — a small
circuit of qubits and quantum gates whose gate angles are trainable
parameters, optimized by gradient descent just like a classical network's
weights. You still write it in Python (this project uses PennyLane), you
still train it with backprop-style optimization, but the actual computation
for that piece happens on a simulated (or, on real hardware, physical)
quantum device instead of as ordinary matrix multiplication.

**How it differs from EfficientNet/ResNet.** Those are convolutional neural
networks — classical, deterministic, well-understood, and *extremely*
well-optimized by a decade of tooling and hardware. A PQC is a fundamentally
different computational substrate: qubits, superposition, entanglement,
measurement. The "dressed hybrid" pattern this project used (frozen CNN
features → PCA → PQC → linear readout) keeps the CNN doing the heavy lifting
of actually *seeing* the image, and only swaps a small classification head
for a quantum one. A "no-CNN" QCNN, by contrast, tries to have the quantum
circuit do convolution-like feature extraction directly on raw pixels — no
classical vision model anywhere in the pipeline.

**What your teacher likely meant.** "Why not use QML?" from a supervisor
almost always means one of: (a) it's a novel, publishable angle for a student
project regardless of whether it wins, (b) they want to see you engage with
an emerging technique, or (c) they're testing whether you understand *why*
you'd reach for it (or not) rather than expecting it to actually beat a
tuned CNN. It essentially never means "replace your CNN, because it will
score higher" — QML on classical, non-quantum data like photographs does not
have any theoretical or empirical track record of beating classical
computer vision at this kind of task in 2024–2026. The realistic
expectation, going in, was a negative or mixed result — which is exactly
what this project found and reported.

**What was actually done about it.** Two full, rigorously-evaluated
experiments, not a token gesture:
- `src/experiments/qml_pqc.py` — the dressed hybrid classifier described
  above, following the closest real precedent found in the literature
  (Ahmed et al. 2024, arXiv:2405.01734, which used the *same datasets* —
  EyePACS/APTOS — and a similar "dressed quantum circuit" pattern). Swept
  4/6/8 qubits × 1/2/3 entangling layers.
- `src/experiments/qcnn_no_cnn.py` — a true Quantum Convolutional Neural
  Network per Cong, Choi & Lukin 2019 (arXiv:1810.03787), operating on raw,
  heavily downsampled pixels with zero classical feature extraction.

Both were evaluated with this project's full rigor: matched data, paired
bootstrap confidence intervals on the difference vs. a classical baseline,
and an honest verdict rather than favorable framing of a loss (see §5–6).

**Is the project on the wrong track?** No. A rigorously-run negative result,
reported honestly with the same statistical standard as every other claim in
this project, *is* the correct outcome of "my teacher told me to try QML."
The wrong track would have been skipping the quantum experiments, or running
them sloppily and reporting a flattering number. Two honestly-negative QML
experiments sitting next to a real, positive methodological finding
(patient-level leakage) is a stronger project than one that only chased a
win.

---

<a name="4-qcnn-performance-crisis"></a>
## 4. The QCNN performance crisis and three real bugs

Running `qcnn_no_cnn.py --mode sweep` initially took **~2,300+ seconds per
epoch** for some configs — projected to run for many hours. Three real,
distinct problems surfaced and were fixed in sequence:

### Bug 1 (mine): recommending `--diff-method adjoint` without reading the script's own warning
`adjoint` differentiation is fast (O(1) per parameter instead of ~2 circuit
evaluations per parameter like `parameter-shift`), so it was the obvious
first suggestion for the slowness. It crashed:
```
QuantumFunctionError: Device ... does not support adjoint with requested circuit
```
Root cause: `adjoint` differentiation does not support `qml.probs()`
measurements on any PennyLane device — only `qml.expval()`. This circuit
returns `qml.probs()`. The script's own `--diff-method` argparse help text
already said this would happen; the mistake was suggesting `adjoint` before
reading that. Owned directly rather than deflected once the crash confirmed
it.

### Fix: `backprop` + `default.qubit`
`backprop` differentiation gives the same O(1)-per-parameter cost as
`adjoint`, but *does* support `qml.probs()` — with one constraint: it only
works on `default.qubit` (PennyLane's native, framework-differentiable
simulator), not `lightning.qubit` (the faster C++ simulator the script was
using). Fixed in `build_qcnn_model()`:
```python
if diff_method == "backprop":
    dev = qml.device("default.qubit", wires=n_qubits)
else:
    dev = qml.device("lightning.qubit", wires=n_qubits)
```
Also updated the `--diff-method` argparse entry with `choices=[...]` and
clearer help text warning about the adjoint/probs incompatibility, so this
mistake can't recur silently. **Result: q8r2 and q8r3 configs that would
have taken hours instead finished in 118.5s and 158.2s.**

### Bug 2 (real, pre-existing): `fit_and_eval()` return-value unpacking reversed
Once the sweep ran fast, `finalize_and_report()` crashed:
```
InvalidParameterError: The 'y2' parameter of cohen_kappa_score must be an array-like. Got -0.0002... instead.
```
`fit_and_eval(...)` (in `claim2_protocols.py`) returns `(qwk, y_pred)` — QWK
first. The call site had it backwards:
```python
pred_baseline, _ = fit_and_eval(...)   # wrong: assigns the QWK float to "pred_baseline"
```
Fixed to:
```python
_, pred_baseline = fit_and_eval(...)
```
Confirmed via grep that this was the only call site of `fit_and_eval` in the
file, so no other instance of the same bug existed.

Both fixes were verified with `py_compile` and pushed. The full sweep then
ran to completion successfully.

---

<a name="5-qcnn-final-results"></a>
## 5. QCNN final results and what they mean

From `results/qcnn_no_cnn_pixels.json`:

| n_qubits | n_conv_reps | val_qwk |
|---|---|---|
| 4 | 1 | 0.0000 |
| 4 | 2 | 0.0000 |
| 4 | 3 | 0.0000 |
| 8 | 1 | 0.0000 |
| 8 | 2 | 0.0000 |
| 8 | 3 | 0.0000 |

**Every single swept configuration converged to QWK 0.0000** — the model
predicts one class for everything, gaining zero ordinal agreement above
chance. Compared against a matched classical baseline (multinomial logistic
regression on the *same* downsampled pixels): baseline QWK
**−0.0002** (patient bootstrap CI [−0.0036, +0.0037]).

**Verdict: `inconclusive`**, not "QCNN loses." The paired difference
(QCNN − baseline) is **+0.0002**, CI **[−0.0034, +0.0033]** — the interval
includes zero, meaning the two are statistically indistinguishable at this
sample size. Both models are equally, completely broken at this task, most
consistent with a **barren-plateau / vanishing-gradient failure** — a
well-known QCNN failure mode where gradients become exponentially small as
circuit depth/qubit count grows, common in small, from-scratch quantum
circuits trained directly on raw, heavily-downsampled pixels with no prior
feature extraction. This is reported as inconclusive precisely because a
0-vs-0 tie doesn't distinguish "quantum circuits are bad at this" from "both
approaches are equally starved of useful signal from 4×4-patch,
heavily-downsampled pixel input" — a fairer, more honest read than declaring
a loser.

---

<a name="6-qml-pqc-results"></a>
## 6. QML (PQC) track results, for reference

From `results/qml_pqc.json` (dressed hybrid classifier: frozen ResNet50 @
224px features → PCA → PQC → linear readout), swept 4/6/8 qubits × 1/2/3
layers, selected by validation QWK:

Best config: **8 qubits, 3 layers** (val_qwk 0.1582). On a matched
subsample (same ~3,000 train / 1,000 val images for all three heads):

| Head | QWK | 95% CI |
|---|---|---|
| Multinomial (classical) | **0.3457** | [0.313, 0.379] |
| Ordinal (classical) | **0.2642** | [0.235, 0.296] |
| QML (q=8, l=3) | **0.1095** | [0.083, 0.135] |

Paired bootstrap differences (QML − classical head), both **significantly
worse**, CI excludes 0 in the negative direction:
- QML − ordinal: **−0.1544**, CI [−0.1936, −0.1169]
- QML − multinomial: **−0.2354**, CI [−0.2730, −0.1957]

**Verdict: `negative_result`.** QML does not beat either classical head at
this qubit/layer budget and training-set size. Reported plainly — a small,
CPU-simulated variational circuit trained on a few thousand images losing to
a Ridge/logistic-style classical head is a plausible, unsurprising outcome,
not a bug to chase. Both this and the QCNN result are surfaced in the app's
**Quantum Lab tab** (§10) exactly as computed here, including the "we lost"
framing — not softened.

Literature grounding (verified by fetching the actual papers, not assumed):
Ahmed et al. 2024 (arXiv:2405.01734) is the closest real precedent — same
datasets (EyePACS/APTOS), a similar "dressed quantum circuit" design,
reporting 97.2–98.5% accuracy in their setup (not independently
reproducible here at the same scale/qubit budget; cited for architectural
precedent, not as a target this run was expected to match). Cong, Choi &
Lukin 2019 (arXiv:1810.03787) is the source of the QCNN conv/pool ansatz
used in `qcnn_no_cnn.py`.

---

<a name="7-retraining-the-app-checkpoint"></a>
## 7. Retraining the app's checkpoint (the converged recipe)

**Motivation:** the deployed app checkpoint (test QWK 0.613) was trained
with the original 15-epoch `src/train/finetune.py` recipe. This project's
*headline* claims were trained with a longer, more-patient "converged"
recipe (`src/train/finetune_converged.py`: 40 epochs, early-stop patience 8,
higher backbone LR) that had already been shown (on the 224px Claim-2b
ablations) to raise test QWK from ~0.55–0.56 (15-epoch) to ~0.63–0.67
(40-epoch/patience-8). The app checkpoint had never gotten that same
treatment.

**New config created:** `configs/finetune_app_converged.yaml` — takes
`finetune_app.yaml` (the real 384px/batch-size-8/grad-accum-4 app recipe)
and applies exactly the two changes `finetune_converged.yaml` made for its
own convergence fix, at 384px instead of 224px:
- `epochs`: 15 → 40
- `early_stop_patience`: 4 → 8
- `lr_backbone`: 3e-5 → 1e-4

Everything else (image size, batch size, grad accumulation, sampler,
augmentation, freeze-backbone-epochs) held constant, so this is an
apples-to-apples "same model, actually converged" comparison, not a
different recipe.

**Honest expectation set beforehand** (documented in the config file's own
comments): based on the 224px ablations, the expectation was an improvement
into the ~0.63–0.67 range — *still* inside the "undertrained" band, not a
jump straight to "correct, proceed." The actual result beat that
expectation (see below) — flagged here specifically so this log doesn't
read as if the good outcome was predicted in advance; it wasn't.

**Result** (`results/finetune_app_converged_p2_class_balanced_seed42.json`):

| | Prior checkpoint (15-epoch) | Converged retrain (40-epoch) |
|---|---|---|
| Best epoch | 11 of 15 | **32 of 40** |
| Best val QWK | 0.6240 | **0.7331** |
| Test QWK | 0.6128 | **0.7160** |
| Referable-DR AUROC | 0.8715 | **0.9117** |
| Acceptance Test 10.1 verdict | undertrained | **"correct -- proceed"** |

The training run itself printed:
```
Training done in 59.3 min. Best val_qwk=0.7331 at epoch 32.
ACCEPTANCE TEST 10.1 VERDICT: test_qwk=0.7160 -> correct -- proceed
```

**A real tradeoff, not a clean win — flagged, not hidden.** Per-class recall
shifted substantially:

| Grade | n | Prior recall | New recall |
|---|---|---|---|
| 0 — No DR | 4,121 | 0.687 | **0.916** |
| 1 — Mild NPDR | 447 | 0.461 | **0.161** |
| 2 — Moderate NPDR | 945 | 0.354 | 0.570 |
| 3 — Severe NPDR | 157 | 0.452 | 0.287 |
| 4 — Proliferative DR | 144 | 0.549 | 0.542 |

Grade-1 recall dropped from 0.461 to **0.161**, well outside the
MASTER_PLAN.md 0.20–0.45 "expected" band for grade-1 (which exists *because*
grade 1 is defined by ~10px microaneurysms destroyed by downsampling, not
because low recall there is normally fine to shrug off at *any* level).
Grade-0 recall rose sharply in the same run. This is a plausible
"stronger healthy-vs-not boundary at grade-1's expense" story on a
class-imbalanced task, but that is a hypothesis, not a verified mechanism —
it has not been separately investigated (e.g., by reading the confusion
matrix row for class 1, which shows most grade-1 errors landing at grade 0).
This is recorded in `app/release/MODEL_CARD.md` as an open, real caveat.

---

<a name="8-winerror-1455"></a>
## 8. The WinError 1455 crash and its fix

The first attempt at this retrain ran the full 59.3 minutes of training
successfully, then crashed at the very last step (final test-set
evaluation):
```
OSError: [WinError 1455] The paging file is too small for this operation to complete.
Error loading "...\torch\lib\curand64_10.dll"
```

**Root cause:** `persistent_workers: true` in the training config keeps the
train/val DataLoader worker processes alive and reused for the entire
59-minute run. The **test** DataLoader's workers, by contrast, are spawned
fresh for the very first time only at the final evaluation step — which
means loading torch's CUDA DLLs (e.g. `curand64_10.dll`) fresh, for the
first time, after 59 minutes of accumulated memory/page-file pressure from
the long-lived train/val workers. On this machine, the OS's page file
couldn't accommodate that fresh DLL load on top of everything already
resident.

**Fix:** set `num_workers: 0` in `configs/finetune_app_converged.yaml` for
the rerun. `finetune_converged.py`'s own DataLoader construction already
guards `persistent_workers` with `persistent = cfg.get("persistent_workers",
True) and n_workers > 0` — so with `num_workers: 0`, `persistent_workers`
becomes inert automatically; no second edit was needed.

**Critically, this cost zero wasted retraining time.** `finetune_converged.py`'s
resume logic reads `ckpt["epoch"]`, sets `start_epoch = ckpt["epoch"] + 1`,
and loops `for epoch in range(start_epoch, cfg["epochs"])`. Since training
had already completed all 40 epochs before the crash, that range was empty
on rerun — the script skipped straight past the (already-done) training
loop to loading `best.pt` and running the final test evaluation, which is
exactly what the confirmation run showed:
```
Resuming at epoch 40, best_val_qwk so far = 0.7331
Training done in 0.0 min.
```

---

<a name="9-phase-3"></a>
## 9. Fundus Console Phase 3: Both Eyes mode + Instrument Card

Built while the checkpoint retrain was running in the background (you asked
whether you'd have to wait for the retrain to finish before moving on to
Phase 3/4 — no, and this was built in parallel).

**Both Eyes tab.** Applies Claim 3's finding (mean-pooling both eyes' CNN
features beats per-eye-then-max grading) live in the app:
- `pooled_embedding(x)` extracts `MODEL.forward_features(x)` →
  `MODEL.forward_head(feats, pre_logits=True)` — verified directly against
  timm's real `EfficientNet` source (fetched from GitHub, not assumed):
  `forward_features` runs `conv_stem → bn1 → blocks → conv_head → bn2`;
  `forward_head(x, pre_logits=True)` applies global pooling (+ dropout only
  in training mode) and returns the pre-classifier embedding without
  applying the final linear layer.
- `predict_both_eyes(left_image, right_image, session_log)` computes both
  eyes' pooled embeddings, mean-pools them, and classifies the mean with
  `MODEL.classifier(...)`.
- Explicitly labeled in the UI as reusing the Claim 3 *idea*, not a
  re-validated number — this app's end-to-end architecture is different
  enough from Claim 3's separately-fit ordinal head that the exact QWK
  0.617-vs-0.543 comparison doesn't transfer directly. The app says this,
  rather than implying the live feature is independently re-measured.

**Instrument Card tab.** Renders MASTER_PLAN.md Part 10's own pre-registered
acceptance-test bands as a horizontal gauge, with a live marker placed at
`RESULTS["test_qwk"]` — read from the results JSON at runtime, not
hardcoded, specifically so it can't go stale the way a hand-typed number
would. (This design decision is exactly what made the later checkpoint swap
in §12 low-risk for this particular number, even though the bands array
itself turned out to have its own bug — see §12.)

**UI plumbing added to support both:** `gr.Tabs()`/`gr.Tab()` multi-mode
layout, `session_state = gr.State([])` threaded through as both input and
output so results from either tab can be logged to one shared session list
(used by Phase 4's Session Log), and `render_cards_from_probs(probs)`
factored out as a shared helper so both the Single Eye and Both Eyes tabs
render their verdict/referral/confidence cards identically.

---

<a name="10-phase-4"></a>
## 10. Fundus Console Phase 4: Quantum Lab tab + Session Log + PDF export

Requested immediately after Phase 3 ("Knock out phase 4 and lmk").

**Quantum Lab tab.** Renders both QML experiment tracks (§5–6) directly from
their sweep JSON files (`results/qml_pqc.json`, `results/qcnn_no_cnn_pixels.json`)
— full sweep tables, the matched-comparison numbers, and each script's own
verdict text, including the "we lost" framing. Neither track is presented as
a win. `qwk_meter_html(label, qwk, max_qwk=0.4)` reuses the same `.fc-meter`
bar component the confidence meter uses, clamped to [0, 100]% so a
near-zero or negative QWK (real values in these sweeps) renders as an
honestly near-empty bar instead of crashing or looking broken.

**Session Log tab.** Every grading this session (Single Eye or Both Eyes)
appends a `make_log_entry(mode, probs)` to the shared `session_log` state;
`render_log_html(session_log)` displays them; `build_pdf_export(session_log)`
exports the log as a PDF via `fpdf2`, lazily imported with graceful
degradation (`(None, status_message)` on any failure) matching the existing
Grad-CAM fail-gracefully pattern.

**A real syntax bug caught before it shipped.** `build_quantum_lab_html()`
originally built a label with a nested f-string:
```python
f'{qwk_meter_html(f"QML head (q={chosen.get("n_qubits")}, ...)}'
```
Double quotes nested inside a double-quoted f-string's `{}` expression are a
`SyntaxError` on Python < 3.12 (PEP 701 only relaxed this in 3.12+), and this
venv runs Python 3.11. Fixed by precomputing the label as its own plain
variable (`qml_head_label = f"QML head (q={chosen.get('n_qubits')}, ..."`)
before using it. Verified via regex grep across the whole file
(`f"[^"]*\{[^}]*"[^}]*\}` and the single-quote equivalent) that no other
instance of the same pattern existed.

**A tooling gap during this phase, disclosed at the time:** the sandbox's
own `Bash` tool was intermittently unavailable, so `py_compile` couldn't be
run to verify `app.py`'s syntax before pushing. Worked around by re-reading
the entire file by eye across several `Read` calls (which is how the
nested-f-string bug above was actually caught), and you were asked to run
`python -m py_compile app\app.py` yourself as a first check.

---

<a name="11-packaging"></a>
## 11. Packaging for GitHub + Hugging Face: README, LICENSE, requirements, runbook

You asked, in one message, to package the project, push to GitHub with a
good README, deploy a demo site, and get it resume-ready — "gimme a super
detailed complex plan to follow." Delivered:

- **`README.md`** — completely rewritten from a stale "Phase 1 complete"
  placeholder into a real project README: a "Results at a glance" table (now
  updated again per §12–13 below), a description of the app's 5 tabs, repo
  layout, reproduction instructions, dataset/licensing notes (EyePACS/APTOS
  are referenced, not redistributed), an honest limitations section, and
  references (Selvaraju 2017 Grad-CAM, Cong/Choi/Lukin 2019 QCNN, Ahmed et
  al. 2024 QML).
- **`LICENSE`** — MIT, with an appended note that EyePACS/APTOS carry their
  own separate terms and that this is a research prototype, not a validated
  medical device.
- **`requirements.txt`** — a best-effort dependency list reconstructed from
  the project's actual imports (numpy, pandas, scipy, scikit-learn,
  matplotlib, Pillow, opencv-python, torch, timm, grad-cam, gradio, fpdf2,
  pennylane, pennylane-lightning, pytest) — explicitly flagged in its own
  comment as *not* generated from `pip freeze`, with the real command
  (`pip freeze > requirements.txt`) given to regenerate it properly from the
  working `.venv` before anyone relies on it for an exact reproduction.
- **A published 6-phase launch-checklist artifact** ("Fundus Console
  Runbook") covering: git-history audit, `.gitignore` hardening (checkpoints,
  cache, processed data, `.venv` already excluded), docs finalization,
  the GitHub push itself, Hugging Face Spaces deployment via a *separate*
  slim folder (kept the main repo checkpoint-free; the Space folder is the
  one place the ~16MB checkpoint actually lives, since HF Spaces need it to
  serve the demo), and resume/share content.
- **A large, self-contained, copy-pasteable prompt** for a local Claude Code
  session to execute the actual GitHub + Hugging Face deployment
  autonomously, with explicit guardrails: ask before any destructive git
  operation, ask before the first push, and never request that a token be
  pasted into chat (credentials always stay on your machine, entered only
  where your own git/HF credential helpers already handle them).

---

<a name="12-checkpoint-swap"></a>
## 12. Swapping in the converged checkpoint + fixing a real bug in my own Instrument Card

After you ran the retrain and it succeeded (§7–8), the deployed app was
still pointing at the *old* checkpoint and results file. Steps taken:

1. **Verified the real verdict field first**, rather than trusting the
   terminal's printed summary line alone (per this project's own
   investigate-before-adjusting standard) — read
   `results/finetune_app_converged_p2_class_balanced_seed42.json` directly
   and confirmed `"acceptance_test_10_1_verdict": "correct -- proceed"`.

2. **Found and fixed a real bug in my own Phase-3 Instrument Card code.**
   The `ACCEPTANCE_BANDS` array in `app.py` had been built by copying
   MASTER_PLAN.md's *prose* table for Acceptance Test 10.1 literally — that
   table's rows print rounded boundaries that skip 0.70–0.75 and 0.85–0.90,
   which I'd rendered as "undefined gap" zones on the gauge. But the
   **actual, executable verdict logic** in `src/train/finetune_converged.py`
   (the code that actually computes and prints this project's official
   verdict) has no such gaps — it's fully continuous:
   ```
   test_qwk < 0.40          -> "BROKEN"
   0.40 <= test_qwk < 0.70  -> "undertrained or preprocessing bug"
   0.70 <= test_qwk <= 0.85 -> "correct -- proceed"
   0.85 <  test_qwk <= 0.95 -> "SUSPICIOUS"
   test_qwk > 0.95          -> "DEFINITELY A LEAK"
   ```
   MASTER_PLAN.md's prose table gaps are a documentation-rounding artifact,
   not a second, stricter rule — confirmed by the fact that the training
   script (which MASTER_PLAN.md's own Part 10 refers to as the actual test)
   never produces an "undefined" verdict for any input. This meant the
   Instrument Card was on track to render your genuinely-correct 0.7160
   result as landing in a gray "undefined" gap, directly contradicting the
   training script's own printed "correct -- proceed" verdict — a new,
   self-inflicted inconsistency in the same spirit as the Acceptance-12.1
   overclaim corrected in §2. Fixed the array to match the real, executable
   rule (bands now: 0.00–0.40 broken / 0.40–0.70 undertrained / 0.70–0.85
   correct / 0.85–0.95 suspicious / 0.95–1.00 leak), with a comment
   explaining why the change was made.

3. **Checkpoint file swapped**:
   `checkpoints/finetune_app_converged_p2_class_balanced_seed42/best.pt` →
   `app/release/best_model.pt` (replacing the old 15-epoch checkpoint of
   the same name).

4. **`RESULTS_JSON_PATH` in `app.py` repointed** from
   `results/finetune_app_p2_seed42.json` to
   `results/finetune_app_converged_p2_class_balanced_seed42.json`.

5. **Hardcoded strings updated** (the "known, accepted debt" flagged by
   `app.py`'s own comments back in Phase 3): `MASTHEAD_HTML`'s spec-strip
   (test QWK 0.613 → 0.716, AUROC 0.872 → 0.912) and `MODEL_INFO_MD`'s
   description text, now describing the converged recipe and the "correct,
   proceed" band instead of "undertrained."

6. **`app/release/config.yaml`** rewritten to record the new checkpoint's
   provenance (source checkpoint path, training script, config file, result
   file, best epoch 32, best val QWK 0.7331, test QWK 0.7160, AUROC
   0.9117), with the old checkpoint's numbers kept as historical context
   where relevant rather than silently deleted.

7. **`app/release/MODEL_CARD.md`** rewritten: new performance table with a
   side-by-side "prior checkpoint" column so the improvement (and the
   grade-1 recall tradeoff — see §7) is visible at a glance, not just
   asserted; the "What Acceptance Test 12.1 found" section reframed to be
   explicit that its findings described the *prior* checkpoint and that a
   re-run against the new one was the natural next step (later completed —
   see §13).

8. **`README.md`'s "Results at a glance" table and "Honest limitations"
   section updated** to match — the deployed-model row now reports 0.716/
   0.912 and the "correct, proceed" band instead of 0.613/"undertrained",
   and the limitations section now leads with the grade-1 recall tradeoff
   as the live open question, rather than the resolved undertraining issue.

All five changed files (`app/app.py`, `app/release/config.yaml`,
`app/release/MODEL_CARD.md`, `README.md`, plus the new `best_model.pt`) were
pushed to the device via file transfer and confirmed written.

---

<a name="13-acceptance-121-rerun"></a>
## 13. Re-running Acceptance Test 12.1 against the new checkpoint — confirmed result

You ran:
```powershell
$env:N_APP_TEST_IMAGES=100; pytest tests\test_app.py -v
```
twice. **Result, both times: 8/100 mismatches** — down from **18/100** on
the prior, undertrained checkpoint. Same signature as before, at reduced
scale: 0/8 carry a non-default EXIF tag, and all 8 have pixel MAE between
0.8–1.2 (out of 255) between the app's freshly-processed image and the
cached evaluation image — i.e., still "near-identical pixels, model flips
anyway," just on roughly half as many borderline cases.

**Verified this is a genuine apples-to-apples comparison, not two different
random samples.** Read `_load_p2_test_sample()` in `tests/test_app.py`
directly: it walks the P2 test split in a *fixed* order (no shuffling, no
seeding) and takes the first `N_APP_TEST_IMAGES` images that have both a
source file and a cached preprocessed file, so `N_APP_TEST_IMAGES=100`
always selects the *identical* 100 images every run. This was confirmed
empirically too — your two consecutive runs against the new checkpoint
produced the exact same 8 image IDs both times. So the 18-vs-8 comparison
is the same 100 images scored by two different checkpoints — a real,
controlled before/after measurement of what the converged retrain's wider
decision margins did, not an artifact of sampling variance.

**What this confirms:** the theory from §2 (retraining to convergence
should widen decision margins and reduce noise-driven flips) held up under
an actual controlled re-test — roughly half the mismatches, same failure
signature. **What it does not claim:** the effect wasn't eliminated. This
checkpoint is measurably more robust to ordinary JPEG-decoder noise, not
perfectly robust to it — 8% of a 100-image sample still flip grade on
sub-1.2/255 pixel noise, which is worth knowing before presenting this as
"solved."

Both `app/release/MODEL_CARD.md` and `README.md` were updated again to
report this confirmed 18→8 result (replacing an intermediate, more
cautious version of the text that had incorrectly assumed the test sample
might differ run-to-run before the fixed-order behavior was verified —
that assumption was checked and corrected in the same pass, per this
project's own standard of not letting an unverified caveat stand once it's
checkable).

---

<a name="14-live-deployment"></a>
## 14. Live deployment

You reported pushing everything to GitHub and deploying live on Hugging
Face Spaces.

- **GitHub repo:** https://github.com/vishalyl/diabetic-retinopathy-detection
- **Hugging Face Space (live demo):** https://huggingface.co/spaces/vishal1829/fundus-console

The deployment plan itself (git-history audit → `.gitignore` hardening →
docs commit → GitHub push → separate `../fundus-space` folder pushed as its
own git repo to a Hugging Face Space with a Gradio-SDK YAML frontmatter
block in its own `README.md`) was handed off as a runbook artifact and a
self-contained Claude-Code prompt (§11) specifically because it needed your
own git/Hugging Face credentials — nothing in that flow was executed by
this assistant directly, by design (credentials never handled on your
behalf).

Once these two links are confirmed, the two remaining `(#)` placeholder
links in `README.md` (the top masthead's "Live demo →" link and the "Model
access" section) should be replaced with the real Space URL — flag this
back to me and I'll make that edit.

---

<a name="15-file-manifest"></a>
## 15. Complete file manifest — everything touched this session

| File | What changed |
|---|---|
| `app/release/MODEL_CARD.md` | Corrected EXIF-overclaim (§2); rewritten for the converged checkpoint with a before/after table and the grade-1 recall tradeoff flagged (§12); updated again with the confirmed 18→8 Acceptance-12.1 re-run and its apples-to-apples verification (§13). |
| `app/release/config.yaml` | Corrected `known_preprocessing_bug_fixed` field (§2); fully rewritten to record the converged checkpoint's provenance and numbers (§12). |
| `configs/finetune_app_converged.yaml` | New file: the 384px converged retrain recipe (§7); `num_workers` dropped to 0 with an explanatory comment after the WinError 1455 crash (§8). |
| `src/experiments/qcnn_no_cnn.py` | `backprop`/`default.qubit` device-selection fix + `--diff-method` argparse warning (§4); `fit_and_eval()` unpacking-order fix (§4). |
| `app/app.py` | Phase 3: Both Eyes tab, `pooled_embedding()`, Instrument Card, shared `render_cards_from_probs()`, `gr.Tabs()` restructure (§9). Phase 4: Quantum Lab tab, Session Log, PDF export, nested-f-string fix (§10). `RESULTS_JSON_PATH` repointed and `ACCEPTANCE_BANDS` gap-zone bug fixed (§12); `MASTHEAD_HTML`/`MODEL_INFO_MD` numbers updated (§12). |
| `README.md` | Fully rewritten from a stale "Phase 1 complete" placeholder (§11); results table and limitations section updated for the converged checkpoint (§12); updated again with the confirmed Acceptance-12.1 re-run (§13). |
| `LICENSE` | New MIT license with a dataset/medical-device disclaimer note (§11). |
| `requirements.txt` | New best-effort dependency list (§11). |
| `app/release/best_model.pt` | Swapped: old 15-epoch checkpoint (test QWK 0.613) → converged 40-epoch checkpoint (test QWK 0.716) (§12). |
| *("Fundus Console Runbook" artifact)* | Published separately (not a repo file): 6-phase GitHub/Hugging Face launch checklist (§11). |

---

<a name="16-open-items"></a>
## 16. Open items / honest state of the project as of now

What's genuinely solid:
- Patient-level leakage finding (Claim 1/1b/2/2b) — strong, well-evidenced,
  the project's real headline.
- Both-eyes ensembling finding (Claim 3) — real, significant, positive.
- Deployed app checkpoint: test QWK 0.716, inside this project's own
  "correct, proceed" band, confirmed by the training script's own verdict
  logic (now correctly rendered in the Instrument Card too).
- Acceptance Test 12.1 mismatch rate: confirmed, controlled 18→8
  improvement from the retrain.
- Two QML experiments: honestly negative/inconclusive, reported with the
  same statistical rigor as every positive claim.

What's still open, in rough priority order:
1. ~~**Grade-1 recall drop (0.461 → 0.161) is unexplained.**~~ **RESOLVED
   (investigated, not fixed) — see `results/grade1_diagnosis.json`.** Four
   decision rules were tested against a pre-registered ≥0.30 recall bar;
   none clear it (best: 0.228, `expected_grade_rounded`). Verdict:
   `H2_representation_ceiling` — the microaneurysm evidence for grade 1
   does not survive this project's 384px representation at any operating
   point, corroborated by the grade-1 partner-transfer rate, the corrupted-
   frame grade distribution, and the 8-run recall floor. The drop itself is
   now explained by that ceiling, not fixed by a better decision rule; one
   free sub-finding did come out of it (ordinal read: +0.0119 QWK, CI
   excludes zero, on raw probabilities — see `app/release/MODEL_CARD.md`).
2. **Acceptance Test 12.1's remaining 8% mismatch rate** is not zero. The
   model is more robust than before, not perfectly robust. Worth deciding
   whether that residual rate matters for whatever this project is being
   presented for.
3. ~~**Confidence calibration** (MASTER_PLAN.md Part 7 — temperature
   scaling + a validated reject threshold) has still not been run for this
   or any deployed checkpoint.~~ **RESOLVED — see
   `results/calibration_reject.json`.** Temperature T=3.3674 fitted on
   validation only (test ECE 0.1628→0.0285); referral threshold 0.13486
   fixed for ≥90% validation sensitivity; uncertainty gate τ=0.59274
   (Acceptance Test 11.1: pass, selective QWK 0.716→0.774 at 80%
   coverage). Applied thresholds live in `app/release/thresholds.json`;
   `app/core/decision.py` applies them at inference with an integrity
   check that falls back to raw uncalibrated confidence (visibly, on
   screen) if the checkpoint hash or the acceptance verdict don't check
   out.
4. **The two `(#)` placeholder links in `README.md`** need the real Hugging
   Face Space URL once confirmed (§14).
5. **`requirements.txt`** is still a best-effort reconstruction, not a real
   `pip freeze` — worth regenerating properly from the working `.venv`
   before anyone else tries to reproduce this from scratch.
