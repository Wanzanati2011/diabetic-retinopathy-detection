# Two Eyes, One Patient

**A diabetic retinopathy grading study built around one methodological question: what happens to your reported accuracy when you split retinal images by image instead of by patient?**

The answer, measured directly on 17,563 patients: a **0.8545 inter-eye quadratic-weighted kappa (QWK)**, and a lookup table that guesses a patient's grade from *only their other eye's label — no image data at all* — scores **QWK 0.838**. That's how much signal leaks across a patient's two eyes into a naive train/test split, and it's the reason this project evaluates everything on a **patient-level split (P2)**, never the image-level split (P1) most tutorials use.

Everything below — the leakage measurement, the final screening model, the both-eyes ensembling method, and two honestly-negative quantum machine learning experiments — was measured with pre-registered acceptance tests, patient-level bootstrap confidence intervals, and one rule enforced throughout: **no claim ships without a check that could have failed it.**

**[Live demo →](#) &nbsp;·&nbsp; [Full methodology (`MASTER_PLAN.md`)](MASTER_PLAN.md) &nbsp;·&nbsp; [Model card](app/release/MODEL_CARD.md)**

---

## Results at a glance

| Question | Finding | Evidence |
|---|---|---|
| Does image-level splitting leak information between a patient's two eyes? | **Yes — badly.** Inter-eye QWK 0.8545 (95% CI 0.846–0.863); a label-only lookup with zero image data scores QWK 0.838. | `results/claim1b_stratified.json`, `src/experiments/claim2_protocols.py` |
| Does grading both eyes together beat grading them separately? | **Yes, significantly.** Mean-pooling both eyes' features beats per-eye-then-max grading: QWK 0.617 vs 0.543 (paired bootstrap CI [+0.046, +0.098], excludes 0). | `results/claim3_both_eyes_effnetb0_384.json` |
| Does a quantum classifier head beat a classical one on the same features? | **No.** Best PQC config (8 qubits, 3 entangling layers) scores QWK 0.109 vs. 0.346/0.264 for classical heads on the same matched data — significantly worse. | `results/qml_pqc.json` |
| Does a fully-quantum, no-CNN model work at all on this task? | **Inconclusive at best.** Every swept config (4–8 qubits) converges to QWK 0.000 — a likely barren-plateau/vanishing-gradient failure, indistinguishable from an equally crippled classical baseline on the same heavily-downsampled pixels. | `results/qcnn_no_cnn_pixels.json` |
| Is the deployed screening model any good? | **Real, but honestly undertrained.** Test QWK 0.613, referable-DR AUROC 0.872 — inside this project's own pre-registered "undertrained" band (0.40–0.70), not its "correct, proceed" band (0.75–0.85). Reported as such, not rounded up. | `results/finetune_app_p2_seed42.json`, `app/release/MODEL_CARD.md` |

Every number above has a patient-level bootstrap confidence interval behind it in the linked file — none of this is a bare point estimate.

---

## What's actually in here

### 1. The leakage study (Claims 1–2)
Two public diabetic retinopathy datasets (EyePACS, APTOS 2019) merged into one manifest, deduplicated with perceptual hashing (calibrated empirically, not just applied at a textbook default — see `results/dedup_threshold_sweep.json`), and split three ways:
- **P1** — image-level split. Deliberately flawed; kept only to *measure* the leakage effect, never optimized against.
- **P2** — patient-level split (both eyes of a patient always in the same fold). The project's honest, primary protocol.
- **P3** — cross-dataset (train on one, test on the other).

`src/experiments/claim2b_partner_ablation.py` gives direct proof of leakage: the same model, same weights, scored with and without seeing a patient's partner eye — a real ablation, not an inference from the correlation number alone.

### 2. The both-eyes model (Claim 3)
`src/experiments/claim3_both_eyes.py` tests three ways to use two eyes at inference time (per-eye-then-max, concatenated features, mean-pooled features) against a patient-level label. Mean-pooling wins, significantly.

### 3. The screening app — *Fundus Console*
A Gradio research-prototype app (`app/app.py`) built around the trained checkpoint, with five tabs:

| Tab | What it does |
|---|---|
| **Single Eye** | Upload a fundus photo → 5-class ICDR grade, referral recommendation, and a **Grad-CAM** overlay showing what the model actually looked at. |
| **Both Eyes** | Applies the Claim 3 finding live — mean-pools both eyes' embeddings before classifying. Explicitly labeled as reusing the *idea*, not a re-validated number (this app's own end-to-end head is architecturally different from Claim 3's separately-fit ordinal head). |
| **Instrument Card** | The project's own pre-registered acceptance bands (`MASTER_PLAN.md` Part 10), rendered as a gauge with a live marker at the deployed checkpoint's real test QWK — reads from the results file, so it can't go stale. |
| **Quantum Lab** | Both QML experiments below, rendered from their actual sweep JSON — including the "we lost" verdicts, not just the numbers. |
| **Session Log** | Every grade this session, exportable as a PDF. |

The whole UI commits to one explicit dark "instrument console" palette — teal for routine, amber for refer, deliberately never red/green (red-green color deficiency affects ~1 in 12 male viewers).

### 4. Two honest quantum machine learning experiments
Built on direct request ("use QML, and parameterize it"), evaluated with the same rigor as everything else — matched data, paired bootstrap CIs, no favorable framing of a loss:

- **`src/experiments/qml_pqc.py`** — a "dressed" hybrid classifier (Ahmed et al. 2024, arXiv:2405.01734): frozen CNN features → PCA → angle-encoded parameterized quantum circuit (PennyLane `StronglyEntanglingLayers`) → linear readout. Swept 4/6/8 qubits × 1/2/3 layers.
- **`src/experiments/qcnn_no_cnn.py`** — a true Quantum Convolutional Neural Network (Cong, Choi & Lukin 2019, arXiv:1810.03787) on raw, heavily downsampled pixels — no classical feature extractor anywhere in the pipeline.

Both lose to their classical baselines. That's reported as the finding.

---

## Repository layout

```
mini/
├── app/                    Gradio app (Fundus Console) + release bundle (checkpoint, model card)
├── src/
│   ├── data/                Manifest building, deduplication, splitting, preprocessing
│   ├── features/            Frozen CNN feature extraction
│   ├── train/                Fine-tuning scripts (fast + "converged" recipes)
│   └── experiments/       Claims 1–3, QML/QCNN tracks, calibration, robustness checks
├── tests/                    pytest suite — acceptance tests for splits, dedup, the app, the QML tracks
├── configs/                 YAML configs for every training run (nothing hand-typed at runtime)
├── results/                 Every experiment's raw JSON output — the actual source of every number above
├── figures/                 Generated plots
├── MASTER_PLAN.md          The full pre-registered methodology, written BEFORE most of these numbers existed
└── PROGRESS.md              Running log of what was done, in what order, and why
```

`data/` (raw images) and `checkpoints/` (trained weights other than the deployed one) are intentionally **not** committed — see [Datasets](#datasets--licensing) below.

---

## Reproducing this

```powershell
git clone https://github.com/<your-username>/two-eyes-one-patient.git
cd two-eyes-one-patient
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Run the test suite (splits, dedup, and app acceptance tests -- skips
# anything that needs data/checkpoints you haven't downloaded)
pytest tests/ -v

# Run the app (needs app/release/best_model.pt -- see Model access below)
python app\app.py
```

### Datasets & licensing
This repo ships **manifests, splits, and results — not raw images.** EyePACS and APTOS 2019 are both public but carry their own terms; download them yourself:
- EyePACS: [Kaggle — Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection)
- APTOS 2019: [Kaggle — APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)

Point `configs/paths.yaml` at wherever you put them, then run `src/data/build_cache.py` to reproduce the processed image cache the manifests reference.

### Model access
The deployed checkpoint (`app/release/best_model.pt`, ~16 MB) is hosted with the [live demo](#) rather than committed to this repo. To run the app locally, download it from the demo Space and drop it in `app/release/`, or train your own with `src/train/finetune.py --config configs/finetune_app.yaml --split p2 --seed 42`.

---

## Honest limitations
This project's standing rule is to report a negative or undertrained result plainly rather than reframe it. In that spirit:
- The deployed checkpoint's test QWK (0.613) sits in this project's own "undertrained or preprocessing bug" acceptance band, not its "correct, proceed" band — see `app/release/MODEL_CARD.md` for the full breakdown and what a converged retrain is expected to do about it.
- Confidence shown in the app is raw, **uncalibrated** softmax output — no temperature scaling or reject option has been fitted yet.
- Grade-1 (Mild NPDR) recall is limited by design: it's defined by ~10px microaneurysms that don't survive downsampling to 384px, a resolution limitation, not a bug.
- The Both Eyes tab's own accuracy hasn't been independently re-measured — see its in-app caveat.

## References
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
- Cong, Choi & Lukin, *Quantum Convolutional Neural Networks*, Nature Physics 15 (2019), arXiv:1810.03787.
- Ahmed et al., *Diabetic Retinopathy Detection Using Quantum Transfer Learning*, arXiv:2405.01734 (2024).

## License
MIT — see [`LICENSE`](LICENSE). The code is free to use; the EyePACS/APTOS datasets are governed by their own separate terms (linked above) and are not included here.
