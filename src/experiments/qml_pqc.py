"""
New experiment track (supervisor request, 2026-09) -- "use QML, and
parameterize it": a real Parameterized Quantum Circuit / Variational
Quantum Classifier head, evaluated as a fourth head alongside this
project's existing multinomial and ordinal heads, on the SAME frozen CNN
features and the SAME P2 (patient-level) split protocol used throughout
Phases 1-6b. This is a dedicated new track, not a footnote.

REUSED, NOT REBUILT (see MASTER_PLAN.md / PROGRESS.md for how these were
produced):
  - Frozen feature caches: features/{backbone}_{size}.npz (Phase 3).
  - data/splits/p2.json (Phase 1) -- train/val/test folds, patient-level.
  - load_active_manifest / build_xy / fit_and_eval (multinomial head) /
    ordinal_qwk (ridge + optimized thresholds) / patient_bootstrap_ci, all
    imported UNCHANGED from src/experiments/claim2_protocols.py.
  - paired_bootstrap_ci_diff, imported UNCHANGED from
    src/experiments/claim3_both_eyes.py -- this is what actually decides
    "QML beats/loses to head X", not a bare point-estimate comparison, per
    this project's standing rule (see claim3_both_eyes.py docstring).

SCOPE (deliberately narrower than the full P2 train set -- read this
before trusting a number below):
  Training a CPU-simulated variational quantum circuit sample-by-sample
  does not scale to the full ~27,100-image P2 training set in reasonable
  wall-clock time on a laptop (unlike the classical heads, which fit that
  in seconds). So the QML head, and ONLY for a fair comparison the
  multinomial/ordinal heads too, are trained on a fixed, seeded, class-
  stratified SUBSAMPLE of the P2 train fold (--n-train-sample, default
  3000). All three heads are evaluated on the FULL, unmodified P2 test
  fold (5,814 images) -- evaluation is never subsampled.
  The already-computed FULL-train-set classical numbers from
  results/claim2_protocols.json are also loaded and reported alongside,
  clearly labeled "cited, not a matched comparison" -- so a reader can see
  both (a) the fair, matched-data three-way comparison this script
  produces, and (b) how much of any gap might just be "10x more training
  data", without confusing the two.

HEAD DESIGN ("dressed quantum classifier", grounded in Ahmed et al.,
arXiv:2405.01734, and PennyLane's own standard quantum-transfer-learning
pattern -- see LITERATURE_NOTES below for what was and wasn't verified by
web search):
  1. PCA (fit on the TRAIN SUBSAMPLE only) reduces the frozen feature
     vector to 2 * n_qubits components.
  2. Angle encoding: first n_qubits components -> RX rotations, second
     n_qubits -> RY rotations (uses both rotation axes the assignment
     asked for, without needing extra qubits for the extra precision).
     Per-column min-max scaling to [-pi, pi], fit on train only.
  3. Ansatz: PennyLane's StronglyEntanglingLayers -- ring-entangled, 3
     trainable rotation angles per qubit per layer. n_qubits and n_layers
     are the swept, explicitly parameterized hyperparameters (this
     directly answers "why these parameter values" -- see the sweep
     table/figure, not a single arbitrary config).
  4. Measurement: PauliZ expectation on every qubit -> a small classical
     linear read-out (n_qubits -> 5 logits) -> softmax -> cross-entropy.
     This mirrors the multinomial head's output structure exactly
     (softmax / argmax / QWK), which is what makes "QML vs multinomial"
     a fair, apples-to-apples comparison and not just a vibes comparison.
  5. Backend: lightning.qubit (CPU simulator). Trained end-to-end via
     PennyLane's qml.qnn.TorchLayer + PyTorch Adam -- real backprop
     through the hybrid classical-quantum model. diff_method defaults to
     "adjoint" (fast, simulator-only) for the sweep; --diff-method
     parameter-shift switches to the gradient rule an actual QPU would
     need (slower here, but the SAME ansatz is hardware-executable either
     way -- only the simulator's gradient bookkeeping differs). See
     tests/test_qml_pqc.py::test_adjoint_matches_parameter_shift_gradient
     for the check that the two agree.

IMPORTANT CAVEAT, stated plainly rather than discovered the hard way: this
file's PennyLane/PyTorch code paths (build_qml_model, train_qml_head,
predict_qml) could NOT be executed where they were written -- pennylane is
not installed in this project's .venv, and this dev sandbox has no network
route to install it either (see chat for the full explanation). They were
written carefully against the documented PennyLane API and reviewed by
hand, but are UNVERIFIED until you run them. This is the same situation
src/train/finetune.py was in when it was written (see that file's
docstring) -- the fix is the same: run --mode debug FIRST (finishes in
well under a minute) before trusting a real sweep. Every OTHER function in
this file (stratified sampling, PCA-slicing-equivalence, angle scaling,
caching) has no such caveat -- it's plain numpy/pandas/sklearn and is unit
-tested in tests/test_qml_pqc.py, which DOES run in this sandbox and does
pass.

LITERATURE_NOTES below records exactly what was and wasn't independently
verified by web search before writing this file -- per this project's
standing rule, never state a citation's specifics beyond what was actually
confirmed.

Usage (see run.ps1 for the exact commands):
    python src\\experiments\\qml_pqc.py --mode debug
    python src\\experiments\\qml_pqc.py --mode sweep  --backbone resnet50 --size 224
    python src\\experiments\\qml_pqc.py --mode final  --backbone tf_efficientnet_b0 --size 224 --n-qubits Q --n-layers L
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.claim2_protocols import (
    load_active_manifest, build_xy, fit_and_eval, ordinal_qwk, patient_bootstrap_ci,
)
from src.experiments.claim3_both_eyes import paired_bootstrap_ci_diff

SEED = 42
N_CLASSES = 5

# What was actually confirmed by web search before this file was written
# (2026-09-11), and what wasn't -- see chat for the full search trail.
LITERATURE_NOTES = {
    "ahmed2024_arxiv_2405.01734": (
        "Ahmed et al., 'Diabetic Retinopathy Detection Using Quantum Transfer "
        "Learning', arXiv:2405.01734. CONFIRMED by fetching the paper: ResNet-18/"
        "34/50/101/152 and Inception V3 frozen features feed a 4-qubit 'dressed "
        "quantum circuit' -- Hadamard superposition init, RY angle encoding, "
        "repeated {CNOT entangling layer + trainable RY rotation layer} blocks "
        "(depth = q_depth), PauliZ measurement, Adam optimizer with a step LR "
        "schedule. Evaluated on APTOS 2019 (the same dataset this project uses). "
        "Reported accuracy 97.2-98.5% across backbones (vs. 85.3-89.8% classical-"
        "only). Closest architectural precedent for this experiment -- this "
        "script's 'dressed classifier' design (classical frozen features -> "
        "angle encoding -> variational ansatz -> classical read-out) follows the "
        "same overall pattern, generalized to RX+RY encoding and a swept "
        "StronglyEntanglingLayers ansatz instead of a fixed RY-only one."
    ),
    "hqcnn2025_arxiv_2509.14277": (
        "'HQCNN: A Hybrid Quantum-Classical Neural Network for Medical Image "
        "Classification', arXiv:2509.14277. CONFIRMED by fetching the paper: "
        "4-qubit circuit, RY angle embedding + cyclic U3 rotations, plus "
        "'Quantum Attention-Fourier' layers (Toffoli-based local attention + "
        "phase-encoded entanglement mimicking a Fourier basis). Evaluated on "
        "MedMNIST v2 (PathMNIST, OrganAMNIST, BloodMNIST, OCTMNIST, "
        "BreastMNIST, PneumoniaMNIST) -- NOT diabetic retinopathy specifically. "
        "Cited here as a more elaborate ansatz for future work; NOT replicated "
        "-- this script uses the simpler, better-precedented StronglyEntanglingLayers."
    ),
    "quantumnet2025": (
        "'QuantumNet: An enhanced diabetic retinopathy detection model using "
        "classical deep learning-quantum transfer learning' (PMC/ScienceDirect, "
        "2025). EXISTENCE confirmed via search (indexed on PubMed/PMC/"
        "ScienceDirect). Full circuit methodology NOT independently confirmed -- "
        "automated fetch was blocked (paywall / bot-check) on every mirror "
        "tried. Cited for topic/existence only; no methodology claim is made "
        "about it in this project."
    ),
    "balanced_multiclass_2025": (
        "'Hybrid quantum-classical deep learning framework for balanced "
        "multiclass diabetic retinopathy classification' (ScienceDirect, 2025). "
        "EXISTENCE confirmed via search. Full methodology NOT independently "
        "confirmed -- fetch returned a 400/blocked response. Cited for topic/"
        "existence only."
    ),
}


# =====================================================================
# Pure, synthetic-data-testable helpers (no pennylane/torch import here --
# see tests/test_qml_pqc.py, which runs these directly with no quantum
# dependency installed).
# =====================================================================

def stratified_sample_ids(ids, grades, n_sample, seed=SEED):
    """Pure function. Deterministic, class-stratified (proportional-to-
    class-share) sample of `n_sample` ids from `ids`. If n_sample >=
    len(ids), returns all ids unchanged (no sampling needed) -- this is
    what makes --n-train-sample a no-op safety valve for small test
    fixtures / --mode debug."""
    ids = list(ids)
    grades = np.asarray(grades)
    n_total = len(ids)
    if n_sample >= n_total:
        return list(ids)
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({"id": ids, "grade": grades})
    out = []
    for _, group in df.groupby("grade"):
        n_g = max(1, round(n_sample * len(group) / n_total))
        n_g = min(n_g, len(group))
        chosen = rng.choice(group["id"].to_numpy(), size=n_g, replace=False)
        out.extend(chosen.tolist())
    rng.shuffle(out)
    if len(out) > n_sample:
        out = out[:n_sample]
    elif len(out) < n_sample:
        remaining = [i for i in ids if i not in set(out)]
        rng.shuffle(remaining)
        out.extend(remaining[: n_sample - len(out)])
    return out


def fit_angle_scaler(X):
    """Pure function. Per-column min/max on TRAIN data only. Apply with
    apply_angle_scaler -- never refit on val/test."""
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    return lo, hi


def apply_angle_scaler(X, lo, hi):
    """Maps each column's fitted [lo, hi] to [-pi, pi]. Values outside the
    train-fitted range (possible on val/test) are clipped, not
    extrapolated -- angles must stay in a physically sensible range for
    the rotation gates. A zero-range (constant) training column maps to a
    constant 0 angle rather than dividing by zero."""
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    span = hi - lo
    span_safe = np.where(span == 0, 1.0, span)
    scaled = np.clip((X - lo) / span_safe, 0.0, 1.0)
    angles = scaled * 2 * np.pi - np.pi
    angles = np.where(span == 0, 0.0, angles)
    return angles


def pca_reduce(X_train, others, n_components, seed=SEED):
    """Fits PCA(n_components) on X_train only; transforms X_train and
    every array in `others` with that SAME fitted transform.

    Why fit once at the sweep's max qubit count and slice, instead of
    refitting per config: sklearn's PCA orders components by explained
    variance and does not re-optimize earlier components when you ask for
    fewer of them -- the first k columns of PCA(K).transform(X) for K > k
    are identical (up to sign flips, which are also deterministic given a
    fixed random_state/solver) to PCA(k).transform(X) fit directly. See
    tests/test_qml_pqc.py::test_pca_slicing_equivalence, which checks this
    numerically rather than assuming it.
    """
    pca = PCA(n_components=n_components, random_state=seed)
    Xtr = pca.fit_transform(X_train)
    out_others = [pca.transform(X) for X in others]
    return pca, Xtr, out_others


def cache_key(backbone, size, n_qubits, n_layers, n_train_sample, seed):
    """Short, deterministic id for a single sweep cell -- used as the
    cache filename so a slow environment can resume a sweep across
    multiple invocations (same convention as claim3_decomposed.py's
    per-stage .npz caches)."""
    raw = f"{backbone}_{size}_q{n_qubits}_l{n_layers}_n{n_train_sample}_s{seed}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def verdict_from_diffs(diff_qml_minus_ordinal, diff_qml_minus_multinomial):
    """Pure function, mirrors the decision logic in claim3_both_eyes.py /
    claim3_decomposed.py: a point-estimate difference is never itself the
    verdict -- only a paired-bootstrap CI that excludes 0 is."""
    beats_ordinal = diff_qml_minus_ordinal["significantly_better"]
    beats_multinomial = diff_qml_minus_multinomial["significantly_better"]
    if beats_ordinal:
        return ("usable_quantum_head", (
            "QML significantly beats the ordinal head (this project's strongest "
            f"performer so far): diff={diff_qml_minus_ordinal['mean_diff']:+.4f}, "
            f"CI {diff_qml_minus_ordinal['ci95']}. Report as a genuine positive "
            "result, and check it on the second backbone before trusting it."
        ))
    if beats_multinomial:
        return ("beats_multinomial_not_ordinal", (
            "QML significantly beats the plain multinomial head "
            f"(diff={diff_qml_minus_multinomial['mean_diff']:+.4f}, "
            f"CI {diff_qml_minus_multinomial['ci95']}) but does NOT significantly "
            f"beat the ordinal head (diff={diff_qml_minus_ordinal['mean_diff']:+.4f}, "
            f"CI {diff_qml_minus_ordinal['ci95']}). Consistent with this project's "
            "recurring finding that the ordinal (regress-then-threshold) trick, "
            "not the head's raw representational power, is what QWK rewards."
        ))
    return ("negative_result", (
        "QML does not significantly beat either classical head at this qubit/"
        "layer budget and training-set size "
        f"(vs ordinal: diff={diff_qml_minus_ordinal['mean_diff']:+.4f}, "
        f"CI {diff_qml_minus_ordinal['ci95']}; vs multinomial: "
        f"diff={diff_qml_minus_multinomial['mean_diff']:+.4f}, "
        f"CI {diff_qml_minus_multinomial['ci95']}). Report honestly -- a negative "
        "result here does not undermine this project's core claims, and a small, "
        "CPU-simulated variational circuit trained on a few thousand images "
        "losing to a Ridge regression is a plausible, unsurprising outcome, not "
        "a bug to chase."
    ))


# =====================================================================
# Quantum model -- needs pennylane + torch. Imported lazily inside these
# functions ONLY, so the rest of this module (and every pure helper above)
# stays importable and testable without either installed. See the module
# docstring's IMPORTANT CAVEAT.
# =====================================================================

def build_qml_model(n_qubits, n_layers, diff_method="adjoint", seed=SEED):
    import pennylane as qml
    import torch
    import torch.nn as nn

    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        # inputs: 2*n_qubits angles, already scaled to [-pi, pi] by
        # apply_angle_scaler. First half -> RX, second half -> RY: both
        # rotation axes the assignment asked for, without extra qubits.
        # NOTE: use [..., :n_qubits] / [..., n_qubits:], NOT [:n_qubits] / [n_qubits:] --
        # PennyLane's TorchLayer passes a BATCHED tensor of shape (batch, 2*n_qubits)
        # straight into this qfunc (it does not loop row-by-row), so slicing the first
        # axis (no ellipsis) slices off *rows of the batch*, not features -- that was
        # the "Features must be of length 4 or less; got length 8" crash. Slicing the
        # LAST axis works correctly for both a single unbatched sample (1D) and a
        # batch (2D).
        qml.AngleEmbedding(inputs[..., :n_qubits], wires=range(n_qubits), rotation="X")
        qml.AngleEmbedding(inputs[..., n_qubits:], wires=range(n_qubits), rotation="Y")
        # The parameterized ansatz being swept: ring-entangled variational
        # layers, 3 trainable rotation angles per qubit per layer.
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    weight_shape = {
        "weights": qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    }
    torch.manual_seed(seed)
    qlayer = qml.qnn.TorchLayer(circuit, weight_shape)

    class QMLHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qlayer
            # dtype=torch.float64 to match qlayer's output dtype -- the quantum
            # node runs in float64/complex128 internally, and TorchLayer's
            # output follows the INPUT dtype (we feed float64 features), while
            # nn.Linear defaults to float32. Left as default, this crashes
            # with "mat1 and mat2 must have the same dtype, but got Double and
            # Float" the first time this runs on real data.
            self.readout = nn.Linear(n_qubits, N_CLASSES, dtype=torch.float64)

        def forward(self, x):
            q_out = self.qlayer(x)      # (batch, n_qubits) PauliZ expectations, each in [-1, 1]
            return self.readout(q_out)  # (batch, 5) logits

    return QMLHead()


def train_qml_head(X_train, y_train, X_val, y_val, n_qubits, n_layers,
                    epochs=40, batch_size=32, lr=0.05, patience=8,
                    diff_method="adjoint", seed=SEED, log=print):
    """Hybrid classical-quantum training loop: PyTorch Adam over a model
    whose forward pass runs a real PennyLane variational circuit. Model
    selection is by VALIDATION QWK (never test), matching this project's
    'never tune on test' rule (MASTER_PLAN.md Part 11). Early-stops on
    patience epochs without a val_qwk improvement; returns the
    best-val_qwk model state, not the final epoch's (same convention as
    finetune.py's best.pt)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    model = build_qml_model(n_qubits, n_layers, diff_method=diff_method, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    Xtr = torch.tensor(X_train, dtype=torch.float64)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xva = torch.tensor(X_val, dtype=torch.float64)

    n = len(Xtr)
    best_val_qwk = -2.0
    best_state = None
    epochs_since_improve = 0
    rng = np.random.RandomState(seed)
    epochs_run = 0

    for epoch in range(epochs):
        epochs_run = epoch + 1
        perm = rng.permutation(n)
        model.train()
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss) * len(idx)

        model.eval()
        with torch.no_grad():
            val_pred = model(Xva).argmax(dim=1).numpy()
        val_qwk = float(cohen_kappa_score(y_val, val_pred, weights="quadratic"))
        log(f"    epoch {epoch + 1:3d}/{epochs}  train_loss={total_loss / n:.4f}  val_qwk={val_qwk:.4f}")

        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                log(f"    early stop at epoch {epoch + 1} (no val_qwk improvement in {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_qwk, epochs_run


def predict_qml(model, X):
    import torch
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float64))
    return logits.argmax(dim=1).numpy()


# =====================================================================
# Orchestration
# =====================================================================

def load_p2_fold_ids(split_map, fold):
    return [i for i, v in split_map.items() if v["fold"] == fold]


def prepare_features(root, args, log):
    """Loads manifest + npz features + P2 split; returns everything the
    rest of main() needs. Shared between --mode sweep and --mode final."""
    active_manifest = load_active_manifest(root / args.manifest)
    grade_by_id = dict(zip(active_manifest["image_id"], active_manifest["grade"]))
    patient_by_id = dict(zip(active_manifest["image_id"], active_manifest["patient_id"]))

    npz_path = root / args.features_dir / f"{args.backbone}_{args.size}.npz"
    if not npz_path.exists():
        log(f"ERROR: {npz_path} not found. Run src/features/extract.py for this config first.")
        sys.exit(1)
    data = np.load(npz_path)
    feats, ids = data["features"], data["image_id"]
    row_of = {i: r for r, i in enumerate(ids.tolist())}

    split_map = json.loads((root / args.p2_split).read_text())
    train_ids_all = load_p2_fold_ids(split_map, "train")
    val_ids_all = load_p2_fold_ids(split_map, "val")
    test_ids_all = load_p2_fold_ids(split_map, "test")

    train_grades_all = np.array([grade_by_id[i] for i in train_ids_all if i in row_of])
    train_ids_all = [i for i in train_ids_all if i in row_of]

    train_ids = stratified_sample_ids(train_ids_all, train_grades_all, args.n_train_sample, seed=args.seed)
    val_ids_full = [i for i in val_ids_all if i in row_of]
    val_grades_full = np.array([grade_by_id[i] for i in val_ids_full])
    val_ids = stratified_sample_ids(val_ids_full, val_grades_full, args.n_val_sample, seed=args.seed)

    X_train, y_train, train_ids = build_xy(train_ids, feats, row_of, grade_by_id)
    X_val, y_val, val_ids = build_xy(val_ids, feats, row_of, grade_by_id)
    X_test, y_test, test_ids = build_xy(test_ids_all, feats, row_of, grade_by_id)
    patient_ids_test = [patient_by_id[i] for i in test_ids]

    log(f"{args.backbone}@{args.size}: train_subsample={len(train_ids)} "
        f"(of {len(train_ids_all)} P2-train), val_subsample={len(val_ids)} "
        f"(of {len(val_ids_full)} P2-val), test={len(test_ids)} (full P2-test, unmodified)")

    return {
        "X_train": X_train, "y_train": y_train, "n_train_full": len(train_ids_all),
        "X_val": X_val, "y_val": y_val, "n_val_full": len(val_ids_full),
        "X_test": X_test, "y_test": y_test, "patient_ids_test": patient_ids_test,
    }


def fit_matched_subsample_heads(d, seed):
    """Multinomial + ordinal, fit on the EXACT SAME train subsample as the
    QML head (not the full P2 train set) -- the fair, matched comparison.
    Returns predictions on the full test set for all three heads to share."""
    _, pred_mult = fit_and_eval(d["X_train"], d["y_train"], d["X_test"], d["y_test"], seed, bootstrap_train=False)
    _, thresholds, pred_ord = ordinal_qwk(d["X_train"], d["y_train"], d["X_test"], d["y_test"], seed=seed)
    return pred_mult, pred_ord, thresholds


def evaluate_head(name, pred, d, n_boot, seed=42):
    qwk = float(cohen_kappa_score(d["y_test"], pred, weights="quadratic"))
    ci = patient_bootstrap_ci(d["patient_ids_test"], d["y_test"], pred, n_boot=n_boot, seed=seed)
    return {"label": name, "qwk": qwk, "patient_bootstrap_ci95": list(ci)}


def load_full_trainset_reference(root, claim2_json_path, backbone, size):
    path = root / claim2_json_path
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    p2 = d.get("protocols", {}).get("P2")
    if p2 is None:
        return None
    return {
        "source_file": str(claim2_json_path),
        "source_backbone_size": f"{d.get('backbone')}@{d.get('size')}",
        "matches_this_run_backbone_size": (d.get("backbone") == backbone and d.get("size") == size),
        "multinomial_qwk_full_train": p2["reference_qwk_seed42"],
        "ordinal_qwk_full_train": p2["ordinal_qwk"],
        "n_train_full": p2["n_train"],
        "note": ("Cited from results/claim2_protocols.json, NOT recomputed here. Trained on the "
                 "FULL P2 train set (not the matched subsample used for the three-way comparison "
                 "above) -- context only, not a like-for-like number."),
    }


def compute_angle_features(d, n_qubits, seed, log):
    """PCA (fit on the train subsample only, to 2*n_qubits components) +
    train-only angle scaling, for one qubit count. Computed ONCE per
    n_qubits and reused across every n_layers in the sweep for that qubit
    count (n_layers doesn't change the encoding at all) -- see pca_reduce's
    docstring / tests/test_qml_pqc.py::test_pca_slicing_equivalence for why
    this is safe rather than just convenient."""
    pca, Xtr_pca, (Xva_pca, Xte_pca) = pca_reduce(
        d["X_train"], [d["X_val"], d["X_test"]], n_components=2 * n_qubits, seed=seed)
    lo, hi = fit_angle_scaler(Xtr_pca)
    log(f"  [q={n_qubits}] PCA+angle-encoding fit once for this qubit count "
        f"(explained_var={pca.explained_variance_ratio_.sum():.3f} of {2 * n_qubits} comps), "
        f"reused across every n_layers below.")
    return {
        "train": apply_angle_scaler(Xtr_pca, lo, hi),
        "val": apply_angle_scaler(Xva_pca, lo, hi),
        "test": apply_angle_scaler(Xte_pca, lo, hi),
    }


def run_pqc_config(d, angle_feats, n_qubits, n_layers, args, cache_dir, log):
    """Trains (or loads from cache) one (n_qubits, n_layers) config on the
    matched train subsample, using the already PCA+angle-encoded features
    for this qubit count (see compute_angle_features). Returns
    (val_qwk, test_pred, epochs_run, seconds)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(args.backbone, args.size, n_qubits, n_layers, args.n_train_sample, args.seed)
    cache_path = cache_dir / f"{key}.npz"
    if cache_path.exists():
        c = np.load(cache_path)
        log(f"  [q={n_qubits} l={n_layers}] loaded from cache {cache_path.name}: "
            f"val_qwk={float(c['val_qwk']):.4f}")
        return float(c["val_qwk"]), c["test_pred"], int(c["epochs_run"]), float(c["seconds"])

    t0 = time.time()
    log(f"  [q={n_qubits} l={n_layers}] training...")
    model, val_qwk, epochs_run = train_qml_head(
        angle_feats["train"], d["y_train"], angle_feats["val"], d["y_val"], n_qubits, n_layers,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience,
        diff_method=args.diff_method, seed=args.seed, log=log)
    test_pred = predict_qml(model, angle_feats["test"])
    seconds = time.time() - t0
    log(f"  [q={n_qubits} l={n_layers}] done in {seconds:.1f}s, best val_qwk={val_qwk:.4f}, "
        f"{epochs_run} epochs")

    np.savez(cache_path, val_qwk=val_qwk, test_pred=test_pred, epochs_run=epochs_run, seconds=seconds)
    return val_qwk, test_pred, epochs_run, seconds


def finalize_and_report(d, chosen_qubits, chosen_layers, chosen_test_pred, args, root, extra):
    pred_mult, pred_ord, thresholds = fit_matched_subsample_heads(d, args.seed)

    cells = {
        "multinomial": evaluate_head("multinomial (matched subsample)", pred_mult, d, args.n_bootstrap_ci),
        "ordinal": evaluate_head("ordinal (matched subsample)", pred_ord, d, args.n_bootstrap_ci),
        "qml": evaluate_head(f"qml (q={chosen_qubits}, l={chosen_layers})", chosen_test_pred, d, args.n_bootstrap_ci),
    }
    cells["ordinal"]["thresholds"] = thresholds

    diff_qml_minus_ord = paired_bootstrap_ci_diff(
        d["patient_ids_test"], d["y_test"], pred_ord, chosen_test_pred, n_boot=args.n_bootstrap_ci, seed=43)
    diff_qml_minus_mult = paired_bootstrap_ci_diff(
        d["patient_ids_test"], d["y_test"], pred_mult, chosen_test_pred, n_boot=args.n_bootstrap_ci, seed=43)

    verdict, verdict_text = verdict_from_diffs(diff_qml_minus_ord, diff_qml_minus_mult)

    ref = load_full_trainset_reference(root, args.claim2_json, args.backbone, args.size)

    print("=" * 78)
    print(f"QML PQC HEAD  ({args.backbone}@{args.size}, P2, matched train subsample n={len(d['y_train'])}, "
          f"full test n={len(d['y_test'])})")
    print("=" * 78)
    for k, v in cells.items():
        print(f"  {k:<12} QWK={v['qwk']:.4f}  95% CI {v['patient_bootstrap_ci95']}")
    if ref is not None:
        flag = "" if ref["matches_this_run_backbone_size"] else "  (DIFFERENT backbone/size -- context only)"
        print(f"  --- cited, full P2 train set (n={ref['n_train_full']}), NOT matched ---{flag}")
        print(f"  multinomial (full train): {ref['multinomial_qwk_full_train']:.4f}")
        print(f"  ordinal     (full train): {ref['ordinal_qwk_full_train']:.4f}")
    print()
    print(f"  QML - ordinal:     diff={diff_qml_minus_ord['mean_diff']:+.4f}  CI {diff_qml_minus_ord['ci95']}  "
          f"significant={diff_qml_minus_ord['significantly_better']}")
    print(f"  QML - multinomial: diff={diff_qml_minus_mult['mean_diff']:+.4f}  CI {diff_qml_minus_mult['ci95']}  "
          f"significant={diff_qml_minus_mult['significantly_better']}")
    print()
    print(f"VERDICT ({verdict}):", verdict_text)

    out = {
        "backbone": args.backbone, "size": args.size,
        "chosen_config": {"n_qubits": chosen_qubits, "n_layers": chosen_layers},
        "n_train_sample": args.n_train_sample, "n_val_sample": args.n_val_sample,
        "n_train_full_p2": d["n_train_full"], "n_val_full_p2": d["n_val_full"],
        "n_test": len(d["y_test"]),
        "diff_method": args.diff_method, "epochs_max": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "patience": args.patience, "seed": args.seed,
        "matched_subsample_heads": cells,
        "paired_diffs": {
            "qml_minus_ordinal": diff_qml_minus_ord,
            "qml_minus_multinomial": diff_qml_minus_mult,
        },
        "full_training_set_reference_cited": ref,
        "verdict": verdict, "verdict_text": verdict_text,
        "literature": LITERATURE_NOTES,
        **extra,
    }
    suffix = "" if (args.backbone == "resnet50" and args.size == 224) else f"_{args.backbone}_{args.size}"
    out_path = root / (args.out or f"results/qml_pqc{suffix}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"\nWrote {out_path}")

    make_figure(args, root, cells, ref, suffix)
    return out


def make_figure(args, root, cells, ref, suffix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["multinomial\n(matched)", "ordinal\n(matched)", f"QML\n(q={cells['qml']['label'].split('q=')[1]}"]
    labels[-1] = labels[-1].rstrip(")") + ")"
    keys = ["multinomial", "ordinal", "qml"]
    heights = [cells[k]["qwk"] for k in keys]
    cis = [cells[k]["patient_bootstrap_ci95"] for k in keys]
    colors = ["#d62728", "#1f77b4", "#9467bd"]
    if ref is not None:
        labels += ["multinomial\n(full train,\ncited)", "ordinal\n(full train,\ncited)"]
        heights += [ref["multinomial_qwk_full_train"], ref["ordinal_qwk_full_train"]]
        cis += [None, None]
        colors += ["#f4a6a6", "#a6c8e8"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    err_lo = [max(0.0, h - c[0]) if c else 0.0 for h, c in zip(heights, cis)]
    err_hi = [max(0.0, c[1] - h) if c else 0.0 for h, c in zip(heights, cis)]
    ax.bar(x, heights, color=colors, yerr=[err_lo, err_hi], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("P2 test QWK (per-image)")
    ax.set_ylim(0, 1)
    ax.set_title(f"QML PQC head vs. classical heads, {args.backbone}@{args.size}\n"
                 f"error bars: 95% patient-bootstrap CI. Faded bars: cited full-train-set reference (not matched).",
                 fontsize=9)
    for i, h in enumerate(heights):
        ax.text(i, h + 0.02, f"{h:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig_path = root / (args.fig or f"figures/figure_qml_pqc{suffix}.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {fig_path}")


def make_sweep_figure(args, root, sweep_rows, suffix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qubits = sorted(set(r["n_qubits"] for r in sweep_rows))
    layers = sorted(set(r["n_layers"] for r in sweep_rows))
    grid = np.full((len(layers), len(qubits)), np.nan)
    for r in sweep_rows:
        grid[layers.index(r["n_layers"]), qubits.index(r["n_qubits"])] = r["val_qwk"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, cmap="viridis", vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    ax.set_xticks(range(len(qubits))); ax.set_xticklabels(qubits)
    ax.set_yticks(range(len(layers))); ax.set_yticklabels(layers)
    ax.set_xlabel("n_qubits"); ax.set_ylabel("n_layers")
    for i in range(len(layers)):
        for j in range(len(qubits)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)
    ax.set_title(f"QML sweep -- validation QWK, {args.backbone}@{args.size}\n"
                 f"(matched train subsample n={args.n_train_sample})", fontsize=9)
    fig.colorbar(im, label="val QWK")
    fig.tight_layout()
    fig_path = root / f"figures/figure_qml_pqc_sweep{suffix}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {fig_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--manifest", default="data/manifests/manifest.csv")
    ap.add_argument("--p2-split", default="data/splits/p2.json")
    ap.add_argument("--features-dir", default="features")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--mode", choices=["sweep", "final", "debug"], default="sweep")
    ap.add_argument("--qubits-grid", default="4,6,8", help="comma list, --mode sweep only")
    ap.add_argument("--layers-grid", default="1,2,3", help="comma list, --mode sweep only")
    ap.add_argument("--n-qubits", type=int, default=None, help="--mode final: fixed qubit count")
    ap.add_argument("--n-layers", type=int, default=None, help="--mode final: fixed layer count")
    ap.add_argument("--n-train-sample", type=int, default=3000)
    ap.add_argument("--n-val-sample", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--diff-method", default="adjoint")
    ap.add_argument("--n-bootstrap-ci", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--claim2-json", default="results/claim2_protocols.json")
    ap.add_argument("--cache-dir", default="cache/qml_pqc")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fig", default=None)
    args = ap.parse_args()

    t0 = time.time()
    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    if args.mode == "debug":
        # Fast smoke test of the WHOLE pipeline (data -> PCA -> angle encode ->
        # quantum circuit -> train -> predict -> CI), same purpose as
        # finetune.py's `--epochs 1 --limit-train 200 ...` debug pass. Run
        # this FIRST -- see the module docstring's IMPORTANT CAVEAT.
        args.n_train_sample = 60
        args.n_val_sample = 30
        args.epochs = 2
        args.patience = 2
        args.n_bootstrap_ci = 50
        args.qubits_grid = "4"
        args.layers_grid = "1"
        log("DEBUG MODE: n_train_sample=60, n_val_sample=30, epochs=2, 1 config, 50 bootstrap draws. "
            "This should finish in well under a minute and only proves the pipeline runs end to end "
            "-- the QWK numbers from this mode are meaningless, do not report them.")

    root = Path(args.project_root).resolve()
    d = prepare_features(root, args, log)
    cache_dir = root / args.cache_dir

    suffix = "" if (args.backbone == "resnet50" and args.size == 224) else f"_{args.backbone}_{args.size}"

    if args.mode == "final":
        if args.n_qubits is None or args.n_layers is None:
            log("ERROR: --mode final requires --n-qubits and --n-layers (use the config chosen by "
                "the sweep on the primary backbone -- MASTER_PLAN's robustness-check pattern is to "
                "re-use, not re-search, hyperparameters on the second backbone).")
            sys.exit(1)
        angle_feats = compute_angle_features(d, args.n_qubits, args.seed, log)
        val_qwk, test_pred, epochs_run, seconds = run_pqc_config(
            d, angle_feats, args.n_qubits, args.n_layers, args, cache_dir, log)
        finalize_and_report(d, args.n_qubits, args.n_layers, test_pred, args, root,
                             extra={"selection_method": "fixed (no sweep this run)",
                                    "val_qwk_at_selection": val_qwk, "epochs_run": epochs_run,
                                    "seconds": seconds})
        return

    # --mode sweep (default) and --mode debug both run the grid (debug's
    # grid is a single trivial cell). PCA+angle-encoding is computed once
    # per n_qubits and reused across every n_layers for that qubit count.
    qubits_grid = [int(x) for x in args.qubits_grid.split(",")]
    layers_grid = [int(x) for x in args.layers_grid.split(",")]
    sweep_rows = []
    best = None
    for nq in qubits_grid:
        angle_feats = compute_angle_features(d, nq, args.seed, log)
        for nl in layers_grid:
            val_qwk, test_pred, epochs_run, seconds = run_pqc_config(
                d, angle_feats, nq, nl, args, cache_dir, log)
            row = {"n_qubits": nq, "n_layers": nl, "val_qwk": val_qwk,
                   "epochs_run": epochs_run, "seconds": seconds}
            sweep_rows.append(row)
            if best is None or val_qwk > best["val_qwk"]:
                best = dict(row, test_pred=test_pred)

    print("\n--- SWEEP RESULTS (selection metric: val QWK, matched train subsample) ---")
    for r in sorted(sweep_rows, key=lambda r: -r["val_qwk"]):
        print(f"  qubits={r['n_qubits']:2d}  layers={r['n_layers']:2d}  val_qwk={r['val_qwk']:.4f}  "
              f"epochs_run={r['epochs_run']:3d}  seconds={r['seconds']:.1f}")
    print(f"\nBest config: qubits={best['n_qubits']}, layers={best['n_layers']} (val_qwk={best['val_qwk']:.4f})")

    if args.mode == "sweep":
        make_sweep_figure(args, root, sweep_rows, suffix)

    out = finalize_and_report(
        d, best["n_qubits"], best["n_layers"], best["test_pred"], args, root,
        extra={"selection_method": f"grid search over qubits={qubits_grid} x layers={layers_grid}, "
                                    "selected by validation QWK",
               "val_qwk_at_selection": best["val_qwk"], "sweep": sweep_rows})
    if args.mode == "debug":
        print("\nDEBUG MODE finished without crashing. The numbers above are NOT meaningful "
              "(60 training images, 2 epochs) -- this only confirms the pipeline runs end to end. "
              "Now run --mode sweep for real.")
    return out


if __name__ == "__main__":
    main()
