"""
SECOND, SEPARATE QML experiment track (2026-09): a TRUE no-classical-CNN
Quantum Convolutional Neural Network (QCNN).

WHY THIS FILE EXISTS, AND HOW IT DIFFERS FROM qml_pqc.py:
  qml_pqc.py ("the hybrid head") keeps ResNet-50/EfficientNet-B0 doing the
  image understanding, and only replaces the final classifier with a real
  PQC. That is what the literature anchor (Ahmed et al., arXiv:2405.01734)
  and every other DR+QML paper found during research actually do, and it's
  this project's primary, reportable QML result.

  This file answers a different, harder question the user asked directly:
  "what if there's no classical CNN doing feature extraction AT ALL?" --
  i.e. a quantum circuit that looks at (heavily downsampled) raw pixels
  itself, with the class boundary drawn by nothing but quantum gates plus
  one small linear readout (the same role StandardScaler+LogisticRegression
  plays as a "head" everywhere else in this project -- it is not a hidden
  CNN, it has no convolutional filters and nothing resembling one).

  READ THIS BEFORE TRUSTING ANY NUMBER FROM THIS SCRIPT: going fully
  quantum forces heavy image downsampling (see "THE ENCODING BOTTLENECK"
  below) which very plausibly destroys the fine lesion detail (micro-
  aneurysms, small hemorrhages) that DR grading actually depends on. A
  weak or negative result here is the EXPECTED, honestly-reportable
  outcome, not a bug to chase -- exactly the standard this project has
  always held itself to (see PROGRESS.md's Claim 2 null result and Claim
  2b's borderline result).

THE ENCODING BOTTLENECK (why raw 224x224 images can't go in directly):
  Loading N classical numbers onto qubits by amplitude encoding needs only
  log2(N) qubits -- efficient in qubit COUNT -- but preparing that quantum
  state from arbitrary classical data is not a free operation, and
  simulating more than a modest number of qubits is exponentially
  expensive on a CPU (the state vector has 2^n_qubits complex entries).
  A 224x224 grayscale image is 50,176 pixels = 2^15.6 -- workable in
  PRINCIPLE at 16 qubits, but combined with batched training and
  gradients, is far past what's practical here. So this script downsamples
  each image to a tiny square patch (4x4=16px for 4 qubits, 16x16=256px
  for 8 qubits) BEFORE the quantum circuit ever sees it. This downsampling
  is a plain, non-trained resize (PIL bilinear) -- not a CNN, not a
  learned filter of any kind -- but it is real, and it is destructive.

ARCHITECTURE -- a real Quantum Convolutional Neural Network, per Cong,
Choi & Lukin, "Quantum Convolutional Neural Networks", arXiv:1810.03787
(published Nature Physics 15, 1273-1278, 2019) -- CONFIRMED by web search
(title, authors, arXiv id, journal, DOI 10.1038/s41567-019-0648-8):
  1. ENCODING: the downsampled, flattened pixel patch is L2-normalized and
     loaded via qml.AmplitudeEmbedding -- genuinely different from
     qml_pqc.py's angle encoding, and for a specific, stated reason: with
     n_qubits qubits, amplitude encoding fits 2**n_qubits pixel values (16
     pixels on 4 qubits, 256 on 8), whereas angle encoding would need one
     qubit PER pixel. This is exactly the tradeoff the original assignment
     asked to be aware of ("use it unless you have a specific reason to
     prefer amplitude encoding") -- here, fitting a whole (tiny) image
     onto few qubits is that reason.
  2. QUANTUM CONVOLUTION: the SAME small parameterized 2-qubit gate (a
     hardware-efficient block: an arbitrary rotation on each of the two
     qubits + two CNOTs) applied to every adjacent pair of qubits, in a
     brick-wall pattern (even pairs, then odd pairs, wrapping in a ring).
     One shared "filter" applied everywhere = translation invariance =
     what makes this a convolution rather than an arbitrary circuit.
     n_conv_reps (repeating this block before each pooling step) is one of
     the two swept hyperparameters.
  3. QUANTUM POOLING: halves the active register. For each adjacent pair,
     a small parameterized two-qubit gate entangles the pair, then the
     second qubit of each pair is DROPPED from every later layer and from
     the final readout. Nothing is physically measured out mid-circuit in
     this simulation -- dropping a wire from later use is the standard
     simulate-a-QCNN shortcut and is mathematically what "pooling" means
     here (the information the dropped qubit carried has already been
     folded into its partner via the entangling gate before being
     excluded).
  4. This repeats (conv x n_conv_reps, then pool) until 2 qubits remain
     (n_qubits, the OTHER swept hyperparameter, sets how many stages that
     takes: 4 qubits -> 1 stage -> 2 left; 8 qubits -> 2 stages -> 2 left).
     Stopping at 2 qubits rather than 1 gives 4 basis states to read out,
     which is more useful for a 5-class problem than a single qubit's 2.
  5. READOUT: qml.probs() on the 2 surviving qubits (4 non-negative numbers
     summing to 1) -> ONE classical Linear(4, 5) layer -> softmax ->
     cross-entropy. This linear layer is the same kind of minimal decision
     boundary the multinomial baseline uses (StandardScaler +
     LogisticRegression is also "just a linear layer") -- it is not a
     hidden CNN and has no spatial/convolutional structure of its own.

FAIR COMPARISON -- three tiers, not confused with each other:
  (a) QCNN itself: quantum circuit on downsampled raw pixels.
  (b) "Downsampled-pixel classical baseline": a plain multinomial
      LogisticRegression (fit_and_eval, reused unchanged from
      claim2_protocols.py) trained on the EXACT SAME flattened,
      downsampled pixel vectors QCNN sees. This is the fair,
      apples-to-apples comparison for THIS experiment -- it isolates "does
      going quantum help or hurt, given identical (crippled) input", from
      "how much does downsampling itself cost you".
  (c) Cited, NOT matched: the full-resolution CNN-feature classical numbers
      from results/claim2_protocols.json (same as qml_pqc.py cites) -- for
      context on how much is lost by abandoning CNN features entirely.
  Every number in (a) and (b) gets a patient-level bootstrap CI
  (patient_bootstrap_ci, imported unchanged from claim2_protocols.py), and
  QCNN vs (b) gets a paired bootstrap CI on the difference
  (paired_bootstrap_ci_diff, imported unchanged from claim3_both_eyes.py)
  -- same non-negotiable statistical framework as everywhere else in this
  project.

SCOPE / TRAINING-SET SIZE: same reasoning and same caveat as qml_pqc.py --
a CPU-simulated quantum circuit does not scale to the full ~27,100-image
P2 training set. Both QCNN and its matched classical baseline train on a
fixed, seeded, class-stratified subsample (--n-train-sample, default
2000 -- smaller than qml_pqc.py's default because this script also has to
load and downsample a real JPEG per image, not just read a row out of a
pre-computed .npz). Evaluation is always on the full, unmodified P2 test
fold.

IMPORTANT CAVEAT (same situation as qml_pqc.py when it was first written):
this file's PennyLane/PyTorch code (build_qcnn_model, train_qcnn,
predict_qcnn) was written carefully against the documented PennyLane API
but could not be executed in the dev sandbox (no pennylane, no network
route to install it there). It reuses the exact TorchLayer batching
pattern qml_pqc.py already hit a real bug on and fixed (see that file's
build_qml_model: slice inputs with [..., :k], never [:k], because
TorchLayer hands the qfunc a BATCHED tensor directly) -- this file applies
that fix from the start. Run --mode debug FIRST.

Usage (see run.ps1 for the exact commands):
    python src\\experiments\\qcnn_no_cnn.py --mode debug
    python src\\experiments\\qcnn_no_cnn.py --mode sweep --backbone-label pixels_224
    python src\\experiments\\qcnn_no_cnn.py --mode final --n-qubits Q --n-conv-reps R
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.claim2_protocols import load_active_manifest, fit_and_eval, patient_bootstrap_ci
from src.experiments.claim3_both_eyes import paired_bootstrap_ci_diff
from src.experiments.qml_pqc import stratified_sample_ids  # reused unchanged, pure function

SEED = 42
N_CLASSES = 5

LITERATURE_NOTES = {
    "cong_choi_lukin_2019_arxiv_1810.03787": (
        "Cong, Choi & Lukin, 'Quantum Convolutional Neural Networks', "
        "arXiv:1810.03787 (published Nature Physics 15, 1273-1278, 2019, "
        "DOI 10.1038/s41567-019-0648-8). CONFIRMED by web search (title, "
        "authors, arXiv id, journal, DOI all matched). This is the source "
        "of the conv/pool architecture used in this file: a shared, "
        "translation-invariant parameterized 2-qubit 'convolution' gate "
        "applied across the qubit register, alternated with parameterized "
        "'pooling' gates that halve the active register each stage. The "
        "original paper targets quantum phase recognition and small "
        "image benchmarks with a SINGLE final qubit for BINARY decisions; "
        "this script adapts the same conv/pool pattern to a 5-class "
        "problem by stopping pooling at 2 surviving qubits (4 basis "
        "outcomes) instead of 1, and generalizes the ansatz's conv/pool "
        "depth (n_conv_reps) and register size (n_qubits) into the two "
        "explicitly swept hyperparameters this project's standard "
        "requires."
    ),
}


# =====================================================================
# Pure, testable helpers
# =====================================================================

def patch_side_for_qubits(n_qubits):
    """n_qubits qubits amplitude-encode 2**n_qubits pixel values. This
    returns the side length of the square patch that uses exactly that
    many pixels -- only defined for even n_qubits (so the patch is an
    actual square, not a rectangle)."""
    if n_qubits % 2 != 0:
        raise ValueError(f"n_qubits must be even for a square patch; got {n_qubits}")
    return 2 ** (n_qubits // 2)


def load_image_patch(path, patch_side):
    """Loads one image, converts to grayscale, and downsamples (PIL
    bilinear -- a plain resize, not a learned filter of any kind) to a
    patch_side x patch_side patch. Returns a flat float64 vector of length
    patch_side**2, raw pixel intensities in [0, 255] (NOT yet normalized
    for amplitude embedding -- see normalize_for_amplitude_embedding)."""
    with Image.open(path) as im:
        gray = im.convert("L")
        small = gray.resize((patch_side, patch_side), Image.BILINEAR)
        arr = np.asarray(small, dtype=np.float64).flatten()
    return arr


def normalize_for_amplitude_embedding(vec):
    """Pure function. L2-normalizes a pixel vector for qml.AmplitudeEmbedding.
    Adds a tiny epsilon before normalizing so an (extremely unlikely, but
    possible) all-zero downsampled patch doesn't divide by zero -- this
    epsilon is negligible next to real pixel intensities (0-255) but keeps
    the function total rather than crashing on a degenerate input."""
    vec = np.asarray(vec, dtype=np.float64) + 1e-6
    norm = np.linalg.norm(vec)
    return vec / norm


def cache_key(label, n_qubits, n_conv_reps, n_train_sample, seed):
    raw = f"{label}_q{n_qubits}_r{n_conv_reps}_n{n_train_sample}_s{seed}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def qcnn_verdict(diff_qcnn_minus_baseline):
    """Pure function. Same 'CI must exclude 0, not just a bigger point
    estimate' rule as everywhere else in this project."""
    if diff_qcnn_minus_baseline["significantly_better"]:
        return ("qcnn_beats_matched_baseline", (
            "The QCNN significantly beats a classical model given the EXACT SAME "
            f"downsampled pixels (diff={diff_qcnn_minus_baseline['mean_diff']:+.4f}, "
            f"CI {diff_qcnn_minus_baseline['ci95']}). Note this is a much lower bar than "
            "beating the full-resolution CNN-feature classical heads -- check the cited "
            "full-train-set reference in the same report before calling this a win overall."
        ))
    mean_diff = diff_qcnn_minus_baseline["mean_diff"]
    if mean_diff < 0:
        return ("qcnn_loses_to_matched_baseline", (
            f"The QCNN does NOT beat the matched classical baseline (diff={mean_diff:+.4f}, "
            f"CI {diff_qcnn_minus_baseline['ci95']}) -- even a plain logistic regression on the "
            "same crippled, heavily downsampled pixels does as well or better. Report this "
            "honestly: it does not mean 'quantum doesn't work', it means a few-qubit circuit "
            "on a 4x4 or 16x16 pixel patch has very little to work with, which was the "
            "expected risk of going fully quantum (see this file's module docstring)."
        ))
    return ("inconclusive", (
        f"QCNN's point estimate is higher (diff={mean_diff:+.4f}) but the paired bootstrap CI "
        f"includes 0 ({diff_qcnn_minus_baseline['ci95']}) -- not distinguishable from the matched "
        "classical baseline at this sample size. Report as inconclusive, not as a win."
    ))


# =====================================================================
# Quantum model -- needs pennylane + torch, imported lazily (see module
# docstring's IMPORTANT CAVEAT).
# =====================================================================

def _conv_unitary(w1, w2, weights):
    """The shared 2-qubit 'convolution filter': an arbitrary single-qubit
    rotation on each wire, then two CNOTs (so information flows both
    ways). 6 trainable parameters, reused at every pair -- translation
    invariance is what makes this a 'convolution' rather than an
    arbitrary circuit."""
    import pennylane as qml
    qml.Rot(weights[0], weights[1], weights[2], wires=w1)
    qml.Rot(weights[3], weights[4], weights[5], wires=w2)
    qml.CNOT(wires=[w1, w2])
    qml.CNOT(wires=[w2, w1])


def qcnn_conv_block(wires, weights):
    """One quantum-convolution layer: the shared _conv_unitary applied in
    a brick-wall pattern (even-indexed pairs, then odd-indexed pairs,
    wrapping around as a ring) across every wire in `wires`. Requires an
    even number of wires (true throughout this file: register sizes are
    always powers of 2 down to 2)."""
    n = len(wires)
    for i in range(0, n, 2):
        _conv_unitary(wires[i], wires[(i + 1) % n], weights)
    for i in range(1, n, 2):
        _conv_unitary(wires[i], wires[(i + 1) % n], weights)


def qcnn_pool_block(wires, weights):
    """One quantum-pooling layer: halves `wires`. For each adjacent pair
    (keep, drop), a parameterized controlled rotation entangles `drop`
    into `keep`, then `drop` is excluded from every later layer. Returns
    the list of surviving wires (half the length of the input, since
    every register size in this file is a power of 2)."""
    import pennylane as qml
    keep = []
    for i in range(0, len(wires), 2):
        w_keep, w_drop = wires[i], wires[i + 1]
        qml.CRot(weights[0], weights[1], weights[2], wires=[w_drop, w_keep])
        keep.append(w_keep)
    return keep


def build_qcnn_model(n_qubits, n_conv_reps, diff_method="parameter-shift", seed=SEED):
    import pennylane as qml
    import torch
    import torch.nn as nn

    if n_qubits & (n_qubits - 1) != 0:
        raise ValueError(f"n_qubits must be a power of 2 for this pooling schedule; got {n_qubits}")
    n_stages = int(np.log2(n_qubits)) - 1  # stop pooling with 2 qubits left, not 1
    if n_stages < 1:
        raise ValueError(f"n_qubits={n_qubits} is too small -- need at least 4 qubits (1 pooling stage)")

    # "adjoint" is NOT usable here (see --diff-method's help text): this
    # circuit's readout is qml.probs(), and PennyLane's adjoint-Jacobian
    # method only supports expectation-value measurements, on ANY device,
    # lightning.qubit included -- that's the "does not support adjoint
    # with requested circuit" error, not a device bug to work around.
    # "backprop" gets the same O(1)-per-parameter speed adjoint would have
    # given, and DOES support qml.probs(), but only on a device that runs
    # its simulation as native, differentiable torch tensor ops rather
    # than lightning's compiled C++ backend -- hence the device switch.
    if diff_method == "backprop":
        dev = qml.device("default.qubit", wires=n_qubits)
    else:
        dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, conv_weights, pool_weights):
        # inputs: flattened, L2-normalized pixel patch, length 2**n_qubits.
        # This IS the encoding of the raw (downsampled) image -- no
        # classical neural network extracts features before this.
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
        active = list(range(n_qubits))
        for stage in range(n_stages):
            for rep in range(n_conv_reps):
                qcnn_conv_block(active, conv_weights[stage][rep])
            active = qcnn_pool_block(active, pool_weights[stage])
        # `active` now has exactly 2 wires (n_stages was chosen to stop here).
        return qml.probs(wires=active)  # 4 non-negative numbers, sum to 1

    conv_shape = (n_stages, n_conv_reps, 6)
    pool_shape = (n_stages, 3)
    weight_shapes = {"conv_weights": conv_shape, "pool_weights": pool_shape}
    torch.manual_seed(seed)
    qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    class QCNNHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.qlayer = qlayer
            # dtype=torch.float64 to match qlayer's output dtype -- the quantum
            # node runs in float64/complex128 internally, and TorchLayer's
            # output follows the INPUT dtype (we feed float64 pixel vectors),
            # while nn.Linear defaults to float32. Left as default, this
            # crashes with "mat1 and mat2 must have the same dtype, but got
            # Double and Float" the first time this runs on real data (same
            # bug and same fix as qml_pqc.py's QMLHead.readout).
            self.readout = nn.Linear(4, N_CLASSES, dtype=torch.float64)  # 4 = probs of the 2 surviving qubits

        def forward(self, x):
            q_out = self.qlayer(x)      # (batch, 4)
            return self.readout(q_out)  # (batch, 5) logits

    return QCNNHead()


def train_qcnn(X_train, y_train, X_val, y_val, n_qubits, n_conv_reps,
               epochs=30, batch_size=32, lr=0.05, patience=6,
               diff_method="parameter-shift", seed=SEED, log=print):
    """Same hybrid classical-quantum training loop as qml_pqc.py's
    train_qml_head: PyTorch Adam over a model whose forward pass runs a
    real PennyLane circuit, model selection by VALIDATION QWK (never
    test), early stopping, returns the best-val_qwk state (not the final
    epoch's)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    model = build_qcnn_model(n_qubits, n_conv_reps, diff_method=diff_method, seed=seed)
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


def predict_qcnn(model, X):
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


def prepare_pixel_data(root, args, patch_side, log):
    """Loads manifest + P2 split, picks the matched train/val subsamples
    (shared with the classical baseline), and loads+downsamples the
    actual JPEGs for train/val/full-test. This is the expensive I/O step
    -- called once per n_qubits (== once per patch_side) and reused across
    every n_conv_reps in the sweep for that qubit count, same optimization
    qml_pqc.py's compute_angle_features makes for PCA."""
    active_manifest = load_active_manifest(root / args.manifest)
    grade_by_id = dict(zip(active_manifest["image_id"], active_manifest["grade"]))
    patient_by_id = dict(zip(active_manifest["image_id"], active_manifest["patient_id"]))

    img_dir = root / args.processed_dir / str(args.proc_size)
    if not img_dir.exists():
        log(f"ERROR: {img_dir} not found. Run src/data/build_cache.py for this size first.")
        sys.exit(1)

    split_map = json.loads((root / args.p2_split).read_text())
    train_ids_all = [i for i in load_p2_fold_ids(split_map, "train")
                     if i in grade_by_id and (img_dir / f"{i}.jpg").exists()]
    val_ids_all = [i for i in load_p2_fold_ids(split_map, "val")
                   if i in grade_by_id and (img_dir / f"{i}.jpg").exists()]
    test_ids_all = [i for i in load_p2_fold_ids(split_map, "test")
                    if i in grade_by_id and (img_dir / f"{i}.jpg").exists()]

    train_grades_all = np.array([grade_by_id[i] for i in train_ids_all])
    train_ids = stratified_sample_ids(train_ids_all, train_grades_all, args.n_train_sample, seed=args.seed)
    val_grades_all = np.array([grade_by_id[i] for i in val_ids_all])
    val_ids = stratified_sample_ids(val_ids_all, val_grades_all, args.n_val_sample, seed=args.seed)
    test_ids = test_ids_all  # full, unmodified P2 test fold

    log(f"loading + downsampling to {patch_side}x{patch_side}: "
        f"{len(train_ids)} train (of {len(train_ids_all)} P2-train), "
        f"{len(val_ids)} val (of {len(val_ids_all)} P2-val), "
        f"{len(test_ids)} test (full P2-test)...")

    def load_all(ids):
        return np.stack([load_image_patch(img_dir / f"{i}.jpg", patch_side) for i in ids])

    X_train = load_all(train_ids)
    X_val = load_all(val_ids)
    X_test = load_all(test_ids)
    y_train = np.array([grade_by_id[i] for i in train_ids])
    y_val = np.array([grade_by_id[i] for i in val_ids])
    y_test = np.array([grade_by_id[i] for i in test_ids])
    patient_ids_test = [patient_by_id[i] for i in test_ids]

    return {
        "X_train": X_train, "y_train": y_train, "n_train_full": len(train_ids_all),
        "X_val": X_val, "y_val": y_val, "n_val_full": len(val_ids_all),
        "X_test": X_test, "y_test": y_test, "patient_ids_test": patient_ids_test,
    }


def run_qcnn_config(d, n_qubits, n_conv_reps, args, cache_dir, log):
    """Trains (or loads from cache) one (n_qubits, n_conv_reps) config on
    the matched pixel-patch train subsample. Returns
    (val_qwk, test_pred, epochs_run, seconds)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(args.backbone_label, n_qubits, n_conv_reps, args.n_train_sample, args.seed)
    cache_path = cache_dir / f"{key}.npz"
    if cache_path.exists():
        c = np.load(cache_path)
        log(f"  [q={n_qubits} r={n_conv_reps}] loaded from cache {cache_path.name}: "
            f"val_qwk={float(c['val_qwk']):.4f}")
        return float(c["val_qwk"]), c["test_pred"], int(c["epochs_run"]), float(c["seconds"])

    t0 = time.time()
    Xtr_amp = np.stack([normalize_for_amplitude_embedding(x) for x in d["X_train"]])
    Xva_amp = np.stack([normalize_for_amplitude_embedding(x) for x in d["X_val"]])
    Xte_amp = np.stack([normalize_for_amplitude_embedding(x) for x in d["X_test"]])

    log(f"  [q={n_qubits} r={n_conv_reps}] training...")
    model, val_qwk, epochs_run = train_qcnn(
        Xtr_amp, d["y_train"], Xva_amp, d["y_val"], n_qubits, n_conv_reps,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience,
        diff_method=args.diff_method, seed=args.seed, log=log)
    test_pred = predict_qcnn(model, Xte_amp)
    seconds = time.time() - t0
    log(f"  [q={n_qubits} r={n_conv_reps}] done in {seconds:.1f}s, best val_qwk={val_qwk:.4f}, "
        f"{epochs_run} epochs")

    np.savez(cache_path, val_qwk=val_qwk, test_pred=test_pred, epochs_run=epochs_run, seconds=seconds)
    return val_qwk, test_pred, epochs_run, seconds


def load_full_trainset_reference(root, claim2_json_path):
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
        "multinomial_qwk_full_resolution_cnn_features": p2["reference_qwk_seed42"],
        "ordinal_qwk_full_resolution_cnn_features": p2["ordinal_qwk"],
        "n_train_full": p2["n_train"],
        "note": ("Cited from results/claim2_protocols.json, NOT recomputed. Trained on FULL-"
                 "RESOLUTION CNN features (not downsampled raw pixels) and the full P2 train "
                 "set -- context only, to show how much is lost by abandoning CNN features "
                 "entirely. Not a matched comparison for this experiment."),
    }


def finalize_and_report(d, chosen_qubits, chosen_reps, chosen_test_pred, args, root, extra):
    # fit_and_eval (claim2_protocols.py) returns (qwk, y_pred) -- QWK first,
    # predictions second -- so this unpacks it in that order (a prior
    # version had it backwards, which fed cohen_kappa_score a bare float
    # where it expected the prediction array).
    _, pred_baseline = fit_and_eval(d["X_train"], d["y_train"], d["X_test"], d["y_test"],
                                     args.seed, bootstrap_train=False)

    qwk_qcnn = float(cohen_kappa_score(d["y_test"], chosen_test_pred, weights="quadratic"))
    ci_qcnn = patient_bootstrap_ci(d["patient_ids_test"], d["y_test"], chosen_test_pred,
                                    n_boot=args.n_bootstrap_ci, seed=42)
    qwk_baseline = float(cohen_kappa_score(d["y_test"], pred_baseline, weights="quadratic"))
    ci_baseline = patient_bootstrap_ci(d["patient_ids_test"], d["y_test"], pred_baseline,
                                        n_boot=args.n_bootstrap_ci, seed=42)

    diff = paired_bootstrap_ci_diff(d["patient_ids_test"], d["y_test"], pred_baseline, chosen_test_pred,
                                     n_boot=args.n_bootstrap_ci, seed=43)
    verdict, verdict_text = qcnn_verdict(diff)
    ref = load_full_trainset_reference(root, args.claim2_json)

    print("=" * 78)
    print(f"QCNN (NO CNN)  patch={patch_side_for_qubits(chosen_qubits)}x{patch_side_for_qubits(chosen_qubits)}px, "
          f"q={chosen_qubits}, reps={chosen_reps}, matched train n={len(d['y_train'])}, full test n={len(d['y_test'])}")
    print("=" * 78)
    print(f"  QCNN:                          QWK={qwk_qcnn:.4f}  95% CI {list(ci_qcnn)}")
    print(f"  matched classical baseline:    QWK={qwk_baseline:.4f}  95% CI {list(ci_baseline)}  "
          f"(logistic regression, SAME downsampled pixels)")
    if ref is not None:
        print(f"  --- cited, full-resolution CNN features + full P2 train set (n={ref['n_train_full']}) ---")
        print(f"  multinomial: {ref['multinomial_qwk_full_resolution_cnn_features']:.4f}   "
              f"ordinal: {ref['ordinal_qwk_full_resolution_cnn_features']:.4f}")
    print()
    print(f"  QCNN - matched baseline: diff={diff['mean_diff']:+.4f}  CI {diff['ci95']}  "
          f"significant={diff['significantly_better']}")
    print()
    print(f"VERDICT ({verdict}):", verdict_text)

    out = {
        "experiment": "qcnn_no_cnn",
        "chosen_config": {"n_qubits": chosen_qubits, "n_conv_reps": chosen_reps,
                           "patch_side": patch_side_for_qubits(chosen_qubits)},
        "n_train_sample": args.n_train_sample, "n_val_sample": args.n_val_sample,
        "n_train_full_p2": d["n_train_full"], "n_val_full_p2": d["n_val_full"],
        "n_test": len(d["y_test"]),
        "proc_size": args.proc_size, "diff_method": args.diff_method,
        "epochs_max": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "patience": args.patience, "seed": args.seed,
        "qcnn": {"qwk": qwk_qcnn, "patient_bootstrap_ci95": list(ci_qcnn)},
        "matched_classical_baseline": {"qwk": qwk_baseline, "patient_bootstrap_ci95": list(ci_baseline),
                                        "note": "multinomial logistic regression on the SAME downsampled pixels"},
        "paired_diff_qcnn_minus_baseline": diff,
        "full_training_set_reference_cited": ref,
        "verdict": verdict, "verdict_text": verdict_text,
        "literature": LITERATURE_NOTES,
        **extra,
    }
    out_path = root / (args.out or f"results/qcnn_no_cnn_{args.backbone_label}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"\nWrote {out_path}")

    make_figure(args, root, qwk_qcnn, ci_qcnn, qwk_baseline, ci_baseline, ref)
    return out


def make_figure(args, root, qwk_qcnn, ci_qcnn, qwk_baseline, ci_baseline, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["QCNN\n(no CNN)", "classical\n(same pixels)"]
    heights = [qwk_qcnn, qwk_baseline]
    cis = [ci_qcnn, ci_baseline]
    colors = ["#9467bd", "#7f7f7f"]
    if ref is not None:
        labels += ["multinomial\n(full-res CNN\nfeatures, cited)", "ordinal\n(full-res CNN\nfeatures, cited)"]
        heights += [ref["multinomial_qwk_full_resolution_cnn_features"],
                    ref["ordinal_qwk_full_resolution_cnn_features"]]
        cis += [None, None]
        colors += ["#f4a6a6", "#a6c8e8"]

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(labels))
    err_lo = [max(0.0, h - c[0]) if c else 0.0 for h, c in zip(heights, cis)]
    err_hi = [max(0.0, c[1] - h) if c else 0.0 for h, c in zip(heights, cis)]
    ax.bar(x, heights, color=colors, yerr=[err_lo, err_hi], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("P2 test QWK (per-image)")
    ax.set_ylim(0, 1)
    ax.set_title("QCNN (no classical CNN) vs. matched classical baseline vs. cited full-res reference\n"
                 "error bars: 95% patient-bootstrap CI. Faded bars are NOT a matched comparison.", fontsize=9)
    for i, h in enumerate(heights):
        ax.text(i, h + 0.02, f"{h:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig_path = root / (args.fig or f"figures/figure_qcnn_no_cnn_{args.backbone_label}.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {fig_path}")


def make_sweep_figure(args, root, sweep_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qubits = sorted(set(r["n_qubits"] for r in sweep_rows))
    reps = sorted(set(r["n_conv_reps"] for r in sweep_rows))
    grid = np.full((len(reps), len(qubits)), np.nan)
    for r in sweep_rows:
        grid[reps.index(r["n_conv_reps"]), qubits.index(r["n_qubits"])] = r["val_qwk"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, cmap="viridis", vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    ax.set_xticks(range(len(qubits))); ax.set_xticklabels(qubits)
    ax.set_yticks(range(len(reps))); ax.set_yticklabels(reps)
    ax.set_xlabel("n_qubits (patch size)"); ax.set_ylabel("n_conv_reps")
    for i in range(len(reps)):
        for j in range(len(qubits)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)
    ax.set_title(f"QCNN sweep -- validation QWK\n(matched train subsample n={args.n_train_sample})", fontsize=9)
    fig.colorbar(im, label="val QWK")
    fig.tight_layout()
    fig_path = root / f"figures/figure_qcnn_no_cnn_sweep_{args.backbone_label}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {fig_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--manifest", default="data/manifests/manifest.csv")
    ap.add_argument("--p2-split", default="data/splits/p2.json")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--proc-size", type=int, default=224, choices=[224, 384])
    ap.add_argument("--backbone-label", default="pixels",
                     help="Label used in output filenames -- this experiment has no CNN backbone, "
                          "so this just tags which run produced the file (e.g. 'pixels').")
    ap.add_argument("--mode", choices=["sweep", "final", "debug"], default="sweep")
    ap.add_argument("--qubits-grid", default="4,8", help="comma list of POWERS OF 2, --mode sweep only")
    ap.add_argument("--conv-reps-grid", default="1,2,3", help="comma list, --mode sweep only")
    ap.add_argument("--n-qubits", type=int, default=None, help="--mode final: fixed qubit count")
    ap.add_argument("--n-conv-reps", type=int, default=None, help="--mode final: fixed conv depth")
    ap.add_argument("--n-train-sample", type=int, default=2000)
    ap.add_argument("--n-val-sample", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--diff-method", default="parameter-shift",
                     choices=["parameter-shift", "adjoint", "backprop"],
                     help="'parameter-shift' (default -- slow: ~2 circuit evals per trainable "
                          "parameter per sample, which is why a 30-parameter, 8-qubit config can "
                          "take 30-40+ min/epoch). 'adjoint' WILL CRASH at runtime on this circuit "
                          "-- its readout is qml.probs(), and PennyLane's adjoint-Jacobian method "
                          "only supports expectation-value measurements, on any device. "
                          "'backprop' is the fast, WORKING alternative: same O(1)-per-parameter "
                          "cost adjoint would have given, and it DOES support qml.probs() -- but "
                          "only runs on default.qubit (a native, torch-differentiable simulator), "
                          "so passing this switches the device automatically (see build_qcnn_model). "
                          "Recommended for --mode sweep at these qubit counts (<=8).")
    ap.add_argument("--n-bootstrap-ci", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--claim2-json", default="results/claim2_protocols.json")
    ap.add_argument("--cache-dir", default="cache/qcnn_no_cnn")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fig", default=None)
    args = ap.parse_args()

    t0 = time.time()
    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    if args.mode == "debug":
        args.n_train_sample = 40
        args.n_val_sample = 20
        args.epochs = 2
        args.patience = 2
        args.n_bootstrap_ci = 50
        args.qubits_grid = "4"
        args.conv_reps_grid = "1"
        log("DEBUG MODE: n_train_sample=40, n_val_sample=20, epochs=2, 1 config, 50 bootstrap draws. "
            "Numbers from this mode are meaningless -- it only proves the pipeline runs end to end.")

    root = Path(args.project_root).resolve()

    if args.mode == "final":
        if args.n_qubits is None or args.n_conv_reps is None:
            log("ERROR: --mode final requires --n-qubits and --n-conv-reps (use the config chosen "
                "by --mode sweep).")
            sys.exit(1)
        patch_side = patch_side_for_qubits(args.n_qubits)
        d = prepare_pixel_data(root, args, patch_side, log)
        val_qwk, test_pred, epochs_run, seconds = run_qcnn_config(
            d, args.n_qubits, args.n_conv_reps, args, root / args.cache_dir, log)
        finalize_and_report(d, args.n_qubits, args.n_conv_reps, test_pred, args, root,
                             extra={"selection_method": "fixed (no sweep this run)",
                                    "val_qwk_at_selection": val_qwk, "epochs_run": epochs_run,
                                    "seconds": seconds})
        return

    # --mode sweep (default) and --mode debug: grid over (n_qubits, n_conv_reps).
    # Pixel data is loaded/downsampled once per n_qubits (== once per patch_side)
    # and reused across every n_conv_reps for that qubit count.
    qubits_grid = [int(x) for x in args.qubits_grid.split(",")]
    for nq in qubits_grid:
        if nq & (nq - 1) != 0 or nq < 4:
            log(f"ERROR: --qubits-grid values must be powers of 2, >= 4 (got {nq}).")
            sys.exit(1)
    reps_grid = [int(x) for x in args.conv_reps_grid.split(",")]

    sweep_rows = []
    best = None
    best_d = None
    for nq in qubits_grid:
        patch_side = patch_side_for_qubits(nq)
        d = prepare_pixel_data(root, args, patch_side, log)
        for reps in reps_grid:
            val_qwk, test_pred, epochs_run, seconds = run_qcnn_config(
                d, nq, reps, args, root / args.cache_dir, log)
            row = {"n_qubits": nq, "n_conv_reps": reps, "val_qwk": val_qwk,
                   "epochs_run": epochs_run, "seconds": seconds}
            sweep_rows.append(row)
            if best is None or val_qwk > best["val_qwk"]:
                best = dict(row, test_pred=test_pred)
                best_d = d

    print("\n--- SWEEP RESULTS (selection metric: val QWK, matched train subsample) ---")
    for r in sorted(sweep_rows, key=lambda r: -r["val_qwk"]):
        print(f"  qubits={r['n_qubits']:2d}  conv_reps={r['n_conv_reps']:2d}  val_qwk={r['val_qwk']:.4f}  "
              f"epochs_run={r['epochs_run']:3d}  seconds={r['seconds']:.1f}")
    print(f"\nBest config: qubits={best['n_qubits']}, conv_reps={best['n_conv_reps']} (val_qwk={best['val_qwk']:.4f})")

    if args.mode == "sweep":
        make_sweep_figure(args, root, sweep_rows)

    out = finalize_and_report(
        best_d, best["n_qubits"], best["n_conv_reps"], best["test_pred"], args, root,
        extra={"selection_method": f"grid search over qubits={qubits_grid} x conv_reps={reps_grid}, "
                                    "selected by validation QWK",
               "val_qwk_at_selection": best["val_qwk"], "sweep": sweep_rows})
    if args.mode == "debug":
        print("\nDEBUG MODE finished without crashing. The numbers above are NOT meaningful "
              "(40 training images, 2 epochs) -- this only confirms the pipeline runs end to end. "
              "Now run --mode sweep for real.")
    return out


if __name__ == "__main__":
    main()
