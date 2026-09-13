"""
Tests for src/experiments/qml_pqc.py (the new QML/PQC head track).

Two tiers, deliberately separated:
  1. Pure-logic tests (stratified sampling, angle scaling, PCA-slicing
     equivalence, cache keys, verdict logic) -- no pennylane/torch needed,
     these ran and passed in the dev sandbox before this file ever reached
     your machine.
  2. Quantum-pipeline tests (marked, and skipped automatically if
     pennylane/torch aren't importable) -- these COULD NOT run in the dev
     sandbox (no pennylane there, see qml_pqc.py's module docstring) and
     are the first real check that build_qml_model/train_qml_head/
     predict_qml actually work. Run this file right after
     `pip install pennylane pennylane-lightning` and BEFORE trusting a
     real sweep -- same convention as this project's other "smoke test
     first" scripts.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.qml_pqc import (
    stratified_sample_ids, fit_angle_scaler, apply_angle_scaler,
    pca_reduce, cache_key, verdict_from_diffs,
)

pennylane = pytest.importorskip("pennylane", reason="pennylane not installed yet")
torch = pytest.importorskip("torch", reason="torch not installed yet")
# Re-import the quantum-dependent functions only once we know the deps exist,
# so tier 1 above still collects and runs even before `pip install pennylane`.
from src.experiments.qml_pqc import build_qml_model, train_qml_head, predict_qml  # noqa: E402


# ---------------------------------------------------------------------
# Tier 1 -- pure logic
# ---------------------------------------------------------------------

def test_stratified_sample_preserves_class_proportions_and_is_deterministic():
    rng = np.random.RandomState(0)
    n = 20000
    grades = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.7, 0.08, 0.14, 0.05, 0.03])
    ids = [f"img_{i}" for i in range(n)]

    sample = stratified_sample_ids(ids, grades, 3000, seed=42)
    assert len(sample) == 3000
    assert len(set(sample)) == 3000  # no duplicates

    grade_by_id = dict(zip(ids, grades))
    sample_grades = [grade_by_id[i] for i in sample]
    full_props = pd.Series(grades).value_counts(normalize=True).sort_index()
    sample_props = pd.Series(sample_grades).value_counts(normalize=True).sort_index()
    for g in full_props.index:
        assert abs(full_props[g] - sample_props[g]) < 0.02, f"grade {g} proportion drifted too much"

    assert stratified_sample_ids(ids, grades, 3000, seed=42) == sample, "not deterministic given same seed"
    assert stratified_sample_ids(ids, grades, 3000, seed=43) != sample, "different seed gave identical sample"


def test_stratified_sample_passthrough_when_n_sample_exceeds_total():
    ids = [f"img_{i}" for i in range(10)]
    grades = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    assert stratified_sample_ids(ids, grades, 999, seed=1) == ids


def test_angle_scaler_range_endpoints_and_clipping():
    rng = np.random.RandomState(1)
    X_train = rng.uniform(-5, 5, size=(500, 6))
    lo, hi = fit_angle_scaler(X_train)
    angles_train = apply_angle_scaler(X_train, lo, hi)
    assert angles_train.min() >= -np.pi - 1e-9
    assert angles_train.max() <= np.pi + 1e-9
    assert np.allclose(apply_angle_scaler(lo.reshape(1, -1), lo, hi), -np.pi, atol=1e-9)
    assert np.allclose(apply_angle_scaler(hi.reshape(1, -1), lo, hi), np.pi, atol=1e-9)
    # out-of-[train-range] test-time values are clipped, not extrapolated
    extreme = np.full((1, 6), 1000.0)
    assert np.allclose(apply_angle_scaler(extreme, lo, hi), np.pi, atol=1e-9)


def test_angle_scaler_zero_range_column_no_nan():
    X_const = np.tile(np.array([1.0, 2, 3, 4, 5, 6]), (100, 1))
    lo, hi = fit_angle_scaler(X_const)
    angles = apply_angle_scaler(X_const, lo, hi)
    assert not np.isnan(angles).any()
    assert np.allclose(angles, 0.0)


def test_cache_key_deterministic_and_distinguishes_configs():
    k1 = cache_key("resnet50", 224, 4, 2, 3000, 42)
    k2 = cache_key("resnet50", 224, 4, 2, 3000, 42)
    k3 = cache_key("resnet50", 224, 4, 3, 3000, 42)
    assert k1 == k2
    assert k1 != k3


def test_pca_slicing_equivalence():
    """The docstring claim in pca_reduce(): the first k columns of a
    PCA(K).transform(X) for K > k are the same (up to a deterministic
    per-component sign, which sklearn fixes via svd_flip) as
    PCA(k).transform(X) fit directly. This is what makes it safe for
    compute_angle_features() to be called once per n_qubits rather than
    once per (n_qubits, n_layers) sweep cell."""
    rng = np.random.RandomState(2)
    X = rng.normal(size=(300, 50))
    _, big, _ = pca_reduce(X, [], n_components=10, seed=42)
    _, small, _ = pca_reduce(X, [], n_components=4, seed=42)
    for j in range(4):
        col_big, col_small = big[:, j], small[:, j]
        same = np.allclose(col_big, col_small, atol=1e-6)
        opp = np.allclose(col_big, -col_small, atol=1e-6)
        assert same or opp, f"component {j} differs beyond a sign flip"


def test_verdict_from_diffs_three_branches():
    sig_pos = {"mean_diff": 0.05, "ci95": [0.01, 0.09], "significantly_better": True}
    not_sig = {"mean_diff": 0.01, "ci95": [-0.02, 0.04], "significantly_better": False}
    neg = {"mean_diff": -0.03, "ci95": [-0.06, -0.005], "significantly_better": False}

    verdict, _ = verdict_from_diffs(sig_pos, sig_pos)
    assert verdict == "usable_quantum_head"

    verdict, _ = verdict_from_diffs(not_sig, sig_pos)
    assert verdict == "beats_multinomial_not_ordinal"

    verdict, _ = verdict_from_diffs(neg, not_sig)
    assert verdict == "negative_result"


# ---------------------------------------------------------------------
# Tier 2 -- quantum pipeline (skipped automatically until pennylane+torch
# are installed; run this file again right after `pip install` to
# actually exercise these).
# ---------------------------------------------------------------------

def _synthetic_angle_data(n_qubits, n_train=60, n_val=30, seed=0):
    rng = np.random.RandomState(seed)
    X_train = rng.uniform(-np.pi, np.pi, size=(n_train, 2 * n_qubits))
    X_val = rng.uniform(-np.pi, np.pi, size=(n_val, 2 * n_qubits))
    y_train = rng.randint(0, 5, size=n_train)
    y_val = rng.randint(0, 5, size=n_val)
    return X_train, y_train, X_val, y_val


def test_qml_head_runs_end_to_end_on_synthetic_data():
    """Not a claim that the model learns anything on random labels -- just
    that build_qml_model / train_qml_head / predict_qml execute without
    crashing, shapes are right, and predictions are valid grades. This is
    the test to run first after installing pennylane, per the module
    docstring's IMPORTANT CAVEAT."""
    n_qubits, n_layers = 4, 1
    X_train, y_train, X_val, y_val = _synthetic_angle_data(n_qubits)
    model, val_qwk, epochs_run = train_qml_head(
        X_train, y_train, X_val, y_val, n_qubits, n_layers,
        epochs=2, batch_size=16, lr=0.05, patience=2, diff_method="adjoint", seed=0,
        log=lambda msg: None)
    assert epochs_run >= 1
    assert isinstance(val_qwk, float)
    preds = predict_qml(model, X_val)
    assert preds.shape == (len(X_val),)
    assert set(np.unique(preds).tolist()).issubset({0, 1, 2, 3, 4})


def test_adjoint_matches_parameter_shift_gradient():
    """The sweep trains with diff_method='adjoint' (fast, simulator-only)
    but the ansatz is hardware-executable and would use parameter-shift on
    a real QPU. Confirms the two gradient rules agree on a tiny circuit,
    so speeding up the CPU sweep with adjoint isn't silently changing what
    gets optimized."""
    import pennylane as qml

    n_qubits, n_layers = 2, 1
    dev = qml.device("lightning.qubit", wires=n_qubits)

    def circuit_factory(diff_method):
        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs[:n_qubits], wires=range(n_qubits), rotation="X")
            qml.AngleEmbedding(inputs[n_qubits:], wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))
        return circuit

    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    rng = np.random.RandomState(0)
    inputs = torch.tensor(rng.uniform(-np.pi, np.pi, size=2 * n_qubits), dtype=torch.float64)
    weights_np = rng.uniform(-np.pi, np.pi, size=shape)

    grads = {}
    for method in ("adjoint", "parameter-shift"):
        circuit = circuit_factory(method)
        weights = torch.tensor(weights_np, dtype=torch.float64, requires_grad=True)
        out = circuit(inputs, weights)
        out.backward()
        grads[method] = weights.grad.detach().numpy().copy()

    assert np.allclose(grads["adjoint"], grads["parameter-shift"], atol=1e-5), (
        "adjoint and parameter-shift gradients disagree -- do not trust the sweep's "
        "fast diff_method until this is resolved"
    )
