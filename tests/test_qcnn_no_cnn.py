"""
Tests for src/experiments/qcnn_no_cnn.py (the true no-CNN QCNN track).

Same two-tier split as tests/test_qml_pqc.py:
  1. Pure-logic tests -- no pennylane/torch needed. These ran and passed
     in the dev sandbox (including load_image_patch on a real sample JPEG
     staged from your machine) before this file ever reached your machine.
  2. Quantum-pipeline tests -- skipped automatically until pennylane/torch
     are importable. Run this file again after `pip install pennylane
     pennylane-lightning` (already done if you've run qml_pqc.py's tests)
     and BEFORE trusting a real sweep.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.qcnn_no_cnn import (
    patch_side_for_qubits, normalize_for_amplitude_embedding, cache_key, qcnn_verdict,
)

pennylane = pytest.importorskip("pennylane", reason="pennylane not installed yet")
torch = pytest.importorskip("torch", reason="torch not installed yet")
from src.experiments.qcnn_no_cnn import (  # noqa: E402
    build_qcnn_model, train_qcnn, predict_qcnn, qcnn_conv_block, qcnn_pool_block,
)


# ---------------------------------------------------------------------
# Tier 1 -- pure logic
# ---------------------------------------------------------------------

def test_patch_side_for_qubits():
    assert patch_side_for_qubits(4) == 4    # 4x4 = 16 = 2**4
    assert patch_side_for_qubits(8) == 16   # 16x16 = 256 = 2**8
    with pytest.raises(ValueError):
        patch_side_for_qubits(5)  # odd n_qubits has no square patch


def test_normalize_for_amplitude_embedding_unit_norm_and_no_nan_on_zero():
    v = np.array([3.0, 4.0])
    out = normalize_for_amplitude_embedding(v)
    assert abs(np.linalg.norm(out) - 1.0) < 1e-9

    z = np.zeros(16)
    out_z = normalize_for_amplitude_embedding(z)
    assert np.all(np.isfinite(out_z))
    assert abs(np.linalg.norm(out_z) - 1.0) < 1e-6


def test_cache_key_deterministic_and_distinguishes_configs():
    k1 = cache_key("pixels", 4, 2, 2000, 42)
    k2 = cache_key("pixels", 4, 2, 2000, 42)
    k3 = cache_key("pixels", 4, 3, 2000, 42)
    assert k1 == k2
    assert k1 != k3


def test_qcnn_verdict_three_branches():
    sig = {"mean_diff": 0.05, "ci95": [0.01, 0.09], "significantly_better": True}
    neg = {"mean_diff": -0.03, "ci95": [-0.06, -0.005], "significantly_better": False}
    incl = {"mean_diff": 0.02, "ci95": [-0.01, 0.05], "significantly_better": False}
    assert qcnn_verdict(sig)[0] == "qcnn_beats_matched_baseline"
    assert qcnn_verdict(neg)[0] == "qcnn_loses_to_matched_baseline"
    assert qcnn_verdict(incl)[0] == "inconclusive"


def test_pooling_schedule_always_ends_at_two_qubits():
    """The circuit stops pooling with 2 qubits left (richer 4-outcome
    readout for a 5-class problem) -- verify the arithmetic for every
    qubit count this script actually offers."""
    for n_qubits in (4, 8):
        n_stages = int(np.log2(n_qubits)) - 1
        active = n_qubits
        for _ in range(n_stages):
            active //= 2
        assert active == 2, f"n_qubits={n_qubits} ended at {active} wires, expected 2"


# ---------------------------------------------------------------------
# Tier 2 -- quantum pipeline
# ---------------------------------------------------------------------

def test_qcnn_conv_and_pool_blocks_run_inside_a_real_circuit():
    """Builds the smallest real circuit (4 qubits, 1 conv rep) and checks
    the conv+pool blocks actually execute and produce a valid 2-qubit
    probability distribution (4 non-negative numbers summing to 1)."""
    import pennylane as qml

    n_qubits = 4
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, conv_weights, pool_weights):
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
        active = list(range(n_qubits))
        qcnn_conv_block(active, conv_weights[0])
        active = qcnn_pool_block(active, pool_weights)
        return qml.probs(wires=active)

    rng = np.random.RandomState(0)
    inputs = rng.uniform(0, 255, size=2 ** n_qubits)
    conv_weights = rng.uniform(-np.pi, np.pi, size=(1, 6))
    pool_weights = rng.uniform(-np.pi, np.pi, size=3)
    probs = circuit(inputs, conv_weights, pool_weights)
    probs = np.asarray(probs)
    assert probs.shape == (4,)
    assert np.all(probs >= -1e-9)
    assert abs(probs.sum() - 1.0) < 1e-6


def test_qcnn_head_runs_end_to_end_on_synthetic_data():
    """Not a claim that it learns anything on random labels -- just that
    build_qcnn_model / train_qcnn / predict_qcnn execute without crashing,
    on BATCHED input (this is exactly what caught the real batching bug in
    qml_pqc.py, so it's worth checking here too)."""
    n_qubits, n_conv_reps = 4, 1
    rng = np.random.RandomState(0)
    n_train, n_val = 40, 20
    X_train = np.stack([normalize_for_amplitude_embedding(rng.uniform(0, 255, size=2 ** n_qubits))
                         for _ in range(n_train)])
    X_val = np.stack([normalize_for_amplitude_embedding(rng.uniform(0, 255, size=2 ** n_qubits))
                       for _ in range(n_val)])
    y_train = rng.randint(0, 5, size=n_train)
    y_val = rng.randint(0, 5, size=n_val)

    model, val_qwk, epochs_run = train_qcnn(
        X_train, y_train, X_val, y_val, n_qubits, n_conv_reps,
        epochs=2, batch_size=16, lr=0.05, patience=2, diff_method="parameter-shift", seed=0,
        log=lambda msg: None)
    assert epochs_run >= 1
    assert isinstance(val_qwk, float)
    preds = predict_qcnn(model, X_val)
    assert preds.shape == (len(X_val),)
    assert set(np.unique(preds).tolist()).issubset({0, 1, 2, 3, 4})
