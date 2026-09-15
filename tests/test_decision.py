"""
Tests for app/core/decision.py (AGENT_EXECUTION_PLAN.md Tasks A1-A3, T-1
through T-5). All tests marked "(no data)" reproduce Part 1.4's reference
numbers straight from the committed test-fold CSV -- no GPU, no checkpoint,
runs everywhere.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core import decision as D

TEST_CSV = (PROJECT_ROOT / "results" /
            "finetune_app_converged_p2_class_balanced_seed42_test_predictions.csv")
THRESHOLDS_JSON = PROJECT_ROOT / "app" / "release" / "thresholds.json"


def _load_test_logits():
    if not TEST_CSV.exists():
        pytest.skip(f"{TEST_CSV} not present")
    df = pd.read_csv(TEST_CSV)
    logits = df[[f"logit_{k}" for k in range(5)]].to_numpy()
    y_true = df["true_grade"].to_numpy()
    return df, logits, y_true


def _thresholds():
    th = D.load_thresholds(THRESHOLDS_JSON)
    if th is None:
        pytest.skip(f"{THRESHOLDS_JSON} not present")
    return th


# ---------------------------------------------------------------------
# T-1 (no data): referral confusion matches Part 1.4 exactly
# ---------------------------------------------------------------------
def test_t1_referral_confusion_matches_reference():
    df, logits, y_true = _load_test_logits()
    th = _thresholds()
    probs = D.calibrated_probs(logits, th)
    referred = D.is_referred(probs, th)
    referable_true = y_true >= 2

    tp = int(((referred) & (referable_true)).sum())
    fn = int(((~referred) & (referable_true)).sum())
    tn = int(((~referred) & (~referable_true)).sum())
    fp = int(((referred) & (~referable_true)).sum())

    assert (tp, fn, tn, fp) == (1116, 130, 3186, 1382)


# ---------------------------------------------------------------------
# T-2 (no data): uncertainty gate kept/rejected counts + selective QWK
# ---------------------------------------------------------------------
def test_t2_uncertainty_gate_matches_reference():
    from sklearn.metrics import cohen_kappa_score

    df, logits, y_true = _load_test_logits()
    th = _thresholds()
    probs = D.calibrated_probs(logits, th)
    uncertain = D.is_uncertain(probs, th)
    kept = ~uncertain

    assert int(kept.sum()) == 4682
    assert int(uncertain.sum()) == 1132

    pred = probs.argmax(axis=-1)
    qwk = cohen_kappa_score(y_true[kept], pred[kept], weights="quadratic")
    assert qwk == pytest.approx(0.77407, abs=1e-4)


# ---------------------------------------------------------------------
# T-3 (no data): ECE of calibrated probs, 15 equal-width bins
# ---------------------------------------------------------------------
def _ece(probs, y_true, n_bins=15):
    """Reproduces src/experiments/calibrate.py's calibration_error() ECE,
    bin edge convention documented in docs/verification/V2_definitions.md:
    bin 0 is closed on both ends, bins 1..14 are (lo, hi]."""
    conf = probs.max(axis=-1)
    pred = probs.argmax(axis=-1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        acc = float(correct[m].mean())
        avg_conf = float(conf[m].mean())
        total += (cnt / n) * abs(acc - avg_conf)
    return total


def test_t3_ece_matches_reference():
    df, logits, y_true = _load_test_logits()
    th = _thresholds()
    probs = D.calibrated_probs(logits, th)
    ece = _ece(probs, y_true)
    assert ece == pytest.approx(0.0285, abs=1e-3)


# ---------------------------------------------------------------------
# T-4 (no data): outcome precedence
# ---------------------------------------------------------------------
def test_t4_ungradable_beats_everything():
    th = _thresholds()
    probs = np.array([0.01, 0.01, 0.01, 0.01, 0.96])  # would be REFER otherwise
    outcome = D.classify_outcome(probs, th, calibration_active=True, is_ungradable=True)
    assert outcome is D.Outcome.UNGRADABLE


def test_t4_low_confidence_beats_referral():
    th = _thresholds()
    # max prob below tau, but referral_score also above referral_threshold
    probs = np.array([0.30, 0.10, 0.30, 0.20, 0.10])
    assert probs.max() < th.reject_tau
    assert D.referral_score(probs) >= th.referral_threshold
    outcome = D.classify_outcome(probs, th, calibration_active=True)
    assert outcome is D.Outcome.UNCERTAIN


def test_t4_referral_threshold_inequality_direction():
    th = _thresholds()
    # craft a probability vector whose referral_score is exactly at the
    # threshold and whose max prob clears tau (so REFER, not UNCERTAIN,
    # is the state under test)
    tail = th.referral_threshold
    p0 = 1.0 - tail
    probs = np.array([p0, 0.0, tail, 0.0, 0.0])
    if probs.max() < th.reject_tau:
        pytest.skip("crafted vector falls in the UNCERTAIN band for this tau; "
                    "direction is still covered by test_t1's exact-count check")
    assert D.referral_score(probs) == pytest.approx(tail)
    assert bool(D.is_referred(probs, th)) is True  # >= is inclusive at the boundary
    outcome = D.classify_outcome(probs, th, calibration_active=True)
    assert outcome is D.Outcome.REFER


def test_t4_routine_when_nothing_else_fires():
    th = _thresholds()
    probs = np.array([0.97, 0.01, 0.01, 0.005, 0.005])
    assert probs.max() >= th.reject_tau
    assert not D.is_referred(probs, th)
    outcome = D.classify_outcome(probs, th, calibration_active=True)
    assert outcome is D.Outcome.ROUTINE


# ---------------------------------------------------------------------
# T-5: integrity checks disable calibration and flag the fallback banner
# ---------------------------------------------------------------------
def test_t5_wrong_checkpoint_hash_disables_calibration(tmp_path):
    ckpt = tmp_path / "best_model.pt"
    ckpt.write_bytes(b"not the real checkpoint")
    sha_path = tmp_path / "checkpoint.sha256"
    sha_path.write_text("0" * 64)
    active, th, reason = D.check_integrity(
        checkpoint_path=ckpt, sha_path=sha_path, thresholds_path=THRESHOLDS_JSON)
    assert active is False
    assert reason == "checkpoint_hash_mismatch"


def test_t5_missing_thresholds_file_disables_calibration(tmp_path):
    active, th, reason = D.check_integrity(
        checkpoint_path=D.CHECKPOINT_PATH, sha_path=D.CHECKPOINT_SHA_PATH,
        thresholds_path=tmp_path / "does_not_exist.json")
    assert active is False
    assert th is None
    assert reason == "missing_thresholds"


def test_t5_verdict_not_pass_disables_calibration(tmp_path):
    import json
    th = _thresholds()
    bad = dict(th.raw)
    bad["acceptance_test_11_1_verdict"] = "fail_confidence_uninformative"
    bad_path = tmp_path / "thresholds.json"
    bad_path.write_text(json.dumps(bad))

    # matching checkpoint hash so only the verdict trips the check
    if not D.CHECKPOINT_PATH.exists():
        pytest.skip("app/release/best_model.pt not present")
    sha_path = tmp_path / "checkpoint.sha256"
    sha_path.write_text(D.checkpoint_sha256(D.CHECKPOINT_PATH))

    active, th2, reason = D.check_integrity(
        checkpoint_path=D.CHECKPOINT_PATH, sha_path=sha_path, thresholds_path=bad_path)
    assert active is False
    assert reason == "acceptance_test_not_pass"


def test_t5_all_three_share_the_same_banner_text():
    # every disabled reason renders the same fallback banner in the UI
    assert "Calibration file does not match this model" in D.INACTIVE_BANNER
