"""
Acceptance test for src/experiments/claim3_both_eyes.py (Phase 5, Claim 3).

Tests only the PURE, no-model-fitting helpers -- build_eyepacs_pairs(),
assign_p2_fold(), referable_sens_spec_at_90_sens() -- with small synthetic
inputs. No npz/features/GPU needed and no heavy sklearn fit, so this is
safe to run anywhere. The actual LogisticRegression/Ridge fits in main()
must be run on real hardware per this project's standing convention (same
handoff pattern as claim2_protocols.py / claim2b_partner_ablation.py).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.claim3_both_eyes import (
    build_eyepacs_pairs, assign_p2_fold, referable_sens_spec_at_90_sens,
)


def _fake_manifest():
    # 3 EyePACS patients with both eyes, 1 EyePACS patient missing right eye
    # (should be dropped), 1 APTOS row (should be excluded entirely).
    rows = [
        {"image_id": "eyepacs_1_left", "patient_id": "eyepacs_1", "eye": "left", "grade": 0, "dataset": "eyepacs"},
        {"image_id": "eyepacs_1_right", "patient_id": "eyepacs_1", "eye": "right", "grade": 2, "dataset": "eyepacs"},
        {"image_id": "eyepacs_2_left", "patient_id": "eyepacs_2", "eye": "left", "grade": 3, "dataset": "eyepacs"},
        {"image_id": "eyepacs_2_right", "patient_id": "eyepacs_2", "eye": "right", "grade": 1, "dataset": "eyepacs"},
        {"image_id": "eyepacs_3_left", "patient_id": "eyepacs_3", "eye": "left", "grade": 4, "dataset": "eyepacs"},
        {"image_id": "eyepacs_3_right", "patient_id": "eyepacs_3", "eye": "right", "grade": 4, "dataset": "eyepacs"},
        {"image_id": "eyepacs_4_left", "patient_id": "eyepacs_4", "eye": "left", "grade": 0, "dataset": "eyepacs"},
        # eyepacs_4's right eye is missing (soft-excluded or never existed) -- must be dropped.
        {"image_id": "aptos_9", "patient_id": "aptos_9", "eye": "unknown", "grade": 2, "dataset": "aptos"},
    ]
    return pd.DataFrame(rows)


def test_missing_partner_eye_is_dropped_not_fabricated():
    pairs = build_eyepacs_pairs(_fake_manifest())
    assert "eyepacs_4" not in set(pairs["patient_id"])
    assert len(pairs) == 3


def test_aptos_rows_never_appear():
    pairs = build_eyepacs_pairs(_fake_manifest())
    assert not any(pid.startswith("aptos") for pid in pairs["patient_id"])


def test_patient_grade_is_max_of_both_eyes():
    pairs = build_eyepacs_pairs(_fake_manifest()).set_index("patient_id")
    assert pairs.loc["eyepacs_1", "patient_grade"] == 2  # max(0, 2)
    assert pairs.loc["eyepacs_2", "patient_grade"] == 3  # max(3, 1)
    assert pairs.loc["eyepacs_3", "patient_grade"] == 4  # max(4, 4)


def test_assign_p2_fold_matching_folds():
    pairs = build_eyepacs_pairs(_fake_manifest())
    split_map = {
        "eyepacs_1_left": {"fold": "train"}, "eyepacs_1_right": {"fold": "train"},
        "eyepacs_2_left": {"fold": "test"}, "eyepacs_2_right": {"fold": "test"},
        "eyepacs_3_left": {"fold": "train"}, "eyepacs_3_right": {"fold": "train"},
    }
    out, mismatches = assign_p2_fold(pairs, split_map)
    assert mismatches == 0
    folds = dict(zip(out["patient_id"], out["fold"]))
    assert folds["eyepacs_1"] == "train"
    assert folds["eyepacs_2"] == "test"
    assert folds["eyepacs_3"] == "train"


def test_assign_p2_fold_catches_a_real_mismatch():
    """If P2 ever put a patient's two eyes on different sides (a real bug,
    since P2 is supposed to guarantee this never happens), this must be
    counted and the patient dropped -- not silently assigned to one side."""
    pairs = build_eyepacs_pairs(_fake_manifest())
    split_map = {
        "eyepacs_1_left": {"fold": "train"}, "eyepacs_1_right": {"fold": "test"},  # bug: mismatch
        "eyepacs_2_left": {"fold": "test"}, "eyepacs_2_right": {"fold": "test"},
        "eyepacs_3_left": {"fold": "train"}, "eyepacs_3_right": {"fold": "train"},
    }
    out, mismatches = assign_p2_fold(pairs, split_map)
    assert mismatches == 1
    row = out[out["patient_id"] == "eyepacs_1"].iloc[0]
    # pandas normalizes a bare None to NaN in some versions/paths when a column
    # mixes None with strings (observed: sandbox pandas 2.3.3 keeps None, the
    # real machine's pandas turned it into float NaN for the same code) -- the
    # actual invariant we care about is "this row is marked missing/dropped",
    # which pd.isna() checks correctly regardless of which sentinel pandas
    # chose to store. `is None` was over-specified to an implementation detail.
    assert pd.isna(row["fold"])


def test_referable_sens_spec_perfect_predictor():
    y_true = np.array([0, 1, 2, 3, 4, 0, 2, 4])
    y_pred = y_true.copy()  # perfect ordinal predictor
    result = referable_sens_spec_at_90_sens(y_true, y_pred)
    assert result["sensitivity"] is not None
    assert result["sensitivity"] >= 0.90
    assert result["specificity"] == 1.0  # perfect predictor -> no false positives at any true-positive-preserving cut


def test_referable_sens_spec_degenerate_all_same_class():
    y_true = np.array([0, 0, 0, 1, 0, 1])  # all grade<2 -- rDR label is all-negative
    y_pred = np.array([0, 1, 0, 1, 0, 0])
    result = referable_sens_spec_at_90_sens(y_true, y_pred)
    assert result["sensitivity"] is None
    assert "degenerate" in result["note"]
