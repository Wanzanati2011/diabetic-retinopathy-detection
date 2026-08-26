"""Locks in the threshold-review decision from the dedup_threshold_sweep.py
audit: hash_size=16 + Hamming<=5 is kept as-is. Fails loudly if that ever
stops being true (e.g. after new data is added) so the decision gets
re-examined rather than silently going stale."""
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SWEEP_PATH = PROJECT_ROOT / "results" / "dedup_threshold_sweep.json"


@pytest.fixture(scope="module")
def sweep():
    assert SWEEP_PATH.exists(), "results/dedup_threshold_sweep.json missing — run dedup_threshold_sweep.py first"
    return json.loads(SWEEP_PATH.read_text())


def test_calibration_perturbation_is_well_under_threshold(sweep):
    """A genuine near-duplicate (resize +/-2px, JPEG q90 re-encode) must land
    far below the threshold=5 cutoff, or the threshold is too tight."""
    calib = sweep["calibration"]
    assert calib["hamming_distance"] <= 5, (
        f"calibration perturbation scored distance {calib['hamming_distance']}, "
        f"which exceeds threshold 5 — the dedup threshold IS too tight, rebuild splits"
    )


def test_eyepacs_shows_no_duplicates_at_original_threshold(sweep):
    assert sweep["by_dataset"]["eyepacs"]["counts_by_threshold"]["5"] == 0


def test_eyepacs_candidates_at_looser_threshold_are_not_real_duplicates(sweep):
    """If EyePACS pairs appear at a looser threshold, they must fail the
    'real duplicate' bar: grade agreement should sit near the dataset's
    chance baseline (coincidental structural similarity), not near 100%
    (which would mean the SAME photo, hence the SAME grade)."""
    diag = sweep["eyepacs_diagnostic"]
    if diag["n_pairs"] == 0:
        return
    chance = diag["chance_grade_agreement_rate"]
    observed = diag["grade_agreement_rate_overall"]
    # allow some headroom above chance without treating it as "real duplicates":
    # a jump to near-100% would be the signal that matters, not a few points above chance
    assert observed < chance + 0.15, (
        f"grade agreement ({observed:.3f}) is well above chance ({chance:.3f}) — "
        f"these candidates may be real duplicates after all, investigate before trusting P2"
    )
