"""
Acceptance test for src/experiments/summary_comparison.py (Part 8.4).

build_summary() is a pure function (dict-in, dict-out, no I/O, no modeling) so
it's tested here with small synthetic dicts shaped like the real result JSONs
-- no npz/features/GPU needed. This checks the AGGREGATION logic (right keys
pulled, right arithmetic) is correct, not the upstream numbers themselves
(those are covered by test_features.py and each claim script's own sanity
checks).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.summary_comparison import build_summary


def _fake_inputs():
    inter_eye = {"label_only_baseline": {"lookup_qwk": 0.838}}
    claim2 = {
        "protocols": {
            "P1": {"reference_qwk_seed42": 0.54, "ordinal_qwk": 0.60, "patient_bootstrap_ci95": [0.50, 0.58]},
            "P2": {"reference_qwk_seed42": 0.56, "ordinal_qwk": 0.62, "patient_bootstrap_ci95": [0.52, 0.60]},
        },
        "p1_vs_p2_paired_permutation_test": {"p_value_two_sided": 0.125},
    }
    claim2b = {
        "multinomial": {
            "raw_qwk_present": 0.49,
            "raw_qwk_absent": 0.44,
            "permutation_test": {"p_value_two_sided": 0.185},
        },
        "ordinal": {
            "raw_qwk_present": 0.55,
            "raw_qwk_absent": 0.49,
            "permutation_test": {"p_value_two_sided": 0.0505},
        },
    }
    return inter_eye, claim2, claim2b


def test_seven_bars_in_expected_order():
    inter_eye, claim2, claim2b = _fake_inputs()
    summary = build_summary(inter_eye, claim2, claim2b)
    assert len(summary["bars"]) == 7
    groups = [b["group"] for b in summary["bars"]]
    assert groups == ["baseline", "P1", "P1", "P2", "P2", "ablation", "ablation"]


def test_bar_values_match_inputs_exactly():
    inter_eye, claim2, claim2b = _fake_inputs()
    summary = build_summary(inter_eye, claim2, claim2b)
    by_label = {b["label"]: b["qwk"] for b in summary["bars"]}
    assert by_label["Label-only lookup\n(no pixels)"] == 0.838
    assert by_label["P1 image-level\n(multinomial)"] == 0.54
    assert by_label["P1 image-level\n(ordinal)"] == 0.60
    assert by_label["P2 patient-level\n(multinomial)"] == 0.56
    assert by_label["P2 patient-level\n(ordinal)"] == 0.62
    assert by_label["Partner present\n(multinomial)"] == 0.49
    assert by_label["Partner absent\n(multinomial)"] == 0.44


def test_diffs_and_pvalues_computed_correctly():
    inter_eye, claim2, claim2b = _fake_inputs()
    summary = build_summary(inter_eye, claim2, claim2b)
    assert abs(summary["p1_vs_p2_diff"] - (0.54 - 0.56)) < 1e-9
    assert summary["p1_vs_p2_p_value"] == 0.125
    assert abs(summary["ablation_diff_multinomial"] - (0.49 - 0.44)) < 1e-9
    assert abs(summary["ablation_diff_ordinal"] - (0.55 - 0.49)) < 1e-9
    assert summary["ablation_p_value_multinomial"] == 0.185
    assert summary["ablation_p_value_ordinal"] == 0.0505


def test_honest_finding_mentions_key_numbers_and_is_a_string():
    inter_eye, claim2, claim2b = _fake_inputs()
    summary = build_summary(inter_eye, claim2, claim2b)
    assert isinstance(summary["honest_finding"], str)
    assert "0.838" in summary["honest_finding"]
    assert "fine-tuning" in summary["honest_finding"]  # flags this as an open question, not a final answer


def test_p1_beats_p2_case_flips_sign_correctly():
    """Sanity: if P1 (hypothetically) DID beat P2, the diff should be positive,
    not silently clipped/absolute-valued anywhere in the aggregation."""
    inter_eye, claim2, claim2b = _fake_inputs()
    claim2["protocols"]["P1"]["reference_qwk_seed42"] = 0.70  # now P1 > P2
    summary = build_summary(inter_eye, claim2, claim2b)
    assert summary["p1_vs_p2_diff"] > 0
