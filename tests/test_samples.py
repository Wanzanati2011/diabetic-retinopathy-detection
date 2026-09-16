"""
Tests for app/data/samples.py (AGENT_EXECUTION_PLAN.md Task B4, T-6).
"""
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data import samples as S


def test_load_samples_missing_file_returns_none(tmp_path):
    result = S.load_samples(path=tmp_path / "does_not_exist.json", samples_dir=tmp_path)
    assert result is None


def test_load_samples_missing_image_file_is_dropped(tmp_path):
    (tmp_path / "samples.json").write_text(json.dumps({
        "routine": {"file": "does_not_exist.jpg", "image_id": "x", "true_grade": 0,
                    "expected_outcome": "ROUTINE"},
    }))
    result = S.load_samples(path=tmp_path / "samples.json", samples_dir=tmp_path)
    assert result is None  # the only entry's image is missing -> empty -> None


def test_load_samples_resolves_paths(tmp_path):
    (tmp_path / "img1.jpg").write_bytes(b"fake jpeg bytes")
    (tmp_path / "samples.json").write_text(json.dumps({
        "routine": {"file": "img1.jpg", "image_id": "x", "true_grade": 0,
                    "expected_outcome": "ROUTINE"},
    }))
    result = S.load_samples(path=tmp_path / "samples.json", samples_dir=tmp_path)
    assert result is not None
    assert "routine" in result
    assert Path(result["routine"]["path"]).exists()


def test_load_samples_pair_slot_needs_both_images(tmp_path):
    (tmp_path / "left.jpg").write_bytes(b"x")
    (tmp_path / "samples.json").write_text(json.dumps({
        "pair": {"left_file": "left.jpg", "right_file": "right.jpg", "patient_id": "p1"},
    }))
    result = S.load_samples(path=tmp_path / "samples.json", samples_dir=tmp_path)
    assert result is None  # right.jpg missing


def test_ground_truth_line_match():
    assert S.ground_truth_line(2, 2) == "Ground truth: Grade 2 · match"


def test_ground_truth_line_off_by_n():
    assert S.ground_truth_line(1, 3) == "Ground truth: Grade 1 · off by 2"
    assert S.ground_truth_line(3, 1) == "Ground truth: Grade 3 · off by 2"


@pytest.mark.skipif(not S.SAMPLES_JSON_PATH.exists(),
                     reason="app/samples_local/samples.json not present (local-only, R6/H1)")
def test_t6_real_curated_samples_match_expected_outcome():
    """T-6: every entry in app/samples_local/samples.json yields its
    expected_outcome when actually run through the app's pipeline."""
    import torch
    from PIL import Image

    from app.core import decision as D
    from app.core.inference import to_model_input
    from app.core.model import MODEL, CALIBRATION_ACTIVE, THRESHOLDS

    data = S.load_samples()
    assert data is not None

    def grade(path):
        with Image.open(path) as im, torch.no_grad():
            x, _ = to_model_input(im.convert("RGB"))
            logits = MODEL(x)[0].numpy()
        probs = D.calibrated_probs(logits, THRESHOLDS) if CALIBRATION_ACTIVE else D.raw_probs(logits)
        return D.classify_outcome(probs, THRESHOLDS, CALIBRATION_ACTIVE)

    for slot, entry in data.items():
        if slot == "pair":
            left_outcome = grade(entry["left_path"])
            right_outcome = grade(entry["right_path"])
            assert left_outcome.value == entry["left_expected_outcome"], slot
            assert right_outcome.value == entry["right_expected_outcome"], slot
        else:
            outcome = grade(entry["path"])
            assert outcome.value == entry["expected_outcome"], slot
