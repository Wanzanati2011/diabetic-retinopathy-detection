"""Acceptance Test 5.1 (MASTER_PLAN.md S5.1) — pytest suite for the manifest."""
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "manifest.csv"

EXPECTED_EYEPACS_GRADE_COUNTS = {0: 25810, 1: 2443, 2: 5292, 3: 873, 4: 708}


@pytest.fixture(scope="module")
def manifest():
    assert MANIFEST_PATH.exists(), f"manifest not found at {MANIFEST_PATH} — run manifest.py first"
    return pd.read_csv(MANIFEST_PATH)


def test_row_count_matches_files_on_disk(manifest):
    ep_dir = PROJECT_ROOT / "resized train 15"
    ap_dir = PROJECT_ROOT / "resized train 19"
    n_ep_files = sum(1 for _ in ep_dir.glob("*.jpg"))
    n_ap_files = sum(1 for _ in ap_dir.glob("*.jpg"))
    assert len(manifest) == n_ep_files + n_ap_files == 38788


def test_no_nulls_in_patient_id_or_grade(manifest):
    assert manifest["patient_id"].isna().sum() == 0
    assert manifest["grade"].isna().sum() == 0


def test_grades_are_exactly_0_to_4(manifest):
    assert set(manifest["grade"].unique()) == {0, 1, 2, 3, 4}


def test_eyepacs_patient_count_and_two_images_each(manifest):
    ep = manifest[manifest["dataset"] == "eyepacs"]
    counts = ep.groupby("patient_id").size()
    assert counts.shape[0] == 17563, f"expected 17563 unique EyePACS patients, got {counts.shape[0]}"
    assert (counts == 2).all(), "every EyePACS patient must have exactly 2 images (left + right)"


def test_every_eyepacs_row_has_partner_id(manifest):
    ep = manifest[manifest["dataset"] == "eyepacs"]
    assert ep["partner_id"].isna().sum() == 0
    # partner_id must point at a real row, and the relationship must be symmetric
    ids = set(ep["image_id"])
    assert set(ep["partner_id"]).issubset(ids)
    lookup = dict(zip(ep["image_id"], ep["partner_id"]))
    for img_id, partner in lookup.items():
        assert lookup[partner] == img_id, f"partner_id not symmetric for {img_id}"


def test_eyepacs_grade_crosstab_matches_published_totals(manifest):
    ep = manifest[manifest["dataset"] == "eyepacs"]
    counts = ep["grade"].value_counts().to_dict()
    for grade, expected in EXPECTED_EYEPACS_GRADE_COUNTS.items():
        assert counts.get(grade) == expected, (
            f"grade {grade}: expected {expected}, got {counts.get(grade)}"
        )


def test_sha256_present_and_looks_valid(manifest):
    assert manifest["sha256"].isna().sum() == 0
    assert (manifest["sha256"].str.len() == 64).all()


def test_no_duplicate_image_ids(manifest):
    assert manifest["image_id"].is_unique
