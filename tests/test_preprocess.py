"""Acceptance Test 6.1 (MASTER_PLAN.md Part 6) — run AFTER build_cache.py
has processed the full manifest (not just a --limit smoke test)."""
import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = PROJECT_ROOT / "results" / "phase2_preprocess.json"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "manifest.csv"
CONTACT_SHEET = PROJECT_ROOT / "figures" / "sanity_crops.png"


@pytest.fixture(scope="module")
def summary():
    assert SUMMARY_PATH.exists(), "results/phase2_preprocess.json missing — run build_cache.py first"
    return json.loads(SUMMARY_PATH.read_text())


def test_every_manifest_image_cached_at_every_size(summary):
    manifest = pd.read_csv(MANIFEST_PATH)
    for size in summary["sizes"]:
        assert summary["per_size_file_counts"][str(size)] == len(manifest), (
            f"size {size}: expected {len(manifest)} cached files, got "
            f"{summary['per_size_file_counts'][str(size)]} — build_cache.py hasn't finished the full manifest yet"
        )


def test_no_processing_errors(summary):
    assert summary["errors"] == 0, f"{summary['errors']} images failed to process — see error_examples in the JSON"


def test_no_low_mean_pixel_crops(summary):
    """Acceptance Test 6.1, post-exclusion: assert zero low-mean crops among
    ACTIVE (non-excluded) images. The 8 known flat-black EyePACS captures
    (results/excluded_images.json, apply_exclusions.py) are documented,
    audited exclusions, not a relaxed threshold — every warning must trace
    to one of those 8 IDs, and no other image may ever appear here."""
    manifest = pd.read_csv(MANIFEST_PATH)
    excluded_ids = set()
    if "excluded" in manifest.columns:
        excluded_ids = set(manifest.loc[manifest["excluded"].fillna(False).astype(bool), "image_id"])

    warnings = summary["low_mean_pixel_warnings"]
    unexplained = [w for w in warnings if w["image_id"] not in excluded_ids]
    assert not unexplained, (
        f"{len(unexplained)} low-mean-pixel crops are NOT in the documented exclusion list — "
        f"these are new/unexplained, investigate before excluding: {unexplained}"
    )

    flagged_ids = {w["image_id"] for w in warnings}
    assert flagged_ids == excluded_ids, (
        f"mismatch between low-mean warnings ({flagged_ids}) and the documented exclusion list "
        f"({excluded_ids}) — every documented exclusion should have caused this warning originally"
    )


def test_exactly_8_excluded_rows():
    """Guards against silent scope creep: if this ever fails, someone added
    (or removed) an exclusion without deliberately updating apply_exclusions.py
    and this test together."""
    manifest = pd.read_csv(MANIFEST_PATH)
    assert "excluded" in manifest.columns, "manifest has no excluded column — run apply_exclusions.py"
    n_excluded = int(manifest["excluded"].fillna(False).astype(bool).sum())
    assert n_excluded == 8, f"expected exactly 8 excluded rows, found {n_excluded}"
    assert manifest.loc[manifest["excluded"].fillna(False).astype(bool), "excluded_reason"].notna().all(), (
        "every excluded row must carry a non-null excluded_reason"
    )


def test_contact_sheet_exists():
    assert CONTACT_SHEET.exists(), (
        "figures/sanity_crops.png missing. Per MASTER_PLAN.md Part 6, a human must look at this "
        "personally before Phase 3 — this check only confirms the file exists, not that it looks right."
    )


def test_cache_size_and_timing_reported(summary):
    assert summary["cache_size_gb"] > 0
    assert summary["median_processing_time_sec_per_image"] is not None
