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
    """Acceptance Test 6.1: assert no processed image has mean pixel value < 5."""
    assert summary["n_low_mean_pixel_warnings"] == 0, (
        f"{summary['n_low_mean_pixel_warnings']} crops have mean pixel value < 5 "
        f"(likely blank/black crops from a circle-detection failure) — inspect low_mean_pixel_warnings in the JSON"
    )


def test_contact_sheet_exists():
    assert CONTACT_SHEET.exists(), (
        "figures/sanity_crops.png missing. Per MASTER_PLAN.md Part 6, a human must look at this "
        "personally before Phase 3 — this check only confirms the file exists, not that it looks right."
    )


def test_cache_size_and_timing_reported(summary):
    assert summary["cache_size_gb"] > 0
    assert summary["median_processing_time_sec_per_image"] is not None
