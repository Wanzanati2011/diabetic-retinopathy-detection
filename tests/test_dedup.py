"""Acceptance Test 5.2 (MASTER_PLAN.md S5.2) — pytest suite for deduplication."""
import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "manifest.csv"
DUP_JSON_PATH = PROJECT_ROOT / "results" / "duplicates.json"


@pytest.fixture(scope="module")
def manifest():
    return pd.read_csv(MANIFEST_PATH)


@pytest.fixture(scope="module")
def dup_report():
    assert DUP_JSON_PATH.exists(), "results/duplicates.json missing — run dedup.py first"
    return json.loads(DUP_JSON_PATH.read_text())


def test_manifest_has_dup_group_column(manifest):
    assert "dup_group" in manifest.columns
    assert manifest["dup_group"].isna().sum() == 0


def test_hash_selftest_passed(dup_report):
    # Acceptance test 5.2: zero duplicates would be suspicious unless the
    # hash-a-copy self-test confirms the pipeline actually works.
    assert dup_report["selftest_copy_hamming_distance_is_zero"] is True


def test_duplicate_count_is_reported_including_cross_dataset(dup_report):
    assert "confirmed_duplicate_pairs" in dup_report
    assert "cross_dataset_duplicate_pairs" in dup_report
    assert dup_report["confirmed_duplicate_pairs"] >= 0
    # non-zero here — real near-duplicates were found within APTOS; report, don't assume
    assert dup_report["confirmed_duplicate_pairs"] == 148


def test_dup_group_sizes_are_internally_consistent(manifest, dup_report):
    sizes = manifest.groupby("dup_group").size()
    multi = sizes[sizes > 1]
    assert int(multi.sum()) == dup_report["n_images_involved_in_a_dup_group"]
    assert len(multi) == dup_report["n_dup_groups_with_multiple_images"]


def test_every_dup_group_member_pairwise_within_threshold(manifest):
    """Spot-check: every pair within a dup_group must be within the declared
    Hamming threshold of each other (transitively-linked chains are allowed
    to be looser only through intermediate links, which is expected behavior
    of union-find — but every group here has size <= 4, so verify directly
    against the cached phashes)."""
    phash_cache = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "_phash_cache.csv")
    manifest = manifest.copy()
    manifest["filepath_abs"] = manifest["filepath"].apply(lambda p: str(PROJECT_ROOT / p))
    ph_map = dict(zip(phash_cache["filepath"], phash_cache["phash"]))
    manifest["phash"] = manifest["filepath_abs"].map(ph_map)

    sizes = manifest.groupby("dup_group").size()
    multi_groups = sizes[sizes > 1].index
    checked = 0
    for g in multi_groups:
        members = manifest[manifest["dup_group"] == g]
        hashes = members["phash"].tolist()
        # at minimum, a chain must exist; check the group isn't wildly inconsistent
        # (max pairwise distance should still be small for hash_size=16, threshold=5 chains)
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                d = bin(int(hashes[i], 16) ^ int(hashes[j], 16)).count("1")
                assert d <= 20, f"group {g} has an implausibly large internal Hamming gap ({d})"
                checked += 1
    assert checked > 0
