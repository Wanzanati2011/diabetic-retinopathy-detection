"""Acceptance Test 5.3 (MASTER_PLAN.md S5.3) — the permanent pytest suite.
All seven tests below must pass before Phase 1 is considered done.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "manifest.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

GRADE_TOLERANCE = 0.03  # 3 percentage points


@pytest.fixture(scope="module")
def manifest():
    df = pd.read_csv(MANIFEST_PATH)
    return df.set_index("image_id", drop=False)


@pytest.fixture(scope="module")
def active_manifest(manifest):
    """Rows NOT marked excluded — the pool splits.py actually assigns folds
    over. See apply_exclusions.py."""
    if "excluded" not in manifest.columns:
        return manifest
    return manifest[~manifest["excluded"].fillna(False).astype(bool)]


@pytest.fixture(scope="module")
def p1():
    return json.loads((SPLITS_DIR / "p1.json").read_text())


@pytest.fixture(scope="module")
def p2():
    return json.loads((SPLITS_DIR / "p2.json").read_text())


@pytest.fixture(scope="module")
def p3a():
    return json.loads((SPLITS_DIR / "p3_eyepacs_to_aptos.json").read_text())


@pytest.fixture(scope="module")
def p3b():
    return json.loads((SPLITS_DIR / "p3_aptos_to_eyepacs.json").read_text())


def _fold_sets(split_map, manifest, key_col):
    """Return {fold: set(of key_col values present in that fold)}."""
    out = {"train": set(), "val": set(), "test": set()}
    for image_id, rec in split_map.items():
        fold = rec["fold"]
        out[fold].add(manifest.loc[image_id, key_col])
    return out


def test_p2_no_patient_overlap(p2, manifest):
    sets = _fold_sets(p2, manifest, "patient_id")
    assert sets["train"] & sets["val"] == set()
    assert sets["train"] & sets["test"] == set()
    assert sets["val"] & sets["test"] == set()


def test_p2_no_dupgroup_overlap(p2, manifest):
    sets = _fold_sets(p2, manifest, "dup_group")
    assert sets["train"] & sets["val"] == set()
    assert sets["train"] & sets["test"] == set()
    assert sets["val"] & sets["test"] == set()


def test_p3_no_dataset_overlap(p3a, p3b, manifest):
    # direction A: train/val must be 100% eyepacs, test must be 100% aptos
    for image_id, rec in p3a.items():
        ds = manifest.loc[image_id, "dataset"]
        if rec["fold"] in ("train", "val"):
            assert ds == "eyepacs", f"{image_id} in {rec['fold']} but dataset={ds}"
        else:
            assert ds == "aptos", f"{image_id} in test but dataset={ds}"
    # direction B: reversed
    for image_id, rec in p3b.items():
        ds = manifest.loc[image_id, "dataset"]
        if rec["fold"] in ("train", "val"):
            assert ds == "aptos", f"{image_id} in {rec['fold']} but dataset={ds}"
        else:
            assert ds == "eyepacs", f"{image_id} in test but dataset={ds}"


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_splits_deterministic(tmp_path):
    """Re-running splits.py with the same seed must produce byte-identical JSON."""
    before = {f.name: _file_hash(f) for f in SPLITS_DIR.glob("*.json")}
    assert before, "no split files found — run splits.py first"

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "data" / "splits.py"),
         "--project-root", str(PROJECT_ROOT),
         "--out-dir", str(tmp_path.relative_to(PROJECT_ROOT)) if tmp_path.is_relative_to(PROJECT_ROOT) else str(tmp_path)],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"splits.py failed:\n{result.stdout}\n{result.stderr}"

    after = {f.name: _file_hash(f) for f in tmp_path.glob("*.json")}
    assert set(before.keys()) == set(after.keys())
    for name in before:
        assert before[name] == after[name], f"{name} differs between runs — split is not deterministic"


def test_all_images_assigned(active_manifest, p1, p2, p3a, p3b):
    all_ids = set(active_manifest["image_id"])
    for name, split_map in [("p1", p1), ("p2", p2), ("p3a", p3a), ("p3b", p3b)]:
        split_ids = set(split_map.keys())
        assert split_ids == all_ids, f"{name}: {len(all_ids - split_ids)} missing, {len(split_ids - all_ids)} extra"
        folds = [rec["fold"] for rec in split_map.values()]
        assert all(f in ("train", "val", "test") for f in folds)


def test_grade_distribution_similar(active_manifest, p1, p2):
    for name, split_map in [("p1", p1), ("p2", p2)]:
        df = active_manifest.copy()
        df["fold"] = df["image_id"].map(lambda i: split_map[i]["fold"])
        tab = pd.crosstab(df["fold"], df["grade"], normalize="index")
        max_diff = (tab.max() - tab.min()).max()
        assert max_diff <= GRADE_TOLERANCE, f"{name}: grade distribution differs by {max_diff:.4f} across folds (tolerance {GRADE_TOLERANCE})"


def test_p1_partner_flag_present(p1, manifest):
    n_checked = 0
    for image_id, rec in p1.items():
        assert "partner_in_train" in rec
        row = manifest.loc[image_id]
        if rec["fold"] == "test" and row["dataset"] == "eyepacs" and pd.notna(row["partner_id"]):
            assert rec["partner_in_train"] in (True, False), (
                f"{image_id}: P1 test image with a partner must have partner_in_train recorded as bool"
            )
            n_checked += 1
    assert n_checked > 0, "no eyepacs P1 test images with partner_in_train recorded — suspicious"


def test_excluded_rows_appear_in_no_split(manifest, p1, p2, p3a, p3b):
    if "excluded" not in manifest.columns:
        pytest.skip("no excluded column yet")
    excluded_ids = set(manifest.loc[manifest["excluded"].fillna(False).astype(bool), "image_id"])
    assert excluded_ids, "expected at least one excluded row"
    for name, split_map in [("p1", p1), ("p2", p2), ("p3a", p3a), ("p3b", p3b)]:
        overlap = excluded_ids & set(split_map.keys())
        assert not overlap, f"{name} contains excluded image_ids: {overlap}"
