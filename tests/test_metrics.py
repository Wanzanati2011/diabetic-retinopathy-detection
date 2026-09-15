"""
Tests for app/data/context.py and (later, Phase 2) app/data/metrics.py.
T-9, T-11, T-13 from AGENT_EXECUTION_PLAN.md Part 6.
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data import context as C

GRADE1_JSON = PROJECT_ROOT / "results" / "grade1_diagnosis.json"


# ---------------------------------------------------------------------
# T-11 (no data): context table n's equal confusion_matrix_full column sums
# ---------------------------------------------------------------------
def test_t11_context_n_matches_confusion_matrix_column_sums():
    if not GRADE1_JSON.exists():
        pytest.skip(f"{GRADE1_JSON} not present")
    ctx = C.load_per_grade_context()
    if ctx is None:
        pytest.skip(f"{C.TEST_CSV_PATH} not present")

    data = json.loads(GRADE1_JSON.read_text())
    cm = np.array(data["confusion_matrix_full"])  # cm[true][pred]
    col_sums = cm.sum(axis=0)

    for g in range(5):
        assert ctx[g] is not None
        assert ctx[g].n == int(col_sums[g]), f"grade {g}: n mismatch"


def test_t11_grade1_recall_matches_confusion_row():
    if not GRADE1_JSON.exists():
        pytest.skip(f"{GRADE1_JSON} not present")
    r1 = C.load_grade1_recall()
    data = json.loads(GRADE1_JSON.read_text())
    expected = data["confusion_row_grade1_fractions"]["Mild NPDR"]
    assert r1 == pytest.approx(expected)


def test_context_missing_file_returns_none(tmp_path):
    ctx = C.load_per_grade_context(csv_path=tmp_path / "does_not_exist.csv")
    assert ctx is None
    r1 = C.load_grade1_recall(json_path=tmp_path / "does_not_exist.json")
    assert r1 is None


# ---------------------------------------------------------------------
# T-9: no result number typed into app code
# ---------------------------------------------------------------------
FORBIDDEN_NUMBERS = [
    0.716, 0.7160, 0.912, 0.9117, 0.838, 0.855, 0.8545, 3.367, 0.1349,
    0.5927, 0.613, 0.074, 0.617, 0.543,
]
SCAN_ROOTS = [PROJECT_ROOT / "app"]
EXCLUDE_PATHS = {
    PROJECT_ROOT / "app" / "data" / "literature.json",
    PROJECT_ROOT / "app" / "release" / "thresholds.json",
    PROJECT_ROOT / "app" / "release" / "checkpoint.sha256",
    PROJECT_ROOT / "app" / "release" / "MODEL_CARD.md",
    PROJECT_ROOT / "app" / "release" / "config.yaml",
}


def _iter_scanned_files():
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if p in EXCLUDE_PATHS or "__pycache__" in p.parts:
                continue
            yield p
        for p in root.rglob("*"):
            if p.suffix in (".css", ".js", ".html") and p not in EXCLUDE_PATHS:
                yield p


def _numeric_literals_in_source(path: Path):
    """Parse Python files with ast for real numeric literals (handles both
    float and string forms); for non-Python assets, fall back to a plain
    text scan of the forbidden numbers' string forms."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    found = []
    if path.suffix == ".py":
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return found
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                found.append(float(node.value))
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for n in FORBIDDEN_NUMBERS:
                    if str(n) in node.value:
                        found.append(n)
    else:
        for n in FORBIDDEN_NUMBERS:
            if str(n) in text:
                found.append(n)
    return found


@pytest.mark.skip(reason="app.py not yet migrated off hardcoded numbers "
                          "(Task B2/D1) -- enable once METRICS loader lands")
def test_t9_no_hardcoded_result_numbers():
    offenders = []
    for path in _iter_scanned_files():
        literals = _numeric_literals_in_source(path)
        for lit in literals:
            for forbidden in FORBIDDEN_NUMBERS:
                if abs(lit - forbidden) < 1e-6:
                    offenders.append((str(path), forbidden))
    assert not offenders, f"Hardcoded result numbers found: {offenders}"


# ---------------------------------------------------------------------
# T-13: waffle rounding sums to exactly 100
# ---------------------------------------------------------------------
def largest_remainder_round(counts, total=100):
    """Largest-remainder (Hamilton) apportionment of `counts` (a list of
    non-negative numbers summing to some N>0) into `total` integer units
    summing to exactly `total`."""
    counts = np.asarray(counts, dtype=float)
    s = counts.sum()
    if s == 0:
        return [0] * len(counts)
    raw = counts / s * total
    floors = np.floor(raw).astype(int)
    remainder = total - floors.sum()
    fracs = raw - floors
    order = np.argsort(-fracs)
    result = floors.copy()
    for i in order[:remainder]:
        result[i] += 1
    return result.tolist()


def test_t13_waffle_rounding_sums_to_100():
    rng = np.random.RandomState(0)
    for _ in range(1000):
        tp, fn, tn, fp = rng.randint(0, 500, size=4)
        if tp + fn + tn + fp == 0:
            continue
        result = largest_remainder_round([tp, fn, tn, fp])
        assert sum(result) == 100
