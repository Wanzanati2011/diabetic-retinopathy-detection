"""
Per-prediction context (AGENT_EXECUTION_PLAN.md Task A4). No Gradio import.

Computes, per predicted (argmax) grade g, what that prediction has
historically meant on the held-out P2 test set: n, % exactly right, %
within +-1, % truly referable, and the most common true grade when wrong.

This will be absorbed into app/data/metrics.py's typed METRICS loader when
Task B2 lands (Phase 2) -- kept as its own small module for now so A4 does
not have to wait on B2's full loader.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
TEST_CSV_PATH = (PROJECT_ROOT / "results" /
                  "finetune_app_converged_p2_class_balanced_seed42_test_predictions.csv")
GRADE1_JSON_PATH = PROJECT_ROOT / "results" / "grade1_diagnosis.json"


@dataclass(frozen=True)
class GradeContext:
    grade: int
    n: int
    pct_exact: float
    pct_within1: float
    pct_referable: float
    most_common_true_when_wrong: Optional[int]


def per_grade_context(df: pd.DataFrame) -> dict:
    """df must have integer columns pred_grade, true_grade (the CSV's
    argmax-derived columns, per DEC-1 -- the displayed grade is always
    argmax, so this context table is keyed on argmax too)."""
    out = {}
    for g in range(5):
        sub = df[df["pred_grade"] == g]
        n = int(len(sub))
        if n == 0:
            out[g] = None
            continue
        exact = float((sub["true_grade"] == g).mean())
        within1 = float((sub["true_grade"] - g).abs().le(1).mean())
        referable = float((sub["true_grade"] >= 2).mean())
        wrong = sub[sub["true_grade"] != g]
        common_wrong = int(wrong["true_grade"].mode().iloc[0]) if len(wrong) else None
        out[g] = GradeContext(
            grade=g, n=n, pct_exact=exact, pct_within1=within1,
            pct_referable=referable, most_common_true_when_wrong=common_wrong,
        )
    return out


def load_per_grade_context(csv_path: Path = TEST_CSV_PATH) -> Optional[dict]:
    """None if the CSV is missing -- callers render 'data unavailable'."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    return per_grade_context(df)


def load_grade1_recall(json_path: Path = GRADE1_JSON_PATH) -> Optional[float]:
    """Grade-1 (Mild NPDR) recall under the argmax rule -- the 'known blind
    spot' figure for A4's grade 0/1 sentence. Source of truth:
    grade1_diagnosis.json -> confusion_row_grade1_fractions['Mild NPDR']."""
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text())
    fractions = data.get("confusion_row_grade1_fractions", {})
    value = fractions.get("Mild NPDR")
    return float(value) if value is not None else None
