"""
Tests for app/core/session.py and app/report/pdf.py (AGENT_EXECUTION_PLAN.md
Task A7, T-12). Rendering-layer tests grow here as Phase 3/4 land.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.decision import Outcome
from app.core.session import make_log_entry
from app.report import pdf as P


def _sample_log():
    entry = make_log_entry(
        mode="Single Eye", grade=2, grade_name="Moderate NPDR",
        outcome=Outcome.REFER, expected_grade=2.3, referral_score=0.81,
        calibrated_conf=0.62, raw_conf=0.94, calibration_active=True,
        model_sha12="8d78cea5c167",
    )
    return [entry.as_dict()]


def test_no_log_entries_returns_none_with_message():
    path, msg = P.build_pdf_export(
        [], model_sha12="8d78cea5c167", calibration_active=True,
        temperature=3.3674, referral_threshold=0.13486, reject_tau=0.59274,
        reject_action="flag 'UNCERTAIN'.")
    assert path is None
    assert "No assessments logged" in msg


def test_t12_pdf_contains_required_fields(tmp_path):
    fpdf = pytest.importorskip("fpdf", reason="fpdf2 not installed")
    log = _sample_log()
    path, msg = P.build_pdf_export(
        log, model_sha12="8d78cea5c167", calibration_active=True,
        temperature=3.3674, referral_threshold=0.13486, reject_tau=0.59274,
        reject_action="flag 'UNCERTAIN -- refer to a human grader'.")
    assert path is not None, msg
    assert Path(path).exists()

    # Extract text back out of the generated PDF to check required content.
    reader = pytest.importorskip("pypdf", reason="pypdf not installed; "
                                  "falling back to a raw-bytes scan below")
    from pypdf import PdfReader
    text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    assert "REFER" in text
    assert "8d78cea5c167" in text
    assert "UNCERTAIN" in text  # from reject_action
    assert "NOT a" in text and "medical device" in text  # disclaimer
    Path(path).unlink(missing_ok=True)


def test_header_line_format():
    line = P.header_line("8d78cea5c167", True, 3.3674, 0.13486, 0.59274)
    assert "8d78cea5c167" in line
    assert "calibration active" in line
    assert "3.3674" in line
    assert "0.13486" in line
    assert "0.59274" in line

    line2 = P.header_line("8d78cea5c167", False, 3.3674, 0.13486, 0.59274)
    assert "calibration inactive" in line2
