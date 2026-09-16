"""
PDF session-log export (AGENT_EXECUTION_PLAN.md Task A7). No Gradio import.

Lazy-imports fpdf2 the same way app.py's old build_pdf_export() and
compute_gradcam_overlay() lazy-import their optional dependencies --
missing or misbehaving, this degrades to a clear message, never a crashed
button (B5's "every stage maps to a user-facing message" rule).
"""
from __future__ import annotations

import datetime
import tempfile
from pathlib import Path
from typing import Optional

DISCLAIMER_TEXT = (
    "Research prototype trained on public datasets (EyePACS + APTOS). NOT a "
    "medical device. NOT validated for clinical use. Not a substitute for "
    "examination by a qualified ophthalmologist."
)


def header_line(model_sha12: str, calibration_active: bool, temperature: float,
                 referral_threshold: float, reject_tau: float) -> str:
    """'Model {sha12} · calibration {active|inactive} · T {T} · referral
    line {thr} · uncertainty threshold {tau}' -- A7 step 2."""
    status = "active" if calibration_active else "inactive"
    return (f"Model {model_sha12} · calibration {status} · "
            f"T {temperature:.4f} · referral line {referral_threshold:.5f} · "
            f"uncertainty threshold {reject_tau:.5f}")


def build_pdf_export(session_log, model_sha12: str, calibration_active: bool,
                      temperature: float, referral_threshold: float,
                      reject_tau: float, reject_action: str):
    """Returns (file_path_or_None, status_markdown). session_log is a list
    of app.core.session.LogEntry.as_dict()-shaped rows."""
    if not session_log:
        return None, "*No assessments logged yet this session -- grade at least one image first.*"
    try:
        from fpdf import FPDF
    except ImportError:
        return None, "*PDF export needs `pip install fpdf2` (not installed in this environment).*"

    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Diabetic Retinopathy Detector -- Session Log")
        pdf.ln(12)

        pdf.set_font("Helvetica", "", 8)
        pdf.multi_cell(0, 4.5, header_line(
            model_sha12, calibration_active, temperature,
            referral_threshold, reject_tau))
        pdf.ln(2)
        pdf.multi_cell(0, 4.5, DISCLAIMER_TEXT)
        pdf.ln(2)
        pdf.multi_cell(0, 4.5, f"Reject action: {reject_action}")
        pdf.ln(2)
        pdf.cell(0, 5, f"Exported {datetime.datetime.now():%Y-%m-%d %H:%M:%S}.")
        pdf.ln(10)

        headers = ["Time", "Mode", "Outcome", "Grade", "Grade name", "Conf. (cal.)"]
        widths = [20, 22, 24, 16, 36, 24]
        pdf.set_font("Helvetica", "B", 9)
        for h, w in zip(headers, widths):
            pdf.cell(w, 7, h, border=1)
        pdf.ln(7)
        pdf.set_font("Helvetica", "", 9)
        for e in session_log:
            pdf.cell(widths[0], 7, str(e["time"]), border=1)
            pdf.cell(widths[1], 7, str(e["mode"]), border=1)
            pdf.cell(widths[2], 7, str(e["outcome"]), border=1)
            pdf.cell(widths[3], 7, str(e["grade"]), border=1)
            pdf.cell(widths[4], 7, str(e["grade_name"]), border=1)
            pdf.cell(widths[5], 7, f'{round(e["calibrated_conf"] * 100)}%', border=1)
            pdf.ln(7)

        out_path = (Path(tempfile.gettempdir()) /
                    f"fundus_session_log_{datetime.datetime.now():%Y%m%d_%H%M%S}.pdf")
        pdf.output(str(out_path))
        return str(out_path), f"*Exported {len(session_log)} assessment(s).*"
    except Exception as e:  # pragma: no cover -- defensive, matches B5's rule
        print(f"PDF export failed ({type(e).__name__}: {e}).")
        return None, f"*PDF export failed ({type(e).__name__}) -- see the terminal for details.*"
