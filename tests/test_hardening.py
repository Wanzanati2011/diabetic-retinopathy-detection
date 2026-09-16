"""
Tests for AGENT_EXECUTION_PLAN.md Task B5 (runtime hardening), T-14.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch", reason="torch not installed yet")
timm = pytest.importorskip("timm", reason="timm not installed yet")

CHECKPOINT_PATH = PROJECT_ROOT / "app" / "release" / "best_model.pt"
if not CHECKPOINT_PATH.exists():
    pytest.skip(f"{CHECKPOINT_PATH} not found", allow_module_level=True)

from PIL import Image  # noqa: E402
import app.app as app_module  # noqa: E402
from app.core import gradcam as G  # noqa: E402


def _sample_image():
    candidates = list((PROJECT_ROOT / "data" / "processed" / "384").glob("*.jpg"))
    if not candidates:
        pytest.skip("no processed sample images available locally")
    return Image.open(candidates[0])


# ---------------------------------------------------------------------
# T-14: Grad-CAM raising doesn't take the grade down with it
# ---------------------------------------------------------------------
def test_t14_gradcam_wrapper_swallows_exceptions_and_predict_still_yields_grade(monkeypatch):
    """Patch the RAW compute_gradcam_overlay (what actually runs inside
    the timeout wrapper's worker thread) to raise. predict() calls the
    wrapper, never the raw function directly -- confirm the wrapper's own
    try/except means predict() still completes normally and every yielded
    frame still carries the grade/outcome that was computed before
    Grad-CAM ever ran."""
    def _boom(x, processed_rgb, grade):
        raise RuntimeError("simulated Grad-CAM failure")

    monkeypatch.setattr(G, "compute_gradcam_overlay", _boom)
    monkeypatch.setattr(app_module, "compute_gradcam_overlay_with_timeout",
                         G.compute_gradcam_overlay_with_timeout)

    img = _sample_image()
    outputs = list(app_module.predict(img, []))
    assert len(outputs) == 3  # scanning, grade-without-heatmap, grade-with-heatmap-attempt
    final = outputs[-1]
    # U4: predict() now yields (result_card_html, image_slider_html, session_log, log_html)
    assert "Grade" in final[0] or "can't be graded" in final[0]  # result card has grade
    assert len(final[1]) > 0  # image_slider_html: HTML string with placeholder
    assert len(final[2]) == 1  # session_log: the grade WAS logged despite Grad-CAM failing


def test_t14_gradcam_timeout_wrapper_never_raises():
    """compute_gradcam_overlay_with_timeout() itself must never raise --
    this is what predict() actually relies on for safety."""
    import numpy as np

    result, reason = G.compute_gradcam_overlay_with_timeout(
        torch.zeros(1, 3, 4, 4), np.zeros((4, 4, 3), dtype="uint8"), 0)
    # A 4x4 input is nonsensical for this model's conv_head target layer,
    # so this is expected to fail internally -- the assertion is that it
    # fails QUIETLY (returns a reason string), not by raising.
    assert reason is None or isinstance(reason, str)


def test_t14_gradcam_timeout_fires(monkeypatch):
    """A Grad-CAM call that runs longer than the configured timeout is
    skipped (returns a 'timed out' reason), not awaited forever."""
    import time

    def _slow(x, processed_rgb, grade):
        time.sleep(2)
        return None

    monkeypatch.setattr(G, "compute_gradcam_overlay", _slow)
    t0 = time.time()
    result, reason = G.compute_gradcam_overlay_with_timeout(
        None, None, 0, timeout_seconds=0.2)
    elapsed = time.time() - t0
    assert result is None
    assert reason == "timed out"
    assert elapsed < 1.0  # did not wait for the full 2s sleep


# ---------------------------------------------------------------------
# T-14: PDF export with the library missing -> message, no exception
# ---------------------------------------------------------------------
def test_t14_pdf_missing_library_returns_message_not_exception(monkeypatch):
    import builtins

    from app.core.session import make_log_entry
    from app.core.decision import Outcome
    from app.report import pdf as P

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fpdf":
            raise ImportError("simulated: fpdf2 not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    entry = make_log_entry(
        mode="Single Eye", grade=2, grade_name="Moderate NPDR", outcome=Outcome.REFER,
        expected_grade=2.3, referral_score=0.81, calibrated_conf=0.62, raw_conf=0.94,
        calibration_active=True, model_sha12="8d78cea5c167",
    )
    path, msg = P.build_pdf_export(
        [entry.as_dict()], model_sha12="8d78cea5c167", calibration_active=True,
        temperature=3.3674, referral_threshold=0.13486, reject_tau=0.59274,
        reject_action="flag 'UNCERTAIN'.")
    assert path is None
    assert "pip install fpdf2" in msg
