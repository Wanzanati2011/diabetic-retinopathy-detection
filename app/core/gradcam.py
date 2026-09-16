"""
Grad-CAM overlay (AGENT_EXECUTION_PLAN.md Task B3/B5). No Gradio import.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import numpy as np

from app.core.model import MODEL

GRADCAM_TIMEOUT_SECONDS = 4.0
_GRADCAM_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gradcam")


def compute_gradcam_overlay(x, processed_rgb, grade):
    """Grad-CAM (Selvaraju et al. 2017) on the backbone's last conv block
    (conv_head -- the standard target layer for a timm EfficientNet, the
    last spatial feature map before global pooling), for the PREDICTED
    class. Needs gradients, so this runs in a normal (non-no_grad) context,
    as a second, short forward+backward pass purely for attribution --
    it never influences the grade itself, which was already decided under
    torch.no_grad() before this is called.

    Returns None if pytorch-grad-cam isn't installed, or if anything about
    this specific model/library-version combination doesn't line up --
    Grad-CAM is explanatory evidence on top of a grade, not the grade
    itself, so a failure here must never take grading down with it. The
    caller hides the evidence panel when this returns None.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        return None
    try:
        cam = GradCAM(model=MODEL, target_layers=[MODEL.conv_head])
        grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(grade)])[0]
        rgb_float = processed_rgb.astype(np.float32) / 255.0
        return show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    except Exception as e:
        print(f"Grad-CAM failed ({type(e).__name__}: {e}) -- showing the grade without it.")
        return None


def compute_gradcam_overlay_with_timeout(x, processed_rgb, grade,
                                          timeout_seconds: float = GRADCAM_TIMEOUT_SECONDS):
    """B5: 'grading result yields first; Grad-CAM yields second. If Grad-
    CAM takes longer than 4s or raises, show "Heatmap skipped" and keep
    the grade.' Runs compute_gradcam_overlay() on a worker thread so a slow
    Grad-CAM can never delay (or crash) the grade the user already has.
    Returns (overlay_or_None, skipped_reason_or_None)."""
    future = _GRADCAM_EXECUTOR.submit(compute_gradcam_overlay, x, processed_rgb, grade)
    try:
        result = future.result(timeout=timeout_seconds)
        return result, (None if result is not None else "unavailable")
    except FutureTimeoutError:
        print(f"Grad-CAM exceeded {timeout_seconds}s -- skipping (grade already computed).")
        return None, "timed out"
    except Exception as e:  # pragma: no cover -- compute_gradcam_overlay already
        # catches its own exceptions; this is a last-resort net.
        print(f"Grad-CAM raised unexpectedly ({type(e).__name__}: {e}) -- skipping.")
        return None, "error"
