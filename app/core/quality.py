"""
Basic image quality checks (AGENT_EXECUTION_PLAN.md Task B1). No Gradio
import. Never claims to catch every bad image -- labelled "basic image
checks" in the UI.

Checks run in order; the first one that fires wins:
  1. Unreadable       -> UNGRADABLE
  2. Tiny (<224px)    -> UNGRADABLE
  3. Huge (>25 MP)    -> UNGRADABLE (never resized -- would break parity
                         with training preprocessing)
  4. Dark frame       -> UNGRADABLE (mean of preprocess() output)
  5. Low contrast     -> UNGRADABLE (std of preprocess() output, vs. the
                         validation fold's own 1st percentile)
  6. Small (<384px)   -> warning only, not blocked
Not RGB is converted silently (not a warning).
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
QUALITY_THRESHOLDS_PATH = APP_DIR / "data" / "quality_thresholds.json"
LOW_MEAN_DIAGNOSTIC_PATH = PROJECT_ROOT / "results" / "low_mean_diagnostic.json"
EXCLUDED_IMAGES_PATH = PROJECT_ROOT / "results" / "excluded_images.json"

MIN_TINY_SIDE = 224
MIN_WARN_SIDE = 384
MAX_MEGAPIXELS = 25.0
# From results/excluded_images.json's exclusion_criterion: "original-source
# image mean pixel value < 5". Computed there on the ORIGINAL source image,
# not preprocess()'s output -- both are 0-255 uint8 scale, so the cutoff
# transfers; see docs/verification/V3_structures.md for the caveat.
DARK_FRAME_MEAN_CUTOFF = 5.0

MESSAGES = {
    "unreadable": "This file isn't an image we can read.",
    "tiny": "Image is too small to grade.",
    "huge": "Image is too large (max 25 MP).",
    "dark_frame": "Image is almost entirely dark; this looks like a failed capture.",
    "low_contrast": "Image is washed out or overexposed.",
}
WARNING_SMALL = "Low-resolution photo; fine lesions may be lost."


@dataclass(frozen=True)
class QualityResult:
    ungradable: bool
    reason: Optional[str]  # key into MESSAGES, or None
    message: Optional[str]  # the user-facing string, or None
    warning: Optional[str]  # non-blocking warning text, or None
    rgb_image: Optional[Image.Image] = None  # RGB-converted PIL image (None if unreadable)
    processed_rgb: Optional[np.ndarray] = None  # preprocess() output, if it ran


def _load_contrast_cutoff(path: Path = QUALITY_THRESHOLDS_PATH) -> Optional[float]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return float(data["contrast_std_p1_cutoff"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _open_image(image_input: Union[Image.Image, bytes, str, Path]) -> Optional[Image.Image]:
    """Returns a PIL Image, or None if it can't be decoded/verified."""
    try:
        if isinstance(image_input, Image.Image):
            image_input.load()  # forces decode of a lazily-opened image
            return image_input
        if isinstance(image_input, (bytes, bytearray)):
            im = Image.open(io.BytesIO(image_input))
            im.load()
            return im
        im = Image.open(image_input)
        im.load()
        return im
    except Exception:
        return None


def check_quality(image_input: Union[Image.Image, bytes, str, Path],
                   preprocess_size: int = 384,
                   contrast_cutoff: Optional[float] = None) -> QualityResult:
    """Runs every check in order (docstring above) and returns the first
    failure, or a passing result with the RGB image and preprocess()
    output attached (so callers don't have to preprocess twice)."""
    pil_image = _open_image(image_input)
    if pil_image is None:
        return QualityResult(True, "unreadable", MESSAGES["unreadable"], None)

    pil_image = ImageOps.exif_transpose(pil_image)
    width, height = pil_image.size
    min_side = min(width, height)
    megapixels = (width * height) / 1_000_000

    if min_side < MIN_TINY_SIDE:
        return QualityResult(True, "tiny", MESSAGES["tiny"], None)
    if megapixels > MAX_MEGAPIXELS:
        return QualityResult(True, "huge", MESSAGES["huge"], None)

    rgb_image = pil_image.convert("RGB")  # silent, not a warning
    warning = WARNING_SMALL if min_side < MIN_WARN_SIDE else None

    # preprocess() is THE SAME FUNCTION used in training -- see app.py's
    # to_model_input(). Imported here, not reimplemented (R2).
    from src.data.preprocess import preprocess

    rgb_arr = np.array(rgb_image)
    bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    processed_rgb = preprocess(bgr_arr, size=preprocess_size)

    mean_val = float(processed_rgb.mean())
    if mean_val < DARK_FRAME_MEAN_CUTOFF:
        return QualityResult(True, "dark_frame", MESSAGES["dark_frame"], None,
                              rgb_image=rgb_image, processed_rgb=processed_rgb)

    cutoff = contrast_cutoff if contrast_cutoff is not None else _load_contrast_cutoff()
    if cutoff is not None:
        std_val = float(processed_rgb.std())
        if std_val < cutoff:
            return QualityResult(True, "low_contrast", MESSAGES["low_contrast"], None,
                                  rgb_image=rgb_image, processed_rgb=processed_rgb)

    return QualityResult(False, None, None, warning, rgb_image=rgb_image,
                          processed_rgb=processed_rgb)
