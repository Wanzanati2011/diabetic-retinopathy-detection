"""
Shared preprocessing (MASTER_PLAN.md Part 6).

*** SINGLE SOURCE OF TRUTH ***
This module is imported by BOTH the training cache-builder (build_cache.py)
AND the Gradio app (app/app.py). Never reimplement this logic anywhere else
-- preprocessing mismatch between training and deployment is the single most
common reason a working model produces garbage in production (Acceptance
Test 12.1 exists specifically to catch that class of bug).

No GPU needed. Pure OpenCV + numpy.
"""
import cv2
import numpy as np


def preprocess(image_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """
    1. Detect the retinal circle:
         gray = channel mean
         mask = gray > 0.1 * gray.max()
         bounding box of the largest connected component
    2. Crop to it (removes black borders — typically 30-40% of pixels)
    3. Resize longer side to `size`, preserving aspect ratio
    4. Pad to square (size, size) with black
    5. Return uint8 RGB
    """
    img = image_bgr
    if img is None or img.size == 0:
        raise ValueError("preprocess() got an empty/unreadable image")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    gray = img.mean(axis=2)
    gmax = gray.max()
    if gmax <= 0:
        # fully black image — nothing to crop to; skip straight to resize/pad
        x, y, w, h = 0, 0, img.shape[1], img.shape[0]
    else:
        mask = (gray > 0.1 * gmax).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            x, y, w, h = 0, 0, img.shape[1], img.shape[0]
        else:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest = 1 + int(np.argmax(areas))
            x = int(stats[largest, cv2.CC_STAT_LEFT])
            y = int(stats[largest, cv2.CC_STAT_TOP])
            w = int(stats[largest, cv2.CC_STAT_WIDTH])
            h = int(stats[largest, cv2.CC_STAT_HEIGHT])

    cropped = img[y:y + h, x:x + w]
    if cropped.size == 0:
        cropped = img  # safety fallback — never emit an empty crop

    ch, cw = cropped.shape[:2]
    scale = size / max(ch, cw)
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def ben_graham(img: np.ndarray, sigma_ratio: float = 30) -> np.ndarray:
    """2015 Kaggle winner's local contrast normalisation.
    Config-controlled ablation — NOT applied by default. img is RGB or BGR,
    either works since this is a per-pixel blend against a blurred copy of
    itself (channel order doesn't matter for this operation).
    """
    blur = cv2.GaussianBlur(img, (0, 0), img.shape[1] / sigma_ratio)
    return cv2.addWeighted(img, 4, blur, -4, 128)
