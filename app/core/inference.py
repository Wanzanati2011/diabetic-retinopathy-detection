"""
Preprocessing and inference helpers (AGENT_EXECUTION_PLAN.md Task B3). No
Gradio import.

to_model_input()'s BODY is byte-identical to the pre-B3 app.py version --
only its module-level dependencies (IMAGENET_MEAN/STD, TRAIN_IMAGE_SIZE,
preprocess) moved with it. See tests/test_app.py (Acceptance Test 12.1)
for the check that this reproduces the evaluation pipeline's predictions.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import ImageOps

from src.data.preprocess import preprocess  # noqa: E402 -- THE SAME FUNCTION USED IN TRAINING
from app.core.model import MODEL, TRAIN_IMAGE_SIZE

IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def to_model_input(pil_image, image_size=None):
    """Reproduces the EXACT eval-time (augment=False) preprocessing path used
    by src/train/finetune_converged.py's FineTuneDataset:
      1. preprocess() (crop retinal circle, resize, pad to square) -- the
         SAME function build_cache.py used to build the training/eval cache.
      2. uint8 RGB -> float tensor in [0, 1] -> ImageNet normalize.
    No augmentation (flip/rotate/brightness jitter) -- that's train-only.

    Returns (model_input_tensor[1,3,H,W], processed_rgb_uint8_array) -- the
    second is handy for a debug preview or a future Grad-CAM overlay.
    """
    size = image_size or TRAIN_IMAGE_SIZE
    # cv2.imread() (what build_cache.py used to build the training/eval cache)
    # auto-applies EXIF orientation for JPEG/TIFF sources; PIL's Image.open()
    # does NOT -- it returns the raw, un-rotated pixel grid unless you call
    # exif_transpose() yourself. Skipping this was a real bug (caught by
    # tests/test_app.py's Acceptance Test 12.1, not assumed): for any source
    # photo with a non-default EXIF Orientation tag, preprocess()'s "find the
    # largest bright region" crop-detection step found a DIFFERENT region on
    # the differently-rotated pixel grid than training saw, so the model got
    # a genuinely different crop, not just JPEG rounding noise.
    pil_image = ImageOps.exif_transpose(pil_image)
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # preprocess() expects BGR, matching
                                                  # cv2.imread()'s convention in build_cache.py
    processed_rgb = preprocess(bgr, size=size)   # returns uint8 RGB, same as the training cache
    arr = torch.from_numpy(processed_rgb).permute(2, 0, 1).float() / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.unsqueeze(0), processed_rgb


def pooled_embedding(x):
    """The penultimate embedding (after global pooling, before the final
    classifier Linear layer) for one preprocessed image -- timm's standard
    forward_features -> forward_head(pre_logits=True) split, stable across
    the EfficientNet family. This is THIS model's own analogue of the
    frozen per-eye feature vectors src/experiments/claim3_both_eyes.py mean-
    pooled (features/{backbone}_{size}.npz) -- same idea (pool each eye's
    embedding, average, classify the average), applied live to this app's
    own fine-tuned end-to-end network rather than to a separately-fit head
    on frozen ImageNet features. See app.py's predict_both_eyes() docstring
    for why the numbers are NOT a reproduction of Claim 3's reported QWK."""
    with torch.no_grad():
        feats = MODEL.forward_features(x)
        return MODEL.forward_head(feats, pre_logits=True)
