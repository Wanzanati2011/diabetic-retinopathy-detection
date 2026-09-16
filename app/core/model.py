"""
Model loading and integrity (AGENT_EXECUTION_PLAN.md Task B3). No Gradio
import. Loading the model and checking its integrity happens exactly once,
at import time -- every other module (inference, gradcam, app.py itself)
imports MODEL/TRAIN_IMAGE_SIZE/CALIBRATION_ACTIVE/THRESHOLDS/MODEL_SHA12
from here rather than reloading anything.
"""
from __future__ import annotations

from pathlib import Path

import torch

from app.core import decision as D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_SIZE = 384  # overridden by the checkpoint's own recorded config, see load_model()
CHECKPOINT_PATH = PROJECT_ROOT / "app" / "release" / "best_model.pt"


def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"{CHECKPOINT_PATH} not found. Copy the trained checkpoint there first, e.g. from "
            f"the project root:\n"
            f'  Copy-Item "checkpoints\\finetune_app_p2_seed42\\best.pt" "app\\release\\best_model.pt"'
        )
    import timm

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    cfg = ckpt.get("config", {}) or {}
    backbone_name = cfg.get("model", "tf_efficientnet_b0")
    image_size = cfg.get("image_size", DEFAULT_IMAGE_SIZE)

    model = timm.create_model(backbone_name, pretrained=False, num_classes=5)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_qwk = ckpt.get("val_qwk")
    val_qwk_str = f"{val_qwk:.4f}" if val_qwk is not None else "unknown"
    print(f"Loaded {backbone_name} @ {image_size}px from {CHECKPOINT_PATH.name} "
          f"(split={ckpt.get('split')}, seed={ckpt.get('seed')}, "
          f"val_qwk={val_qwk_str} at epoch {ckpt.get('epoch')})")
    return model, image_size


def warm_up(model, image_size):
    """B5: one forward pass on a zero tensor at startup, so the FIRST real
    user request doesn't pay for lazy CUDA/cuDNN kernel init, timm's first-
    call overhead, etc. on top of the actual grading latency."""
    import time

    t0 = time.time()
    with torch.inference_mode():
        model(torch.zeros(1, 3, image_size, image_size))
    print(f"Warm-up forward pass: {time.time() - t0:.2f}s")


MODEL, TRAIN_IMAGE_SIZE = load_model()
warm_up(MODEL, TRAIN_IMAGE_SIZE)

# A1: calibration integrity check, once at startup. If the checkpoint hash,
# thresholds file, or acceptance-test verdict don't check out, the app
# falls back to raw uncalibrated confidence and says so on screen -- it
# never fakes a calibrated number it hasn't actually verified.
CALIBRATION_ACTIVE, THRESHOLDS, CALIBRATION_INACTIVE_REASON = D.check_integrity()
MODEL_SHA12 = D.checkpoint_sha256()[:12] if CHECKPOINT_PATH.exists() else "unknown"
print(f"Calibration active: {CALIBRATION_ACTIVE}"
      f"{'' if CALIBRATION_ACTIVE else f' (reason: {CALIBRATION_INACTIVE_REASON})'}"
      f"  model sha12={MODEL_SHA12}")
