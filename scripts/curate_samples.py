"""
B4: curate demo sample images from the EyePACS 2015 Kaggle TEST set
(`resized test 15/` + `labels/testLabels15.csv`), which is NOT used
anywhere in this project's training or evaluation (the manifest only
covers `train 15` + `train 19` -- see AGENT_EXECUTION_PLAN.md 1.3).

Local-only (R6/H1): copies chosen files to app/samples_local/, which is
gitignored. Until the owner confirms the Kaggle competition licence
permits showing a handful of test images in a public demo, this folder
never leaves this machine.

Method (per AGENT_EXECUTION_PLAN.md B4): iterate images in SORTED
filename order, run the real app pipeline (quality check -> calibrated
probs -> four-state outcome) on CPU, and take the FIRST image matching
each of six rules. Stop scanning after 3,000 images -- if a slot is still
empty at that point, STOP (Part 7 condition #7) rather than silently
shipping with a hole.

Usage (from the project root, venv active):
    python scripts\\curate_samples.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core import decision as D  # noqa: E402
from app.core import quality as Q  # noqa: E402
from app.core.inference import to_model_input  # noqa: E402
from app.core.model import MODEL, CALIBRATION_ACTIVE, THRESHOLDS  # noqa: E402

LABELS_CSV = PROJECT_ROOT / "labels" / "testLabels15.csv"
IMAGE_DIR = PROJECT_ROOT / "resized test 15"
OUT_DIR = PROJECT_ROOT / "app" / "samples_local"
MAX_SCAN = 3000
GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]


def grade_image(path):
    """Runs the SAME pipeline the live app uses: quality gate, then (if
    gradable) preprocessing + forward pass + calibration + four-state
    outcome. Returns a dict; never raises on a bad image (quality catches
    that)."""
    quality = Q.check_quality(path)
    if quality.ungradable:
        return {"gradable": False, "quality_reason": quality.reason}

    with Image.open(path) as im:
        x, _ = to_model_input(im.convert("RGB"))
    with torch.no_grad():
        logits = MODEL(x)[0].numpy()
    probs = D.calibrated_probs(logits, THRESHOLDS) if CALIBRATION_ACTIVE else D.raw_probs(logits)
    outcome = D.classify_outcome(probs, THRESHOLDS, CALIBRATION_ACTIVE)
    return {"gradable": True, "grade": int(probs.argmax()), "outcome": outcome}


def parse_id(image_id):
    if image_id.endswith("_left"):
        return image_id[:-len("_left")], "left"
    if image_id.endswith("_right"):
        return image_id[:-len("_right")], "right"
    return image_id, "unknown"


def main():
    if not LABELS_CSV.exists():
        raise SystemExit(f"{LABELS_CSV} not found.")
    if not IMAGE_DIR.exists():
        raise SystemExit(f"{IMAGE_DIR} not found.")

    df = pd.read_csv(LABELS_CSV).sort_values("image").reset_index(drop=True)
    print(f"Scanning up to {MAX_SCAN} of {len(df)} labeled images "
          f"(calibration_active={CALIBRATION_ACTIVE})...")

    slots = {k: None for k in
             ["routine", "refer", "proliferative", "failure", "uncertain", "pair"]}
    pending_left = {}  # patient_id -> (image_id, true_level, result)

    n_scanned = 0
    for _, row in df.iterrows():
        if n_scanned >= MAX_SCAN:
            break
        image_id = str(row["image"])
        true_level = int(row["level"])
        img_path = IMAGE_DIR / f"{image_id}.jpg"
        if not img_path.exists():
            continue

        n_scanned += 1
        if n_scanned % 200 == 0:
            print(f"  scanned {n_scanned}...  filled: "
                  f"{[k for k, v in slots.items() if v is not None]}")

        result = grade_image(img_path)
        patient_id, eye = parse_id(image_id)

        if result["gradable"]:
            grade, outcome = result["grade"], result["outcome"]

            if (slots["routine"] is None and true_level == 0
                    and outcome is D.Outcome.ROUTINE and grade == 0):
                slots["routine"] = {
                    "slot": "routine", "file": f"{image_id}.jpg", "source": "EyePACS 2015 Kaggle test set",
                    "image_id": image_id, "true_grade": true_level, "eye": eye,
                    "expected_outcome": outcome.value, "selection_rule":
                        "true level 0, outcome ROUTINE, argmax 0",
                }
            if (slots["refer"] is None and true_level in (2, 3)
                    and outcome is D.Outcome.REFER and grade == true_level):
                slots["refer"] = {
                    "slot": "refer", "file": f"{image_id}.jpg", "source": "EyePACS 2015 Kaggle test set",
                    "image_id": image_id, "true_grade": true_level, "eye": eye,
                    "expected_outcome": outcome.value, "selection_rule":
                        "true level 2 or 3, outcome REFER, argmax == true level",
                }
            if slots["proliferative"] is None and true_level == 4 and outcome is D.Outcome.REFER:
                slots["proliferative"] = {
                    "slot": "proliferative", "file": f"{image_id}.jpg", "source": "EyePACS 2015 Kaggle test set",
                    "image_id": image_id, "true_grade": true_level, "eye": eye,
                    "expected_outcome": outcome.value, "selection_rule":
                        "true level 4, outcome REFER",
                }
            if slots["failure"] is None and true_level == 1 and grade == 0:
                slots["failure"] = {
                    "slot": "failure", "file": f"{image_id}.jpg", "source": "EyePACS 2015 Kaggle test set",
                    "image_id": image_id, "true_grade": true_level, "eye": eye,
                    "expected_outcome": outcome.value, "selection_rule":
                        "true level 1, argmax 0",
                }
            if slots["uncertain"] is None and outcome is D.Outcome.UNCERTAIN:
                slots["uncertain"] = {
                    "slot": "uncertain", "file": f"{image_id}.jpg", "source": "EyePACS 2015 Kaggle test set",
                    "image_id": image_id, "true_grade": true_level, "eye": eye,
                    "expected_outcome": outcome.value, "selection_rule":
                        "outcome UNCERTAIN",
                }

        if eye == "left":
            pending_left[patient_id] = (image_id, true_level, result)
        elif eye == "right" and slots["pair"] is None:
            left = pending_left.get(patient_id)
            if left is not None:
                left_id, left_true, left_result = left
                if (left_result["gradable"] and result["gradable"]
                        and (left_true >= 2 or true_level >= 2)):
                    slots["pair"] = {
                        "slot": "pair", "left_file": f"{left_id}.jpg", "right_file": f"{image_id}.jpg",
                        "source": "EyePACS 2015 Kaggle test set", "patient_id": patient_id,
                        "left_true_grade": left_true, "right_true_grade": true_level,
                        "left_expected_outcome": left_result["outcome"].value,
                        "right_expected_outcome": result["outcome"].value,
                        "selection_rule": "first patient with both eyes gradable and "
                                          "at least one eye true level >= 2",
                    }

    print(f"\nScanned {n_scanned} images.")
    missing = [k for k, v in slots.items() if v is None]
    if missing:
        print(f"\nSTOP (Part 7 condition #7): no image found for slot(s) {missing} "
              f"within {MAX_SCAN} images. Not writing samples.json -- report this "
              f"to the owner rather than shipping with a hole.")
        raise SystemExit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for slot_name, entry in slots.items():
        if slot_name == "pair":
            shutil.copy2(IMAGE_DIR / entry["left_file"], OUT_DIR / entry["left_file"])
            shutil.copy2(IMAGE_DIR / entry["right_file"], OUT_DIR / entry["right_file"])
        else:
            shutil.copy2(IMAGE_DIR / entry["file"], OUT_DIR / entry["file"])

    (OUT_DIR / "samples.json").write_text(json.dumps(slots, indent=2))
    print(f"\nWrote {OUT_DIR / 'samples.json'} and copied "
          f"{sum(2 if k == 'pair' else 1 for k in slots)} image file(s) to {OUT_DIR}.")
    for slot_name, entry in slots.items():
        print(f"  {slot_name}: {entry}")


if __name__ == "__main__":
    main()
