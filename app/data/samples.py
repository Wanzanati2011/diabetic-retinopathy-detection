"""
Curated demo samples loader (AGENT_EXECUTION_PLAN.md Task B4). No Gradio
import.

Loads app/samples_local/samples.json (written by scripts/curate_samples.py)
if present. That folder is gitignored and local-only until the owner
confirms the Kaggle EyePACS 2015 test-set licence permits showing a
handful of test images in a public demo (R6/H1) -- so the public app must
work with ZERO bundled samples: if the file is absent, callers hide the
sample chips and show the upload prompt instead (DEC-5).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

APP_DIR = Path(__file__).resolve().parents[1]
SAMPLES_LOCAL_DIR = APP_DIR / "samples_local"
SAMPLES_JSON_PATH = SAMPLES_LOCAL_DIR / "samples.json"

SAMPLES_CAPTION = "held-out images the model never saw during training or evaluation"


def load_samples(path: Path = SAMPLES_JSON_PATH,
                  samples_dir: Path = SAMPLES_LOCAL_DIR) -> Optional[dict]:
    """None if the curated-samples file (or its images) aren't present --
    the only way samples are ever missing from a public deploy, by design."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None

    resolved = {}
    for slot, entry in data.items():
        if slot == "pair":
            left = samples_dir / entry["left_file"]
            right = samples_dir / entry["right_file"]
            if not (left.exists() and right.exists()):
                continue
            resolved[slot] = {**entry, "left_path": str(left), "right_path": str(right)}
        else:
            p = samples_dir / entry["file"]
            if not p.exists():
                continue
            resolved[slot] = {**entry, "path": str(p)}
    return resolved or None


def ground_truth_line(true_grade: int, predicted_grade: int) -> str:
    """'Ground truth: Grade {g} · {match | off by n}' -- B4 step 5."""
    diff = predicted_grade - true_grade
    verdict = "match" if diff == 0 else f"off by {abs(diff)}"
    return f"Ground truth: Grade {true_grade} · {verdict}"
