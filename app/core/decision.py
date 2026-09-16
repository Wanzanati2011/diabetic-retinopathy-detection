"""
Calibration, referral and uncertainty-gate decisions (AGENT_EXECUTION_PLAN.md
Tasks A1-A3). No Gradio import here (Rule B3) -- this module is pure numerics
so it can be unit-tested and imported by app.py without pulling in the UI.

Reference implementation for every definition below:
src/experiments/calibrate.py. Do not change a formula here without re-reading
that file and docs/verification/V2_definitions.md -- the app must reproduce
calibrate.py's decisions exactly, not approximate them.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
THRESHOLDS_PATH = APP_DIR / "release" / "thresholds.json"
CHECKPOINT_PATH = APP_DIR / "release" / "best_model.pt"
CHECKPOINT_SHA_PATH = APP_DIR / "release" / "checkpoint.sha256"

INACTIVE_BANNER = (
    "Calibration file does not match this model. Showing uncalibrated scores; "
    "referral and uncertainty checks are disabled."
)


# --------------------------------------------------------------------------
# A1: thresholds + integrity
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Thresholds:
    temperature: float
    referral_threshold: float
    reject_tau: float
    reject_signal: str
    reject_action: str
    acceptance_test_11_1_verdict: str
    source_checkpoint: str = ""
    val_image_id_sha256_16: str = ""
    raw: dict = field(default_factory=dict, repr=False, compare=False)


def load_thresholds(path: Path = THRESHOLDS_PATH) -> Optional[Thresholds]:
    """None if the file is missing or malformed -- callers fall back to
    CALIBRATION_ACTIVE = False, never raise."""
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        return Thresholds(
            temperature=float(raw["temperature"]),
            referral_threshold=float(raw["referral_threshold"]),
            reject_tau=float(raw["reject_tau"]),
            reject_signal=str(raw["reject_signal"]),
            reject_action=str(raw["reject_action"]),
            acceptance_test_11_1_verdict=str(raw["acceptance_test_11_1_verdict"]),
            source_checkpoint=str(raw.get("source_checkpoint", "")),
            val_image_id_sha256_16=str(raw.get("val_image_id_sha256_16", "")),
            raw=raw,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def checkpoint_sha256(path: Path = CHECKPOINT_PATH) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_integrity(checkpoint_path: Path = CHECKPOINT_PATH,
                     sha_path: Path = CHECKPOINT_SHA_PATH,
                     thresholds_path: Path = THRESHOLDS_PATH):
    """Returns (calibration_active, thresholds_or_None, reason_or_None).

    reason is one of: "missing_thresholds", "missing_checkpoint_hash",
    "checkpoint_hash_mismatch", "acceptance_test_not_pass" -- or None when
    calibration_active is True."""
    thresholds = load_thresholds(thresholds_path)
    if thresholds is None:
        return False, None, "missing_thresholds"
    if not sha_path.exists():
        return False, thresholds, "missing_checkpoint_hash"
    expected = sha_path.read_text().strip().split()[0]
    if not checkpoint_path.exists():
        return False, thresholds, "missing_checkpoint_hash"
    actual = checkpoint_sha256(checkpoint_path)
    if actual != expected:
        return False, thresholds, "checkpoint_hash_mismatch"
    if thresholds.acceptance_test_11_1_verdict != "pass":
        return False, thresholds, "acceptance_test_not_pass"
    return True, thresholds, None


# --------------------------------------------------------------------------
# A1: temperature scaling
# --------------------------------------------------------------------------
def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """softmax(logits / T), numerically stable, row-wise. Matches
    src/experiments/calibrate.py's softmax_T() exactly."""
    z = np.asarray(logits, dtype=np.float64) / float(T)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def raw_probs(logits: np.ndarray) -> np.ndarray:
    """Uncalibrated softmax (T=1), for the raw-vs-calibrated disclosure."""
    return softmax(logits, T=1.0)


def calibrated_probs(logits: np.ndarray, thresholds: Thresholds) -> np.ndarray:
    """softmax(logits / T) using the fitted temperature."""
    return softmax(logits, T=thresholds.temperature)


def expected_grade(probs: np.ndarray) -> np.ndarray:
    """Sigma k * p_k -- the continuous severity marker (DEC-1/A5). Always
    call this on CALIBRATED probs for the app's dial; see
    docs/verification/V2_definitions.md for why grade1_diagnosis.py's own
    +0.0119 finding used raw probs instead and must not be conflated with
    this marker."""
    weights = np.arange(probs.shape[-1])
    return (probs * weights).sum(axis=-1)


# --------------------------------------------------------------------------
# A2: referral rule
# --------------------------------------------------------------------------
def referral_score(probs: np.ndarray) -> np.ndarray:
    """P(grade>=2) = p2 + p3 + p4 -- the whole referable-DR tail, not argmax."""
    return probs[..., 2:].sum(axis=-1)


def is_referred(probs: np.ndarray, thresholds: Thresholds) -> np.ndarray:
    """REFER iff referral_score >= referral_threshold (inequality direction
    confirmed against calibrate.py's referable_stats(): `scores >= threshold`)."""
    return referral_score(probs) >= thresholds.referral_threshold


# --------------------------------------------------------------------------
# A3: four-state outcome
# --------------------------------------------------------------------------
class Outcome(Enum):
    UNGRADABLE = "UNGRADABLE"
    UNCERTAIN = "UNCERTAIN"
    REFER = "REFER"
    ROUTINE = "ROUTINE"


def is_uncertain(probs: np.ndarray, thresholds: Thresholds) -> np.ndarray:
    """Gate on max CALIBRATED probability < tau (kept iff >= tau, per
    calibrate.py: `keep = conf_test >= tau`)."""
    return probs.max(axis=-1) < thresholds.reject_tau


def classify_outcome(probs: Optional[np.ndarray], thresholds: Optional[Thresholds],
                      calibration_active: bool, is_ungradable: bool = False) -> Outcome:
    """Precedence per DEC-7: UNGRADABLE > UNCERTAIN > REFER > ROUTINE.

    When calibration_active is False, the UNCERTAIN gate is disabled (there
    is no valid tau to gate on) and referral falls back to the uncalibrated
    argmax >= 2 rule -- the old pre-Phase-7 behaviour, labelled
    "uncalibrated fallback" by the caller.
    """
    if is_ungradable:
        return Outcome.UNGRADABLE
    if not calibration_active or thresholds is None or probs is None:
        # uncalibrated fallback: argmax >= 2 => REFER, else ROUTINE
        if probs is not None and int(probs.argmax()) >= 2:
            return Outcome.REFER
        return Outcome.ROUTINE
    if bool(is_uncertain(probs, thresholds)):
        return Outcome.UNCERTAIN
    if bool(is_referred(probs, thresholds)):
        return Outcome.REFER
    return Outcome.ROUTINE


# --------------------------------------------------------------------------
# A6: Both Eyes -- patient-level outcome (DEC-2) and identical-image guard
# --------------------------------------------------------------------------
def patient_outcome(pooled: Outcome, left: Outcome, right: Outcome) -> Optional[Outcome]:
    """Combine pooled + per-eye outcomes into one patient-level result, per
    DEC-2 ("REFER if the pooled result or either individual eye is REFER")
    extended to the other three states in the only order consistent with
    routing safety: a REFER anywhere can never be suppressed by a better
    result elsewhere, and a missing (UNGRADABLE) eye can never silently
    resolve to ROUTINE just because the other eye/pool looked fine.

    Returns None for "joint outcome unavailable" -- an eye is UNGRADABLE and
    nothing already forced a REFER; callers show the gradable eye's own
    result and mark the joint/patient row as unavailable (T-10)."""
    states = (pooled, left, right)
    if Outcome.REFER in states:
        return Outcome.REFER
    if Outcome.UNGRADABLE in states:
        return None
    if Outcome.UNCERTAIN in states:
        return Outcome.UNCERTAIN
    return Outcome.ROUTINE


def worse_eye_grade(left_grade: int, right_grade: int) -> int:
    """max(left, right) -- DEC-2's 'worse-eye grade' row."""
    return max(left_grade, right_grade)


def images_look_identical(pil_image_a, pil_image_b, hamming_threshold: int = 5) -> bool:
    """True if the two uploads are near-duplicates by the SAME method
    src/data/dedup.py uses for the dataset's own duplicate detection
    (perceptual hash, hash_size=16 / 256-bit, Hamming distance <= 5) --
    reused, not reimplemented, per AGENT_EXECUTION_PLAN.md A6 step 5."""
    import imagehash

    from src.data.dedup import hamming_hex

    hash_a = str(imagehash.phash(pil_image_a, hash_size=16))
    hash_b = str(imagehash.phash(pil_image_b, hash_size=16))
    return hamming_hex(hash_a, hash_b) <= hamming_threshold
