"""
Acceptance Test 12.1 (MASTER_PLAN.md Part 12.3) -- "the critical one":
run real test images through BOTH app/app.py's predict() path and the
evaluation pipeline's path, and check they agree. If they don't, it's a
preprocessing mismatch -- the single most common way a model that works
in evaluation produces garbage after "deployment."

Two tiers, same pattern as the other test files in this project:
  1. Pure-logic tests -- no torch/timm/gradio/checkpoint needed.
  2. The real acceptance test -- needs torch + timm installed AND the
     checkpoint copied to app/release/best_model.pt (see app/app.py's
     error message for the exact copy command). Skipped automatically
     if either is missing, so this file is always safe to run.

HONEST CAVEAT on "match exactly": the evaluation pipeline reads the
ALREADY-CACHED, JPEG-quality-95-compressed image from
data/processed/384/{image_id}.jpg. The app's predict() path runs
preprocess() freshly on the ORIGINAL source image (as a real user's
upload would be) and never touches that JPEG. JPEG compression is lossy,
so the two pixel arrays are not bit-identical -- this test therefore
checks that the PREDICTED GRADE (argmax) matches on every sampled image,
which is what actually matters for the app being trustworthy, rather than
requiring bit-identical softmax outputs, which the JPEG cache step makes
impossible in principle. If this test starts failing, the first thing to
check is still a genuine preprocessing mismatch (wrong color channel
order, wrong normalization, wrong crop) -- not this JPEG-rounding
caveat, which has always been true and is not itself a bug.

Default sample size is 20 test images (fast). MASTER_PLAN.md Part 12.3
suggests 100 for the real acceptance run -- override with an environment
variable (a plain pytest CLI flag needs a conftest.py to register, which
this project doesn't otherwise have -- an env var needs no new file):
    $env:N_APP_TEST_IMAGES=100; pytest tests\\test_app.py -v
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

N_APP_TEST_IMAGES = int(os.environ.get("N_APP_TEST_IMAGES", "20"))


# ---------------------------------------------------------------------
# Tier 1 -- pure logic, no heavy deps
# ---------------------------------------------------------------------

def test_referral_logic_matches_master_plan_rule():
    """grade >= 2 (Moderate NPDR or worse) is 'referable DR' and must
    trigger a REFER recommendation; grades 0-1 must not."""
    from app.app import GRADE_NAMES
    assert GRADE_NAMES == ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
    for grade in range(5):
        referral = "REFER (referable DR)" if grade >= 2 else "Routine annual follow-up"
        expect_refer = grade >= 2
        assert ("REFER" in referral) == expect_refer


def test_checkpoint_missing_raises_clear_error(tmp_path, monkeypatch):
    """If someone runs the app before copying the checkpoint into
    app/release/, they should get a clear instruction, not a stack trace
    from deep inside torch.load(). B3: load_model()/CHECKPOINT_PATH live in
    app.core.model now, not app.app -- patch and call them there."""
    import app.core.model as model_module
    monkeypatch.setattr(model_module, "CHECKPOINT_PATH", tmp_path / "does_not_exist.pt")
    with pytest.raises(FileNotFoundError, match="Copy the trained checkpoint"):
        model_module.load_model()


# ---------------------------------------------------------------------
# Tier 2 -- the real acceptance test (needs torch, timm, cv2, and the
# checkpoint actually copied to app/release/best_model.pt)
# ---------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="torch not installed yet")
timm = pytest.importorskip("timm", reason="timm not installed yet")
cv2 = pytest.importorskip("cv2", reason="opencv (cv2) not installed yet")

CHECKPOINT_PATH = PROJECT_ROOT / "app" / "release" / "best_model.pt"
if not CHECKPOINT_PATH.exists():
    pytest.skip(
        f"{CHECKPOINT_PATH} not found -- copy it first (from the project root):\n"
        f'  Copy-Item "checkpoints\\finetune_app_p2_seed42\\best.pt" "app\\release\\best_model.pt"',
        allow_module_level=True,
    )

from PIL import Image  # noqa: E402
# B3: to_model_input/MODEL/TRAIN_IMAGE_SIZE moved to app.core.inference /
# app.core.model. app.app still re-exports them (it imports these names at
# module level), but import from their canonical home per B3's instruction.
from app.core.inference import to_model_input  # noqa: E402
from app.core.model import MODEL, TRAIN_IMAGE_SIZE  # noqa: E402


def _exif_orientation(path):
    """Diagnostic only (not part of the pass/fail check): the raw EXIF
    Orientation tag (0x0112) on the ORIGINAL source file, or None if absent
    or the default (1, "normal"). cv2.imread() auto-applies this rotation;
    PIL's Image.open() does not unless exif_transpose() is called -- if a
    mismatch below correlates with a non-None value here, that confirms
    (rather than assumes) EXIF orientation as the mechanism, per this
    project's standing rule to investigate before adjusting anything."""
    try:
        with Image.open(path) as im:
            orientation = im.getexif().get(0x0112)
        return orientation if orientation not in (None, 1) else None
    except Exception:
        return "unreadable"


def _load_p2_test_sample(n):
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "manifests" / "manifest.csv")
    if "excluded" in manifest.columns:
        manifest = manifest[~manifest["excluded"].fillna(False).astype(bool)]
    split_map = json.loads((PROJECT_ROOT / "data" / "splits" / "p2.json").read_text())
    test_ids = [i for i, v in split_map.items() if v["fold"] == "test"]

    filepath_by_id = dict(zip(manifest["image_id"], manifest["filepath"]))
    cached_dir = PROJECT_ROOT / "data" / "processed" / str(TRAIN_IMAGE_SIZE)

    rows = []
    for image_id in test_ids:
        src = filepath_by_id.get(image_id)
        cached = cached_dir / f"{image_id}.jpg"
        if src is not None and (PROJECT_ROOT / src).exists() and cached.exists():
            rows.append((image_id, PROJECT_ROOT / src, cached))
        if len(rows) >= n:
            break
    return rows


@torch.no_grad()
def _eval_pipeline_result(cached_jpeg_path):
    """Same path as FineTuneDataset with augment=False: open the ALREADY
    cached/preprocessed image, normalize, forward pass. This is what
    finetune_converged.py's evaluate() actually did to produce the numbers
    in results/finetune_app_p2_seed42.json. Returns (grade, processed_rgb)."""
    from app.core.inference import IMAGENET_MEAN, IMAGENET_STD

    with Image.open(cached_jpeg_path) as im:
        im = im.convert("RGB")
        cached_rgb = np.array(im)
        arr = torch.from_numpy(cached_rgb).permute(2, 0, 1).float() / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    logits = MODEL(arr.unsqueeze(0))
    grade = int(torch.softmax(logits, dim=-1)[0].argmax())
    return grade, cached_rgb


@torch.no_grad()
def _app_pipeline_result(source_image_path):
    """The app's actual code path: load the ORIGINAL (uncached) image, as a
    real user upload would arrive, and run it through to_model_input() --
    the exact function app.app.predict() calls. Returns (grade, processed_rgb)."""
    with Image.open(source_image_path) as im:
        x, processed_rgb = to_model_input(im.convert("RGB"))
    logits = MODEL(x)
    grade = int(torch.softmax(logits, dim=-1)[0].argmax())
    return grade, processed_rgb


def test_app_pipeline_matches_eval_pipeline_on_real_test_images():
    rows = _load_p2_test_sample(N_APP_TEST_IMAGES)
    assert len(rows) >= 1, (
        "No P2 test images with both a source file and a cached preprocessed file were found -- "
        "check data/processed/384/ exists and data/splits/p2.json is the real split file."
    )

    mismatches = []
    for image_id, source_path, cached_path in rows:
        eval_grade, cached_rgb = _eval_pipeline_result(cached_path)
        app_grade, app_rgb = _app_pipeline_result(source_path)
        if eval_grade != app_grade:
            # Pixel-level mean absolute difference between the two PROCESSED
            # images, same shape by construction (both are TRAIN_IMAGE_SIZE
            # square). This is the evidence that tells apart two very
            # different explanations for a mismatch, instead of guessing
            # again after EXIF orientation turned out not to be it:
            #   - small MAE (~single digits out of 255): the two processed
            #     images are essentially the same crop, just decoder-noise
            #     apart (cv2/libjpeg vs PIL/libjpeg can decode the same JPEG
            #     to slightly different RGB values) -- the flip then comes
            #     from this 0.613-QWK model being genuinely non-robust to
            #     that noise on some images, which is a model-quality
            #     finding to report, not a preprocessing bug to chase.
            #   - large MAE: the crop-detection step genuinely produced a
            #     DIFFERENT region on the original vs. what's cached, which
            #     IS a real preprocessing bug (e.g. cache/source drift, or
            #     preprocess() behaving differently than when the cache was
            #     built) and needs fixing, not explaining away.
            mae = float(np.abs(cached_rgb.astype(np.int16) - app_rgb.astype(np.int16)).mean())
            mismatches.append((image_id, eval_grade, app_grade, _exif_orientation(source_path), round(mae, 1)))

    n_exif = sum(1 for m in mismatches if m[3] not in (None, "unreadable"))
    n_large_mae = sum(1 for m in mismatches if m[4] > 15)
    assert not mismatches, (
        f"Acceptance Test 12.1 FAILED: {len(mismatches)}/{len(rows)} images got a different "
        f"predicted grade from the app's pipeline vs. the evaluation pipeline.\n"
        f"  {n_exif}/{len(mismatches)} carry a non-default EXIF orientation tag (ruled out as the "
        f"sole cause if this is 0 or small).\n"
        f"  {n_large_mae}/{len(mismatches)} have pixel MAE > 15 (out of 255) between the app's "
        f"freshly-processed image and the cached one -- if that's most/all of them, the CROP "
        f"itself genuinely differs (real preprocessing/cache-drift bug); if it's 0 or few, the "
        f"crops are essentially identical and this model is simply not robust to small "
        f"decoder-level pixel noise on these images (a model-quality finding, not a code bug).\n"
        f"Mismatches (image_id, eval_grade, app_grade, exif_orientation, pixel_mae): {mismatches[:10]}"
    )
    print(f"\nAcceptance Test 12.1: {len(rows)}/{len(rows)} images matched "
          f"(app pipeline == evaluation pipeline).")


# ---------------------------------------------------------------------
# B6: decision-level Acceptance Test 12.1 -- compares the app's actual
# four-state OUTCOME (not just argmax) against the outcome computed from
# the dumped CSV logits, on the same fixed sample _load_p2_test_sample()
# already uses. Extends the argmax-only check above; does not replace or
# weaken it.
# ---------------------------------------------------------------------
def test_decision_level_acceptance_12_1():
    from app.core.model import CALIBRATION_ACTIVE, THRESHOLDS
    from app.core import decision as D

    test_csv_path = (PROJECT_ROOT / "results" /
                      "finetune_app_converged_p2_class_balanced_seed42_test_predictions.csv")
    if not test_csv_path.exists():
        pytest.skip(f"{test_csv_path} not present")

    rows = _load_p2_test_sample(N_APP_TEST_IMAGES)
    assert len(rows) >= 1, "no sample images found -- see the argmax-level test above"

    csv_df = pd.read_csv(test_csv_path)
    logit_cols = [f"logit_{k}" for k in range(5)]
    logits_by_id = dict(zip(csv_df["image_id"],
                             csv_df[logit_cols].to_numpy(dtype=np.float64)))

    argmax_mismatches = 0
    outcome_mismatches = 0
    max_abs_delta_logit = 0.0
    details = []

    for image_id, source_path, cached_path in rows:
        csv_logits = logits_by_id.get(image_id)
        if csv_logits is None:
            continue  # image not in this fold's CSV (shouldn't happen for P2 test ids)

        with torch.no_grad(), Image.open(source_path) as im:
            x, _ = to_model_input(im.convert("RGB"))
            app_logits = MODEL(x).numpy()[0].astype(np.float64)

        delta = float(np.abs(app_logits - csv_logits).max())
        max_abs_delta_logit = max(max_abs_delta_logit, delta)

        if CALIBRATION_ACTIVE:
            csv_probs = D.calibrated_probs(csv_logits, THRESHOLDS)
            app_probs = D.calibrated_probs(app_logits, THRESHOLDS)
        else:
            csv_probs = D.raw_probs(csv_logits)
            app_probs = D.raw_probs(app_logits)

        csv_outcome = D.classify_outcome(csv_probs, THRESHOLDS, CALIBRATION_ACTIVE)
        app_outcome = D.classify_outcome(app_probs, THRESHOLDS, CALIBRATION_ACTIVE)

        if int(csv_probs.argmax()) != int(app_probs.argmax()):
            argmax_mismatches += 1
        if csv_outcome != app_outcome:
            outcome_mismatches += 1
            details.append({
                "image_id": image_id, "csv_outcome": csv_outcome.value,
                "app_outcome": app_outcome.value,
            })

    n = len(rows)
    out = {
        "n_images": n,
        "argmax_mismatches": argmax_mismatches,
        "outcome_mismatches": outcome_mismatches,
        "max_abs_delta_logit": max_abs_delta_logit,
        "calibration_active_at_test_time": CALIBRATION_ACTIVE,
        "mismatch_details": details,
    }
    out_path = PROJECT_ROOT / "results" / "acceptance_12_1.json"
    out_path.write_text(json.dumps(out, indent=2))

    print(f"\nDecision-level Acceptance Test 12.1: {outcome_mismatches}/{n} outcome mismatches, "
          f"{argmax_mismatches}/{n} argmax mismatches, max|delta logit|={max_abs_delta_logit:.4f}")
    print(f"Wrote {out_path}")

    assert outcome_mismatches <= 10, (
        f"Decision-level Acceptance Test 12.1: {outcome_mismatches}/{n} outcome mismatches "
        f"exceeds the 10/100 bound. Details: {details[:10]}"
    )
