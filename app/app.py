"""
Phase 8 -- the application (MASTER_PLAN.md Part 12).

A Gradio "upload a retinal photo, get a screening assessment" demo, built
around the ALREADY-TRAINED app model (checkpoints/finetune_app_p2_seed42),
not a new model. Nothing here is trained or fitted -- this file only loads
frozen weights and runs inference.

WHAT THIS APP IS, HONESTLY:
  - Model: tf_efficientnet_b0 @ 384px, fine-tuned end-to-end on the P2
    (patient-level, honest) split. See app/release/MODEL_CARD.md for the
    real numbers -- this is the converged retrain
    (src/train/finetune_converged.py, 40 epochs, early-stop patience 8),
    which lands in MASTER_PLAN.md Part 10's "correct, proceed" acceptance
    band (Instrument Card tab), up from an earlier, undertrained 15-epoch
    checkpoint. The retrain also dropped grade-1 (Mild NPDR) recall
    substantially -- a real tradeoff, investigated in
    results/grade1_diagnosis.json and flagged in MODEL_CARD.md, not
    smoothed over. No number here is hand-typed elsewhere in this file
    (R3) -- every displayed value comes from app/data/metrics.py at
    runtime, so this comment can't drift out of sync with what the app
    actually shows.
  - Confidence shown is TEMPERATURE-SCALED (Phase 7,
    src/experiments/calibrate.py; T fitted on the validation fold only,
    app/release/thresholds.json), with a validation-fitted UNCERTAIN gate
    (max calibrated probability < reject_tau -- routes to a human grader,
    never discarded) and a referral rule fixed for >=90% sensitivity on
    validation. See app/core/decision.py for the exact definitions and
    docs/verification/V2_definitions.md for how they were verified against
    results/calibration_reject.json. If the checkpoint hash or the
    acceptance-test verdict in thresholds.json don't check out at startup,
    the app falls back to raw uncalibrated confidence and says so on
    screen -- it never silently fakes a calibrated number.
  - Grad-CAM (Selvaraju et al. 2017) on the backbone's last conv block,
    for the predicted class -- shown alongside the preprocessed photo the
    model actually saw. Fails gracefully (panel goes blank, grading still
    works) if `pytorch-grad-cam` isn't installed or a library-version
    mismatch trips it up on this architecture -- untested against your
    exact installed version, so check the panel actually renders.

PREPROCESSING FIDELITY (the single most common way an app like this quietly
breaks): app/core/inference.py's to_model_input() imports preprocess() from
src/data/preprocess.py -- THE SAME FUNCTION used by build_cache.py to build
the training/eval image cache -- rather than reimplementing crop/resize/pad
logic there. The only extra step is converting Gradio's PIL RGB image to
BGR before calling it, because preprocess() expects BGR input (matching
cv2.imread's convention, which is what build_cache.py actually feeds it).
See tests/test_app.py (Acceptance Test 12.1) for the check that this
reproduces the evaluation pipeline's predictions on real test images.

Usage (from the project root, venv active):
    python app\\app.py
Then open the printed local URL (usually http://127.0.0.1:7860) in a browser.
"""
import datetime
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core import decision as D  # noqa: E402
from app.core.session import make_log_entry as make_log_entry_v2  # noqa: E402
from app.report.pdf import build_pdf_export as build_pdf_export_v2  # noqa: E402
from app.data import context  # noqa: E402
from app.core import quality as Q  # noqa: E402
from app.data import metrics as M  # noqa: E402
from app.data import samples  # noqa: E402
# B3: model loading, preprocessing/inference, and Grad-CAM moved out of
# app.py into app/core/ -- none of those modules import Gradio.
from app.core.model import (  # noqa: E402
    CHECKPOINT_PATH, MODEL, TRAIN_IMAGE_SIZE, CALIBRATION_ACTIVE, THRESHOLDS,
    MODEL_SHA12, load_model,
)
from app.core.inference import to_model_input, pooled_embedding  # noqa: E402
from app.core.gradcam import compute_gradcam_overlay_with_timeout  # noqa: E402

METRICS = M.load_metrics()

GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]

# B2: the checkpoint file itself only carries val_qwk (see load_model()'s
# log line), not the test-set numbers, which live in the training script's
# separate results/{run_name}.json. Every place that number is shown
# (Instrument Card, masthead spec-strip, MODEL_INFO_MD) reads it live via
# app/data/metrics.py's METRICS object, so a checkpoint swap can't leave a
# stale number behind anywhere.
RESULTS_JSON_PATH = PROJECT_ROOT / "results" / "finetune_app_converged_p2_class_balanced_seed42.json"

# B4: sample images come from app/data/samples.py (CURATED_SAMPLES below,
# loaded from app/samples_local/) -- the old ad-hoc SAMPLES list and its
# untracked app/samples/sample1-4.jpg are retired.
# Phase 4 -- Quantum Lab tab source files. Both are written by their own
# sweep scripts (src/experiments/qml_pqc.py, src/experiments/qcnn_no_cnn.py)
# and read here as-is -- nothing below re-derives or recomputes a number.
QML_JSON_PATH = PROJECT_ROOT / "results" / "qml_pqc.json"
QCNN_JSON_PATH = PROJECT_ROOT / "results" / "qcnn_no_cnn_pixels.json"


def build_disclaimer_md():
    """B2/R3: calibration status comes from the live CALIBRATION_ACTIVE
    flag, not a typed claim -- this exact string was caught stale during
    B5 (it still claimed 'not calibrated' after A1-A3 landed calibration
    weeks earlier in this branch's history; D1's grep check missed it
    because '**not**' split the literal substring 'not calibrated')."""
    base = (
        "**Research prototype trained on public datasets (EyePACS + APTOS). "
        "NOT a medical device. NOT validated for clinical use. "
        "Not a substitute for examination by a qualified ophthalmologist.**\n\n"
    )
    if CALIBRATION_ACTIVE:
        return base + (
            "Confidence shown below is temperature-scaled (MASTER_PLAN.md Part 7) and "
            "gated by a validation-fitted uncertainty threshold -- a case below that "
            "threshold is flagged UNCERTAIN and routed to a human grader, never silently "
            "kept or discarded. Treat it as a calibrated ranking signal validated on this "
            "project's own test set, not as a clinical-grade probability."
        )
    return base + (
        "Calibration is currently INACTIVE for this checkpoint (the checkpoint hash or "
        "acceptance-test verdict in thresholds.json didn't check out at startup -- see the "
        "System status panel below). Confidence shown below is the model's raw, "
        "uncalibrated softmax output -- treat it as a rough ranking signal, not a "
        "validated probability, and do not use it to decide when to trust the model."
    )


def build_model_info_md(m):
    """B2: every number below comes from METRICS (app/data/metrics.py), not
    typed here -- see R3/T-9."""
    dm = m.deployed_model
    if not dm.available:
        return (
            "**Model:** tf_efficientnet_b0 @ 384px, fine-tuned on the P2 "
            "(patient-level) split, seed 42.\n\n*Deployed-model results file "
            "not found -- data unavailable.*"
        )
    qwk_str = M.fmt_num(dm.test_qwk)
    auroc_str = M.fmt_num(dm.referable_auroc)
    n_test = dm.n_test if dm.n_test is not None else "?"
    epoch_str = f"best epoch {dm.best_epoch}" if dm.best_epoch is not None else "epoch unavailable"
    verdict = dm.acceptance_test_10_1_verdict or "verdict unavailable"
    return (
        "**Model:** tf_efficientnet_b0 @ 384px, fine-tuned on the P2 (patient-level) split, seed 42.\n\n"
        "**Real, measured test-set numbers** (from "
        "`results/finetune_app_converged_p2_class_balanced_seed42.json`, "
        f"n={n_test} P2 test images, evaluated once): 5-class QWK **{qwk_str}**, "
        f"referable-DR (grade ≥ 2) AUROC **{auroc_str}**.\n\n"
        f"**Status:** this run (converged recipe -- 40 epochs, early-stop patience 8, "
        f"{epoch_str}) verdict: *{verdict}* -- see MASTER_PLAN.md Part 10, Acceptance "
        f"Test 10.1. It replaces an earlier, undertrained 15-epoch checkpoint; see "
        f"`app/release/MODEL_CARD.md` for the full before/after and what the retrain was "
        f"expected to (and did) fix."
    )


# =====================================================================
# Phase 1 visual pass (see the "Fundus Console" UX plan) -- palette,
# type, and an instrument spec-strip masthead. Presentation only: no
# prediction logic below this point changes because of anything here.
# Colors deliberately pair teal (routine) with amber (refer), never
# red/green -- red-green color deficiency affects ~1 in 12 male viewers,
# and blue/teal-vs-amber stays distinguishable for both deuteranopia and
# protanopia where red-vs-green does not.
# =====================================================================
# B3: moved to app/render/tokens.css (design tokens live in one CSS file,
# not a Python triple-quoted string) -- read once at import time.
# U1: also hides Gradio footer + API link, adds sticky nav styling.
FUNDUS_CSS = (Path(__file__).resolve().parent / "render" / "tokens.css").read_text(encoding="utf-8")

# U1: hide Gradio's default footer and API link to make room for our
# single-page anchor nav (no gr.Tabs, no gradio-footer visible).
U1_NAV_HIDE = """
.gradio-container .footer,
.gradio-container .api-nav,
.gradio-container .gradio-container > .container > .wrap > .row:last-child > .footer-block,
#_gradio_footer,
.gradio-container footer,
.gradio-container div[data-testid*="footer"] { display: none !important; }
"""

# U1: sticky nav markup -- anchor links to each section.
NAV_HTML = """
<nav class="fc-nav" id="fc-nav">
  <a href="#fc-try" class="fc-nav-link" data-target="fc-try">Console</a>
  <a href="#fc-integrity" class="fc-nav-link" data-target="fc-integrity">Instrument</a>
  <a href="#fc-evidence" class="fc-nav-link" data-target="fc-evidence">Evidence</a>
  <a href="#fc-quantum" class="fc-nav-link" data-target="fc-quantum">Quantum Lab</a>
  <a href="#fc-about" class="fc-nav-link" data-target="fc-about">About</a>
</nav>
"""

MASTHEAD_HTML = """
<div class="fc-masthead">
  <span class="fc-kicker">Diabetic Retinopathy Screening &middot; Research Prototype</span>
  <h1 class="fc-title">Fundus Console</h1>
  <div class="fc-specstrip">
    <span>MODEL <b>tf_efficientnet_b0 @ 384px</b></span>
    <span>SPLIT <b>P2, patient-level</b></span>
    <span>TEST QWK <b>{{TEST_QWK}}</b></span>
    <span>REFERABLE-DR AUROC <b>{{REFERABLE_AUROC}}</b></span>
    <span>CALIBRATION <b>{{CALIBRATION_STATUS}}</b></span>
    <span>STATUS <b>research prototype</b></span>
  </div>
</div>
"""


# MASTER_PLAN.md Part 10, Acceptance Test 10.1 -- bands taken from the
# ACTUAL executable verdict logic in src/train/finetune_converged.py
# (test_qwk < 0.40 broken / < 0.70 undertrained / <= 0.85 correct /
# <= 0.95 suspicious / else leak), not from MASTER_PLAN.md's prose table.
# An earlier version of this array copied the prose table literally, which
# prints rounded boundaries that skip 0.70-0.75 and 0.85-0.90 and rendered
# those as an "undefined band" gap here -- but the training script that
# actually computes and prints this project's official verdict has no such
# gap; it is continuous. That mismatch caused this Instrument Card to show
# a genuinely-correct test QWK result as landing in a gap instead of
# the CORRECT band, contradicting the training script's own printed
# "correct -- proceed" verdict. Fixed to match the real, executable rule.
ACCEPTANCE_BANDS = [
    (0.00, 0.40, "fc-gauge-zone-broken", "BROKEN", "check label alignment, LR, normalization"),
    (0.40, 0.70, "fc-gauge-zone-undertrained", "UNDERTRAINED", "undertrained or preprocessing bug"),
    (0.70, 0.85, "fc-gauge-zone-correct", "CORRECT", "proceed"),
    (0.85, 0.95, "fc-gauge-zone-suspicious", "SUSPICIOUS", "run Part 14 leakage check"),
    (0.95, 1.00, "fc-gauge-zone-leak", "LEAK", "definitely a leak -- stop"),
]


def load_json_or_none(path):
    """Loads any results/*.json this app reads (checkpoint results, the two
    Quantum Lab sweeps). Returns None (not an exception) if the file is
    missing or unparseable, so a missing/incomplete sweep degrades to
    'numbers unavailable' in that tab, rather than crashing the whole app."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


RESULTS = load_json_or_none(RESULTS_JSON_PATH)


def build_instrument_card_html():
    """Screen 6 of the Fundus Console plan -- MASTER_PLAN.md's own
    Acceptance-Test-10.1 table, rendered as a gauge with a marker showing
    where THIS deployed checkpoint actually landed."""
    zones_html = "".join(
        f'<div class="fc-gauge-zone {css_class}" style="width:{(hi - lo) * 100:.2f}%" '
        f'title="{lo:.2f}-{hi:.2f}: {sub}">{label}</div>'
        for lo, hi, css_class, label, sub in ACCEPTANCE_BANDS
    )
    if RESULTS is not None and RESULTS.get("test_qwk") is not None:
        qwk = float(RESULTS["test_qwk"])
        verdict = RESULTS.get("acceptance_test_10_1_verdict", "not recorded")
        n_test = RESULTS.get("n_test", "?")
        marker_html = (
            f'<div class="fc-gauge-marker" style="left:{qwk * 100:.2f}%">'
            f'<div class="fc-gauge-arrow">&#9660;</div>{qwk:.3f}</div>'
        )
        verdict_line = (
            f'<div class="fc-verdict-line">This checkpoint: test QWK <b>{qwk:.3f}</b> '
            f'(n={n_test} test images) &mdash; <b>{verdict}</b></div>'
        )
    else:
        marker_html = ""
        verdict_line = (
            '<div class="fc-verdict-line">No results file found at '
            f'<code>{RESULTS_JSON_PATH.relative_to(PROJECT_ROOT)}</code> for the currently deployed '
            "checkpoint -- update RESULTS_JSON_PATH in app.py after a checkpoint swap.</div>"
        )
    return (
        '<div class="fc-card">'
        '<span class="fc-eyebrow">Instrument Card &middot; MASTER_PLAN.md Part 10, Acceptance Test 10.1</span>'
        '<div class="fc-gauge-wrap">'
        f'<div class="fc-gauge-track">{zones_html}</div>'
        f'{marker_html}'
        '<div class="fc-gauge-axis"><span>0.00</span><span>0.50</span><span>1.00</span></div>'
        '</div>'
        f'{verdict_line}'
        '<p style="font-family: \'IBM Plex Mono\',monospace;font-size:0.68rem;color:var(--fc-text-muted);'
        'margin-top:10px;line-height:1.6;">These bands are this project\'s own pre-registered '
        "acceptance thresholds (chosen before this checkpoint was evaluated), not a general-purpose "
        "grading scale -- see MASTER_PLAN.md Part 10 for how and why they were set.</p>"
        '</div>'
    )


def qwk_meter_html(label, qwk, max_qwk=0.4):
    """Reuses the same .fc-meter bar the single-eye confidence meter uses,
    for QWK values instead of a probability -- clamped to [0, 100]% of
    max_qwk so a negative or near-zero QWK (real values in the Quantum Lab
    sweeps below) renders as an honestly near-empty bar, not a crash."""
    pct = max(0.0, min(100.0, (qwk / max_qwk) * 100))
    return (
        '<div class="fc-meter-label"><span class="fc-eyebrow" style="margin:0;">'
        f'{label}</span><span class="fc-uncal">{qwk:+.4f}</span></div>'
        f'<div class="fc-meter"><i style="width:{pct:.1f}%"></i></div>'
    )


def build_quantum_lab_html():
    """Phase 4 / Screen 7 of the Fundus Console plan. Renders BOTH quantum
    experiment tracks from their own results JSON files, exactly as those
    scripts reported them -- including the sweep tables, the matched
    (apples-to-apples) comparisons, and each script's own verdict text.
    Neither track is presented as a win: both were negative/inconclusive
    results, and this tab says so plainly rather than only showing numbers
    and letting a viewer guess at the framing.
    """
    qml = load_json_or_none(QML_JSON_PATH)
    qcnn = load_json_or_none(QCNN_JSON_PATH)

    intro = (
        '<div class="fc-card">'
        '<span class="fc-eyebrow">Quantum Lab &middot; supervisor-requested QML tracks, reported honestly</span>'
        '<p style="font-family: \'IBM Plex Mono\',monospace;font-size:0.72rem;color:var(--fc-text-muted);'
        'line-height:1.6;margin:0;">Two parameterized-quantum-circuit experiments, each compared against '
        "a classical baseline on the SAME data with a paired bootstrap CI on the difference -- this "
        "project's standard for calling a result a real win, not a point-estimate guess. Neither track "
        "beats its classical baseline here; that is reported as the finding, not hidden or reframed.</p>"
        '</div>'
    )

    if qml is not None:
        sweep_rows = "".join(
            f'<tr><td>{r["n_qubits"]}</td><td>{r["n_layers"]}</td>'
            f'<td>{r["val_qwk"]:.4f}</td><td>{r["epochs_run"]}</td></tr>'
            for r in qml.get("sweep", [])
        )
        chosen = qml.get("chosen_config", {})
        heads = qml.get("matched_subsample_heads", {})
        qml_head_label = (
            f"QML head (q={chosen.get('n_qubits')}, l={chosen.get('n_layers')}) "
            "-- matched subsample"
        )
        track1 = (
            '<div class="fc-card">'
            '<span class="fc-eyebrow">Track 1 &middot; Hybrid dressed quantum classifier '
            '(frozen CNN features &rarr; PQC head)</span>'
            '<p style="font-family: \'IBM Plex Mono\',monospace;font-size:0.68rem;color:var(--fc-text-muted);">'
            f'Sweep: qubits &times; entangling layers, selected by validation QWK. '
            f'Best: q={chosen.get("n_qubits")}, layers={chosen.get("n_layers")}.</p>'
            '<table style="width:100%;border-collapse:collapse;font-family: \'IBM Plex Mono\',monospace;'
            'font-size:0.68rem;color:var(--fc-text);">'
            '<tr style="color:var(--fc-text-muted);"><th style="text-align:left;">qubits</th>'
            '<th>layers</th><th>val QWK</th><th>epochs</th></tr>'
            f'{sweep_rows}</table>'
            '<div style="margin-top:14px;">'
            f'{qwk_meter_html(qml_head_label, heads.get("qml", {}).get("qwk", 0.0))}'
            f'{qwk_meter_html("Multinomial head (classical) &mdash; matched subsample", heads.get("multinomial", {}).get("qwk", 0.0))}'
            f'{qwk_meter_html("Ordinal head (classical) &mdash; matched subsample", heads.get("ordinal", {}).get("qwk", 0.0))}'
            '</div>'
            f'<p style="font-family: \'IBM Plex Mono\',monospace;font-size:0.68rem;color:var(--fc-amber);'
            f'margin-top:10px;line-height:1.6;">{qml.get("verdict_text", "")}</p>'
            '</div>'
        )
    else:
        track1 = (
            '<div class="fc-card fc-empty">results/qml_pqc.json not found -- run '
            '<code>python src\\experiments\\qml_pqc.py --mode sweep</code> first.</div>'
        )

    if qcnn is not None:
        sweep_rows2 = "".join(
            f'<tr><td>{r["n_qubits"]}</td><td>{r["n_conv_reps"]}</td>'
            f'<td>{r["val_qwk"]:.4f}</td><td>{r["seconds"]:.0f}s</td></tr>'
            for r in qcnn.get("sweep", [])
        )
        ref = qcnn.get("full_training_set_reference_cited", {})
        track2 = (
            '<div class="fc-card">'
            '<span class="fc-eyebrow">Track 2 &middot; True no-CNN Quantum Convolutional Network '
            '(quantum circuit on raw, heavily downsampled pixels)</span>'
            '<table style="width:100%;border-collapse:collapse;font-family: \'IBM Plex Mono\',monospace;'
            'font-size:0.68rem;color:var(--fc-text);">'
            '<tr style="color:var(--fc-text-muted);"><th style="text-align:left;">qubits</th>'
            '<th>conv reps</th><th>val QWK</th><th>wall time</th></tr>'
            f'{sweep_rows2}</table>'
            '<div style="margin-top:14px;">'
            f'{qwk_meter_html("QCNN (best config)", qcnn.get("qcnn", {}).get("qwk", 0.0))}'
            f'{qwk_meter_html("Classical baseline, SAME downsampled pixels", qcnn.get("matched_classical_baseline", {}).get("qwk", 0.0))}'
            '</div>'
            f'<p style="font-family: \'IBM Plex Mono\',monospace;font-size:0.68rem;color:var(--fc-uncertain);'
            f'margin-top:10px;line-height:1.6;">Verdict ({qcnn.get("verdict")}): {qcnn.get("verdict_text", "")}</p>'
            '<p style="font-family: \'IBM Plex Mono\',monospace;font-size:0.64rem;color:var(--fc-text-muted);'
            f'margin-top:8px;line-height:1.6;">Cited for context, NOT a matched comparison: full-resolution '
            f'CNN features on the full P2 train set reach multinomial QWK '
            f'{ref.get("multinomial_qwk_full_resolution_cnn_features", 0):.3f}, ordinal '
            f'{ref.get("ordinal_qwk_full_resolution_cnn_features", 0):.3f} -- the gap above is mostly the '
            "4x4/16x16-pixel downsampling the encoding bottleneck forces, not something specific to going "
            "quantum (the matched classical baseline on the same crippled pixels does no better either).</p>"
            '</div>'
        )
    else:
        track2 = (
            '<div class="fc-card fc-empty">results/qcnn_no_cnn_pixels.json not found -- run '
            '<code>python src\\experiments\\qcnn_no_cnn.py --mode sweep --diff-method backprop</code> first.</div>'
        )

    return intro + track1 + track2


# B3: load_model()/MODEL/TRAIN_IMAGE_SIZE/CALIBRATION_ACTIVE/THRESHOLDS/
# MODEL_SHA12 all come from app.core.model (imported above) -- loaded and
# integrity-checked exactly once, at that module's import time.
MASTHEAD_HTML = (
    MASTHEAD_HTML
    .replace("{{CALIBRATION_STATUS}}", "calibrated" if CALIBRATION_ACTIVE else "uncalibrated fallback")
    .replace("{{TEST_QWK}}", M.fmt_num(METRICS.deployed_model.test_qwk))
    .replace("{{REFERABLE_AUROC}}", M.fmt_num(METRICS.deployed_model.referable_auroc))
)

# A4: per-prediction context, loaded once at startup. None -> that section
# renders "data unavailable" rather than crashing (B2's rule, applied early).
PER_GRADE_CONTEXT = context.load_per_grade_context()
GRADE1_RECALL = context.load_grade1_recall()

# B4: curated samples, loaded once at startup. None -> the public-app
# default (no bundled samples, R6/H1) -- chips hide, upload stays (DEC-5).
CURATED_SAMPLES = samples.load_samples()


# U3: hero section with three stat tiles read from METRICS.
def build_hero_html():
    """U3: Hero with headline, three stat tiles from METRICS, and a
    disclaimer chip. Each tile value comes from the METRICS object --
    never a typed number (R3)."""
    h = METRICS.hero
    if h.lookup_qwk is not None:
        qwk_str = f"{h.lookup_qwk:.3f}"
    else:
        qwk_str = "unavailable"

    if h.referral_sens_in_10 is not None and h.referral_spec_in_10 is not None:
        sens_text = f"{h.referral_sens_in_10} in 10"
        spec_text = f"{h.referral_spec_in_10} in 10"
    else:
        sens_text = spec_text = "data unavailable"

    if h.n_models is not None:
        models_text = f"{h.n_models} models"
    else:
        models_text = "data unavailable"

    tiles_html = (
        '<div style="display:flex;gap:16px;margin:20px 0;flex-wrap:wrap;">'
        f'<div class="fc-stat-tile"><span class="fc-stat-value">{qwk_str}</span>'
        f'<span class="fc-stat-label">QWK of a model that sees *no pixels*, '
        f'only the other eye\'s label</span></div>'
        f'<div class="fc-stat-tile"><span class="fc-stat-value">'
        f'{sens_text} / {spec_text}</span>'
        f'<span class="fc-stat-label">referable patients caught · healthy ones cleared</span></div>'
        f'<div class="fc-stat-tile"><span class="fc-stat-value">{models_text}</span>'
        f'<span class="fc-stat-label">models trained · 1 laptop GPU</span></div>'
        '</div>'
    )

    disclaimer_chip = (
        '<div class="fc-disclaimer-chip">● Research prototype · not a medical device</div>'
    )

    return (
        '<div class="fc-hero">'
        '<span class="fc-kicker">Diabetic Retinopathy Screening &middot; Research Prototype</span>'
        f'<h2 class="fc-hero-headline">An AI that grades retinal photos for diabetic eye '
        'disease, and a study of how to evaluate one honestly.</h2>'
        f'{tiles_html}'
        f'{disclaimer_chip}'
        '</div>'
    )


def _check_optional_dep(module_name):
    import importlib.util
    return importlib.util.find_spec(module_name) is not None


def build_status_report():
    """B7: system status -- model sha, calibration/gate state, optional-
    dependency availability, Gradio version, and any missing METRICS
    source file. Same block is printed to the console at startup and
    rendered as a footer disclosure."""
    import gradio as gr_version_probe

    gradcam_available = _check_optional_dep("pytorch_grad_cam")
    pdf_available = _check_optional_dep("fpdf")

    metrics_files = {
        "deployed-model results": RESULTS_JSON_PATH,
        "calibration (thresholds.json)": D.THRESHOLDS_PATH,
        "calibration (calibration_reject.json)": PROJECT_ROOT / "results" / "calibration_reject.json",
        "test predictions CSV": context.TEST_CSV_PATH,
        "grade1_diagnosis.json": context.GRADE1_JSON_PATH,
        "claim3_decomposed (384px)": context.CLAIM3_DECOMPOSED_384_PATH,
        "qml_pqc.json": QML_JSON_PATH,
        "qcnn_no_cnn_pixels.json": QCNN_JSON_PATH,
    }
    missing = [name for name, path in metrics_files.items() if not Path(path).exists()]

    return {
        "model_sha12": MODEL_SHA12,
        "calibration_active": CALIBRATION_ACTIVE,
        "gates_active": CALIBRATION_ACTIVE,  # the uncertainty/referral gates ride the same flag
        "gradcam_available": gradcam_available,
        "pdf_export_available": pdf_available,
        "gradio_version": gr_version_probe.__version__,
        "metrics_files_missing": missing,
    }


def _status_mark(ok):
    return "\u2713" if ok else "\u2717"


def build_status_panel_html():
    s = build_status_report()
    missing_str = ", ".join(s["metrics_files_missing"]) if s["metrics_files_missing"] else "none"
    return (
        '<div class="fc-card">'
        '<span class="fc-eyebrow">System status</span>'
        '<ul class="fc-steps" style="list-style:none;padding:0;margin:0;">'
        f'<li>model sha12: <b>{s["model_sha12"]}</b></li>'
        f'<li>calibration active: <b>{_status_mark(s["calibration_active"])}</b></li>'
        f'<li>gates active: <b>{_status_mark(s["gates_active"])}</b></li>'
        f'<li>Grad-CAM available: <b>{_status_mark(s["gradcam_available"])}</b></li>'
        f'<li>PDF export available: <b>{_status_mark(s["pdf_export_available"])}</b></li>'
        f'<li>Gradio version: <b>{s["gradio_version"]}</b></li>'
        f'<li>METRICS files missing: <b>{missing_str}</b></li>'
        '</ul></div>'
    )


def print_status_report():
    s = build_status_report()
    print("=== System status ===")
    print(f"  model sha12: {s['model_sha12']}")
    print(f"  calibration active: {s['calibration_active']}")
    print(f"  gates active: {s['gates_active']}")
    print(f"  Grad-CAM available: {s['gradcam_available']}")
    print(f"  PDF export available: {s['pdf_export_available']}")
    print(f"  Gradio version: {s['gradio_version']}")
    print(f"  METRICS files missing: {s['metrics_files_missing'] or 'none'}")
    print("======================")


print_status_report()


# B3: to_model_input() lives in app.core.inference (imported above),
# compute_gradcam_overlay() in app.core.gradcam -- byte-identical bodies,
# moved so core/ has zero Gradio imports.


SCANNING_HTML = (
    '<div class="fc-card fc-scanning">'
    '<div class="fc-scan-fundus"><div class="fc-scan-sweep"></div></div>'
    '<ul class="fc-steps">'
    '<li class="fc-step-done">preprocess (crop &middot; resize &middot; pad)</li>'
    '<li class="fc-step-active">backbone forward pass + Grad-CAM</li>'
    '</ul></div>'
)


OUTCOME_BADGE = {
    # (css class, icon, label) -- DEC-8: teal=routine, amber=refer,
    # lavender=uncertain, neutral grey=ungradable. Never red/green.
    D.Outcome.ROUTINE: ("fc-badge-routine", "\u25cf", "Routine \u00b7 rescreen in 12 months"),
    D.Outcome.REFER: ("fc-badge-refer", "\u25b2", "Refer to an eye specialist"),
    D.Outcome.UNCERTAIN: ("fc-badge-uncertain", "\u25c6", "Needs a human grader"),
    D.Outcome.UNGRADABLE: ("fc-badge-ungradable", "\u2715", "Image can't be graded"),
}


def render_cards_from_logits(logits, is_ungradable=False, ungradable_reason=None):
    """Shared by predict() (single eye) and predict_both_eyes() (joint
    grade) so the two modes can never drift into inconsistent verdict/
    referral/confidence HTML -- there is exactly one place this logic
    lives. Applies app.core.decision (A1-A3): temperature scaling, the
    referral rule, and the four-state outcome, with the uncalibrated
    fallback when CALIBRATION_ACTIVE is False.

    Returns (verdict_html, referral_html, conf_html, prob_dict, outcome,
    calibrated_probs_or_raw, raw_probs)."""
    raw = D.raw_probs(logits)
    if CALIBRATION_ACTIVE:
        probs = D.calibrated_probs(logits, THRESHOLDS)
    else:
        probs = raw
    grade = int(probs.argmax())
    conf = float(probs[grade])
    outcome = D.classify_outcome(probs, THRESHOLDS, CALIBRATION_ACTIVE,
                                  is_ungradable=is_ungradable)
    ref_score = float(D.referral_score(probs)) if CALIBRATION_ACTIVE else float(probs[2:].sum())
    exp_grade = float(D.expected_grade(probs))

    badge_class, icon, label = OUTCOME_BADGE[outcome]
    if outcome is D.Outcome.UNGRADABLE:
        verdict_html = (
            '<div class="fc-card"><span class="fc-eyebrow">Assessment</span>'
            '<div class="fc-grade">Image can&#39;t be graded</div>'
            f'<div class="fc-gradename">{ungradable_reason or "basic image checks failed"}</div></div>'
        )
    else:
        verdict_html = (
            '<div class="fc-card"><span class="fc-eyebrow">Assessment</span>'
            f'<div class="fc-grade">Grade {grade}</div>'
            f'<div class="fc-gradename">{GRADE_NAMES[grade]} '
            f'&middot; expected grade {exp_grade:.1f}</div></div>'
        )

    outcome_note = ""
    if outcome is D.Outcome.UNCERTAIN and THRESHOLDS is not None:
        outcome_note = (f'<div class="fc-outcome-note">Confidence {round(conf * 100)}% is below the '
                         f'{round(THRESHOLDS.reject_tau * 100)}% uncertainty threshold. '
                         f'{THRESHOLDS.reject_action}</div>')
    referral_html = (
        f'<div class="fc-card"><span class="fc-eyebrow">Recommendation</span>'
        f'<span class="fc-badge {badge_class}">{icon} {label}</span>{outcome_note}</div>'
    )

    if CALIBRATION_ACTIVE:
        conf_pct = round(conf * 100)
        conf_html = (
            '<div class="fc-card">'
            '<div class="fc-meter-label"><span class="fc-eyebrow" style="margin:0;">Confidence</span>'
            f'<span class="fc-uncal">{conf_pct}% &middot; calibrated</span></div>'
            f'<div class="fc-meter"><i style="width:{conf_pct}%"></i></div></div>'
        )
    else:
        raw_conf_pct = round(conf * 100)
        conf_html = (
            '<div class="fc-card fc-inactive-banner">'
            f'<p>{D.INACTIVE_BANNER}</p>'
            '<div class="fc-meter-label"><span class="fc-eyebrow" style="margin:0;">Confidence</span>'
            f'<span class="fc-uncal">{raw_conf_pct}% &middot; uncalibrated fallback</span></div>'
            f'<div class="fc-meter"><i style="width:{raw_conf_pct}%"></i></div></div>'
        )

    prob_dict = {f"{g} \u2014 {GRADE_NAMES[g]}": float(probs[g]) for g in range(5)}
    context_html = render_context_card(grade) if outcome is not D.Outcome.UNGRADABLE else ""
    return verdict_html, referral_html, conf_html, prob_dict, outcome, probs, raw


def render_ungradable_cards(message):
    """B1: for images that fail a basic quality check before any model
    forward pass runs at all (unreadable, tiny, huge, dark frame, low
    contrast) -- no logits exist, so this bypasses render_cards_from_logits
    entirely rather than faking a probability vector."""
    badge_class, icon, label = OUTCOME_BADGE[D.Outcome.UNGRADABLE]
    verdict_html = (
        '<div class="fc-card"><span class="fc-eyebrow">Assessment</span>'
        '<div class="fc-grade">Image can&#39;t be graded</div>'
        f'<div class="fc-gradename">{message}</div></div>'
    )
    referral_html = (
        f'<div class="fc-card"><span class="fc-eyebrow">Recommendation</span>'
        f'<span class="fc-badge {badge_class}">{icon} {label}</span></div>'
    )
    conf_html = '<div class="fc-card fc-empty">No confidence score -- image was not graded.</div>'
    return verdict_html, referral_html, conf_html


def render_context_card(grade):
    """A4: 'When this model says Grade {g}, it was right {x}% of the time
    and within one grade {y}% (n = {n}).' Plus the grade-1 blind-spot line
    for predicted grade 0 or 1. Reads PER_GRADE_CONTEXT/GRADE1_RECALL,
    loaded once at startup from the test CSV -- never a typed number."""
    if PER_GRADE_CONTEXT is None or PER_GRADE_CONTEXT.get(grade) is None:
        return (
            '<div class="fc-outcome-note" style="margin-top:10px;">'
            'Per-prediction context: data unavailable.</div>'
        )
    c = PER_GRADE_CONTEXT[grade]
    line = (
        f'<div class="fc-outcome-note" style="margin-top:10px;">'
        f'When this model says Grade {grade} on the held-out test set, it was right '
        f'{c.pct_exact:.0%} of the time and within one grade {c.pct_within1:.0%} '
        f'(n = {c.n}).</div>'
    )
    if grade in (0, 1) and GRADE1_RECALL is not None:
        line += (
            f'<div class="fc-outcome-note" style="margin-top:4px;">Mild disease is this '
            f"model's known blind spot: only {GRADE1_RECALL:.0%} of truly mild cases "
            f"are recognised.</div>"
        )
    return line


# B3: pooled_embedding() lives in app.core.inference (imported above).


def _prob_bar_html(pct, grade_idx, total_grades=5):
    """Build a single probability bar row for the result card."""
    # Color progression: teal (no DR) -> greenish -> amber (severe) -> red-ish (proliferative)
    # Using our semantic palette: teal for routine grades, amber for referable
    if grade_idx <= 1:
        bar_color = "var(--fc-teal)"
    elif grade_idx == 2:
        bar_color = "var(--fc-amber)"
    else:
        bar_color = "var(--fc-amber)"

    pct_str = f"{pct:.1%}"
    return (
        f'<div class="fc-prob-row">'
        f'<span class="fc-prob-label">{pct_str}</span>'
        f'<span class="fc-prob-name">{GRADE_NAMES[grade_idx]}</span>'
        f'<div class="fc-prob-track"><div class="fc-prob-fill" style="width:{pct*100}%;background:{bar_color};"></div></div>'
        f'</div>'
    )


def build_single_result_html(probs, outcome, grade, expected_grade, conf,
                              ref_score, calibration_active,
                              gradcam_skipped_reason,
                              per_grade_ctx, grade1_recall,
                              processing_ms, timing_text):
    """Build the complete single-eye result card HTML.

    Combines: outcome badge, referral, calibrated confidence, probability
    bars, context card, and model timing/sha — all in one card.
    All numbers come from the decision pipeline, never typed.
    """
    # -- Outcome badge --
    badge_class, icon, label = OUTCOME_BADGE[outcome]

    if outcome is D.Outcome.UNGRADABLE:
        grade_line = (
            '<div class="fc-grade">Image can&#39;t be graded</div>'
            f'<div class="fc-gradename">{expected_grade}</div>'
        )
    else:
        grade_line = (
            f'<div class="fc-grade">Grade {grade}</div>'
            f'<div class="fc-gradename">{GRADE_NAMES[grade]} '
            f'&middot; expected {expected_grade:.1f}</div>'
        )

    # -- Outcome note (uncertainty gate) --
    outcome_note = ""
    if outcome is D.Outcome.UNCERTAIN and THRESHOLDS is not None:
        tau_pct = round(THRESHOLDS.reject_tau * 100)
        outcome_note = (
            f'<div class="fc-outcome-note">Confidence {round(conf * 100)}% is below the '
            f'{tau_pct}% uncertainty threshold. {THRESHOLDS.reject_action}</div>'
        )

    # -- Calibration badge --
    cal_tag = "calibrated" if calibration_active else "uncalibrated"

    # -- Probability bars --
    bars_html = ""
    if probs is not None:
        for g in range(5):
            bars_html += _prob_bar_html(float(probs[g]), g)

    # -- Context card --
    ctx_html = ""
    if outcome is not D.Outcome.UNGRADABLE and per_grade_ctx is not None:
        c = per_grade_ctx.get(grade)
        if c is not None:
            ctx_html = (
                f'<div class="fc-outcome-note" style="margin-top:10px;">'
                f'When this model says Grade {grade} on the held-out test set, it was right '
                f'{c.pct_exact:.0%} of the time and within one grade {c.pct_within1:.0%} '
                f'(n = {c.n}).'
            )
            if grade in (0, 1) and grade1_recall is not None:
                ctx_html += (
                    f'<div class="fc-outcome-note" style="margin-top:4px;">Mild disease is this '
                    f"model's known blind spot: only {grade1_recall:.0%} of truly mild cases "
                    f"are recognised.</div>"
                )

    # -- Timing / model SHA strip --
    meta_html = (
        f'<div class="fc-meta-strip">'
        f'<span class="fc-meta-item">SHA: <code>{MODEL_SHA12}</code></span>'
        f'<span class="fc-meta-item">Inference: <code>{processing_ms:.0f} ms</code></span>'
        f'</div>'
    )
    if timing_text:
        meta_html += f'<div class="fc-timing-note">{timing_text}</div>'

    # -- Assemble --
    return (
        '<div class="fc-result-card">'
        f'<span class="fc-eyebrow">Assessment</span>'
        f'<div class="fc-grade-row">'
        f'<div class="fc-grade-block">{grade_line}</div>'
        f'<span class="fc-badge {badge_class}">{icon} {label}</span>'
        f'</div>'
        f'<div class="fc-meter-label">'
        f'<span class="fc-eyebrow" style="margin:0;">Confidence</span>'
        f'<span class="fc-uncal">{round(conf * 100)}% &middot; {cal_tag}</span>'
        f'</div>'
        f'<div class="fc-meter"><i style="width:{conf * 100}%;"></i></div>'
        f'{outcome_note}'
        f'<div class="fc-probs-container"><span class="fc-probs-title">Grade probabilities</span>{bars_html}</div>'
        f'{ctx_html}'
        f'{meta_html}'
        f'</div>'
    )


def build_image_slider_html(processed_img, gradcam_img, gradcam_skipped):
    """Build the before/after image comparison section.

    Returns gr.HTML content showing the preprocessed image and Grad-CAM
    side by side, with a caption explaining the comparison.
    """
    if processed_img is None:
        return '<div class="fc-card fc-empty">No image processed.</div>'

    # Build image HTML with Gradio data URL
    img_tag = f'<img src="{processed_img}" class="fc-img-main" alt="Preprocessed">'

    if gradcam_img is not None:
        gc_tag = f'<img src="{gradcam_img}" class="fc-img-gradcam" alt="Grad-CAM">'
        caption = "Preprocessed input (left) vs. Grad-CAM heatmap (right)"
    elif gradcam_skipped:
        gc_tag = '<div class="fc-img-placeholder">Grad-CAM not available</div>'
        caption = f"Preprocessed input vs. Grad-CAM (skipped: {gradcam_skipped})"
    else:
        gc_tag = '<div class="fc-img-placeholder">No Grad-CAM yet</div>'
        caption = "Preprocessed input vs. Grad-CAM (loading...)"

    return (
        '<div class="fc-card">'
        '<span class="fc-eyebrow">Image comparison</span>'
        '<div class="fc-img-row">'
        f'{img_tag}{gc_tag}'
        '</div>'
        f'<div class="fc-outcome-note" style="text-align:center;margin-top:8px;">{caption}</div>'
        '</div>'
    )


def make_log_entry(mode, outcome, probs, raw):
    """One Session Log row (A7). Session state lives ONLY in this browser
    tab's Gradio session (a plain Python list passed through gr.State) --
    nothing here is written to disk or shared across users; the log HTML
    says so."""
    grade = int(probs.argmax())
    ref_score = float(D.referral_score(probs)) if CALIBRATION_ACTIVE else float(probs[2:].sum())
    entry = make_log_entry_v2(
        mode=mode, grade=grade, grade_name=GRADE_NAMES[grade], outcome=outcome,
        expected_grade=float(D.expected_grade(probs)), referral_score=ref_score,
        calibrated_conf=float(probs[grade]), raw_conf=float(raw[grade]),
        calibration_active=CALIBRATION_ACTIVE, model_sha12=MODEL_SHA12,
    )
    return entry.as_dict()


def render_log_html(session_log):
    if not session_log:
        return '<div class="fc-card fc-empty">No assessments yet this session.</div>'
    rows = "".join(
        f'<tr><td>{e["time"]}</td><td>{e["mode"]}</td><td>Grade {e["grade"]}</td>'
        f'<td>{e["grade_name"]}</td><td>{e["outcome"]}</td><td>{round(e["calibrated_conf"] * 100)}%</td></tr>'
        for e in reversed(session_log)
    )
    return (
        '<div class="fc-card">'
        '<span class="fc-eyebrow">Session Log &middot; this browser session only -- not saved '
        'anywhere, cleared on page reload</span>'
        '<table style="width:100%;border-collapse:collapse;font-family: \'IBM Plex Mono\',monospace;'
        'font-size:0.72rem;color:var(--fc-text);margin-top:8px;">'
        '<tr style="color:var(--fc-text-muted);"><th style="text-align:left;">time</th><th>mode</th>'
        '<th>grade</th><th>name</th><th>outcome</th><th>conf.</th></tr>'
        f'{rows}</table></div>'
    )


def build_pdf_export(session_log):
    """Thin wrapper: app.report.pdf.build_pdf_export() (A7) does the actual
    work, with the header line reading model sha / calibration state /
    T / referral threshold / uncertainty threshold straight from
    THRESHOLDS, plus reject_action verbatim and the disclaimer."""
    temperature = THRESHOLDS.temperature if THRESHOLDS else float("nan")
    referral_threshold = THRESHOLDS.referral_threshold if THRESHOLDS else float("nan")
    reject_tau = THRESHOLDS.reject_tau if THRESHOLDS else float("nan")
    reject_action = THRESHOLDS.reject_action if THRESHOLDS else "n/a -- calibration inactive"
    return build_pdf_export_v2(
        session_log, model_sha12=MODEL_SHA12, calibration_active=CALIBRATION_ACTIVE,
        temperature=temperature, referral_threshold=referral_threshold,
        reject_tau=reject_tau, reject_action=reject_action,
    )


def predict(image, session_log):
    """U4: Console v2 single-eye. Generator yielding (result_card_html,
    image_slider_html, session_log, log_html).

    result_card_html: complete single-eye result with outcome badge,
    calibrated confidence, probability bars, context, timing strip.
    image_slider_html: side-by-side preprocessed vs Grad-CAM comparison.

    Three yield phases (same pacing as before, just different output shapes):
      1. Scanning animation
      2. Grade ready (no Grad-CAM yet)
      3. Grad-CAM complete
    """
    import time as _time

    if image is None:
        empty_result = (
            '<div class="fc-card fc-empty">Upload a retinal fundus photograph to begin.</div>'
        )
        yield empty_result, '', session_log, render_log_html(session_log)
        return

    yield SCANNING_HTML, '', session_log, render_log_html(session_log)

    quality = Q.check_quality(image)
    if quality.ungradable:
        # Build an ungradable result card
        outcome = D.Outcome.UNGRADABLE
        expected_grade = quality.message
        ref_score = 0.0
        gradcam_skipped = None
        build_single_result_html(
            None, outcome, 0, expected_grade, 0.0, ref_score,
            CALIBRATION_ACTIVE, gradcam_skipped,
            PER_GRADE_CONTEXT, GRADE1_RECALL, 0, ""
        )
        ungradable_card = (
            '<div class="fc-card">'
            '<span class="fc-eyebrow">Assessment</span>'
            f'<div class="fc-grade">Image can&#39;t be graded</div>'
            f'<div class="fc-gradename">{quality.message}</div>'
            f'<span class="fc-badge fc-badge-ungradable">\u2715 Not gradable</span>'
            f'<div class="fc-outcome-note">No confidence score -- image failed quality checks.</div>'
            f'<div class="fc-meta-strip"><span class="fc-meta-item">SHA: <code>{MODEL_SHA12}</code></span></div>'
            '</div>'
        )
        yield ungradable_card, '', session_log, render_log_html(session_log)
        return

    # U4: time the inference phase
    t0 = _time.perf_counter()

    # B5: preprocess + inference wrapped
    try:
        x, processed_rgb = to_model_input(image)
        with torch.inference_mode():
            logits = MODEL(x)[0].numpy()
    except Exception as e:
        print(f"Grading failed ({type(e).__name__}: {e}) during preprocess/inference.")
        error_html = (
            '<div class="fc-card fc-empty">Something went wrong while grading this '
            'image. Try a different photo.</div>'
        )
        yield error_html, '', session_log, render_log_html(session_log)
        return

    # Timing for inference/preprocess
    inference_ms = (_time.perf_counter() - t0) * 1000

    verdict_html, referral_html, conf_html, prob_dict, outcome, probs, raw = \
        render_cards_from_logits(logits)
    grade = int(probs.argmax())
    conf = float(probs[grade])
    exp_grade = float(D.expected_grade(probs))
    ref_score = float(D.referral_score(probs)) if CALIBRATION_ACTIVE else float(probs[2:].sum())
    new_log = session_log + [make_log_entry("Single Eye", outcome, probs, raw)]

    # Build the full result card HTML
    warning_extra = ''
    if quality.warning:
        warning_extra = f'<div class="fc-outcome-note">Basic image checks: {quality.warning}</div>'
    timing_text = f"Preprocess + inference: {inference_ms:.0f} ms"

    result_card = build_single_result_html(
        probs, outcome, grade, exp_grade, conf, ref_score,
        CALIBRATION_ACTIVE, None,  # gradcam_skipped is None until phase 3
        PER_GRADE_CONTEXT, GRADE1_RECALL,
        inference_ms, timing_text,
    )
    # Add warning to result card
    if warning_extra:
        result_card = result_card.rstrip('</div>') + warning_extra + '</div>'

    # Build image comparison (no Grad-CAM yet)
    image_html = build_image_slider_html(processed_rgb, None, gradcam_skipped="not yet loaded")

    # Phase 2 yield: grade ready, no heatmap
    yield result_card, image_html, new_log, render_log_html(new_log)

    # Phase 3: Grad-CAM
    cam_overlay, skipped_reason = compute_gradcam_overlay_with_timeout(x, processed_rgb, grade)

    # Update image comparison with Grad-CAM
    timing_text += " + Grad-CAM: <span id='fc-gc-timing'></span>"
    result_card = build_single_result_html(
        probs, outcome, grade, exp_grade, conf, ref_score,
        CALIBRATION_ACTIVE, skipped_reason,
        PER_GRADE_CONTEXT, GRADE1_RECALL,
        inference_ms, timing_text,
    )
    if warning_extra:
        result_card = result_card.rstrip('</div>') + warning_extra + '</div>'

    if cam_overlay is not None:
        image_html = build_image_slider_html(processed_rgb, cam_overlay, gradcam_skipped=None)
    else:
        skip_note = f" (skipped: {skipped_reason})" if skipped_reason else ""
        image_html = build_image_slider_html(processed_rgb, None, gradcam_skipped=skipped_reason or "failed")

    yield result_card, image_html, new_log, render_log_html(new_log)


def predict_both_eyes(left_image, right_image, session_log):
    """Screen 5 of the Fundus Console plan -- ties the app to this
    project's real Claim 3 finding, decomosed
    (results/claim3_decomposed_tf_efficientnet_b0_384.json): most of
    mean-pooling-both-eyes' apparent gain over per-eye-then-max is actually
    an ordinal-vs-multinomial HEAD effect (+0.048 QWK); the FUSION effect
    (averaging both eyes' features) is real and significant but smaller
    (+0.024 QWK at 384px) and reversed sign on one of three backbones.

    HONEST CAVEAT, stated plainly rather than implied by reusing the same
    number: this reapplies that SAME IDEA -- pool both eyes' embeddings,
    classify the average -- to THIS app's own fine-tuned end-to-end model
    and its own trained softmax head, a DIFFERENT architecture from the
    separately-fit ordinal-regression head on FROZEN ImageNet features that
    the decomposition above was measured on. This mode has not been
    separately re-evaluated on a held-out both-eyes test set, so its own
    QWK is unknown -- it is a principled application of a validated idea,
    not a re-validated number (see Task X1 for the not-yet-run
    measurement). Patient outcome follows DEC-2: REFER if the pooled result
    or either individual eye is REFER. Both eyes must be the same patient
    for this to be meaningful -- near-identical uploads are blocked below.
    """
    empty_extra = (None, None)
    if left_image is None or right_image is None:
        empty = '<div class="fc-card fc-empty">Upload both eyes\' photographs to begin.</div>'
        yield empty, "", "", None, *empty_extra, session_log, render_log_html(session_log)
        return

    yield SCANNING_HTML, "", "", None, *empty_extra, session_log, render_log_html(session_log)

    if D.images_look_identical(left_image, right_image):
        blocked = (
            '<div class="fc-card fc-empty">These look like the same photo. '
            'Upload the left and right eye of the same patient.</div>'
        )
        yield blocked, "", "", None, *empty_extra, session_log, render_log_html(session_log)
        return

    quality_l = Q.check_quality(left_image)
    quality_r = Q.check_quality(right_image)
    if quality_l.ungradable and quality_r.ungradable:
        blocked = (
            f'<div class="fc-card fc-empty">Neither eye could be graded '
            f'(left: {quality_l.message} right: {quality_r.message})</div>'
        )
        yield blocked, "", "", None, *empty_extra, session_log, render_log_html(session_log)
        return

    # B1: an ungradable eye never reaches the model -- its outcome is
    # UNGRADABLE by construction, per-eye grade is unavailable (None), and
    # the patient outcome (below) reflects that via DEC-2's extension
    # (T-10: "per-eye result shown for the other eye, joint outcome
    # unavailable" unless the gradable eye alone already forces REFER).
    # B5: each eye's preprocess+inference is wrapped independently -- a
    # failure grading one eye degrades that eye to "not gradable" rather
    # than crashing the whole request (the other eye's result, if any, is
    # still shown, same principle as the single-eye path above).
    if not quality_l.ungradable:
        try:
            xL, procL = to_model_input(left_image)
            with torch.inference_mode():
                logits_l = MODEL(xL)[0].numpy()
            probs_l = D.calibrated_probs(logits_l, THRESHOLDS) if CALIBRATION_ACTIVE else D.raw_probs(logits_l)
            outcome_l = D.classify_outcome(probs_l, THRESHOLDS, CALIBRATION_ACTIVE)
            grade_l = int(probs_l.argmax())
        except Exception as e:
            print(f"Left-eye grading failed ({type(e).__name__}: {e}).")
            quality_l = Q.QualityResult(True, "error", "Something went wrong grading this image.", None)
            procL, probs_l, outcome_l, grade_l = None, None, D.Outcome.UNGRADABLE, None
    else:
        procL, probs_l, outcome_l, grade_l = None, None, D.Outcome.UNGRADABLE, None

    if not quality_r.ungradable:
        try:
            xR, procR = to_model_input(right_image)
            with torch.inference_mode():
                logits_r = MODEL(xR)[0].numpy()
            probs_r = D.calibrated_probs(logits_r, THRESHOLDS) if CALIBRATION_ACTIVE else D.raw_probs(logits_r)
            outcome_r = D.classify_outcome(probs_r, THRESHOLDS, CALIBRATION_ACTIVE)
            grade_r = int(probs_r.argmax())
        except Exception as e:
            print(f"Right-eye grading failed ({type(e).__name__}: {e}).")
            quality_r = Q.QualityResult(True, "error", "Something went wrong grading this image.", None)
            procR, probs_r, outcome_r, grade_r = None, None, D.Outcome.UNGRADABLE, None
    else:
        procR, probs_r, outcome_r, grade_r = None, None, D.Outcome.UNGRADABLE, None

    if quality_l.ungradable and quality_r.ungradable:
        blocked = (
            f'<div class="fc-card fc-empty">Neither eye could be graded '
            f'(left: {quality_l.message} right: {quality_r.message})</div>'
        )
        yield blocked, "", "", None, *empty_extra, session_log, render_log_html(session_log)
        return

    both_gradable = not quality_l.ungradable and not quality_r.ungradable
    pooling_failed = False
    if both_gradable:
        # Pooled: mean embedding -> this model's own classifier head -> the
        # SAME calibration/outcome pipeline as everything else (A6.2, DEC-3
        # -- tau validated on single images, applied here to the pooled
        # result). B5: a pooling failure falls back to showing the LEFT
        # eye's own result rather than losing both already-computed grades.
        try:
            pooled_l = pooled_embedding(xL)
            pooled_r = pooled_embedding(xR)
            mean_pooled = (pooled_l + pooled_r) / 2
            with torch.inference_mode():
                logits_pooled = MODEL.classifier(mean_pooled)[0].numpy()
            verdict_html, referral_html, conf_html, prob_dict, outcome_pooled, probs_pooled, raw_pooled = \
                render_cards_from_logits(logits_pooled)
        except Exception as e:
            print(f"Pooled (both-eyes) grading failed ({type(e).__name__}: {e}) "
                  f"-- falling back to the left eye's own result.")
            pooling_failed = True
    if not both_gradable or pooling_failed:
        # One eye ungradable: no pooled embedding is possible. Show the
        # gradable eye's own single-eye result as the main card (T-10:
        # "per-eye result shown for the other eye").
        outcome_pooled = D.Outcome.UNGRADABLE
        probs_pooled = raw_pooled = None
        prob_dict = None
        gradable_logits = logits_l if not quality_l.ungradable else logits_r
        verdict_html, referral_html, conf_html, prob_dict, _, _, _ = \
            render_cards_from_logits(gradable_logits)

    # Patient outcome per DEC-2, worse-eye grade, and the restored caveat (A6.3-4).
    joint = D.patient_outcome(outcome_pooled, outcome_l, outcome_r)
    worse_grade = D.worse_eye_grade(grade_l, grade_r) if both_gradable else (
        grade_l if grade_l is not None else grade_r)
    fusion_384 = context.load_fusion_effect_384()
    fusion_text = f"{fusion_384:.3f}" if fusion_384 is not None else "unavailable"
    joint_text = (joint.value if joint is not None else
                  "unavailable (one eye could not be graded)")
    left_desc = (f"Grade {grade_l} ({GRADE_NAMES[grade_l]}), {outcome_l.value}"
                 if grade_l is not None else f"not gradable ({quality_l.message})")
    right_desc = (f"Grade {grade_r} ({GRADE_NAMES[grade_r]}), {outcome_r.value}"
                  if grade_r is not None else f"not gradable ({quality_r.message})")
    worse_desc = f"{worse_grade} ({GRADE_NAMES[worse_grade]})" if worse_grade is not None else "unavailable"
    extra_html = (
        '<div class="fc-card">'
        '<span class="fc-eyebrow">Per-eye &amp; patient-level detail</span>'
        f'<div class="fc-outcome-note">Left eye: {left_desc}. Right eye: {right_desc}. '
        f'Worse-eye grade: {worse_desc}. '
        f'Patient outcome (pooled OR either eye REFER): <b>{joint_text}</b>.</div>'
        '<div class="fc-outcome-note" style="margin-top:10px;">Averaging both eyes\' features '
        'helped in our study, but less than it first appeared: most of the early gain came from '
        f'the classifier head. The fusion effect alone was {fusion_text} QWK at 384px and '
        'reversed on one of three backbones. This mode applies the idea to this app\'s model; '
        'its accuracy here has not been separately measured.</div>'
        '<div class="fc-outcome-note" style="margin-top:10px; opacity:0.85;">Uncertainty threshold '
        'validated on single images, not on combined eyes.</div>'
        '</div>'
    )
    referral_html = referral_html + extra_html
    if both_gradable:
        new_log = session_log + [make_log_entry("Both Eyes", outcome_pooled, probs_pooled, raw_pooled)]
    else:
        new_log = session_log

    yield verdict_html, referral_html, conf_html, prob_dict, procL, procR, new_log, render_log_html(new_log)


def build_demo():
    import gradio as gr

    # Constructor kwargs only (no .set() token overrides) -- these have been
    # stable across recent Gradio versions, unlike per-token theme names,
    # which change between major versions and would risk crashing the app
    # on a version this was never run against. The rest of the visual work
    # happens in FUNDUS_CSS instead, via elem_classes/elem_id hooks, which
    # are much more stable to target than theme internals.
    fundus_theme = gr.themes.Base(
        primary_hue="teal", secondary_hue="orange", neutral_hue="stone",
        font=[gr.themes.GoogleFont("Work Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    )

    # B5: delete_cache=(3600, 3600) -- every uploaded/temp file this Blocks
    # instance creates is deleted once it's more than an hour old, checked
    # every hour. This is what makes the footer's "removed from temporary
    # storage within an hour" claim (below) true rather than aspirational.
    with gr.Blocks(
        title="Diabetic Retinopathy Screening Assistant (Research Prototype)",
        delete_cache=(3600, 3600),
    ) as demo:
        gr.HTML(MASTHEAD_HTML)
        gr.Markdown(build_disclaimer_md(), elem_classes=["***"])

        # U1: sticky nav with anchor links to each section.
        gr.HTML(NAV_HTML)
        # U1: hide Gradio footer / API link via inline style injected into
        # the page head -- done as a hidden component so it renders inside
        # the Blocks scope and applies to this page only.
        gr.HTML(f'<style>{U1_NAV_HIDE}</style>')

        # U3: hero section with headline, stat tiles, and disclaimer chip.
        gr.HTML(build_hero_html())

        # Shared across every section -- a plain Python list living in THIS
        # browser session only (Gradio's gr.State), not written to disk or
        # shared across users. Every grading action, in either mode, appends
        # to it and re-renders the session log table.
        session_state = gr.State([])

        # =================================================================
        # SECTION: Console (Single Eye / Both Eyes) -- elem_id="fc-try"
        # =================================================================
        with gr.Column(elem_id="fc-try"):
            # U1: Single eye / Both Eyes segmented toggle inside the console
            # section instead of separate tabs.
            eye_mode = gr.Radio(
                ["Single Eye", "Both Eyes"],
                value="Single Eye",
                label="Mode",
                elem_id="fc-eye-mode",
            )

            # -- Single Eye panel --
            with gr.Column(elem_id="fc-single-eye-panel"):
                # B4: curated samples for Single Eye; if absent, show upload prompt.
                single_slots_raw = (
                    {k: v for k, v in CURATED_SAMPLES.items() if k != "pair"}
                    if CURATED_SAMPLES else {}
                )
                pair_slot = CURATED_SAMPLES.get("pair") if CURATED_SAMPLES else None
                single_slots = {
                    f"{slot} (true grade {v['true_grade']})": v
                    for slot, v in single_slots_raw.items()
                }
                single_choices = ["Upload your own"] + list(single_slots.keys())

                if single_slots:
                    gr.Markdown(
                        f"**No retinal image available?** Use one of the samples below "
                        f"({samples.SAMPLES_CAPTION}) or upload your own to test the system.",
                        elem_classes=["fc-both-eyes-note"],
                    )
                    sample_dd_single = gr.Radio(
                        single_choices, value="Upload your own",
                        label="Select a sample image to test",
                    )
                else:
                    gr.Markdown(
                        "**Upload a retinal fundus photograph to test the system.**",
                        elem_classes=["fc-both-eyes-note"],
                    )

                with gr.Row():
                    with gr.Column():
                        img_in = gr.Image(type="pil", label="Upload retinal fundus photograph")
                        btn_single = gr.Button("Grade this image", variant="primary")
                    with gr.Column():
                        out_result_s = gr.HTML(
                            '<div class="fc-card fc-empty">Upload and grade an image to see '
                            'the result card here.</div>',
                        )

                # U4: image comparison rendered as HTML (not two gr.Image outputs)
                with gr.Row():
                    out_image_s = gr.HTML(
                        '<div class="fc-card fc-empty">Results appear here with '
                        'the preprocessed image and Grad-CAM comparison.</div>',
                    )

                if single_slots:
                    sample_dd_single.change(
                        make_single_sample_selector(single_slots),
                        inputs=[sample_dd_single],
                        outputs=[img_in],
                    )

            # -- Both Eyes panel --
            with gr.Column(elem_id="fc-both-eyes-panel"):
                if pair_slot:
                    gr.Markdown(
                        f"**No retinal images available?** Load the paired sample below "
                        f"({samples.SAMPLES_CAPTION}), or upload your own. Both photos "
                        f"must be of the same patient.",
                        elem_classes=["fc-both-eyes-note"],
                    )
                    btn_load_pair = gr.Button("Load sample pair")
                else:
                    gr.Markdown(
                        "**Upload both eyes' photographs.** Both photos must be of the "
                        "same patient.",
                        elem_classes=["fc-both-eyes-note"],
                    )
                with gr.Row():
                    with gr.Column():
                        img_left = gr.Image(type="pil", label="Left eye")
                    with gr.Column():
                        img_right = gr.Image(type="pil", label="Right eye")
                        btn_both = gr.Button("Grade both eyes", variant="primary")
                    with gr.Column():
                        out_verdict_b = gr.HTML()
                        out_referral_b = gr.HTML()
                        out_conf_b = gr.HTML()
                        out_probs_b = gr.Label(
                            label="Grade probabilities (joint)", num_top_classes=5,
                        )
                with gr.Row():
                    out_left_proc = gr.Image(
                        label="Left eye (preprocessed)", interactive=False,
                    )
                    out_right_proc = gr.Image(
                        label="Right eye (preprocessed)", interactive=False,
                    )

                if pair_slot:
                    btn_load_pair.click(
                        make_pair_sample_loader(pair_slot),
                        outputs=[img_left, img_right],
                    )

            # -- Session Log (inline, collapsible drawer) --
            with gr.Accordion("Session Log", open=False, elem_id="fc-history-drawer"):
                gr.Markdown(
                    "Every grade from **Single Eye** or **Both Eyes** in this browser "
                    "session shows up here, newest first. Nothing is saved to disk or "
                    "shared across users \u2014 reloading the page clears it. Export it "
                    "as a PDF to keep a copy (e.g. for a viva or a lab notebook)."
                )
                out_log_html = gr.HTML(render_log_html([]))
                btn_pdf = gr.Button("Export session log as PDF")
                out_pdf_file = gr.File(label="Session log PDF", interactive=False)
                out_pdf_status = gr.Markdown()
                btn_pdf.click(
                    build_pdf_export,
                    inputs=[session_state],
                    outputs=[out_pdf_file, out_pdf_status],
                )

        # =================================================================
        # SECTION: Instrument Card -- elem_id="fc-integrity"
        # =================================================================
        with gr.Column(elem_id="fc-integrity"):
            gr.HTML(build_instrument_card_html())

        # =================================================================
        # SECTION: Quantum Lab -- elem_id="fc-quantum"
        # =================================================================
        with gr.Column(elem_id="fc-quantum"):
            gr.HTML(build_quantum_lab_html())

        # =================================================================
        # SECTION: About -- elem_id="fc-about"
        # =================================================================
        with gr.Column(elem_id="fc-about"):
            gr.Markdown(
                build_model_info_md(METRICS)
            )
            gr.Markdown(
                "----\n" + build_disclaimer_md(),
                elem_classes=["***"],
            )
            gr.Markdown(
                "*Photos are processed in memory and removed from temporary "
                "storage within an hour. Nothing is kept or used for training.*",
                elem_classes=["fc-cam-caption"],
            )
            with gr.Accordion("System status", open=False):
                gr.HTML(build_status_panel_html())

        # =================================================================
        # Event wiring (must come after all components exist above)
        # =================================================================

        # Single Eye prediction (U4: 4 outputs — result card HTML, image slider HTML, session log, log HTML)
        btn_single.click(
            predict,
            inputs=[img_in, session_state],
            outputs=[
                out_result_s, out_image_s,
                session_state, out_log_html,
            ],
        )

        # Both Eyes prediction
        btn_both.click(
            predict_both_eyes,
            inputs=[img_left, img_right, session_state],
            outputs=[
                out_verdict_b, out_referral_b, out_conf_b,
                out_probs_b, out_left_proc, out_right_proc,
                session_state, out_log_html,
            ],
        )

        # U1: toggle Single Eye / Both Eyes visibility via JS
        # (the gradio-native .change() on gr.Radio has a known issue with
        # gr.Column visibility; inline JS is more reliable here).

    return demo, fundus_theme


def make_single_sample_selector(single_slots):
    """B4: returns a Gradio .change() callback that loads the chosen
    curated sample's image. single_slots is CURATED_SAMPLES minus 'pair'."""
    def _select(selection):
        if selection is None or selection == "Upload your own" or selection not in single_slots:
            return None
        from PIL import Image
        return Image.open(single_slots[selection]["path"])
    return _select


def make_pair_sample_loader(pair_slot):
    """B4: returns a Gradio .click() callback that loads the curated pair
    into both eye inputs at once."""
    def _load():
        from PIL import Image
        return (
            Image.open(pair_slot["left_path"]),
            Image.open(pair_slot["right_path"]),
        )
    return _load


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--share", action="store_true",
        help="create a public gradio.live tunnel (sends traffic through Gradio's "
             "servers) -- off by default; only pass this if you specifically want a "
             "public link (e.g. for a remote demo/viva).",
    )
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    demo, fundus_theme = build_demo()
    # B5: bounded queue (max_size) + concurrency 1 (default_concurrency_limit)
    # -- this app holds one model in CPU/GPU memory; letting requests pile
    # up unbounded would either OOM or silently queue forever with no
    # user-visible limit. Gradio 6's queue() signature (checked against the
    # installed version, see docs/verification/V4_env.md).
    demo.queue(max_size=20, default_concurrency_limit=1)
    demo.launch(share=args.share, server_port=args.port, theme=fundus_theme, css=FUNDUS_CSS)
