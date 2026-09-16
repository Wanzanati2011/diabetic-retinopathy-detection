"""
Phase 8 -- the application (MASTER_PLAN.md Part 12).

A Gradio "upload a retinal photo, get a screening assessment" demo, built
around the ALREADY-TRAINED app model (checkpoints/finetune_app_p2_seed42),
not a new model. Nothing here is trained or fitted -- this file only loads
frozen weights and runs inference.

WHAT THIS APP IS, HONESTLY:
  - Model: tf_efficientnet_b0 @ 384px, fine-tuned end-to-end on the P2
    (patient-level, honest) split. See app/release/MODEL_CARD.md for the
    real numbers (test QWK 0.716, referable-DR AUROC 0.912) -- this is the
    converged retrain (src/train/finetune_converged.py, 40 epochs,
    early-stop patience 8), which lands in MASTER_PLAN.md Part 10's
    0.70-0.85 "correct, proceed" band, up from an earlier 15-epoch
    checkpoint that scored 0.613 ("undertrained"). The retrain also
    dropped grade-1 (Mild NPDR) recall (0.461 -> 0.161) -- a real,
    unexplained tradeoff, flagged in MODEL_CARD.md, not smoothed over.
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
breaks): this file imports preprocess() from src/data/preprocess.py --
THE SAME FUNCTION used by build_cache.py to build the training/eval image
cache -- rather than reimplementing crop/resize/pad logic here. The only
extra step is converting Gradio's PIL RGB image to BGR before calling it,
because preprocess() expects BGR input (matching cv2.imread's convention,
which is what build_cache.py actually feeds it). See
tests/test_app.py (Acceptance Test 12.1) for the check that this
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

import cv2
import numpy as np
import torch
from PIL import ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import preprocess  # noqa: E402 -- THE SAME FUNCTION USED IN TRAINING
from app.core import decision as D  # noqa: E402
from app.core.session import make_log_entry as make_log_entry_v2  # noqa: E402
from app.report.pdf import build_pdf_export as build_pdf_export_v2  # noqa: E402
from app.data import context  # noqa: E402

GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
DEFAULT_IMAGE_SIZE = 384  # overridden by the checkpoint's own recorded config, see load_model()

CHECKPOINT_PATH = Path(__file__).resolve().parent / "release" / "best_model.pt"
# Update this alongside CHECKPOINT_PATH whenever the deployed checkpoint changes
# (e.g. after a retrain) -- the checkpoint file itself only carries val_qwk
# (see load_model()'s log line), not the test-set numbers, which live in the
# training script's separate results/{run_name}.json. Rather than hardcode a
# test QWK string that goes stale the moment the checkpoint is swapped (as
# MODEL_INFO_MD/MASTHEAD_HTML below currently do -- known, accepted debt),
# the Instrument Card tab reads this file live, so IT stays honest even if
# those older hardcoded strings are forgotten during a swap.
RESULTS_JSON_PATH = PROJECT_ROOT / "results" / "finetune_app_converged_p2_class_balanced_seed42.json"

# Sample retinal images for quick demo/testing
SAMPLES = [
    str(Path(__file__).resolve().parent / "samples" / "sample1.jpg"),
    str(Path(__file__).resolve().parent / "samples" / "sample2.jpg"),
    str(Path(__file__).resolve().parent / "samples" / "sample3.jpg"),
    str(Path(__file__).resolve().parent / "samples" / "sample4.jpg"),
]
# Phase 4 -- Quantum Lab tab source files. Both are written by their own
# sweep scripts (src/experiments/qml_pqc.py, src/experiments/qcnn_no_cnn.py)
# and read here as-is -- nothing below re-derives or recomputes a number.
QML_JSON_PATH = PROJECT_ROOT / "results" / "qml_pqc.json"
QCNN_JSON_PATH = PROJECT_ROOT / "results" / "qcnn_no_cnn_pixels.json"

DISCLAIMER = (
    "**Research prototype trained on public datasets (EyePACS + APTOS). "
    "NOT a medical device. NOT validated for clinical use. "
    "Not a substitute for examination by a qualified ophthalmologist.**\n\n"
    "Confidence shown below is the model's raw softmax output and has **not** been "
    "calibrated (temperature scaling / a validation-fitted reject threshold, "
    "MASTER_PLAN.md Part 7, has not been run for this checkpoint yet) -- treat it as "
    "a rough ranking signal, not a validated probability, and do not use it to decide "
    "when to trust the model."
)

MODEL_INFO_MD = (
    "**Model:** tf_efficientnet_b0 @ 384px, fine-tuned on the P2 (patient-level) split, seed 42.\n\n"
    "**Real, measured test-set numbers** (from "
    "`results/finetune_app_converged_p2_class_balanced_seed42.json`, n=5,814 P2 test "
    "images, evaluated once): 5-class QWK **0.716**, referable-DR (grade ≥ 2) AUROC "
    "**0.912**.\n\n"
    "**Status:** this run (converged recipe -- 40 epochs, early-stop patience 8, best "
    "epoch 32) lands in this project's own \"correct, proceed\" acceptance band "
    "(0.70-0.85 QWK) -- see MASTER_PLAN.md Part 10, Acceptance Test 10.1. It replaces an "
    "earlier 15-epoch checkpoint that scored 0.613 (undertrained band); see "
    "`app/release/MODEL_CARD.md` for the full before/after and what the retrain was "
    "expected to (and did) fix."
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
FUNDUS_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');

/* Committed to ONE explicit dark "instrument console" palette, deliberately --
   not a light/dark switch. Gradio decides its own light/dark mode through a
   mechanism that did not line up with a plain @media(prefers-color-scheme)
   check here (that's the bug from the last screenshot: light-theme text
   landing on Gradio's dark chrome, and vice versa in the disclaimer box).
   Rather than chase Gradio's internal theme-state signal version-by-version,
   every custom block below sets its OWN background AND text color together
   from this one token set, so it is legible regardless of what Gradio's
   native components are doing around it. */
:root {
  --fc-bg: #131311; --fc-surface: #1D1B17; --fc-border: #38342D;
  --fc-text: #ECE6DA; --fc-text-muted: #A79D8C;
  --fc-teal: #3FC3BC; --fc-teal-soft: #17302D;
  --fc-amber: #E8A33D; --fc-amber-soft: #392A15;
  --fc-uncertain: #A99DDF; --fc-uncertain-soft: #292440;
  --fc-ungradable: #A79D8C; --fc-ungradable-soft: #2A2823;
}

.fc-masthead {
  background: var(--fc-surface); border: 1px solid var(--fc-border); border-radius: 12px;
  padding: 20px 22px; margin-bottom: 4px;
}
.fc-kicker {
  font-family: "IBM Plex Mono", monospace; font-size: 0.72rem; letter-spacing: 0.09em;
  text-transform: uppercase; color: var(--fc-teal); display: block; margin-bottom: 8px;
}
.fc-title {
  font-family: "Fraunces", ui-serif, Georgia, serif; font-weight: 560; font-size: 2.1rem;
  margin: 0 0 16px; color: var(--fc-text);
}
.fc-specstrip { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 4px; }
.fc-specstrip span {
  font-family: "IBM Plex Mono", monospace; font-size: 0.72rem; color: var(--fc-text-muted);
  background: var(--fc-bg); border: 1px solid var(--fc-border); border-radius: 4px; padding: 4px 9px;
}
.fc-specstrip b { color: var(--fc-text); font-weight: 500; }

.fc-disclaimer {
  background: var(--fc-surface) !important; color: var(--fc-text) !important;
  border-radius: 8px; padding: 14px 16px !important; border: 1px solid var(--fc-uncertain);
}
.fc-disclaimer p, .fc-disclaimer strong { color: var(--fc-text) !important; }

.fc-card {
  background: var(--fc-surface); border: 1px solid var(--fc-border); border-radius: 10px;
  padding: 14px 16px; margin-bottom: 4px;
}
.fc-eyebrow {
  display: block; font-family: "IBM Plex Mono", monospace; font-size: 0.66rem; letter-spacing: 0.07em;
  text-transform: uppercase; color: var(--fc-text-muted); margin-bottom: 6px;
}
.fc-grade { font-family: "Fraunces", serif; font-weight: 560; font-size: 1.5rem; color: var(--fc-text); line-height: 1.1; }
.fc-gradename { color: var(--fc-text-muted); font-size: 0.95rem; margin-top: 2px; }

.fc-badge {
  display: inline-flex; align-items: center; font-family: "IBM Plex Mono", monospace;
  font-size: 0.78rem; letter-spacing: 0.02em; border-radius: 5px; padding: 5px 10px;
}
.fc-badge-refer { background: var(--fc-amber-soft); color: var(--fc-amber); }
.fc-badge-routine { background: var(--fc-teal-soft); color: var(--fc-teal); }
.fc-badge-uncertain { background: var(--fc-uncertain-soft); color: var(--fc-uncertain); }
.fc-badge-ungradable { background: var(--fc-ungradable-soft); color: var(--fc-ungradable); }
.fc-outcome-note {
  font-family: "IBM Plex Mono", monospace; font-size: 0.7rem; color: var(--fc-text-muted);
  margin-top: 8px; line-height: 1.5;
}
.fc-inactive-banner {
  border: 1px solid var(--fc-amber);
}
.fc-inactive-banner p {
  color: var(--fc-amber); font-family: "IBM Plex Mono", monospace; font-size: 0.72rem;
  margin: 0 0 10px; line-height: 1.5;
}

.fc-meter-label { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }
.fc-uncal { font-family: "IBM Plex Mono", monospace; font-size: 0.7rem; color: var(--fc-uncertain); }
.fc-meter { height: 8px; background: var(--fc-border); border-radius: 4px; overflow: hidden; }
.fc-meter > i { display: block; height: 100%; background: var(--fc-uncertain); border-radius: 4px 0 0 4px; }

.fc-empty { color: var(--fc-text-muted); font-style: italic; }

.fc-scanning { display: flex; align-items: center; gap: 18px; }
.fc-scan-fundus {
  width: 56px; height: 56px; border-radius: 50%; flex: none; position: relative; overflow: hidden;
  background: radial-gradient(circle at 38% 34%, #E8A33D 0%, #C9762A 32%, #7A3A16 72%, #401E0D 100%);
  box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
}
.fc-scan-sweep {
  position: absolute; inset: 0;
  background: linear-gradient(180deg, transparent 0%, rgba(63,195,188,0.65) 48%, transparent 56%);
  animation: fc-sweep 1.4s ease-in-out infinite;
}
@keyframes fc-sweep { 0% { transform: translateY(-100%); } 100% { transform: translateY(100%); } }
@media (prefers-reduced-motion: reduce) { .fc-scan-sweep { animation: none; top: 44%; } }
.fc-steps { list-style: none; margin: 0; padding: 0; font-family: "IBM Plex Mono", monospace; font-size: 0.72rem; }
.fc-steps li { padding: 3px 0; color: var(--fc-text-muted); }
.fc-steps li.fc-step-done { color: var(--fc-teal); }
.fc-steps li.fc-step-active { color: var(--fc-text); }

.fc-cam-caption { font-family: "IBM Plex Mono", monospace; font-size: 0.7rem; color: var(--fc-text-muted); margin-top: 4px; }

/* Instrument Card -- MASTER_PLAN.md Part 10, Acceptance Test 10.1's own
   QWK bands, rendered as a gauge instead of a table, with a marker showing
   where THIS deployed checkpoint actually landed. */
.fc-gauge-wrap { margin: 10px 0 18px; }
.fc-gauge-track {
  position: relative; height: 34px; border-radius: 6px; overflow: visible;
  display: flex; border: 1px solid var(--fc-border);
}
.fc-gauge-zone {
  height: 100%; display: flex; align-items: center; justify-content: center;
  font-family: "IBM Plex Mono", monospace; font-size: 0.62rem; letter-spacing: 0.03em;
  color: var(--fc-text-muted); border-right: 1px solid var(--fc-bg); white-space: nowrap; overflow: hidden;
}
.fc-gauge-zone:last-child { border-right: none; }
.fc-gauge-zone-broken { background: #3A1414; }
.fc-gauge-zone-undertrained { background: var(--fc-amber-soft); color: var(--fc-amber); }
.fc-gauge-zone-gap { background: var(--fc-bg); }
.fc-gauge-zone-correct { background: var(--fc-teal-soft); color: var(--fc-teal); }
.fc-gauge-zone-suspicious { background: var(--fc-uncertain-soft); color: var(--fc-uncertain); }
.fc-gauge-zone-leak { background: #3A1414; color: #E8746A; }
.fc-gauge-marker {
  position: absolute; top: -9px; transform: translateX(-50%); text-align: center;
  font-family: "IBM Plex Mono", monospace; font-size: 0.68rem; color: var(--fc-text);
}
.fc-gauge-marker .fc-gauge-arrow { color: var(--fc-text); font-size: 0.8rem; line-height: 1; }
.fc-gauge-axis { display: flex; justify-content: space-between; font-family: "IBM Plex Mono", monospace;
  font-size: 0.62rem; color: var(--fc-text-muted); margin-top: 3px; }
.fc-verdict-line { font-family: "IBM Plex Mono", monospace; font-size: 0.78rem; margin-top: 10px; }

.fc-both-eyes-note {
  font-family: "IBM Plex Mono", monospace; font-size: 0.7rem; color: var(--fc-text-muted);
  background: var(--fc-surface); border: 1px solid var(--fc-border); border-radius: 8px;
  padding: 10px 12px; margin-bottom: 8px; line-height: 1.5;
}
"""

MASTHEAD_HTML = """
<div class="fc-masthead">
  <span class="fc-kicker">Diabetic Retinopathy Screening &middot; Research Prototype</span>
  <h1 class="fc-title">Fundus Console</h1>
  <div class="fc-specstrip">
    <span>MODEL <b>tf_efficientnet_b0 @ 384px</b></span>
    <span>SPLIT <b>P2, patient-level</b></span>
    <span>TEST QWK <b>0.716</b></span>
    <span>REFERABLE-DR AUROC <b>0.912</b></span>
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
# a genuinely-correct 0.7160 test QWK result as landing in a gap instead of
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
    Acceptance-Test-10.1 table, rendered as a gauge with a marker at this
    checkpoint's REAL test QWK (read live from RESULTS, not hardcoded), so
    this stays honest across a checkpoint swap even if MASTHEAD_HTML/
    MODEL_INFO_MD's hardcoded strings above are forgotten."""
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
        '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;color:var(--fc-text-muted);'
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
        '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.72rem;color:var(--fc-text-muted);'
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
        # Built as a plain variable, not inlined as a nested f-string below --
        # an f-string delimited by " " cannot itself contain a "..." literal
        # in its {} expression on Python < 3.12 (SyntaxError), and this file
        # should run on whatever Python the venv already has.
        qml_head_label = (
            f"QML head (q={chosen.get('n_qubits')}, l={chosen.get('n_layers')}) "
            "&mdash; matched subsample"
        )
        track1 = (
            '<div class="fc-card">'
            '<span class="fc-eyebrow">Track 1 &middot; Hybrid dressed quantum classifier '
            '(frozen CNN features &rarr; PQC head)</span>'
            '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;color:var(--fc-text-muted);">'
            f'Sweep: qubits &times; entangling layers, selected by validation QWK. '
            f'Best: q={chosen.get("n_qubits")}, layers={chosen.get("n_layers")}.</p>'
            '<table style="width:100%;border-collapse:collapse;font-family:\'IBM Plex Mono\',monospace;'
            'font-size:0.68rem;color:var(--fc-text);">'
            '<tr style="color:var(--fc-text-muted);"><th style="text-align:left;">qubits</th>'
            '<th>layers</th><th>val QWK</th><th>epochs</th></tr>'
            f'{sweep_rows}</table>'
            '<div style="margin-top:14px;">'
            f'{qwk_meter_html(qml_head_label, heads.get("qml", {}).get("qwk", 0.0))}'
            f'{qwk_meter_html("Multinomial head (classical) &mdash; matched subsample", heads.get("multinomial", {}).get("qwk", 0.0))}'
            f'{qwk_meter_html("Ordinal head (classical) &mdash; matched subsample", heads.get("ordinal", {}).get("qwk", 0.0))}'
            '</div>'
            f'<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;color:var(--fc-amber);'
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
            '<table style="width:100%;border-collapse:collapse;font-family:\'IBM Plex Mono\',monospace;'
            'font-size:0.68rem;color:var(--fc-text);">'
            '<tr style="color:var(--fc-text-muted);"><th style="text-align:left;">qubits</th>'
            '<th>conv reps</th><th>val QWK</th><th>wall time</th></tr>'
            f'{sweep_rows2}</table>'
            '<div style="margin-top:14px;">'
            f'{qwk_meter_html("QCNN (best config)", qcnn.get("qcnn", {}).get("qwk", 0.0))}'
            f'{qwk_meter_html("Classical baseline, SAME downsampled pixels", qcnn.get("matched_classical_baseline", {}).get("qwk", 0.0))}'
            '</div>'
            f'<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;color:var(--fc-uncertain);'
            f'margin-top:10px;line-height:1.6;">Verdict ({qcnn.get("verdict")}): {qcnn.get("verdict_text", "")}</p>'
            '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.64rem;color:var(--fc-text-muted);'
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


MODEL, TRAIN_IMAGE_SIZE = load_model()

# A1: calibration integrity check, once at startup. If the checkpoint hash,
# thresholds file, or acceptance-test verdict don't check out, the app
# falls back to raw uncalibrated confidence and says so on screen -- it
# never fakes a calibrated number it hasn't actually verified.
CALIBRATION_ACTIVE, THRESHOLDS, _CALIBRATION_INACTIVE_REASON = D.check_integrity()
MODEL_SHA12 = D.checkpoint_sha256()[:12] if CHECKPOINT_PATH.exists() else "unknown"
print(f"Calibration active: {CALIBRATION_ACTIVE}"
      f"{'' if CALIBRATION_ACTIVE else f' (reason: {_CALIBRATION_INACTIVE_REASON})'}"
      f"  model sha12={MODEL_SHA12}")
MASTHEAD_HTML = MASTHEAD_HTML.replace(
    "{{CALIBRATION_STATUS}}", "calibrated" if CALIBRATION_ACTIVE else "uncalibrated fallback")


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
    D.Outcome.ROUTINE: ("fc-badge-routine", "●", "Routine · rescreen in 12 months"),
    D.Outcome.REFER: ("fc-badge-refer", "▲", "Refer to an eye specialist"),
    D.Outcome.UNCERTAIN: ("fc-badge-uncertain", "◆", "Needs a human grader"),
    D.Outcome.UNGRADABLE: ("fc-badge-ungradable", "✕", "Image can't be graded"),
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
            f'<div class="fc-grade">Image can&#39;t be graded</div>'
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

    prob_dict = {f"{g} — {GRADE_NAMES[g]}": float(probs[g]) for g in range(5)}
    return verdict_html, referral_html, conf_html, prob_dict, outcome, probs, raw


def pooled_embedding(x):
    """The penultimate embedding (after global pooling, before the final
    classifier Linear layer) for one preprocessed image -- timm's standard
    forward_features -> forward_head(pre_logits=True) split, stable across
    the EfficientNet family. This is THIS model's own analogue of the
    frozen per-eye feature vectors src/experiments/claim3_both_eyes.py mean-
    pooled (features/{backbone}_{size}.npz) -- same idea (pool each eye's
    embedding, average, classify the average), applied live to this app's
    own fine-tuned end-to-end network rather than to a separately-fit head
    on frozen ImageNet features. See predict_both_eyes()'s docstring for
    why the numbers are NOT a reproduction of Claim 3's reported QWK."""
    with torch.no_grad():
        feats = MODEL.forward_features(x)
        return MODEL.forward_head(feats, pre_logits=True)


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
        '<table style="width:100%;border-collapse:collapse;font-family:\'IBM Plex Mono\',monospace;'
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
    """A generator, not a plain function: Gradio streams each yielded tuple
    to the UI as it arrives, which is what makes the single scan-sweep
    frame in Screen 02 of the design plan real rather than a fake spinner.
    Nothing is padded to feel slower -- if the forward pass is fast, the
    reveal is fast; the point was pacing the animation to real work, not
    manufacturing a delay. session_log is a gr.State list, threaded through
    as both input and output so every grade gets appended to the Session
    Log tab regardless of which tab produced it."""
    if image is None:
        empty = '<div class="fc-card fc-empty">Upload a retinal fundus photograph to begin.</div>'
        yield empty, "", "", None, None, None, session_log, render_log_html(session_log)
        return

    yield SCANNING_HTML, "", "", None, None, None, session_log, render_log_html(session_log)

    x, processed_rgb = to_model_input(image)
    with torch.no_grad():
        logits = MODEL(x)[0].numpy()
    verdict_html, referral_html, conf_html, prob_dict, outcome, probs, raw = \
        render_cards_from_logits(logits)
    grade = int(probs.argmax())
    cam_overlay = compute_gradcam_overlay(x, processed_rgb, grade)
    new_log = session_log + [make_log_entry("Single Eye", outcome, probs, raw)]

    yield verdict_html, referral_html, conf_html, prob_dict, processed_rgb, cam_overlay, new_log, render_log_html(new_log)


def predict_both_eyes(left_image, right_image, session_log):
    """Screen 5 of the Fundus Console plan -- ties the app to this
    project's real Claim 3 finding, decomposed
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
        blocked = ('<div class="fc-card fc-empty">These look like the same photo. '
                   'Upload the left and right eye of the same patient.</div>')
        yield blocked, "", "", None, *empty_extra, session_log, render_log_html(session_log)
        return

    xL, procL = to_model_input(left_image)
    xR, procR = to_model_input(right_image)

    # Per-eye: same pipeline as Single Eye, independently on each image (A6.1).
    with torch.no_grad():
        logits_l = MODEL(xL)[0].numpy()
        logits_r = MODEL(xR)[0].numpy()
    probs_l = D.calibrated_probs(logits_l, THRESHOLDS) if CALIBRATION_ACTIVE else D.raw_probs(logits_l)
    probs_r = D.calibrated_probs(logits_r, THRESHOLDS) if CALIBRATION_ACTIVE else D.raw_probs(logits_r)
    outcome_l = D.classify_outcome(probs_l, THRESHOLDS, CALIBRATION_ACTIVE)
    outcome_r = D.classify_outcome(probs_r, THRESHOLDS, CALIBRATION_ACTIVE)
    grade_l, grade_r = int(probs_l.argmax()), int(probs_r.argmax())

    # Pooled: mean embedding -> this model's own classifier head -> the
    # SAME calibration/outcome pipeline as everything else (A6.2, DEC-3 --
    # tau validated on single images, applied here to the pooled result).
    pooled_l = pooled_embedding(xL)
    pooled_r = pooled_embedding(xR)
    mean_pooled = (pooled_l + pooled_r) / 2
    with torch.no_grad():
        logits_pooled = MODEL.classifier(mean_pooled)[0].numpy()
    verdict_html, referral_html, conf_html, prob_dict, outcome_pooled, probs_pooled, raw_pooled = \
        render_cards_from_logits(logits_pooled)

    # Patient outcome per DEC-2, worse-eye grade, and the restored caveat (A6.3-4).
    joint = D.patient_outcome(outcome_pooled, outcome_l, outcome_r)
    worse_grade = D.worse_eye_grade(grade_l, grade_r)
    fusion_384 = context.load_fusion_effect_384()
    fusion_text = f"{fusion_384:.3f}" if fusion_384 is not None else "unavailable"
    joint_text = (joint.value if joint is not None else
                  "unavailable (one eye could not be graded)")
    extra_html = (
        '<div class="fc-card">'
        '<span class="fc-eyebrow">Per-eye &amp; patient-level detail</span>'
        f'<div class="fc-outcome-note">Left eye: Grade {grade_l} ({GRADE_NAMES[grade_l]}), '
        f'{outcome_l.value}. Right eye: Grade {grade_r} ({GRADE_NAMES[grade_r]}), '
        f'{outcome_r.value}. Worse-eye grade: {worse_grade} ({GRADE_NAMES[worse_grade]}). '
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
    new_log = session_log + [make_log_entry("Both Eyes", outcome_pooled, probs_pooled, raw_pooled)]

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

    # theme= and css= are passed to demo.launch() below, not here -- Gradio 6.0
    # moved them from the Blocks constructor to launch() (this app warned about
    # exactly that the first time it ran); build_demo() hands the theme back to
    # __main__ so it can be passed where this installed version actually wants it.
    with gr.Blocks(
        title="Diabetic Retinopathy Screening Assistant (Research Prototype)",
    ) as demo:
        gr.HTML(MASTHEAD_HTML)
        gr.Markdown(DISCLAIMER, elem_classes=["fc-disclaimer"])

        # Shared across every tab -- a plain Python list living in THIS
        # browser session only (Gradio's gr.State), not written to disk or
        # shared across users. Every grading action, in either tab, appends
        # to it and re-renders the Session Log tab's table.
        session_state = gr.State([])

        with gr.Tabs():
            with gr.Tab("Single Eye"):
                gr.Markdown(
                    "**No retinal image available?** Use one of the samples below "
                    "or upload your own to test the system.",
                    elem_classes=["fc-both-eyes-note"],
                )
                with gr.Row():
                    sample_dd = gr.Radio(
                        ["Upload your own", "Sample 1", "Sample 2", "Sample 3", "Sample 4"],
                        value="Sample 2",
                        label="Select a sample image to test",
                    )
                with gr.Row():
                    with gr.Column():
                        img_in = gr.Image(type="pil", value=SAMPLES[1], label="Upload retinal fundus photograph")
                        btn = gr.Button("Grade this image", variant="primary")
                    with gr.Column():
                        out_verdict = gr.HTML()
                        out_referral = gr.HTML()
                        out_conf = gr.HTML()
                        out_probs = gr.Label(label="Grade probabilities", num_top_classes=5)
                gr.Markdown("### Evidence")
                with gr.Row():
                    out_original = gr.Image(label="What the model actually saw (preprocessed)", interactive=False)
                    out_cam = gr.Image(label="Grad-CAM — where the model looked", interactive=False)
                gr.Markdown(
                    "*Blank if Grad-CAM isn't installed (`pip install grad-cam`) or fails on this model/"
                    "library combination — grading itself never depends on it.*",
                    elem_classes=["fc-cam-caption"],
                )
                sample_dd.change(update_single_sample, inputs=[sample_dd], outputs=[img_in])
            with gr.Tab("Both Eyes"):
                gr.Markdown(
                    "**No retinal images available?** Use the samples below for left and right eyes, "
                    "or upload your own. Both photos must be of the same patient.",
                    elem_classes=["fc-both-eyes-note"],
                )
                with gr.Row():
                    with gr.Column():
                        sample_dd_left = gr.Radio(
                            ["Upload your own", "Sample 1", "Sample 2", "Sample 3", "Sample 4"],
                            value="Sample 2",
                            label="Left eye sample",
                        )
                        img_left = gr.Image(type="pil", value=SAMPLES[1], label="Left eye")
                    with gr.Column():
                        sample_dd_right = gr.Radio(
                            ["Upload your own", "Sample 1", "Sample 2", "Sample 3", "Sample 4"],
                            value="Sample 2",
                            label="Right eye sample",
                        )
                        img_right = gr.Image(type="pil", value=SAMPLES[1], label="Right eye")
                        btn_both = gr.Button("Grade both eyes", variant="primary")
                    with gr.Column():
                        out_verdict_b = gr.HTML()
                        out_referral_b = gr.HTML()
                        out_conf_b = gr.HTML()
                        out_probs_b = gr.Label(label="Grade probabilities (joint)", num_top_classes=5)
                with gr.Row():
                    out_left_proc = gr.Image(label="Left eye (preprocessed)", interactive=False)
                    out_right_proc = gr.Image(label="Right eye (preprocessed)", interactive=False)
                sample_dd_left.change(update_single_sample, inputs=[sample_dd_left], outputs=[img_left])
                sample_dd_right.change(update_single_sample, inputs=[sample_dd_right], outputs=[img_right])

            with gr.Tab("Instrument Card"):
                gr.HTML(build_instrument_card_html())

            with gr.Tab("Quantum Lab"):
                gr.HTML(build_quantum_lab_html())

            with gr.Tab("Session Log"):
                gr.Markdown(
                    "Every grade from **Single Eye** or **Both Eyes** in this browser session shows up "
                    "here, newest first. Nothing is saved to disk or shared across users -- reloading "
                    "the page clears it. Export it as a PDF to keep a copy (e.g. for a viva or a lab "
                    "notebook)."
                )
                out_log_html = gr.HTML(render_log_html([]))
                btn_pdf = gr.Button("Export session log as PDF")
                out_pdf_file = gr.File(label="Session log PDF", interactive=False)
                out_pdf_status = gr.Markdown()
                btn_pdf.click(
                    build_pdf_export, inputs=[session_state], outputs=[out_pdf_file, out_pdf_status],
                )

        # Wired here, after every tab's components exist, rather than inline
        # inside each gr.Tab(...) block -- Single Eye's and Both Eyes'
        # predictions both need to write to out_log_html, which lives in the
        # Session Log tab defined afterwards.
        btn.click(
            predict, inputs=[img_in, session_state],
            outputs=[out_verdict, out_referral, out_conf, out_probs, out_original, out_cam,
                     session_state, out_log_html],
        )
        btn_both.click(
            predict_both_eyes, inputs=[img_left, img_right, session_state],
            outputs=[out_verdict_b, out_referral_b, out_conf_b, out_probs_b,
                     out_left_proc, out_right_proc, session_state, out_log_html],
        )

        gr.Markdown("---\n" + MODEL_INFO_MD)
    return demo, fundus_theme


def update_single_sample(selection):
    """Return the sample file path for the selected option."""
    if selection is None or selection == "Upload your own":
        return None
    idx = int(selection.split()[-1]) - 1
    return SAMPLES[idx]


def update_both_samples(selection_left, selection_right):
    """Return sample file paths for left and right eye selection."""
    def resolve(sel):
        if sel is None or sel == "Upload your own":
            return None
        idx = int(sel.split()[-1]) - 1
        return SAMPLES[idx]
    return resolve(selection_left), resolve(selection_right)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true",
                     help="create a public gradio.live tunnel (sends traffic through Gradio's "
                          "servers) -- off by default; only pass this if you specifically want a "
                          "public link (e.g. for a remote demo/viva).")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    demo, fundus_theme = build_demo()
    demo.launch(share=args.share, server_port=args.port, theme=fundus_theme, css=FUNDUS_CSS)
