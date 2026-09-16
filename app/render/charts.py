"""
SVG chart builders for Evidence sections C1-C9.

All charts are pure Python -> SVG strings. They read from results/*.json and
results/*.csv at build time. Every chart caption states n and its source file.

R3 compliance: No numeric literals from the metrics list appear here.
Every number is read from a JSON/CSV file.
"""
from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Optional, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"


# ====================================================================
# SVG helpers — all accept a attrs dict
# ====================================================================

def _el(tag: str, attrs: dict, children: str = "") -> str:
    parts = ['%s="%s"' % (k, v) for k, v in attrs.items()]
    return '<%s %s>%s</%s>' % (tag, ' '.join(parts), children, tag)


def _svg(tag, children):
    return '<svg %s xmlns="http://www.w3.org/2000/svg">%s</svg>' % (
        ' '.join('%s="%s"' % (k, v) for k, v in tag.items()), children)


def txt(x, y, label, **kw):
    d = dict(x=x, y=y, fill="#A39A8A",
             **{"font-family": '"IBM Plex Mono", monospace',
                "font-size": "11"})
    d.update(kw)
    return _el("text", d, label)


def lne(x1, y1, x2, y2, **kw):
    d = dict(x1=x1, y1=y1, x2=x2, y2=y2,
             stroke="#3D3932", **{"stroke-width": "1"})
    d.update(kw)
    return _el("line", d)


def rct(x, y, w, h, **kw):
    d = dict(x=x, y=y, width=w, height=h, fill="#3D3932")
    d.update(kw)
    return _el("rect", d)


def cir(cx, cy, r, **kw):
    d = dict(cx=cx, cy=cy, r=r,
             fill="#8AB4FF", stroke="#3D3932", **{"stroke-width": "1"})
    d.update(kw)
    return _el("circle", d)


# ====================================================================
# C1: Waffle chart (10x10 grid from TP/FN/TN/FP)
# ====================================================================

def build_waffle_svg(tp, fn, tn, fp, with_gate=False, rejected=0, source=""):
    """10x10 waffle chart showing confusion matrix cells."""
    total = tp + fn + tn + fp
    if total == 0:
        return _svg(dict(width="200", height="120", viewBox="0 0 200 120"),
                    txt(10, 20, "No data available"))

    gw, gh, gap, pad = 16, 16, 1, 10
    svg_w = pad * 2 + 10 * (gw + gap) - gap
    svg_h = pad * 2 + 10 * (gh + gap) - gap + 60

    correct_fill = "#3FC3BC"
    error_fill = "#E8A33D"
    gate_fill = "#B3A6EC"

    cells = []
    for _ in range(tp):
        cells.append(correct_fill)
    for _ in range(fp):
        cells.append(error_fill)
    for _ in range(tn):
        cells.append(correct_fill)
    for _ in range(fn):
        cells.append(error_fill)

    if with_gate and rejected > 0 and rejected <= len(cells):
        cells = cells[:-rejected] + [gate_fill] * rejected

    p = []
    p.append('<svg width="%d" height="%d" viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg">' % (svg_w, svg_h, svg_w, svg_h))

    for idx, color in enumerate(cells):
        col = idx % 10
        row = idx // 10
        x = pad + col * (gw + gap)
        y = pad + row * (gh + gap)
        p.append(rct(x, y, gw, gh, fill=color, **{"rx": "2"}))

    ly = pad + 10 * (gh + gap) + 12
    for lbl, clr in [("Correct", correct_fill), ("Error", error_fill)]:
        p.append(rct(10, ly, 10, 10, fill=clr, **{"rx": "2"}))
        p.append(txt(24, ly + 9, lbl))

    if with_gate and rejected > 0:
        p.append(rct(10, ly + 16, 10, 10, fill=gate_fill, **{"rx": "2"}))
        p.append(txt(24, ly + 25, "Rejected (%d)" % rejected))

    p.append(txt(10, svg_h - 6, "n = %d test images . source: %s" % (total, source or "test CSV")))
    p.append("</svg>")
    return "\n".join(p)


# ====================================================================
# C1: Mini ROC curve from referral thresholds
# ====================================================================

def build_roc_svg(points, operating_point=None, source=""):
    """Mini ROC. points = list of (1-spec, 1-sens) in [0,1]."""
    if not points:
        return _svg(dict(width="200", height="150", viewBox="0 0 200 150"),
                    txt(10, 20, "ROC data unavailable"))

    pad = 30
    size = 120
    svg_w = pad * 2 + size
    svg_h = pad * 2 + size + 50

    p = []
    p.append('<svg width="%d" height="%d" viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg">' % (svg_w, svg_h, svg_w, svg_h))

    p.append(lne(pad, pad, pad, pad + size))
    p.append(lne(pad, pad + size, pad + size, pad + size))
    p.append(txt(pad - 30, pad + size // 2, "Sens", transform="rotate(-90 %d %d)" % (pad + 10, pad + size // 2)))
    p.append(txt(pad + size // 2, pad + size + 30, "1 - Spec"))
    dash = dict(**{"stroke-dasharray": "4,3"})
    p.append(lne(pad, pad + size, pad + size, pad, stroke="#6F685C", **dash))

    for i, (x, y) in enumerate(points):
        cx = pad + x * size
        cy = pad + y * size
        fill = "#E8A33D" if i == len(points) - 1 else "#3FC3BC"
        p.append(cir(cx, cy, 3, fill=fill))
        if i > 0:
            px = pad + points[i - 1][0] * size
            py = pad + points[i - 1][1] * size
            sw = dict(**{"stroke-width": "1.5"})
            p.append(lne(px, py, cx, cy, stroke="#8AB4FF", **sw))

    if operating_point:
        ox = pad + operating_point[0] * size
        oy = pad + operating_point[1] * size
        p.append(cir(ox, oy, 5, fill="#E8A33D", **{"stroke-width": "2"}))

    # points are (1-spec, 1-sens): integrating (1-sens) over d(1-spec) gives
    # the area ABOVE the real ROC curve, i.e. (1 - AUROC) -- correct for
    # that so the printed number is the actual AUROC, not its complement.
    area_above = 0
    for i in range(len(points) - 1):
        dx = (points[i + 1][0] - points[i][0])
        h = (points[i][1] + points[i + 1][1]) / 2
        area_above += dx * h
    auroc = 1 - area_above
    p.append(txt(10, svg_h - 6, "AUC ~ %.3f . source: %s" % (auroc, source or "calibration_reject.json")))
    p.append("</svg>")
    return "\n".join(p)


# ====================================================================
# C2: Reliability diagram
# ====================================================================

def build_reliability_svg(bins, dataset="All"):
    """bins = [(bin_mid, calibration, count)]."""
    if not bins:
        return _svg(dict(width="220", height="150", viewBox="0 0 220 150"),
                    txt(10, 20, "Calibration data unavailable"))

    pad = 40
    size = 120
    svg_w = pad * 2 + size + 20
    svg_h = pad * 2 + size + 30

    p = []
    p.append('<svg width="%d" height="%d" viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg">' % (svg_w, svg_h, svg_w, svg_h))

    p.append(lne(pad, pad, pad, pad + size))
    p.append(lne(pad, pad + size, pad + size, pad + size))
    p.append(txt(5, pad + size // 2, "Cal", transform="rotate(-90 %d %d)" % (pad + 10, pad + size // 2)))
    p.append(txt(pad + size // 2, pad + size + 25, "Predicted prob"))

    for i in range(11):
        v = i / 10.0
        p.append(lne(pad + v * size, pad, pad + v * size, pad + size, stroke="#2E2B26"))
        p.append(lne(pad, pad + v * size, pad + size, pad + v * size, stroke="#2E2B26"))
    p.append(lne(pad, pad + size, pad + size, pad, stroke="#2E2B26", **{"stroke-dasharray": "4,3"}))

    for mid, cal, count in bins:
        y = pad + size - cal * size
        p.append(rct(pad + mid * size - 1, y, 2, cal * size, fill="#3FC3BC", **{"rx": "1"}))

    p.append(rct(pad, pad + size + 12, 10, 10, fill="#3FC3BC", **{"rx": "2"}))
    p.append(txt(pad + 14, pad + size + 22, "After cal (%s)" % dataset))
    p.append("</svg>")
    return "\n".join(p)


# ====================================================================
# C3: Literature dot plot (AUROC comparison)
# ====================================================================

def build_literature_dot_svg(entries):
    """entries = [(label, auroc, color)]."""
    if not entries:
        return _svg(dict(width="240", height="100", viewBox="0 0 240 100"),
                    txt(10, 20, "Literature data unavailable"))

    svg_w = 240
    svg_h = 30 + len(entries) * 22
    pad_left = 110
    x_scale = 300

    p = []
    p.append('<svg width="%d" height="%d" viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg">' % (svg_w, svg_h, svg_w, svg_h))

    for i, (label, auroc, color) in enumerate(entries):
        y = 20 + i * 22
        x = pad_left + (auroc - 0.8) * x_scale
        p.append(cir(x, y, 5, fill=color or "#8AB4FF"))
        ta = dict(**{"text-anchor": "end", "font-size": "10"})
        fs = dict(**{"font-size": "10"})
        p.append(txt(pad_left - 5, y + 4, label, **ta))
        p.append(txt(x + 8, y + 4, "%.3f" % auroc, **fs))

    p.append("</svg>")
    return "\n".join(p)


# ====================================================================
# C4: Split diagram (P1 vs P2)
# ====================================================================

def build_split_diagram_svg(n_p1, n_p2):
    """n = dict(train, val, test)."""
    svg_w, svg_h = 240, 120
    p = []
    p.append('<svg width="%d" height="%d" viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg">' % (svg_w, svg_h, svg_w, svg_h))

    p.append(txt(10, 15, "P1 (image-level)"))
    p.append(txt(10, 30, "train: %d" % n_p1.get("train", 0)))
    p.append(txt(10, 45, "val:   %d" % n_p1.get("val", 0)))
    p.append(txt(10, 60, "test:  %d" % n_p1.get("test", 0)))

    p.append(txt(140, 15, "P2 (patient-level)"))
    p.append(txt(140, 30, "train: %d" % n_p2.get("train", 0)))
    p.append(txt(140, 45, "val:   %d" % n_p2.get("val", 0)))
    p.append(txt(140, 60, "test:  %d" % n_p2.get("test", 0)))

    p.append(lne(75, 37, 130, 37, stroke="#8AB4FF", **{"stroke-width": "2"}))
    p.append(txt(100, 33, "->", fill="#8AB4FF"))
    p.append("</svg>")
    return "\n".join(p)


# ====================================================================
# C5: Waterfall chart
# ====================================================================

def build_waterfall_svg(steps, source=""):
    """steps = [(label, value, color)]."""
    if not steps:
        return _svg(dict(width="200", height="80", viewBox="0 0 200 80"),
                    txt(10, 20, "Waterfall data unavailable"))

    svg_w = 220
    svg_h = 40 + len(steps) * 24
    pad_left = 80
    bar_max = 100

    p = []
    p.append('<svg width="%d" height="%d" viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg">' % (svg_w, svg_h, svg_w, svg_h))

    for i, (label, value, color) in enumerate(steps):
        y = 20 + i * 24
        w = value * bar_max
        p.append(rct(pad_left, y, max(w, 2), 14, fill=color or "#3FC3BC", **{"rx": "2"}))
        ta = dict(**{"text-anchor": "end", "font-size": "10"})
        p.append(txt(pad_left - 5, y + 11, label, **ta))
        p.append(txt(pad_left + w + 4, y + 11, "+%.3f" % value, **{"font-size": "10"}))

    p.append(txt(10, svg_h - 6, "source: %s" % source))
    p.append("</svg>")
    return "\n".join(p)


# ====================================================================
# C5: Forest plot (design levers)
# ====================================================================

def build_forest_plot_svg(levers):
    """levers = [(label, point, ci_low, ci_high, color)]."""
    if not levers:
        return _svg(dict(width="200", height="80", viewBox="0 0 200 80"),
                    txt(10, 20, "Forest plot data unavailable"))

    svg_w = 240
    svg_h = 30 + len(levers) * 22
    x_zero = 130
    x_scale = 200

    p = []
    p.append('<svg width="%d" height="%d" viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg">' % (svg_w, svg_h, svg_w, svg_h))

    for label, point, ci_low, ci_high, color in levers:
        y = 20
        x_point = x_zero + point * x_scale
        x_low = max(x_zero + ci_low * x_scale, 10)
        x_high = min(x_zero + ci_high * x_scale, svg_w - 10)
        sw = dict(**{"stroke-width": "1.5"})
        p.append(lne(x_low, y, x_high, y, stroke=color or "#8AB4FF", **sw))
        p.append(lne(x_low, y - 3, x_low, y + 3))
        p.append(lne(x_high, y - 3, x_high, y + 3))
        p.append(cir(x_point, y, 4, fill=color or "#3FC3BC"))
        p.append(txt(10, y + 4, label, **{"font-size": "10"}))

    p.append(lne(x_zero, 10, x_zero, svg_h - 10, stroke="#2E2B26", **{"stroke-dasharray": "4,3"}))
    p.append(txt(x_zero - 5, svg_h - 2, "0", **{"text-anchor": "middle", "font-size": "9"}))
    p.append("</svg>")
    return "\n".join(p)


# ====================================================================
# C7: Codebase stats strip
# ====================================================================

def build_codebase_stats_svg(n_lines, n_files, n_python, n_js):
    """Four stat chips laid out left to right. The previous version drew
    every chip's rect+text at the SAME (10,10) coordinates -- each one
    completely covered the last, so only the final chip ("js") was ever
    visible (exactly what the reported screenshot showed: one lone chip)."""
    chips = [("%d LOC" % n_lines, "#3FC3BC"),
             ("%d files" % n_files, "#8AB4FF"),
             ("%d py" % n_python, "#E8A33D"),
             ("%d js/css" % n_js, "#B3A6EC")]
    chip_w, chip_h, gap, pad = 92, 30, 10, 10
    svg_w = pad * 2 + len(chips) * chip_w + (len(chips) - 1) * gap
    svg_h = pad * 2 + chip_h

    p = [f'<svg width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}" '
         'xmlns="http://www.w3.org/2000/svg">']
    for i, (text, color) in enumerate(chips):
        x = pad + i * (chip_w + gap)
        p.append(rct(x, pad, chip_w, chip_h, fill=color, **{"rx": "6"}))
        ta = dict(**{"text-anchor": "middle", "font-size": "13", "font-weight": "600", "fill": "#0C0C0B"})
        p.append(txt(x + chip_w / 2, pad + chip_h / 2 + 5, text, **ta))
    p.append("</svg>")
    return "\n".join(p)


# ====================================================================
# C7: Architecture diagram (5 layers)
# ====================================================================

def build_arch_svg():
    """5-layer architecture stack, top (what the browser talks to) to
    bottom (frozen weights on disk). Widened and enlarged from the
    original 200x160/opacity-0.3 version, whose same-hue text-on-tinted-
    background combination read as nearly blank at the sizes this renders
    at in practice."""
    layers = [
        ("Presentation", "app.py -- Gradio Blocks/Tabs, event wiring", "#8AB4FF"),
        ("App logic", "app/core/decision.py, quality.py -- calibration, referral, gates", "#3FC3BC"),
        ("Data", "app/data/metrics.py, context.py, samples.py -- METRICS loader", "#E8A33D"),
        ("Core inference", "app/core/model.py, inference.py, gradcam.py", "#B3A6EC"),
        ("Frozen weights", "app/release/best_model.pt -- never modified at runtime", "#6F685C"),
    ]
    svg_w = 420
    bar_h, gap, pad_top = 44, 8, 10
    svg_h = pad_top * 2 + len(layers) * bar_h + (len(layers) - 1) * gap

    p = [f'<svg width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}" '
         'xmlns="http://www.w3.org/2000/svg">']
    for i, (name, detail, color) in enumerate(layers):
        y = pad_top + i * (bar_h + gap)
        p.append(rct(10, y, svg_w - 20, bar_h, fill=color, **{"rx": "6"}))
        ta_name = dict(**{"font-size": "14", "font-weight": "700", "fill": "#0C0C0B"})
        p.append(txt(22, y + 19, name, **ta_name))
        ta_detail = dict(**{"font-size": "10", "fill": "#0C0C0B", "opacity": "0.75"})
        p.append(txt(22, y + 34, detail, **ta_detail))
    p.append("</svg>")
    return "\n".join(p)


# ====================================================================
# C7: Training curve
# ====================================================================

def build_training_curve_svg(epochs, qwk_vals, best_epoch=None):
    if not epochs:
        return _svg(dict(width="200", height="100", viewBox="0 0 200 100"),
                    txt(10, 20, "Training data unavailable"))

    svg_w, svg_h = 200, 100
    pad = 30
    size = 50
    best_qwk = max(qwk_vals)

    p = []
    p.append('<svg width="%d" height="%d" viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg">' % (svg_w, svg_h, svg_w, svg_h))

    p.append(lne(pad, pad, pad, pad + size))
    p.append(lne(pad, pad + size, pad + size, pad + size))

    n = len(epochs)
    px, py = None, None
    for i in range(n):
        x = pad + (i / max(n - 1, 1)) * size
        y = pad + (1 - qwk_vals[i] / best_qwk) * size
        if px is not None:
            sw = dict(**{"stroke-width": "1.5"})
            p.append(lne(px, py, x, y, stroke="#3FC3BC", **sw))
        px, py = x, y

    if best_epoch is not None and 0 <= best_epoch < n:
        bx = pad + (best_epoch / max(n - 1, 1)) * size
        by = pad + (1 - best_qwk / best_qwk) * size
        p.append(cir(bx, by, 4, fill="#E8A33D"))
        p.append(txt(bx, by - 8, "%.3f" % best_qwk, **{"font-size": "8", "fill": "#E8A33D"}))

    p.append("</svg>")
    return "\n".join(p)


# ====================================================================
# Loaders: read JSON/CSV -> structured data for chart builders
# ====================================================================

def load_confusion_counts():
    """Return dict with tp, fn, tn, fp -- read directly from
    calibration_reject.json's referral_threshold.test_full_coverage, the
    SAME validated numbers T-1 checks exactly (TP 1116, FN 130, TN 3186,
    FP 1382). Not recomputed from the CSV here: the CSV's own
    referable_score column is RAW (pre-temperature) probability, not the
    calibrated one the app's referral rule actually uses, so recomputing
    from it under a naive argmax>=2 rule (the previous version of this
    function) silently produced a different, wrong, unvalidated number."""
    path = RESULTS_DIR / "calibration_reject.json"
    if not path.exists():
        return dict(tp=0, fn=0, tn=0, fp=0)
    data = json.loads(path.read_text())
    full_cov = data.get("referral_threshold", {}).get("test_full_coverage", {})
    return dict(
        tp=full_cov.get("tp", 0), fn=full_cov.get("fn", 0),
        tn=full_cov.get("tn", 0), fp=full_cov.get("fp", 0),
    )


def load_referral_roc_data():
    """Return (points, op, source). points = [(1-spec, sens), ...] -- a
    real ROC sweep computed from the test CSV's per-image CALIBRATED
    referral score (temperature-scaled, matching what app/core/decision.py
    actually applies at inference -- NOT the CSV's own referable_score
    column, which is raw/pre-temperature). op is the validated operating
    point from calibration_reject.json (same sens/spec T-1 checks)."""
    csv_path = RESULTS_DIR / "finetune_app_converged_p2_class_balanced_seed42_test_predictions.csv"
    thresholds_path = PROJECT_ROOT / "app" / "release" / "thresholds.json"
    cal_path = RESULTS_DIR / "calibration_reject.json"
    if not (csv_path.exists() and thresholds_path.exists()):
        return [], None, "calibration_reject.json"

    import numpy as np
    import pandas as pd

    th = json.loads(thresholds_path.read_text())
    T = th.get("temperature", 1.0)

    df = pd.read_csv(csv_path)
    logits = df[[f"logit_{k}" for k in range(5)]].to_numpy(dtype=float)
    z = logits / float(T)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    probs = e / e.sum(axis=1, keepdims=True)
    scores = probs[:, 2:].sum(axis=1)
    y_true = (df["true_grade"].to_numpy() >= 2).astype(int)

    # Sweep unique score values (downsampled to ~40 points for the SVG).
    order = np.argsort(-scores)
    thresholds = np.unique(scores)
    if len(thresholds) > 40:
        idx = np.linspace(0, len(thresholds) - 1, 40).astype(int)
        thresholds = thresholds[idx]
    points = []
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    # build_roc_svg's plotting convention (unchanged from before this fix):
    # points are (1-spec, 1-sens), with y=0 at the top of the SVG meaning
    # sens=1 -- i.e. the curve should bow toward the top-left.
    for t in sorted(thresholds):
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        sens = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        points.append((fpr, 1 - sens))
    points.sort()

    op = None
    if cal_path.exists():
        cal = json.loads(cal_path.read_text())
        full_cov = cal.get("referral_threshold", {}).get("test_full_coverage", {})
        s, sp = full_cov.get("sensitivity"), full_cov.get("specificity")
        if s is not None and sp is not None:
            op = (1 - sp, 1 - s)

    return points, op, "finetune_app_converged_..._test_predictions.csv (calibrated)"


def load_acceptance_12_1():
    path = RESULTS_DIR / "acceptance_12_1.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_claim3_decomposed():
    path = RESULTS_DIR / "claim3_decomposed_tf_efficientnet_b0_384.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_grade1_diagnosis():
    path = RESULTS_DIR / "grade1_diagnosis.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())
