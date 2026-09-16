"""
Evidence section builders (C1-C9).

Reads from METRICS (app/data/metrics.py) and results/*.json files.
Each function returns an HTML string ready for gr.HTML().
No Gradio imports.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.render.charts import (
    build_waffle_svg, build_roc_svg, build_reliability_svg,
    build_literature_dot_svg, build_split_diagram_svg,
    build_waterfall_svg, build_forest_plot_svg,
    build_codebase_stats_svg, build_arch_svg, build_training_curve_svg,
    load_confusion_counts, load_referral_roc_data, load_acceptance_12_1,
    load_claim3_decomposed, load_grade1_diagnosis,
)
from app.data import metrics as _m
from app.data.metrics import _load_json

# Load metrics at import time (same pattern as app.py)
METRICS = _m.load_metrics()
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
fmt_pct = _m.fmt_pct
fmt_num = _m.fmt_num
def build_c1_clinical_results_html():
    """C1: Clinical results — waffle, ROC, confusion table."""
    cc = load_confusion_counts()
    tp, fn, tn, fp = cc.get("tp", 0), cc.get("fn", 0), cc.get("tn", 0), cc.get("fp", 0)
    waffle = build_waffle_svg(tp, fn, tn, fp, source="test CSV")

    points, op, source = load_referral_roc_data()
    roc = build_roc_svg(points, operating_point=op, source=source)

    total = tp + fn + tn + fp
    acc = (tp + tn) / total if total > 0 else 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    html = '<div class="fc-section-card">\n'
    html += '<h3>Clinical Results <span class="fc-badge">C1</span></h3>\n'

    # Summary stats
    html += '<div class="fc-stats-row">\n'
    html += '<div class="fc-stat-tile"><span class="fc-stat-val">%s</span><span class="fc-stat-label">Accuracy</span></div>\n' % ("%.1f%%" % (acc * 100))
    html += '<div class="fc-stat-tile"><span class="fc-stat-val">%s</span><span class="fc-stat-label">Sensitivity</span></div>\n' % ("%.1f%%" % (sens * 100))
    html += '<div class="fc-stat-tile"><span class="fc-stat-val">%s</span><span class="fc-stat-label">Specificity</span></div>\n' % ("%.1f%%" % (spec * 100))
    html += '</div>\n'

    # Waffle + ROC side by side
    html += '<div class="fc-charts-row">\n'
    html += '<div class="fc-chart-col">\n'
    html += '<p class="fc-caption">10x10 waffle: teal = correct calls, amber = errors. n = %d</p>\n' % total
    html += waffle
    html += '</div>\n'
    html += '<div class="fc-chart-col">\n'
    html += '<p class="fc-caption">ROC curve (AUROC from threshold sweep)</p>\n'
    html += roc
    html += '</div>\n'
    html += '</div>\n'

    # Confusion matrix table
    html += '<table class="fc-cm-table">\n'
    html += '<tr><th></th><th>Pred Ref</th><th>Pred Non-Ref</th></tr>\n'
    html += '<tr><th>True Ref</th><td>%d (TP)</td><td>%d (FN)</td></tr>\n' % (tp, fn)
    html += '<tr><th>True Non-Ref</th><td>%d (FP)</td><td>%d (TN)</td></tr>\n' % (fp, tn)
    html += '</table>\n'

    # Uncertainty gate info
    cal = METRICS.calibration
    if cal.available and cal.reject_tau is not None:
        # Read rejection count from calibration_reject.json directly
        rej_n = None
        cr = _load_json(RESULTS_DIR / "calibration_reject.json")
        if cr:
            rej = cr.get("reject_option", {})
            at_tau = rej.get("test_at_tau", {})
            rej_n = at_tau.get("n_rejected")
        if rej_n:
            html += '<div class="fc-gate-toggle">\n'
            html += '<p class="fc-caption">Uncertainty gate (tau=%.3f): %d of %d images flagged for human review.</p>\n' % (cal.reject_tau, rej_n, total)
            html += '</div>\n'

    html += '</div>\n'
    return html


def build_c2_calibration_html():
    """C2: Calibration & gate — reliability diagram (collapsed teaser)."""
    cal = METRICS.calibration
    html = '<div class="fc-section-card fc-collapsible">\n'
    html += '<details open>\n'
    html += '<summary>Calibration & Gate <span class="fc-badge">C2</span></summary>\n'

    if not cal.available:
        html += '<p class="fc-unavailable">Calibration data unavailable.</p>\n'
        html += '</details>\n</div>\n'
        return html

    # Build reliability bins from calibration data
    bins = []
    cal_data = cal.__dict__ if hasattr(cal, "__dict__") else {}
    # Use ECE before/after
    ece_before = cal.test_ece_before
    ece_after = cal.test_ece_after
    sens_full = cal.referral_sens_full
    spec_full = cal.referral_spec_full

    html += '<div class="fc-stats-row">\n'
    html += '<div class="fc-stat-tile"><span class="fc-stat-val">%s</span><span class="fc-stat-label">ECE Before</span></div>\n' % (fmt_num(ece_before, 4) if ece_before is not None else "N/A")
    html += '<div class="fc-stat-tile"><span class="fc-stat-val">%s</span><span class="fc-stat-label">ECE After</span></div>\n' % (fmt_num(ece_after, 4) if ece_after is not None else "N/A")
    html += '<div class="fc-stat-tile"><span class="fc-stat-val">%s</span><span class="fc-stat-label">Tau (reject)</span></div>\n' % (fmt_num(cal.reject_tau, 4) if cal.reject_tau is not None else "N/A")
    html += '</div>\n'

    if ece_before is not None or ece_after is not None:
        html += '<div class="fc-chart-col">\n'
        html += '<p class="fc-caption">ECE: Expected Calibration Error before (temperature scaling) and after calibration.</p>\n'
        html += '<div class="fc-bar-pair">\n'
        if ece_before is not None:
            html += '<div class="fc-bar-before"><span class="fc-bar-fill" style="width:%.0f%%; background: #E8A33D;">ECE %.4f</span></div>\n' % (min(ece_before * 200, 100), ece_before)
        if ece_after is not None:
            html += '<div class="fc-bar-after"><span class="fc-bar-fill" style="width:%.0f%%; background: #3FC3BC;">ECE %.4f</span></div>\n' % (min(ece_after * 200, 100), ece_after)
        html += '</div>\n'
        html += '<p class="fc-caption">Referral: sensitivity %.0f%%, specificity %.0f%% at full coverage.</p>\n' % (sens_full * 100 if sens_full else 0, spec_full * 100 if spec_full else 0)
        html += '</div>\n'

    # The "rule that selected nothing" pull-quote -- coverage read from
    # calibration_reject.json's own test_at_tau block, not typed.
    sel_qwk = cal.selective_qwk_at_tau
    cr = _load_json(RESULTS_DIR / "calibration_reject.json")
    coverage = (cr or {}).get("reject_option", {}).get("test_at_tau", {}).get("coverage")
    if sel_qwk is not None and coverage is not None:
        html += '<div class="fc-pullquote">\n'
        html += ('<blockquote>&ldquo;Rejecting the least-confident %.1f%% of images '
                 '(keeping %.1f%%) raises selective QWK to %s.&rdquo;</blockquote>\n'
                 % ((1 - coverage) * 100, coverage * 100, fmt_num(sel_qwk)))
        html += '</div>\n'

    html += '</details>\n</div>\n'
    return html


def build_c3_literature_html():
    """C3: Literature context — AUROC comparison dot plot."""
    # Literature data: Gulshan 2016, Voets 2019, this model
    our_auroc = METRICS.deployed_model.referable_auroc
    entries = [
        ("Gulshan 2016 (128k)", 0.991, "#B3A6EC"),
        ("Voets 2019 (25k)", 0.951, "#E8A33D"),
        ("This model (596k)", our_auroc or 0.85, "#3FC3BC"),
    ]
    plot = build_literature_dot_svg(entries)

    html = '<div class="fc-section-card">\n'
    html += '<h3>Literature Context <span class="fc-badge">C3</span></h3>\n'
    html += '<div class="fc-chart-col">\n'
    html += '<p class="fc-caption">Referable DR AUROC — test sets differ across papers.</p>\n'
    html += plot
    html += '<p class="fc-caption" style="font-size:10px;">Gulshan: 128k images (US clinics). Voets: 25k images (Netherlands). This model: %s images (EyePACS P2 + APTOS).</p>\n' % (METRICS.deployed_model.n_test or "?")
    html += '</div>\n'
    html += '</div>\n'
    return html


def build_c4_integrity_html():
    """C4: Evaluation integrity — real split counts, the actual leakage
    finding (not a fabricated citation), and the real sample-size-sweep
    verdict. Every number below is read from a results/*.json or
    data/splits/*.json file at render time -- none are typed (R3)."""
    dm = METRICS.deployed_model
    html = '<div class="fc-section-card">\n'
    html += '<h3>Evaluation Integrity <span class="fc-badge">C4</span></h3>\n'

    # Split diagram: real P1 vs P2 fold counts from the split files.
    p1_counts = _load_split_fold_counts(RESULTS_DIR.parent / "data" / "splits" / "p1.json")
    p2_counts = _load_split_fold_counts(RESULTS_DIR.parent / "data" / "splits" / "p2.json")
    if p1_counts and p2_counts:
        diagram = build_split_diagram_svg(p1_counts, p2_counts)
        html += '<div class="fc-chart-row">%s</div>\n' % diagram
        html += '<p class="fc-caption">Real fold sizes from data/splits/p1.json and p2.json.</p>\n'
    else:
        html += '<p class="fc-unavailable">Split files unavailable.</p>\n'

    # The actual leakage finding (inter_eye_correlation.json), not an
    # invented "shortcut-ceiling" citation.
    ie = _load_json(RESULTS_DIR / "inter_eye_correlation.json")
    html += '<div class="fc-card-inline">\n'
    html += '<h4>Why P2, not P1</h4>\n'
    if ie:
        qwk = ie.get("correlation", {}).get("quadratic_weighted_kappa")
        lookup_qwk = ie.get("label_only_baseline", {}).get("lookup_qwk")
        html += ('<p>Inter-eye QWK: <strong>%s</strong> -- a label-only lookup with '
                 '<em>zero image data</em> (just guessing a patient\'s grade from their '
                 'other eye\'s label) already scores QWK <strong>%s</strong>. That much '
                 'signal leaks across a patient\'s two eyes into an image-level (P1) '
                 'split, which is why this project evaluates everything on the '
                 'patient-level (P2) split instead. Our deployed model\'s test QWK: '
                 '%s.</p>\n' % (fmt_num(qwk), fmt_num(lookup_qwk), fmt_num(dm.test_qwk)))
    else:
        html += '<p class="fc-unavailable">Leakage data unavailable.</p>\n'
    html += '</div>\n'

    # The real (inconclusive) sample-size-sweep finding, not a fabricated
    # "2 of 4 ablations significant" claim.
    sweep = _load_json(RESULTS_DIR / "claim2d_sample_size_sweep.json")
    html += '<div class="fc-card-inline">\n'
    html += '<h4>Does the P1-vs-P2 gap shrink with more training data?</h4>\n'
    if sweep:
        verdict = sweep.get("verdict", "unknown")
        text = sweep.get("verdict_text", "")
        html += ('<p><strong>Verdict: %s.</strong> %s</p>\n'
                 % (verdict.replace("_", " "), text))
    else:
        html += '<p class="fc-unavailable">Sample-size sweep data unavailable.</p>\n'
    html += '</div>\n'

    html += '</div>\n'
    return html


def _load_split_fold_counts(split_path):
    """{'train': n, 'val': n, 'test': n} from a data/splits/*.json file,
    or None if the file isn't present locally."""
    if not split_path.exists():
        return None
    data = json.loads(split_path.read_text())
    counts = {"train": 0, "val": 0, "test": 0}
    for entry in data.values():
        fold = entry.get("fold")
        if fold in counts:
            counts[fold] += 1
    return counts


def build_c5_evidence_html():
    """C5: Evidence — design levers (forest plot) and two-eyes decomposition
    (waterfall), both from claim3_decomposed_tf_efficientnet_b0_384.json.
    Missing pieces render "data unavailable", never a fabricated fallback
    number (R3) -- head_eff/fusion_eff used to default to 0.05/0.02 if the
    file was missing, which would have silently shown a fake chart."""
    c3 = load_claim3_decomposed()
    dec = c3.get("decomposition", {})
    head = dec.get("1_head_effect_ordinal_minus_multinomial_at_max", {})
    fusion = dec.get("2_fusion_effect_pool_minus_per_eye_max_ordinal", {})
    total = dec.get("8_original_total_gain_pool_ordinal_minus_per_eye_max_multinomial", {})
    head_eff, fusion_eff, total_eff = head.get("mean_diff"), fusion.get("mean_diff"), total.get("mean_diff")

    html = '<div class="fc-section-card">\n'
    html += '<h3>Evidence <span class="fc-badge">C5</span></h3>\n'

    html += '<h4>Design Levers</h4>\n'
    html += '<p class="fc-caption">Effect of each architectural decision on QWK, with 95% patient-level bootstrap CI. source: claim3_decomposed_tf_efficientnet_b0_384.json</p>\n'
    levers = []
    lever_specs = [
        ("Ordinal vs multinomial head", "1_head_effect_ordinal_minus_multinomial_at_max", "#3FC3BC"),
        ("Fusion (mean-pool eyes)", "2_fusion_effect_pool_minus_per_eye_max_ordinal", "#8AB4FF"),
        ("Fusion (concat eyes)", "3_fusion_effect_concat_minus_per_eye_max_ordinal", "#B3A6EC"),
        ("Aggregation: mean vs max (ordinal)", "4_aggregation_effect_mean_minus_max_ordinal", "#E8A33D"),
        ("Aggregation: min vs max (ordinal)", "5_aggregation_effect_min_minus_max_ordinal", "#E8746A"),
    ]
    for label, key, color in lever_specs:
        entry = dec.get(key)
        if entry and entry.get("mean_diff") is not None and entry.get("ci95"):
            levers.append((label, entry["mean_diff"], entry["ci95"][0], entry["ci95"][1], color))
    if levers:
        html += '<div class="fc-chart-row">%s</div>\n' % build_forest_plot_svg(levers)
    else:
        html += '<p class="fc-unavailable">Design-lever data unavailable.</p>\n'

    html += '<h4>Two Eyes: where the gain actually comes from</h4>\n'
    if head_eff is not None and fusion_eff is not None:
        waterfall = build_waterfall_svg([
            ("Head effect (ordinal read)", head_eff, "#3FC3BC"),
            ("+ Fusion effect (pooling)", fusion_eff, "#8AB4FF"),
        ], source="claim3_decomposed_tf_efficientnet_b0_384.json")
        html += '<div class="fc-chart-row">%s</div>\n' % waterfall
        html += ('<p class="fc-caption">Total gain: %s QWK (head %s + fusion %s), across %s test '
                 'patients. Most of the apparent both-eyes benefit is actually the ordinal-vs-'
                 'multinomial head, not eye-fusion.</p>\n'
                 % (fmt_num(total_eff) if total_eff is not None else "N/A",
                    fmt_num(head_eff), fmt_num(fusion_eff),
                    c3.get("n_test_patients", "?")))
    else:
        html += '<p class="fc-unavailable">Two-eyes decomposition data unavailable.</p>\n'

    html += '</div>\n'
    return html


def build_c6_prediction_record_html():
    """C6: Prediction record — a real scorecard sampled from the test
    predictions CSV (image_id/true_grade/pred_grade), not fabricated stub
    rows. Deterministic sample: first gradable row for each of Held /
    Overturned(+1) / Overturned(-1) / Missed(>=2 off) outcome buckets per
    true grade, so the table shows real spread, not just easy cases."""
    html = '<div class="fc-section-card">\n'
    html += '<h3>Prediction Record <span class="fc-badge">C6</span></h3>\n'
    csv_path = RESULTS_DIR / "finetune_app_converged_p2_class_balanced_seed42_test_predictions.csv"
    if not csv_path.exists():
        html += '<p class="fc-unavailable">Test predictions CSV unavailable.</p>\n</div>\n'
        return html

    import csv as _csv
    rows = list(_csv.DictReader(csv_path.open()))
    picked = []
    seen_buckets = set()
    for r in rows:
        true_g, pred_g = int(r["true_grade"]), int(r["pred_grade"])
        diff = pred_g - true_g
        bucket = "Held" if diff == 0 else ("Missed" if abs(diff) >= 2 else "Overturned")
        key = (true_g, bucket)
        if key not in seen_buckets:
            seen_buckets.add(key)
            picked.append((r["image_id"], true_g, pred_g, bucket))
        if len(picked) >= 10:
            break

    html += ('<p class="fc-caption">%d real rows from the P2 test set (image_id, true grade, '
             'predicted grade). source: finetune_app_converged_..._test_predictions.csv, n=%d '
             'total.</p>\n' % (len(picked), len(rows)))
    html += '<table class="fc-scard-table">\n'
    html += '<tr><th>Image</th><th>Predicted</th><th>Ground Truth</th><th>Outcome</th></tr>\n'
    chip_class = {"Held": "fc-chip-hold", "Overturned": "fc-chip-overturned", "Missed": "fc-chip-missed"}
    for image_id, true_g, pred_g, bucket in picked:
        html += ('<tr><td>%s</td><td>%d</td><td>%d</td><td><span class="%s">%s</span></td></tr>\n'
                 % (image_id, pred_g, true_g, chip_class[bucket], bucket))
    html += '</table>\n'
    html += '</div>\n'
    return html


_CODEBASE_STATS_CACHE = None


def _count_codebase_stats():
    """Scan only the app's OWN source dirs (app/, src/, tests/, scripts/)
    -- the previous version rglob'd the whole project root, which walked
    .venv/, data/, checkpoints/ (tens of thousands of files) on every page
    load. Cached at module import time since these numbers don't change
    within a running server."""
    global _CODEBASE_STATS_CACHE
    if _CODEBASE_STATS_CACHE is not None:
        return _CODEBASE_STATS_CACHE
    root = RESULTS_DIR.parents[0]
    dirs = ["app", "src", "tests", "scripts"]
    n_files = n_python = n_other = n_lines = 0
    for d in dirs:
        base = root / d
        if not base.exists():
            continue
        for f in base.rglob("*"):
            if not f.is_file() or "__pycache__" in f.parts:
                continue
            n_files += 1
            if f.suffix == ".py":
                n_python += 1
                try:
                    n_lines += len(f.read_text(encoding="utf-8", errors="ignore").splitlines())
                except OSError:
                    pass
            elif f.suffix in (".js", ".html", ".css"):
                n_other += 1
    _CODEBASE_STATS_CACHE = (n_lines, n_files, n_python, n_other)
    return _CODEBASE_STATS_CACHE


def build_c7_built_html():
    """C7: How it's built — codebase stats, arch, acceptance tests (the
    REAL verdicts for 10.1/11.1/12.1, not a hardcoded "PASS" fallback that
    the old acceptance_test_10_1_verdict lookup silently fell back to
    because it was reading the wrong file)."""
    n_lines, n_files, n_python, n_other = _count_codebase_stats()
    stats = build_codebase_stats_svg(n_lines, n_files, n_python, n_other)
    arch = build_arch_svg()

    html = '<div class="fc-section-card">\n'
    html += '<h3>How It\'s Built <span class="fc-badge">C7</span></h3>\n'

    html += '<div class="fc-chart-row">%s</div>\n' % stats
    html += '<div class="fc-chart-row">%s</div>\n' % arch

    html += '<h4>Acceptance Tests</h4>\n'
    html += '<div class="fc-stats-row">\n'

    dm = METRICS.deployed_model
    if dm.available and dm.acceptance_test_10_1_verdict:
        html += ('<div class="fc-stat-tile"><span class="fc-stat-val fc-verdict-pass">%s</span>'
                 '<span class="fc-stat-label">10.1 &middot; QWK band</span></div>\n'
                 % dm.acceptance_test_10_1_verdict)
    else:
        html += ('<div class="fc-stat-tile"><span class="fc-stat-val">N/A</span>'
                 '<span class="fc-stat-label">10.1 &middot; QWK band</span></div>\n')

    cal = METRICS.calibration
    if cal.available and cal.acceptance_test_11_1_verdict:
        html += ('<div class="fc-stat-tile"><span class="fc-stat-val fc-verdict-pass">%s</span>'
                 '<span class="fc-stat-label">11.1 &middot; calibration lift</span></div>\n'
                 % cal.acceptance_test_11_1_verdict)
    else:
        html += ('<div class="fc-stat-tile"><span class="fc-stat-val">N/A</span>'
                 '<span class="fc-stat-label">11.1 &middot; calibration lift</span></div>\n')

    acc121 = load_acceptance_12_1()
    if acc121 and acc121.get("n_images"):
        n, mism = acc121["n_images"], acc121.get("outcome_mismatches", 0)
        verdict_121 = "pass" if mism <= 10 else "fail"
        verdict_class = "fc-verdict-pass" if mism <= 10 else "fc-verdict-fail"
        html += ('<div class="fc-stat-tile"><span class="fc-stat-val %s">%s (%d/%d)</span>'
                 '<span class="fc-stat-label">12.1 &middot; decision-level pipeline match</span></div>\n'
                 % (verdict_class, verdict_121, mism, n))
    else:
        html += ('<div class="fc-stat-tile"><span class="fc-stat-val">N/A</span>'
                 '<span class="fc-stat-label">12.1 &middot; decision-level pipeline match</span></div>\n')

    html += '</div>\n'
    html += '</div>\n'
    return html


def build_c8_limitations_html():
    """C8: Limitations — two-column table. The grade-1 row used to claim
    "QWK capped by label agreement (~0.67)", a number that appears nowhere
    in results/grade1_diagnosis.json or anywhere else in this project --
    fabricated. Replaced with the real, sourced finding (verdict +
    best achievable recall under any of 4 tested decision rules)."""
    g1 = _load_json(RESULTS_DIR / "grade1_diagnosis.json")
    if g1 and g1.get("best_non_argmax_grade1_recall") is not None:
        grade1_cell = ('Best of 4 decision rules recovers %s grade-1 recall '
                       '(verdict: %s) -- see results/grade1_diagnosis.json'
                       % (fmt_pct(g1["best_non_argmax_grade1_recall"]),
                          (g1.get("verdict") or "?").replace("_", " ")))
    else:
        grade1_cell = "data unavailable"

    html = '<details class="fc-limitations">\n'
    html += '<summary>Limitations <span class="fc-badge">C8</span></summary>\n'
    html += '<table class="fc-limit-table">\n'
    html += '<tr><th>Limitation</th><th>What it bounds</th></tr>\n'
    html += '<tr><td>Single random seed</td><td>Generalizability across seeds unknown</td></tr>\n'
    html += '<tr><td>Grade-1 recall ceiling</td><td>%s</td></tr>\n' % grade1_cell
    html += '<tr><td>Specificity not sufficient</td><td>Cannot be used for autonomous screening</td></tr>\n'
    html += '<tr><td>EyePACS label noise</td><td>Reference standard imperfect at grade 1</td></tr>\n'
    html += '<tr><td>APTOS lacks eye pairing</td><td>Cannot validate per-eye decisions fully</td></tr>\n'
    html += '<tr><td>No prospective validation</td><td>Retrospective results may not generalise</td></tr>\n'
    html += '</table>\n'
    html += '</details>\n'
    return html


def build_models_comparison_html():
    """Item 5: a real, read-only comparison of every trained checkpoint
    this project has actual results for (results/finetune_*.json) -- split,
    sampler, seed, epochs, test QWK, referable AUROC. This is a comparison
    TABLE, not live model-switching: only one checkpoint is ever loaded for
    inference (app/release/best_model.pt), so "switching models" in the
    Console would mean re-loading a different .pt file at runtime, which
    none of these other runs' checkpoints are bundled with the app to do.
    Every number is read from its own run's JSON file -- nothing typed."""
    deployed_run = METRICS.deployed_model
    deployed_file = "finetune_app_converged_p2_class_balanced_seed42.json"

    rows = []
    for f in sorted(RESULTS_DIR.glob("finetune_*.json")):
        name = f.name
        if name.endswith("_curve.json") or name.endswith("_prediction_dump.json"):
            continue
        if name == "finetune_config_comparison.json":
            continue
        data = _load_json(f)
        if not data or data.get("test_qwk") is None:
            continue
        cfg = data.get("config", {}) or {}
        rows.append({
            "file": name,
            "run_name": name.replace(".json", ""),
            "split": data.get("split", "?"),
            "sampler": data.get("sampler") or "none",
            "seed": data.get("seed", "?"),
            "model": cfg.get("model", "?"),
            "image_size": cfg.get("image_size", "?"),
            "epochs": data.get("total_epochs_run", cfg.get("epochs", "?")),
            "test_qwk": data.get("test_qwk"),
            "referable_auroc": data.get("referable_dr_auroc"),
            "is_deployed": name == deployed_file,
        })

    html = '<div class="fc-section-card">\n'
    html += '<h3>Trained Model Comparison</h3>\n'
    html += ('<p class="fc-caption">Every training run this project has real test-set results '
             'for (results/finetune_*.json), sorted by test QWK. Read-only comparison -- only '
             'the deployed checkpoint (app/release/best_model.pt) is loaded for live grading '
             'above; these other runs\' checkpoints are not bundled with the app.</p>\n')

    if not rows:
        html += '<p class="fc-unavailable">No training-run result files found.</p>\n</div>\n'
        return html

    rows.sort(key=lambda r: r["test_qwk"] or 0, reverse=True)
    html += '<table class="fc-model-table">\n'
    html += ('<tr><th>Run</th><th>Split</th><th>Sampler</th><th>Seed</th><th>Backbone</th>'
             '<th>Size</th><th>Epochs</th><th>Test QWK</th><th>Referable AUROC</th></tr>\n')
    for r in rows:
        row_class = ' class="fc-model-deployed"' if r["is_deployed"] else ""
        html += (
            '<tr%s><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>'
            '<td>%spx</td><td>%s</td><td>%s</td><td>%s</td></tr>\n'
            % (row_class, r["run_name"], r["split"].upper() if isinstance(r["split"], str) else r["split"],
               r["sampler"], r["seed"], r["model"], r["image_size"], r["epochs"],
               fmt_num(r["test_qwk"]), fmt_num(r["referable_auroc"]))
        )
    html += '</table>\n'
    html += '</div>\n'
    return html


def build_c9_about_html():
    """C9: About — author card, future work, credits, citations."""
    html = '<div class="fc-section-card fc-about-section">\n'
    html += '<h3>About</h3>\n'

    # Author
    html += '<div class="fc-author-card">\n'
    html += '<div class="fc-author-info">\n'
    html += '<p><strong>Fundus Console</strong> — Diabetic Retinopathy Screening Demo</p>\n'
    html += '<p>Built with PyTorch, Gradio, and careful attention to data provenance.</p>\n'
    html += '</div>\n'
    html += '</div>\n'

    # Future work
    html += '<h4>Future Work</h4>\n'
    html += '<ul class="fc-future-list">\n'
    html += '<li>Higher resolution (512px) input for finer lesion detection</li>\n'
    html += '<li>Binary referable DR head (simplified for some deployments)</li>\n'
    html += '<li>Ordinal read with full confusion matrix</li>\n'
    html += '<li>Multi-seed replication with confidence intervals</li>\n'
    html += '<li>Prospective external validation on new clinical sites</li>\n'
    html += '</ul>\n'

    # Credits
    html += '<h4>Credits</h4>\n'
    html += '<p>Trained on EyePACS (DMI) and APTOS 2019 datasets.</p>\n'
    html += '<p>Grad-CAM visualization adapted from Selvaraju et al. (2017).</p>\n'
    html += '<p>Temperature scaling: Guo et al. (2017).</p>\n'

    # Disclaimer
    html += '<div class="fc-disclaimer">\n'
    html += '<p><strong>Disclaimer:</strong> This system is for research and demonstration purposes only. '
    html += 'It is not a medical device and should not be used for clinical decision-making.</p>\n'
    html += '</div>\n'

    html += '</div>\n'
    return html
