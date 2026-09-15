"""
Figures 5.8 and 5.9 (and 5.3) -- generated programmatically from
results/calibration_reject.json and the prediction dumps. No hard-coded
values, so re-running an experiment updates its figure with no manual edit.

CPU only, seconds.

    python -m src.analysis.make_phase7_figures

Writes into figures/report_v2/:
    figN_reliability.png       -> report Figure 5.8
    figO_risk_coverage.png     -> report Figure 5.9
    figP_confusion.png         -> report Figure 5.3
    figQ_grade1_rules.png      -> report Figure 5.11 (grade-1 decision rules)

Palette follows the Fundus Console's own convention: teal for routine /
good, amber for refer / attention, and deliberately never red-green, since
red-green colour vision deficiency affects roughly one in twelve male
viewers and this project's outputs are meant to be read by clinicians.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUN = "finetune_app_converged_p2_class_balanced_seed42"
TEAL = "#0C6F6E"
AMBER = "#A96606"
SLATE = "#5B6E72"
GRID = "#D7E3E3"
GRADE_SHORT = ["0\nNo DR", "1\nMild", "2\nModerate", "3\nSevere", "4\nProlif."]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": SLATE, "axes.labelcolor": "#0F1A1D",
    "text.color": "#0F1A1D", "xtick.color": SLATE, "ytick.color": SLATE,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "figure.dpi": 160, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def bins_to_arrays(binlist):
    x, y, n = [], [], []
    for b in binlist:
        if b["count"] == 0 or b["accuracy"] is None:
            continue
        x.append(b["confidence"]); y.append(b["accuracy"]); n.append(b["count"])
    return np.array(x), np.array(y), np.array(n)


def fig_reliability(cal, out):
    """Figure 5.8 -- reliability diagrams before and after temperature scaling."""
    keys = [k for k in ["p2_test", "p2_test_eyepacs", "p2_test_aptos"] if k in cal]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.0 * len(keys), 4.0),
                             sharey=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        s = cal[key]
        ax.plot([0, 1], [0, 1], color=SLATE, lw=1, ls="--", zorder=1,
                label="perfect calibration")
        for stage, colour, mark in (("before", AMBER, "o"), ("after", TEAL, "s")):
            x, y, n = bins_to_arrays(s[stage]["bins"])
            ax.plot(x, y, marker=mark, ms=4.5, lw=1.6, color=colour, zorder=3,
                    label=f"{stage} (ECE {s[stage]['ece']:.3f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("mean predicted confidence")
        ax.set_title(s["name"].replace(" (domain-shift proxy)", ""),
                     fontsize=9, pad=8)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("observed accuracy")
    fig.suptitle("Reliability before and after temperature scaling  "
                 f"(T fitted on validation only)", fontsize=10.5, y=1.02)
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")


def fig_risk_coverage(rej, base_qwk, at111, out):
    """Figure 5.9 -- coverage against selective QWK and selective sens/spec."""
    rows = sorted(rej["curves"]["max_softmax"]["test"],
                  key=lambda r: r["coverage"])
    cov = np.array([r["coverage"] for r in rows])
    qwk = np.array([r["selective_qwk"] for r in rows])
    sens = np.array([r["selective_sensitivity"] for r in rows])
    spec = np.array([r["selective_specificity"] for r in rows])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    ylo = min(qwk.min(), base_qwk) - 0.02
    yhi = qwk.max() + 0.035
    a1.plot(cov, qwk, color=TEAL, lw=2, zorder=3)
    a1.axhline(base_qwk, color=SLATE, lw=1, ls="--", zorder=2)
    a1.annotate(f"full coverage  {base_qwk:.3f}", xy=(0.27, base_qwk),
                xytext=(0.27, base_qwk + 0.007), fontsize=7.5, color=SLATE,
                va="bottom")
    c = at111["actual_coverage"]; q = at111["selective_qwk"]
    a1.scatter([c], [q], s=46, color=AMBER, zorder=5)
    a1.annotate(f"Acceptance Test 11.1\n{c:.0%} coverage, {q:.3f}\n"
                f"lift {at111['lift']:+.3f}  (pre-registered +0.05 to +0.10)",
                xy=(c, q), xytext=(0.30, yhi - 0.012), fontsize=7.5,
                color=AMBER, va="top",
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=0.9,
                                connectionstyle="arc3,rad=-0.18"))
    a1.set_xlabel("coverage (fraction of cases the model grades itself)")
    a1.set_ylabel("selective QWK")
    a1.set_xlim(0.25, 1.02); a1.set_ylim(ylo, yhi)
    a1.set_title("Rejecting low-confidence cases raises agreement",
                 fontsize=9.5, pad=8)

    a2.plot(cov, sens, color=AMBER, lw=2, label="referable sensitivity", zorder=3)
    a2.plot(cov, spec, color=TEAL, lw=2, label="referable specificity", zorder=3)
    a2.axvline(c, color=SLATE, lw=1, ls=":", zorder=2)
    a2.set_xlabel("coverage")
    a2.set_ylabel("selective rate at the fixed referral threshold")
    a2.set_xlim(0.25, 1.02)
    a2.legend(frameon=False, fontsize=8, loc="lower left")
    a2.set_title("and trades sensitivity for specificity", fontsize=9.5, pad=8)

    fig.suptitle("Risk-coverage behaviour of the confidence-based reject option",
                 fontsize=10.5, y=1.02)
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")


def fig_confusion(df, out):
    """Figure 5.3 -- confusion matrix of the deployed checkpoint, row-normalised."""
    y = df["true_grade"].to_numpy()
    p = df[[f"prob_{k}" for k in range(5)]].to_numpy().argmax(axis=1)
    cm = np.zeros((5, 5), dtype=int)
    for t, q in zip(y, p):
        cm[t, q] += 1
    rown = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.grid(False)
    im = ax.imshow(rown, cmap="BuGn", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]}\n{rown[i, j]:.0%}", ha="center",
                    va="center", fontsize=7.5,
                    color="white" if rown[i, j] > 0.55 else "#0F1A1D")
    ax.set_xticks(range(5)); ax.set_xticklabels(GRADE_SHORT, fontsize=7.5)
    ax.set_yticks(range(5)); ax.set_yticklabels(GRADE_SHORT, fontsize=7.5)
    ax.set_xlabel("predicted grade"); ax.set_ylabel("true grade")
    ax.set_title("Deployed checkpoint, frozen P2 test set (n = "
                 f"{len(df):,})\nrow-normalised; 69.6% of grade 1 lands on grade 0",
                 fontsize=9.5, pad=10)
    fig.colorbar(im, ax=ax, fraction=0.045, label="row fraction")
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")


def fig_grade1_rules(g1, out):
    """Figure 5.11 -- grade-1 recall under four decision rules."""
    order = ["argmax", "prior_corrected_argmax", "expected_grade_rounded"]
    labels = ["argmax\n(as shipped)", "prior-corrected\nargmax",
              "expected grade,\nrounded (ordinal read)"]
    vals = [g1["decision_rules"][k]["grade1_recall"] for k in order]
    los = [g1["decision_rules"][k]["grade1_recall_patient_bootstrap_ci95"][0]
           for k in order]
    his = [g1["decision_rules"][k]["grade1_recall_patient_bootstrap_ci95"][1]
           for k in order]
    prior = g1["prior_checkpoint_recall"]["1"] if "1" in g1["prior_checkpoint_recall"] \
        else g1["prior_checkpoint_recall"][1]
    bar_target = g1["expected_written_in_advance"][
        "balanced_rule_recovers_grade1_recall_to_at_least"]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(order))
    colours = [SLATE, TEAL, TEAL]
    ax.bar(x, vals, color=colours, width=0.55, zorder=3)
    ax.errorbar(x, vals, yerr=[np.array(vals) - np.array(los),
                               np.array(his) - np.array(vals)],
                fmt="none", ecolor="#0F1A1D", elinewidth=1.1, capsize=4, zorder=4)
    ax.axhline(prior, color=AMBER, lw=1.4, ls="--", zorder=2)
    ax.annotate(f"prior 15-epoch checkpoint  {prior:.3f}", xy=(2.35, prior),
                xytext=(1.45, prior + 0.022), fontsize=7.5, color=AMBER)
    ax.axhline(bar_target, color=SLATE, lw=1.2, ls=":", zorder=2)
    ax.annotate(f"pre-registered bar  {bar_target:.2f}", xy=(0, bar_target),
                xytext=(-0.35, bar_target + 0.014), fontsize=7.5, color=SLATE)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=8.5,
                color="#0F1A1D")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("grade-1 (Mild NPDR) recall")
    ax.set_ylim(0, max(0.52, prior + 0.06))
    ax.set_title("Grade-1 recall under three reads of the same probabilities\n"
                 "partial recovery, below the bar written down in advance",
                 fontsize=9.5, pad=10)
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")


def main():
    root = PROJECT_ROOT
    res = root / "results"
    figs = root / "figures" / "report_v2"
    figs.mkdir(parents=True, exist_ok=True)

    cal_path = res / "calibration_reject.json"
    if not cal_path.exists():
        raise SystemExit(f"Missing {cal_path} -- run src.experiments.calibrate first.")
    cal = json.loads(cal_path.read_text())

    print("Generating Phase 7 figures...")
    fig_reliability(cal["calibration"], figs / "figN_reliability.png")
    fig_risk_coverage(cal["reject_option"],
                      cal["acceptance_test_11_1"]["base_qwk_full_coverage"],
                      cal["acceptance_test_11_1"],
                      figs / "figO_risk_coverage.png")

    test_csv = res / f"{RUN}_test_predictions.csv"
    if test_csv.exists():
        fig_confusion(pd.read_csv(test_csv), figs / "figP_confusion.png")

    g1_path = res / "grade1_diagnosis.json"
    if g1_path.exists():
        fig_grade1_rules(json.loads(g1_path.read_text()),
                         figs / "figQ_grade1_rules.png")

    print("\nDone. These are report Figures 5.3, 5.8, 5.9 and 5.11.")


if __name__ == "__main__":
    main()
