"""
Part 8.4 -- Three-way comparison plot (per MASTER_PLAN.md).

One figure, one panel, that tells the whole Phase 4 story at a glance:
  1. Label-only lookup (no pixels at all) -- the shortcut ceiling.
  2. P1 (image-level, leaky split) model  -- multinomial + ordinal heads.
  3. P2 (patient-level, honest split) model -- multinomial + ordinal heads.
  4. Partner-present vs partner-absent (P1 test set) -- the leakage ablation.

This script computes NOTHING new. It only pulls numbers out of the Phase 4
result JSONs already written by earlier scripts, per MASTER_PLAN.md S13.1:
"Generate every table/figure from results/*.json with a script. Never
hand-copy numbers."

Inputs (produced by earlier Phase 4 / Step 0 scripts):
  results/inter_eye_correlation.json    (Step 0 -- label_only_baseline.lookup_qwk)
  results/claim2_protocols.json         (Claim 2 -- P1/P2 multinomial+ordinal QWK, CIs)
  results/claim2b_partner_ablation.json (Claim 2b -- partner present/absent QWK)

Output:
  results/summary_comparison.json
  figures/figure1_summary_comparison.png

NOTE ON FIGURE NUMBERING: MASTER_PLAN.md S13.1 lists this as the paper's
"Figure 1". The existing figures/figure1_claim2_protocols.png (from
claim2_protocols.py) is a different, more detailed figure (5-seed breakdown
per protocol). Keeping both files for now -- deciding which is the actual
paper Figure 1 vs. a supplementary figure is a Phase 9 (paper writing)
decision, not made here.

No GPU, no heavy compute -- pure JSON aggregation + matplotlib. Safe to run
anywhere once the three input JSONs exist.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_summary(inter_eye, claim2, claim2b):
    """Pure function: pull the headline numbers out of the three already-computed
    result dicts. No new modeling/statistics happens here -- see module docstring.
    Kept separate from main() so it can be unit-tested with synthetic dicts."""
    lookup = inter_eye["label_only_baseline"]
    p1 = claim2["protocols"]["P1"]
    p2 = claim2["protocols"]["P2"]
    mult = claim2b["multinomial"]
    ordi = claim2b["ordinal"]

    bars = [
        {"label": "Label-only lookup\n(no pixels)", "qwk": lookup["lookup_qwk"], "ci": None, "group": "baseline"},
        {"label": "P1 image-level\n(multinomial)", "qwk": p1["reference_qwk_seed42"], "ci": p1["patient_bootstrap_ci95"], "group": "P1"},
        {"label": "P1 image-level\n(ordinal)", "qwk": p1["ordinal_qwk"], "ci": None, "group": "P1"},
        {"label": "P2 patient-level\n(multinomial)", "qwk": p2["reference_qwk_seed42"], "ci": p2["patient_bootstrap_ci95"], "group": "P2"},
        {"label": "P2 patient-level\n(ordinal)", "qwk": p2["ordinal_qwk"], "ci": None, "group": "P2"},
        {"label": "Partner present\n(multinomial)", "qwk": mult["raw_qwk_present"], "ci": None, "group": "ablation"},
        {"label": "Partner absent\n(multinomial)", "qwk": mult["raw_qwk_absent"], "ci": None, "group": "ablation"},
    ]

    p1_vs_p2_diff = p1["reference_qwk_seed42"] - p2["reference_qwk_seed42"]
    ablation_diff_mult = mult["raw_qwk_present"] - mult["raw_qwk_absent"]
    ablation_diff_ord = ordi["raw_qwk_present"] - ordi["raw_qwk_absent"]

    p1_vs_p2_p = claim2["p1_vs_p2_paired_permutation_test"]["p_value_two_sided"]
    ablation_p_mult = mult["permutation_test"]["p_value_two_sided"]
    ablation_p_ord = ordi["permutation_test"]["p_value_two_sided"]

    honest_finding = (
        f"Label-only lookup reaches QWK={lookup['lookup_qwk']:.3f} using zero pixels -- "
        f"a strong shortcut ceiling. Under frozen features, P1 (leaky split, "
        f"QWK={p1['reference_qwk_seed42']:.3f}) does NOT outperform P2 (honest split, "
        f"QWK={p2['reference_qwk_seed42']:.3f}); the difference is {p1_vs_p2_diff:+.3f} "
        f"and not significant (paired permutation p={p1_vs_p2_p:.3g}). "
        f"The partner-eye ablation on P1 test images shows a {ablation_diff_mult:+.3f} QWK "
        f"gap (present vs absent, multinomial) that is also not significant "
        f"(p={ablation_p_mult:.3g}; ordinal gap {ablation_diff_ord:+.3f}, p={ablation_p_ord:.3g}). "
        "Together: with frozen (non-fine-tuned) features, patient-level leakage does not "
        "measurably inflate scores. This does not rule out leakage under fine-tuning "
        "(Phase 6), where a model can memorize image-specific detail -- that open question "
        "is left for later phases, not answered by this figure."
    )

    return {
        "bars": bars,
        "p1_vs_p2_diff": p1_vs_p2_diff,
        "p1_vs_p2_p_value": p1_vs_p2_p,
        "ablation_diff_multinomial": ablation_diff_mult,
        "ablation_diff_ordinal": ablation_diff_ord,
        "ablation_p_value_multinomial": ablation_p_mult,
        "ablation_p_value_ordinal": ablation_p_ord,
        "honest_finding": honest_finding,
    }


def make_figure(summary, backbone, size, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bars = summary["bars"]
    colors = {"baseline": "#888888", "P1": "#d62728", "P2": "#1f77b4", "ablation": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(bars))
    heights = [b["qwk"] for b in bars]
    bar_colors = [colors[b["group"]] for b in bars]

    yerr = np.zeros((2, len(bars)))
    for i, b in enumerate(bars):
        if b["ci"] is not None:
            lo, hi = b["ci"]
            yerr[0, i] = max(0.0, b["qwk"] - lo)
            yerr[1, i] = max(0.0, hi - b["qwk"])

    ax.bar(x, heights, color=bar_colors, yerr=yerr, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([b["label"] for b in bars], fontsize=8)
    ax.set_ylabel("QWK")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Figure 1 (per MASTER_PLAN.md S13.1) -- the whole Phase 4 story, {backbone}@{size}\n"
        f"P1 vs P2 diff={summary['p1_vs_p2_diff']:+.3f} (p={summary['p1_vs_p2_p_value']:.3g});  "
        f"partner ablation diff={summary['ablation_diff_multinomial']:+.3f} "
        f"(p={summary['ablation_p_value_multinomial']:.3g})",
        fontsize=10,
    )
    for i, h in enumerate(heights):
        ax.text(i, h + 0.02, f"{h:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--backbone", default="tf_efficientnet_b0")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--inter-eye-json", default="results/inter_eye_correlation.json")
    ap.add_argument("--claim2-json", default="results/claim2_protocols.json")
    ap.add_argument("--claim2b-json", default="results/claim2b_partner_ablation.json")
    ap.add_argument("--out", default="results/summary_comparison.json")
    ap.add_argument("--fig", default="figures/figure1_summary_comparison.png")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()

    paths = {
        "inter_eye": root / args.inter_eye_json,
        "claim2": root / args.claim2_json,
        "claim2b": root / args.claim2b_json,
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        print("ERROR: missing required result file(s):", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        print("Run inter_eye.py / claim2_protocols.py / claim2b_partner_ablation.py first.", file=sys.stderr)
        sys.exit(1)

    inter_eye = json.loads(paths["inter_eye"].read_text())
    claim2 = json.loads(paths["claim2"].read_text())
    claim2b = json.loads(paths["claim2b"].read_text())

    if claim2.get("backbone") != args.backbone or claim2.get("size") != args.size:
        print(f"WARNING: {paths['claim2'].name} was computed for "
              f"{claim2.get('backbone')}@{claim2.get('size')}, not {args.backbone}@{args.size}. "
              f"Pass matching --claim2-json/--claim2b-json if comparing a different config.",
              file=sys.stderr)

    summary = build_summary(inter_eye, claim2, claim2b)

    print("=" * 72)
    print(f"PART 8.4 -- THREE-WAY COMPARISON  ({args.backbone}@{args.size})")
    print("=" * 72)
    for b in summary["bars"]:
        print(f"  {b['label'].replace(chr(10), ' '):32s} QWK={b['qwk']:.4f}")
    print()
    print(summary["honest_finding"])

    fig_path = root / args.fig
    make_figure(summary, args.backbone, args.size, fig_path)
    print(f"\nWrote {fig_path}")

    out = {"backbone": args.backbone, "size": args.size, **summary}
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
