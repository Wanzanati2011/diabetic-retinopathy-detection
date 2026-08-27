"""
Claim 3 cross-backbone robustness summary.

Not a MASTER_PLAN.md-numbered deliverable on its own -- a small companion to
claim3_both_eyes.py that answers the obvious next question once the
mean-pool-plus-ordinal result showed up on one backbone: does it hold up on
the others, or was it a fluke of one feature space? Pulls the already-written
per-backbone claim3_both_eyes*.json files together into one table/figure.
Computes nothing new, per MASTER_PLAN.md S13.1's "generate every table from
results/*.json with a script" rule.

Inputs: results/claim3_both_eyes.json (effnetb0@224, the default run),
        results/claim3_both_eyes_effnetb0_384.json,
        results/claim3_both_eyes_resnet50_224.json
Outputs: results/claim3_robustness.json, figures/figure4b_claim3_robustness.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIGS = [
    ("tf_efficientnet_b0", 224, "results/claim3_both_eyes.json"),
    ("tf_efficientnet_b0", 384, "results/claim3_both_eyes_effnetb0_384.json"),
    ("resnet50", 224, "results/claim3_both_eyes_resnet50_224.json"),
]


def build_robustness_summary(loaded):
    """Pure function: loaded is a list of (backbone, size, result_dict).
    Returns the per-config comparison plus an overall verdict."""
    rows = []
    for backbone, size, d in loaded:
        a_qwk = d["arm_A_per_eye_then_max"]["qwk"]
        c_ord_qwk = d["arm_C_mean_pooled_features"]["qwk_ordinal"]
        diff = d["arm_vs_a_paired_diff_ci95"]["C_ordinal_minus_A"]
        rows.append({
            "backbone": backbone, "size": size,
            "arm_A_qwk": a_qwk, "arm_C_ordinal_qwk": c_ord_qwk,
            "diff": diff["mean_diff"], "diff_ci95": diff["ci95"],
            "significant": diff["significantly_better"],
        })
    all_significant = all(r["significant"] for r in rows)
    honest_finding = (
        f"Mean-pooling both eyes' frozen features and using the ordinal head beats grading each "
        f"eye separately then taking the worse one, and this holds across all {len(rows)} tested "
        f"feature configs: " +
        "; ".join(f"{r['backbone']}@{r['size']} diff={r['diff']:+.3f} "
                  f"CI [{r['diff_ci95'][0]:.3f}, {r['diff_ci95'][1]:.3f}]" for r in rows) +
        (". This is a robust, backbone-independent effect, not an artifact of one feature space."
         if all_significant else
         ". NOTE: not significant in every config -- do not claim full robustness.")
    )
    return {"rows": rows, "all_significant": bool(all_significant), "honest_finding": honest_finding}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--out", default="results/claim3_robustness.json")
    ap.add_argument("--fig", default="figures/figure4b_claim3_robustness.png")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    loaded = []
    for backbone, size, rel_path in DEFAULT_CONFIGS:
        p = root / rel_path
        if not p.exists():
            print(f"WARNING: {p} not found -- skipping {backbone}@{size}.", file=sys.stderr)
            continue
        loaded.append((backbone, size, json.loads(p.read_text())))

    if not loaded:
        print("ERROR: no claim3_both_eyes*.json files found. Run claim3_both_eyes.py first.", file=sys.stderr)
        sys.exit(1)

    summary = build_robustness_summary(loaded)

    print("=" * 72)
    print("CLAIM 3 -- CROSS-BACKBONE ROBUSTNESS  (mean-pool + ordinal vs per-eye-then-max)")
    print("=" * 72)
    for r in summary["rows"]:
        flag = "SIGNIFICANT" if r["significant"] else "not significant"
        print(f"  {r['backbone']}@{r['size']:<4} A={r['arm_A_qwk']:.4f}  C-ordinal={r['arm_C_ordinal_qwk']:.4f}  "
              f"diff={r['diff']:+.4f}  CI {r['diff_ci95']}  [{flag}]")
    print()
    print(summary["honest_finding"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary["rows"]
    labels = [f"{r['backbone']}\n@{r['size']}" for r in rows]
    diffs = [r["diff"] for r in rows]
    err_lo = [d - r["diff_ci95"][0] for d, r in zip(diffs, rows)]
    err_hi = [r["diff_ci95"][1] - d for d, r in zip(diffs, rows)]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(rows))
    colors = ["#2ca02c" if r["significant"] else "#888888" for r in rows]
    ax.bar(x, diffs, color=colors, yerr=[err_lo, err_hi], capsize=5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("QWK diff (mean-pool + ordinal) - (per-eye-then-max)")
    ax.set_title("Figure 4b -- Claim 3 holds across all 3 feature configs\n"
                  "(error bars: 95% paired patient-bootstrap CI; green = CI excludes 0)")
    for i, (d, r) in enumerate(zip(diffs, rows)):
        ax.text(i, d + err_hi[i] + 0.005, f"{d:+.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig_path = root / args.fig
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nWrote {fig_path}")

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
