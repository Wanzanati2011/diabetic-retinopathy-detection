"""
Headline fine-tune (MASTER_PLAN.md Part 10, Run 1) aggregation.

Not itself a fine-tuning script -- pulls the 4 already-written
finetune_headline_p{1,2}_seed{42,43}.json result files together into one
table/figure, per MASTER_PLAN.md S13.1's "generate every table from
results/*.json with a script" rule. Companion to claim3_robustness.py --
same shape, same conventions.

Inputs: results/finetune_headline_p1_seed42.json
        results/finetune_headline_p1_seed43.json
        results/finetune_headline_p2_seed42.json
        results/finetune_headline_p2_seed43.json
Outputs: results/headline_summary.json, figures/figure1_headline_summary.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUNS = [
    ("p1", 42, "results/finetune_headline_p1_seed42.json"),
    ("p1", 43, "results/finetune_headline_p1_seed43.json"),
    ("p2", 42, "results/finetune_headline_p2_seed42.json"),
    ("p2", 43, "results/finetune_headline_p2_seed43.json"),
]


def build_headline_summary(loaded):
    """Pure function: loaded is a list of (split, seed, result_dict).
    Returns the per-run table, per-split stats, and an overall verdict."""
    rows = []
    for split, seed, d in loaded:
        rows.append({
            "split": split, "seed": seed,
            "test_qwk": d["test_qwk"],
            "best_val_qwk": d["best_val_qwk"],
            "best_epoch": d["best_epoch"],
            "total_epochs_run": d["total_epochs_run"],
            "referable_dr_auroc": d.get("referable_dr_auroc"),
            "verdict": d["acceptance_test_10_1_verdict"],
        })

    by_split = {}
    for split in ("p1", "p2"):
        qwks = [r["test_qwk"] for r in rows if r["split"] == split]
        if qwks:
            by_split[split] = {
                "mean_qwk": float(np.mean(qwks)),
                "std_qwk": float(np.std(qwks, ddof=1)) if len(qwks) > 1 else 0.0,
                "n_seeds": len(qwks),
                "values": qwks,
            }

    overall_qwks = [r["test_qwk"] for r in rows]
    overall = {
        "mean_qwk": float(np.mean(overall_qwks)),
        "std_qwk": float(np.std(overall_qwks, ddof=1)) if len(overall_qwks) > 1 else 0.0,
        "n_runs": len(overall_qwks),
    }

    p1_mean = by_split.get("p1", {}).get("mean_qwk")
    p2_mean = by_split.get("p2", {}).get("mean_qwk")
    # A single-seed-pair mean-vs-mean comparison is noisy (n=2 per split) -- don't call a few
    # thousandths of QWK "inflation". Require a gap clearly bigger than the observed seed-to-seed
    # spread before claiming the split behaves as MASTER_PLAN.md Part 10 expects.
    INFLATION_MARGIN = 0.02
    p1_p2_gap = (p1_mean - p2_mean) if (p1_mean is not None and p2_mean is not None) else None
    p1_inflated_as_expected = (p1_p2_gap is not None and p1_p2_gap > INFLATION_MARGIN)

    all_below_target = all(r["test_qwk"] < 0.70 for r in rows)

    if p1_p2_gap is None:
        p1_p2_note = ""
    elif p1_inflated_as_expected:
        p1_p2_note = f"P1 looks inflated relative to P2 (gap={p1_p2_gap:+.4f}), as MASTER_PLAN.md Part 10 predicts."
    else:
        p1_p2_note = (
            f"NOTE: P1 vs P2 gap is only {p1_p2_gap:+.4f} (n=2 seeds/split) -- essentially "
            f"indistinguishable, NOT the clear inflation MASTER_PLAN.md Part 10 predicts for P1. "
            f"Do not report this as 'P1 inflated as expected' with this little data; either run more "
            f"seeds or treat the two splits as equivalent here."
        )

    honest_finding = (
        f"Headline fine-tune (effnetb0@224, fast config, 2 splits x 2 seeds): "
        f"test QWK = {overall['mean_qwk']:.4f} +/- {overall['std_qwk']:.4f} across {overall['n_runs']} runs "
        f"(p1 mean={p1_mean:.4f}, p2 mean={p2_mean:.4f}). " +
        ("All 4 runs fall in the 0.40-0.70 'undertrained or preprocessing bug' acceptance-test band "
         "-- none reached the 0.70 'correct -- proceed' bar. "
         if all_below_target else
         "At least one run reached the 0.70 'correct -- proceed' bar. ") +
        p1_p2_note
    )

    return {
        "rows": rows,
        "by_split": by_split,
        "overall": overall,
        "all_below_target": bool(all_below_target),
        "p1_inflated_as_expected": bool(p1_inflated_as_expected),
        "honest_finding": honest_finding,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--out", default="results/headline_summary.json")
    ap.add_argument("--fig", default="figures/figure1_headline_summary.png")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    loaded = []
    for split, seed, rel_path in RUNS:
        p = root / rel_path
        if not p.exists():
            print(f"WARNING: {p} not found -- skipping {split} seed{seed}.", file=sys.stderr)
            continue
        loaded.append((split, seed, json.loads(p.read_text())))

    if not loaded:
        print("ERROR: no finetune_headline_*.json files found. Run finetune.py first.", file=sys.stderr)
        sys.exit(1)

    summary = build_headline_summary(loaded)

    print("=" * 72)
    print("HEADLINE FINE-TUNE (Part 10, Run 1) -- SUMMARY")
    print("=" * 72)
    for r in summary["rows"]:
        print(f"  {r['split']} seed{r['seed']:<3} test_qwk={r['test_qwk']:.4f}  "
              f"best_val_qwk={r['best_val_qwk']:.4f} (epoch {r['best_epoch']})  "
              f"rDR_auroc={r['referable_dr_auroc']}  [{r['verdict']}]")
    print()
    for split, s in summary["by_split"].items():
        print(f"  {split} mean={s['mean_qwk']:.4f} +/- {s['std_qwk']:.4f}  (n={s['n_seeds']})")
    print(f"  overall mean={summary['overall']['mean_qwk']:.4f} +/- {summary['overall']['std_qwk']:.4f}")
    print()
    print(summary["honest_finding"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary["rows"]
    labels = [f"{r['split']}\nseed{r['seed']}" for r in rows]
    qwks = [r["test_qwk"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(rows))
    colors = ["#1f77b4" if r["split"] == "p1" else "#ff7f0e" for r in rows]
    ax.bar(x, qwks, color=colors)
    ax.axhline(0.70, color="green", linestyle="--", linewidth=1, label="0.70 'correct -- proceed' bar")
    ax.axhline(0.40, color="red", linestyle="--", linewidth=1, label="0.40 'BROKEN' bar")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("test QWK")
    ax.set_ylim(0, 1)
    ax.set_title("Figure 1 -- Headline fine-tune (effnetb0@224) test QWK\nby split and seed")
    ax.legend(loc="upper right", fontsize=8)
    for i, q in enumerate(qwks):
        ax.text(i, q + 0.02, f"{q:.3f}", ha="center", fontsize=9)
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
