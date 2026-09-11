"""
Headline vs. app-model fine-tune comparison (MASTER_PLAN.md Part 10, Run 1 vs Run 2).

Not itself a fine-tuning script -- pulls the 4 headline result files and the
1 app-model result file together to answer: did the higher-resolution "best
config" (384px, app model) actually beat the fast "headline config" (224px),
and did either clear the acceptance-test 0.70 "correct -- proceed" bar?
Same conventions as claim3_robustness.py / aggregate_headline.py.

Inputs: results/finetune_headline_p1_seed42.json
        results/finetune_headline_p1_seed43.json
        results/finetune_headline_p2_seed42.json
        results/finetune_headline_p2_seed43.json
        results/finetune_app_p2_seed42.json
Outputs: results/finetune_config_comparison.json,
         figures/figure1b_headline_vs_app.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HEADLINE_RUNS = [
    ("p1", 42, "results/finetune_headline_p1_seed42.json"),
    ("p1", 43, "results/finetune_headline_p1_seed43.json"),
    ("p2", 42, "results/finetune_headline_p2_seed42.json"),
    ("p2", 43, "results/finetune_headline_p2_seed43.json"),
]
APP_RUNS = [
    ("p2", 42, "results/finetune_app_p2_seed42.json"),
]

TARGET_QWK = 0.70


def build_comparison(headline_loaded, app_loaded):
    """Pure function: each *_loaded is a list of (split, seed, result_dict)."""
    def summarize(loaded, label):
        qwks = [d["test_qwk"] for _, _, d in loaded]
        return {
            "label": label,
            "n_runs": len(qwks),
            "values": qwks,
            "mean_qwk": float(np.mean(qwks)),
            "std_qwk": float(np.std(qwks, ddof=1)) if len(qwks) > 1 else 0.0,
            "any_cleared_target": bool(any(q >= TARGET_QWK for q in qwks)),
            "all_cleared_target": bool(all(q >= TARGET_QWK for q in qwks)),
        }

    headline = summarize(headline_loaded, "headline (224px, fast config, n=4)")
    app = summarize(app_loaded, "app (384px, best config, n=1)")

    gain = app["mean_qwk"] - headline["mean_qwk"]
    # n=1 for the app run -- there's no variance to compare against, so this is a single point
    # estimate, not a statistically tested claim. Say so plainly rather than implying significance.
    honest_finding = (
        f"Headline config (224px, fast, 4 runs): mean test QWK = {headline['mean_qwk']:.4f} "
        f"+/- {headline['std_qwk']:.4f}. App config (384px, best, 1 run): test QWK = "
        f"{app['mean_qwk']:.4f}. Gain from switching to the higher-res best config: "
        f"{gain:+.4f} QWK. " +
        ("Neither config cleared the 0.70 'correct -- proceed' acceptance-test bar. "
         if not headline["any_cleared_target"] and not app["any_cleared_target"] else
         "At least one run cleared the 0.70 bar. ") +
        (f"The gain is in the expected direction (higher res + longer effective training helps), "
         f"which argues against a systemic preprocessing bug -- a broken pipeline would not scale "
         f"predictably with a better config. But this is a SINGLE app-model run (n=1, no seed replicate, "
         f"no CI) -- do not report {gain:+.4f} as a statistically established gain. Either run a second "
         f"seed for the app config before claiming the improvement is real and not run-to-run noise, or "
         f"report it explicitly as a single-run observation."
         if gain > 0 else
         f"The app config did NOT outperform headline here ({gain:+.4f}) -- unexpected if 384px/best-config "
         f"is supposed to be strictly better; worth a second look before writing this up.")
    )

    return {"headline": headline, "app": app, "gain_app_minus_headline": gain, "honest_finding": honest_finding}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--out", default="results/finetune_config_comparison.json")
    ap.add_argument("--fig", default="figures/figure1b_headline_vs_app.png")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()

    def load(runs):
        loaded = []
        for split, seed, rel_path in runs:
            p = root / rel_path
            if not p.exists():
                print(f"WARNING: {p} not found -- skipping.", file=sys.stderr)
                continue
            loaded.append((split, seed, json.loads(p.read_text())))
        return loaded

    headline_loaded = load(HEADLINE_RUNS)
    app_loaded = load(APP_RUNS)

    if not headline_loaded or not app_loaded:
        print("ERROR: need at least one headline result and the app result. Run finetune.py first.",
              file=sys.stderr)
        sys.exit(1)

    summary = build_comparison(headline_loaded, app_loaded)

    print("=" * 72)
    print("HEADLINE vs APP-MODEL FINE-TUNE COMPARISON")
    print("=" * 72)
    for key in ("headline", "app"):
        s = summary[key]
        print(f"  {s['label']}: mean={s['mean_qwk']:.4f} +/- {s['std_qwk']:.4f}  values={s['values']}")
    print(f"\n  gain (app - headline) = {summary['gain_app_minus_headline']:+.4f}")
    print()
    print(summary["honest_finding"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["headline\n(224px, n=4)", "app\n(384px, n=1)"]
    means = [summary["headline"]["mean_qwk"], summary["app"]["mean_qwk"]]
    errs = [summary["headline"]["std_qwk"], 0.0]
    colors = ["#1f77b4", "#9467bd"]
    x = np.arange(2)
    ax.bar(x, means, yerr=errs, capsize=6, color=colors)
    ax.axhline(TARGET_QWK, color="green", linestyle="--", linewidth=1, label=f"{TARGET_QWK} 'correct -- proceed' bar")
    ax.axhline(0.40, color="red", linestyle="--", linewidth=1, label="0.40 'BROKEN' bar")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("test QWK")
    ax.set_ylim(0, 1)
    ax.set_title("Figure 1b -- Headline (fast) vs App (best) config\ntest QWK, effnetb0, P2 split")
    ax.legend(loc="upper right", fontsize=8)
    for i, m in enumerate(means):
        ax.text(i, m + errs[i] + 0.02, f"{m:.3f}", ha="center", fontsize=9)
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
