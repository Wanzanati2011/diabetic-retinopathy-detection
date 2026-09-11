"""
Referable-DR operating point -- the project's PRIMARY declared endpoint.

README.md declares the primary endpoint as referable-DR sensitivity at a fixed
operating point under Protocol 2, and the secondary endpoint as five-class QWK.
Every result reported so far has been the secondary endpoint. This script
computes the primary one.

Why it needs its own script rather than falling out of the confusion matrix:
the confusion matrices reported in Section 3.5 are taken at the model's own
argmax, which is an arbitrary operating point with no clinical meaning. A
screening model is not deployed at argmax -- it is deployed at a threshold
chosen to hit a required sensitivity, because in DR screening a missed
referable case costs far more than a false alarm. The right question is
therefore "at the threshold where this model catches 90% of referable
patients, how many healthy patients does it needlessly refer?", and that
cannot be read off an argmax confusion matrix.

Definitions used throughout:
  referable DR := true grade >= 2 (ICDR Moderate NPDR or worse), matching the
    clinical referral threshold used in the rest of this project.
  referable score := P(grade>=2) = prob_2 + prob_3 + prob_4, taken from the
    model's own softmax over the five grades. Summing the tail is the correct
    reduction of a 5-class head to a binary decision -- it uses the whole
    distribution rather than throwing away everything except the argmax.

Reported at three operating points, because different literatures fix
different things and a single number invites the wrong comparison:
  1. SENSITIVITY FIXED AT 0.90 -- the project's declared primary endpoint.
     Report the specificity achieved and the threshold that achieves it.
  2. SPECIFICITY FIXED AT 0.90 -- the convention several published DR papers
     report instead. Included so the comparison in the report is like-for-like.
  3. ARGMAX -- the model's default, for continuity with the Section 3.5
     confusion matrices.

Also reported: full ROC AUC, and the count of referable patients missed at the
90%-sensitivity threshold, because "10% of referable cases missed" is the
number a clinician would actually ask about.

NOTE ON TEST-SET USE: this reads the already-frozen, already-scored Phase 6b
test predictions. It computes a different metric on the same predictions; it
does not re-fit, re-tune, or select anything on test, and no threshold chosen
here is fed back into any model.

Inputs : results/finetune_converged_*_test_predictions.csv
Writes : results/clinical_operating_point.json
         figures/report_v2/figM_operating_point.png

Usage:
  python src/experiments/clinical_operating_point.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ["p1", "p2"]
SAMPLERS = ["class_balanced", "none"]
SEEDS = [42, 43]
BLUE, ORANGE, INK, MUTED = "#0072B2", "#D55E00", "#222222", "#888888"


def patient_bootstrap_ci(pid, y, score, stat_fn, n_boot=1000, seed=0):
    """Resample patients with replacement; recompute stat_fn(y, score)."""
    df = pd.DataFrame({"pid": pid, "y": y, "s": score})
    g = df.groupby("pid").indices
    pats = np.array(list(g.keys()))
    Y, S = df["y"].to_numpy(), df["s"].to_numpy()
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([g[p] for p in rng.choice(pats, len(pats), replace=True)])
        if len(np.unique(Y[idx])) < 2:
            continue
        out.append(stat_fn(Y[idx], S[idx]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return [float(lo), float(hi)]


def at_fixed_sensitivity(y, score, target=0.90):
    """Lowest threshold whose sensitivity is still >= target."""
    fpr, tpr, thr = roc_curve(y, score)
    ok = np.where(tpr >= target)[0]
    i = ok[0]
    return {"threshold": float(thr[i]), "sensitivity": float(tpr[i]),
            "specificity": float(1 - fpr[i])}


def at_fixed_specificity(y, score, target=0.90):
    fpr, tpr, thr = roc_curve(y, score)
    ok = np.where((1 - fpr) >= target)[0]
    i = ok[-1]
    return {"threshold": float(thr[i]), "sensitivity": float(tpr[i]),
            "specificity": float(1 - fpr[i])}


def counts(y, pred):
    tp = int((y & pred).sum()); fn = int((y & ~pred).sum())
    tn = int((~y & ~pred).sum()); fp = int((~y & pred).sum())
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "ppv": tp / (tp + fp) if tp + fp else 0.0,
            "npv": tn / (tn + fn) if tn + fn else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="results/clinical_operating_point.json")
    ap.add_argument("--fig", default="figures/report_v2/figM_operating_point.png")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()

    print("=" * 78)
    print("REFERABLE-DR OPERATING POINT  (primary declared endpoint)")
    print("=" * 78)

    out = {"definition": "referable DR = true grade >= 2; score = prob_2+prob_3+prob_4",
           "n_boot": args.n_boot, "runs": {}}
    roc_store = {}

    for split in SPLITS:
        for sampler in SAMPLERS:
            for seed in SEEDS:
                f = root / "results" / f"finetune_converged_{split}_{sampler}_seed{seed}_test_predictions.csv"
                if not f.exists():
                    print(f"  SKIP (missing): {f.name}")
                    continue
                d = pd.read_csv(f)
                y = (d["true_grade"].to_numpy() >= 2)
                score = (d["prob_2"] + d["prob_3"] + d["prob_4"]).to_numpy()
                pid = d["patient_id"].to_numpy()

                auc = float(roc_auc_score(y, score))
                auc_ci = patient_bootstrap_ci(pid, y, score, roc_auc_score,
                                              args.n_boot, seed)
                sens90 = at_fixed_sensitivity(y, score, 0.90)
                spec90 = at_fixed_specificity(y, score, 0.90)
                c_sens90 = counts(y, score >= sens90["threshold"])
                c_spec90 = counts(y, score >= spec90["threshold"])
                c_argmax = counts(y, d["pred_grade"].to_numpy() >= 2)

                spec_at_sens90_ci = patient_bootstrap_ci(
                    pid, y, score,
                    lambda a, b: at_fixed_sensitivity(a, b, 0.90)["specificity"],
                    args.n_boot, seed)

                key = f"{split}_{sampler}_seed{seed}"
                out["runs"][key] = {
                    "split": split, "sampler": sampler, "seed": seed,
                    "n_test": int(len(d)),
                    "n_referable": int(y.sum()), "n_non_referable": int((~y).sum()),
                    "referable_auroc": auc, "referable_auroc_ci95": auc_ci,
                    "at_sensitivity_90": {**sens90, **c_sens90,
                                          "specificity_ci95": spec_at_sens90_ci},
                    "at_specificity_90": {**spec90, **c_spec90},
                    "at_argmax": c_argmax,
                }
                fpr, tpr, _ = roc_curve(y, score)
                roc_store[key] = (fpr, tpr, auc)
                print(f"  {key:34s} AUROC {auc:.4f} [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]")
                print(f"      @sens 90%: spec {c_sens90['specificity']*100:5.1f}%  "
                      f"PPV {c_sens90['ppv']*100:4.1f}%  missed {c_sens90['fn']:3d}/"
                      f"{int(y.sum())} referable  (thr {sens90['threshold']:.4f})")
                print(f"      @spec 90%: sens {c_spec90['sensitivity']*100:5.1f}%   "
                      f"@argmax: sens {c_argmax['sensitivity']*100:5.1f}% "
                      f"spec {c_argmax['specificity']*100:5.1f}%")

    # Primary endpoint = P2 (patient-level), sampler=none (the winning config)
    prim = [k for k in out["runs"] if k.startswith("p2_none")]
    if prim:
        sens = [out["runs"][k]["at_sensitivity_90"]["specificity"] for k in prim]
        aucs = [out["runs"][k]["referable_auroc"] for k in prim]
        out["primary_endpoint"] = {
            "protocol": "P2 (patient-level)", "sampler": "none",
            "runs": prim,
            "mean_specificity_at_90_sensitivity": float(np.mean(sens)),
            "mean_referable_auroc": float(np.mean(aucs)),
        }
        out["headline"] = (
            f"On the frozen patient-level test set, the converged model reaches referable-DR "
            f"AUROC {np.mean(aucs):.3f}. Held at the clinically required 90% sensitivity, it "
            f"achieves {np.mean(sens)*100:.1f}% specificity -- i.e. it catches 9 of every 10 "
            f"referable patients while correctly clearing {np.mean(sens)*100:.0f}% of those who "
            f"do not need referral.")
        print(f"\nPRIMARY ENDPOINT (P2, no resampling):")
        print(f"  referable AUROC          {np.mean(aucs):.4f}")
        print(f"  specificity @ 90% sens   {np.mean(sens)*100:.1f}%")

    (root / args.out).parent.mkdir(parents=True, exist_ok=True)
    (root / args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {root / args.out}")

    # ---- figure: ROC curves + the operating point ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.4))

    for key, (fpr, tpr, auc) in roc_store.items():
        if not key.startswith("p2_none"):
            continue
        a1.plot(fpr, tpr, lw=2, color=BLUE,
                label=f"P2, no resampling, seed {key[-2:]} (AUC {auc:.3f})")
    for key, (fpr, tpr, auc) in roc_store.items():
        if not key.startswith("p2_class"):
            continue
        a1.plot(fpr, tpr, lw=1.6, color=ORANGE, alpha=0.85, ls="--",
                label=f"P2, class-balanced, seed {key[-2:]} (AUC {auc:.3f})")
    a1.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
    a1.axhline(0.90, color=INK, lw=1.1, ls="--")
    a1.text(0.98, 0.915, "90% sensitivity — the clinical requirement",
            ha="right", fontsize=8.4, color=INK)
    a1.set_xlabel("1 − specificity (false referrals)")
    a1.set_ylabel("sensitivity (referable cases caught)")
    a1.set_xlim(0, 1); a1.set_ylim(0, 1.02)
    a1.grid(color="#eeeeee"); a1.legend(frameon=False, fontsize=7.8, loc="lower right")
    a1.set_title("Referable-DR ROC on the frozen patient-level test set", fontsize=10.5)

    keys = [k for k in out["runs"] if k.startswith("p2")]
    labs = [k.replace("p2_", "").replace("_seed", "\nseed ").replace("_", " ") for k in keys]
    sensv = [out["runs"][k]["at_sensitivity_90"]["sensitivity"] * 100 for k in keys]
    specv = [out["runs"][k]["at_sensitivity_90"]["specificity"] * 100 for k in keys]
    x = np.arange(len(keys)); w = 0.38
    a2.bar(x - w/2, sensv, w, color=BLUE, label="sensitivity (held at 90%)", zorder=3)
    a2.bar(x + w/2, specv, w, color=ORANGE, label="specificity achieved", zorder=3)
    for xi, v in zip(x + w/2, specv):
        a2.text(xi, v + 1.2, f"{v:.1f}%", ha="center", fontsize=9, weight="bold")
    a2.set_xticks(x); a2.set_xticklabels(labs, fontsize=8.4)
    a2.set_ylim(0, 105); a2.set_ylabel("percent")
    a2.grid(axis="y", color="#e8e8e8", zorder=0)
    a2.legend(frameon=False, fontsize=8.6, loc="upper right")
    a2.set_title("At the clinical operating point: what specificity\n"
                 "the model buys for 90% sensitivity", fontsize=10.5)
    fig.tight_layout()
    (root / args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / args.fig, dpi=200)
    print(f"Wrote {root / args.fig}")


if __name__ == "__main__":
    main()
