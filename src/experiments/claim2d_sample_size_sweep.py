"""
Claim 2d -- does leakage inflation depend on TRAINING-SET SIZE?

This is the experiment that tries to reconcile this project's null result
with the published positive ones, instead of just noting that they disagree.

The disagreement, restated: Tampu et al. (2022) measured 0.07-0.43 MCC
inflation from image-level splitting on three public OCT datasets, and Yagis
et al. (2021) measured a large effect on brain MRI. This project measured
nothing detectable on EyePACS. The obvious difference between those settings
and this one is SCALE. Their datasets are small; EyePACS gives 27,145
training images. A model with 27k examples has no reason to memorise patient
identity -- there is plenty of genuine signal to fit. A model with 500 might
have every reason to.

If that is the explanation, the P1-minus-P2 gap should be large at small N
and shrink toward zero as N grows. That is a curve, and it is falsifiable:

  - GAP GROWS AS N SHRINKS  => leakage inflation is a small-data phenomenon.
    The null in Section 3.3 is then not a contradiction of Tampu/Yagis but a
    boundary condition on them, and this project's contribution becomes the
    curve rather than the null.
  - GAP FLAT AND ~ZERO AT EVERY N => the size hypothesis is dead, the null is
    stronger than before (the obvious explanation has been ruled out), and
    the disagreement with the OCT/MRI literature must be modality- or
    pipeline-specific instead.

Both outcomes are worth reporting. That is the point of running it.

Design:
  - Frozen cached features, so this is a fitting loop, not a training run.
  - At each size N, subsample the P1 train fold and the P2 train fold
    independently to N images, fit both heads, evaluate.
  - EVALUATION IS ON THE COMMON TEST IMAGES (intersection of the P1 and P2
    test folds), so the P1-vs-P2 comparison isolates what is in TRAINING and
    is not confounded by the two protocols having different test sets.
  - R independent replicates per size (different subsamples, different seeds)
    give a spread, and the paired patient-level bootstrap gives a CI at each
    size on the largest-N and smallest-N points.
  - A second, more direct mechanism curve is computed alongside: within the
    P1 model at each N, the partner-present minus partner-absent gap. This
    measures the mechanism itself rather than its whole-protocol shadow, and
    it is the cleaner signal if the two ever disagree.

EyePACS-only throughout: APTOS has no eye pairing and cannot express the
mechanism under test.

Writes results/claim2d_sample_size_sweep.json
       figures/report_v2/figK_sample_size_sweep.png

Usage:
  python src/experiments/claim2d_sample_size_sweep.py
  python src/experiments/claim2d_sample_size_sweep.py --sizes 500 1000 2000 5000 10000 20000 --replicates 5
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLUE, ORANGE, INK, MUTED = "#0072B2", "#D55E00", "#222222", "#888888"


def qwk(y, p):
    return float(cohen_kappa_score(y, p, weights="quadratic"))


def fit_multinomial(Xtr, ytr, Xte, seed):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))


def fit_ordinal(Xtr, ytr, Xte, seed):
    """Ridge on the integer grade, thresholds optimised on TRAIN only."""
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    reg = Ridge(alpha=1.0, random_state=seed).fit(A, ytr.astype(float))
    s_tr, s_te = reg.predict(A), reg.predict(B)

    def neg(t):
        return -cohen_kappa_score(ytr, np.clip(np.digitize(s_tr, np.sort(t)), 0, 4),
                                  weights="quadratic")
    res = minimize(neg, np.array([0.5, 1.5, 2.5, 3.5]), method="Nelder-Mead")
    return np.clip(np.digitize(s_te, np.sort(res.x)), 0, 4)


def paired_patient_bootstrap_diff(pid, y, pa, pb, n_boot=1000, seed=0):
    df = pd.DataFrame({"pid": pid, "y": y, "a": pa, "b": pb})
    g = df.groupby("pid").indices
    pats = np.array(list(g.keys()))
    Y, A, B = df["y"].to_numpy(), df["a"].to_numpy(), df["b"].to_numpy()
    rng = np.random.RandomState(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([g[p] for p in rng.choice(pats, len(pats), replace=True)])
        d[i] = qwk(Y[idx], A[idx]) - qwk(Y[idx], B[idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--manifest", default="data/manifests/manifest.csv")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--features-dir", default="features")
    ap.add_argument("--backbone", default="tf_efficientnet_b0")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[500, 1000, 2000, 5000, 10000, 20000])
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="results/claim2d_sample_size_sweep.json")
    ap.add_argument("--fig", default="figures/report_v2/figK_sample_size_sweep.png")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()

    npz = root / args.features_dir / f"{args.backbone}_{args.size}.npz"
    if not npz.exists():
        raise SystemExit(f"ERROR: {npz} not found. Run src/features/extract.py first.")

    man = pd.read_csv(root / args.manifest)
    if "excluded" in man.columns:
        man = man[~man["excluded"].fillna(False).astype(bool)]
    grade_of = dict(zip(man["image_id"], man["grade"]))
    patient_of = dict(zip(man["image_id"], man["patient_id"]))
    partner_of = dict(zip(man["image_id"], man["partner_id"]))

    data = np.load(npz)
    feats, ids = data["features"], data["image_id"]
    row_of = {i: r for r, i in enumerate(ids.tolist())}

    p1 = json.loads((root / args.splits_dir / "p1.json").read_text())
    p2 = json.loads((root / args.splits_dir / "p2.json").read_text())
    ey = lambda i: str(i).startswith("eyepacs_")
    tr1 = [i for i, v in p1.items() if v["fold"] == "train" and ey(i) and i in row_of]
    tr2 = [i for i, v in p2.items() if v["fold"] == "train" and ey(i) and i in row_of]
    te1 = {i for i, v in p1.items() if v["fold"] == "test" and ey(i) and i in row_of}
    te2 = {i for i, v in p2.items() if v["fold"] == "test" and ey(i) and i in row_of}
    common_test = sorted(te1 & te2)
    tr1_set = set(tr1)

    Xte = np.stack([feats[row_of[i]] for i in common_test])
    yte = np.array([grade_of[i] for i in common_test])
    pid_te = [patient_of[i] for i in common_test]
    partner_present = np.array([partner_of.get(i) in tr1_set for i in common_test])

    print("=" * 74)
    print(f"CLAIM 2d -- SAMPLE-SIZE SWEEP  ({args.backbone}@{args.size}, EyePACS only)")
    print("=" * 74)
    print(f"P1 train pool {len(tr1):,} | P2 train pool {len(tr2):,} | "
          f"common test {len(common_test):,} imgs, {len(set(pid_te)):,} patients")
    print(f"  of the common test images, {int(partner_present.sum()):,} have their partner eye "
          f"in the P1 train fold and {int((~partner_present).sum()):,} do not")

    sizes = [s for s in args.sizes if s <= min(len(tr1), len(tr2))] + [min(len(tr1), len(tr2))]
    sizes = sorted(set(sizes))
    rows = []
    for N in sizes:
        for rep in range(args.replicates):
            seed = 1000 * rep + N % 997
            rng = np.random.RandomState(seed)
            s1 = [tr1[k] for k in rng.choice(len(tr1), N, replace=False)]
            s2 = [tr2[k] for k in rng.choice(len(tr2), N, replace=False)]
            X1 = np.stack([feats[row_of[i]] for i in s1]); y1 = np.array([grade_of[i] for i in s1])
            X2 = np.stack([feats[row_of[i]] for i in s2]); y2 = np.array([grade_of[i] for i in s2])
            rec = {"n_train": int(N), "replicate": rep, "seed": int(seed)}
            for head, fn in (("multinomial", fit_multinomial), ("ordinal", fit_ordinal)):
                a = fn(X1, y1, Xte, seed)   # P1-trained
                b = fn(X2, y2, Xte, seed)   # P2-trained
                rec[head] = {
                    "qwk_p1": qwk(yte, a), "qwk_p2": qwk(yte, b),
                    "gap_p1_minus_p2": qwk(yte, a) - qwk(yte, b),
                    # mechanism curve: within the P1 model only
                    "qwk_partner_present": qwk(yte[partner_present], a[partner_present]),
                    "qwk_partner_absent": qwk(yte[~partner_present], a[~partner_present]),
                }
                rec[head]["partner_gap"] = (rec[head]["qwk_partner_present"]
                                            - rec[head]["qwk_partner_absent"])
            rows.append(rec)
        m = np.mean([r["ordinal"]["gap_p1_minus_p2"] for r in rows if r["n_train"] == N])
        pg = np.mean([r["ordinal"]["partner_gap"] for r in rows if r["n_train"] == N])
        print(f"  N={N:6,}  ordinal  P1-P2 gap {m:+.4f}   partner-present-minus-absent {pg:+.4f}")

    # paired CI at the smallest and largest N (ordinal head, replicate 0)
    ci = {}
    for N in (sizes[0], sizes[-1]):
        rng = np.random.RandomState(N % 997)
        s1 = [tr1[k] for k in rng.choice(len(tr1), N, replace=False)]
        s2 = [tr2[k] for k in rng.choice(len(tr2), N, replace=False)]
        X1 = np.stack([feats[row_of[i]] for i in s1]); y1 = np.array([grade_of[i] for i in s1])
        X2 = np.stack([feats[row_of[i]] for i in s2]); y2 = np.array([grade_of[i] for i in s2])
        a = fit_ordinal(X1, y1, Xte, 42); b = fit_ordinal(X2, y2, Xte, 42)
        lo, hi = paired_patient_bootstrap_diff(pid_te, yte, a, b, args.n_boot, 42)
        ci[str(N)] = {"gap": qwk(yte, a) - qwk(yte, b), "ci95": [lo, hi],
                      "excludes_zero": bool(lo > 0 or hi < 0)}
        print(f"  paired CI at N={N:,}: gap {ci[str(N)]['gap']:+.4f} "
              f"CI [{lo:+.4f}, {hi:+.4f}]  {'EXCLUDES 0' if ci[str(N)]['excludes_zero'] else 'includes 0'}")

    df = pd.DataFrame([{**{"n_train": r["n_train"], "replicate": r["replicate"]},
                        **{f"{h}_{k}": v for h in ("multinomial", "ordinal")
                           for k, v in r[h].items()}} for r in rows])
    agg = df.groupby("n_train").agg(["mean", "std"])

    small = float(df[df.n_train == sizes[0]]["ordinal_gap_p1_minus_p2"].mean())
    large = float(df[df.n_train == sizes[-1]]["ordinal_gap_p1_minus_p2"].mean())
    small_pg = float(df[df.n_train == sizes[0]]["ordinal_partner_gap"].mean())
    large_pg = float(df[df.n_train == sizes[-1]]["ordinal_partner_gap"].mean())
    trend = small - large

    out = {"backbone": args.backbone, "size": args.size, "sizes": sizes,
           "replicates": args.replicates, "scope": "eyepacs_only",
           "n_common_test_images": len(common_test),
           "n_common_test_patients": len(set(pid_te)),
           "n_partner_present": int(partner_present.sum()),
           "n_partner_absent": int((~partner_present).sum()),
           "rows": rows, "paired_ci_endpoints": ci,
           "gap_at_smallest_N": small, "gap_at_largest_N": large,
           "partner_gap_at_smallest_N": small_pg, "partner_gap_at_largest_N": large_pg,
           "gap_shrinkage_small_minus_large": trend}

    if trend > 0.02 and small > 0.02:
        out["verdict"] = "leakage_is_a_small_data_effect"
        out["verdict_text"] = (
            f"The P1-minus-P2 gap is {small:+.4f} QWK at N={sizes[0]:,} and {large:+.4f} at "
            f"N={sizes[-1]:,}, shrinking by {trend:.4f} as training data grows. This RECONCILES "
            f"this project's null with Tampu et al. and Yagis et al.: image-level leakage inflates "
            f"scores when data is scarce and washes out at EyePACS scale. Report the curve as the "
            f"headline finding -- it is a boundary condition on the published results, not a "
            f"contradiction of them.")
    elif abs(trend) <= 0.02 and abs(small) <= 0.02:
        out["verdict"] = "no_size_dependence_null_strengthened"
        out["verdict_text"] = (
            f"The gap is {small:+.4f} at N={sizes[0]:,} and {large:+.4f} at N={sizes[-1]:,} -- flat "
            f"and near zero throughout. The small-data explanation for the disagreement with "
            f"Tampu/Yagis is RULED OUT, which makes the null result stronger, not weaker: the most "
            f"obvious confound has been tested and eliminated. The remaining explanation must be "
            f"modality- or pipeline-specific.")
    else:
        out["verdict"] = "inconclusive_trend"
        out["verdict_text"] = (
            f"Gap {small:+.4f} at N={sizes[0]:,} vs {large:+.4f} at N={sizes[-1]:,} (change "
            f"{trend:+.4f}). The trend is not clean enough to call either way -- report the curve "
            f"with its spread and do not claim a size dependence.")

    (root / args.out).write_text(json.dumps(out, indent=2))
    print(f"\nVERDICT: {out['verdict']}\n  {out['verdict_text']}")
    print(f"\nWrote {root / args.out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.4))
    for ax, col, title, ylab in (
        (a1, "gap_p1_minus_p2", "Whole-protocol gap\nP1 (image-level) minus P2 (patient-level)",
         "QWK gap"),
        (a2, "partner_gap", "Mechanism gap, within the P1 model\npartner eye in training minus absent",
         "QWK gap")):
        for head, c in (("ordinal", BLUE), ("multinomial", ORANGE)):
            m = agg[(f"{head}_{col}", "mean")]; s = agg[(f"{head}_{col}", "std")].fillna(0)
            ax.plot(m.index, m.values, "-o", color=c, ms=5, lw=2, label=f"{head} head", zorder=3)
            ax.fill_between(m.index, m - s, m + s, color=c, alpha=0.15, zorder=2)
        ax.axhline(0, color=INK, lw=1.1, zorder=2)
        ax.set_xscale("log"); ax.set_xlabel("training-set size (images, log scale)")
        ax.set_ylabel(ylab); ax.grid(color="#eeeeee", zorder=0)
        ax.legend(frameon=False, fontsize=9); ax.set_title(title, fontsize=10.5)
    fig.suptitle("Does image-level leakage depend on how much training data you have? "
                 "Shaded band = ±1 SD over replicates.", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    (root / args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / args.fig, dpi=200)
    print(f"Wrote {root / args.fig}")


if __name__ == "__main__":
    main()
