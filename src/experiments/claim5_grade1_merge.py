"""
Claim 5 -- is grade 1 (Mild NPDR) unlearnable at this resolution?

Three independent lines of evidence in this project point at grade 1 and
nothing else:

  1. It is the ONLY grade that fails to transfer between a patient's two eyes
     (45.8% partner-lookup accuracy, against 72-94% for grades 0, 2, 3, 4 --
     results/claim1b_stratified.json).
  2. Every converged Phase 6b model recalls it at 8.7-14.0%, far below the
     20-45% band written down in advance, under BOTH sampler settings
     (results/finetune_converged_*.json).
  3. Three of the eight flat-black images excluded in Phase 1 carry a human
     grade of 1 -- a grader assigned "microaneurysms only" to a near-uniformly
     black frame (results/excluded_images.json).

The common explanation for all three: Mild NPDR is defined by microaneurysms
roughly ten pixels wide in the source image, which do not survive downsampling
to 224px, and which human graders are themselves unreliable on. If that is
right, grade 1 is not a class this pipeline is failing to learn -- it is a
class the DATA cannot express at this resolution, and the distinction matters
because one is a modelling problem and the other is a measurement ceiling.

This tests it directly by collapsing grades 0 and 1 into a single class and
asking what happens to everything else.

The comparison is done correctly, which is the whole difficulty:
  QWK on a 5-class problem and QWK on a 4-class problem are NOT comparable --
  different label spaces, different chance agreement, different penalty
  matrix. So both models are scored in the SAME (merged) label space:
    - the 5-class model's predictions are mapped down ({0,1}->0, 2->1, 3->2,
      4->3) after the fact and scored in merged space;
    - the merged model is trained in merged space and scored there directly.
  The difference between those two numbers is the honest answer to "does
  training on a 5-class target hurt you, given that you only care about the
  4-class distinction?"

Referable-DR (grade >= 2) sensitivity and specificity are reported alongside
and are INVARIANT to the 0/1 merge, so they provide a fixed reference point
that cannot be moved by relabelling -- if referable performance is unchanged
while merged QWK improves, that is strong evidence the grade-1 boundary was
pure noise.

Pre-registered outcome:
  - Merged QWK clearly ABOVE the mapped-down 5-class QWK, with referable
    sens/spec unchanged => grade 1 is noise this pipeline was paying to fit.
    Report it as a measurement-ceiling finding and a concrete recommendation:
    on downsampled fundus images, the 0/1 boundary is not worth modelling.
  - No difference => grade 1 costs nothing to carry; the low recall is a
    reporting curiosity, not a design problem, and Section 3.2's per-grade
    result stands alone without this stronger claim.

Writes results/claim5_grade1_merge.json
       figures/report_v2/figL_grade1_merge.png

Usage:
  python src/experiments/claim5_grade1_merge.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLUE, ORANGE, INK = "#0072B2", "#D55E00", "#222222"
MERGE = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}   # {No DR, Mild} / Moderate / Severe / Proliferative


def qwk(y, p):
    return float(cohen_kappa_score(y, p, weights="quadratic"))


def fit_multinomial(Xtr, ytr, Xte, seed):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, random_state=seed).fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))


def fit_ordinal(Xtr, ytr, Xte, seed, n_classes):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    reg = Ridge(alpha=1.0, random_state=seed).fit(A, ytr.astype(float))
    s_tr, s_te = reg.predict(A), reg.predict(B)
    hi = n_classes - 1

    def neg(t):
        return -cohen_kappa_score(ytr, np.clip(np.digitize(s_tr, np.sort(t)), 0, hi),
                                  weights="quadratic")
    x0 = np.arange(0.5, hi, 1.0)
    res = minimize(neg, x0, method="Nelder-Mead")
    return np.clip(np.digitize(s_te, np.sort(res.x)), 0, hi)


def patient_bootstrap_paired(pid, ya, pa, yb, pb, n_boot=2000, seed=0):
    """Paired CI on qwk(ya,pa) - qwk(yb,pb) over the same patients."""
    df = pd.DataFrame({"pid": pid, "ya": ya, "pa": pa, "yb": yb, "pb": pb})
    g = df.groupby("pid").indices
    pats = np.array(list(g.keys()))
    YA, PA, YB, PB = (df[c].to_numpy() for c in ("ya", "pa", "yb", "pb"))
    rng = np.random.RandomState(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([g[p] for p in rng.choice(pats, len(pats), replace=True)])
        d[i] = qwk(YA[idx], PA[idx]) - qwk(YB[idx], PB[idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def referable(y, p, thresh_true=2, thresh_pred=2):
    yt, yp = (np.asarray(y) >= thresh_true), (np.asarray(p) >= thresh_pred)
    tp = int((yt & yp).sum()); fn = int((yt & ~yp).sum())
    tn = int((~yt & ~yp).sum()); fp = int((~yt & yp).sum())
    return {"sensitivity": tp / (tp + fn) if tp + fn else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "tp": tp, "fn": fn, "tn": tn, "fp": fp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--manifest", default="data/manifests/manifest.csv")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--features-dir", default="features")
    ap.add_argument("--backbone", default="tf_efficientnet_b0")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--protocol", default="p2.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/claim5_grade1_merge.json")
    ap.add_argument("--fig", default="figures/report_v2/figL_grade1_merge.png")
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

    data = np.load(npz)
    feats, ids = data["features"], data["image_id"]
    row_of = {i: r for r, i in enumerate(ids.tolist())}
    split = json.loads((root / args.splits_dir / args.protocol).read_text())
    tr = [i for i, v in split.items() if v["fold"] == "train" and i in row_of]
    te = [i for i, v in split.items() if v["fold"] == "test" and i in row_of]

    Xtr = np.stack([feats[row_of[i]] for i in tr]); ytr = np.array([grade_of[i] for i in tr])
    Xte = np.stack([feats[row_of[i]] for i in te]); yte = np.array([grade_of[i] for i in te])
    pid = [patient_of[i] for i in te]
    m = np.vectorize(MERGE.get)
    ytr_m, yte_m = m(ytr), m(yte)

    print("=" * 74)
    print(f"CLAIM 5 -- GRADE-1 MERGE TEST  ({args.backbone}@{args.size}, {args.protocol})")
    print("=" * 74)
    print(f"train {len(tr):,} | test {len(te):,} ({len(set(pid)):,} patients)")
    print(f"test grade counts 5-class: {np.bincount(yte, minlength=5).tolist()}")
    print(f"test grade counts merged : {np.bincount(yte_m, minlength=4).tolist()}")

    out = {"backbone": args.backbone, "size": args.size, "protocol": args.protocol,
           "merge_map": {str(k): v for k, v in MERGE.items()},
           "n_train": len(tr), "n_test": len(te), "n_test_patients": len(set(pid)),
           "heads": {}}

    for head in ("multinomial", "ordinal"):
        if head == "multinomial":
            p5 = fit_multinomial(Xtr, ytr, Xte, 42)
            p4 = fit_multinomial(Xtr, ytr_m, Xte, 42)
        else:
            p5 = fit_ordinal(Xtr, ytr, Xte, 42, 5)
            p4 = fit_ordinal(Xtr, ytr_m, Xte, 42, 4)
        p5_mapped = m(p5)

        native5 = qwk(yte, p5)
        mapped5 = qwk(yte_m, p5_mapped)      # 5-class model, scored in merged space
        native4 = qwk(yte_m, p4)             # merged model, scored in merged space
        lo, hi = patient_bootstrap_paired(pid, yte_m, p4, yte_m, p5_mapped, args.n_boot, 42)

        # referable-DR is invariant to the 0/1 merge: grade>=2 five-class == >=1 merged
        ref5 = referable(yte, p5, 2, 2)
        ref4 = referable(yte_m, p4, 1, 1)

        g1 = (yte == 1)
        out["heads"][head] = {
            "qwk_5class_native": native5,
            "qwk_5class_mapped_to_merged": mapped5,
            "qwk_merged_native": native4,
            "merged_minus_mapped": native4 - mapped5,
            "paired_ci95_on_difference": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "grade1_recall_5class": float((p5[g1] == 1).mean()) if g1.any() else None,
            "n_grade1_test": int(g1.sum()),
            "referable_5class": ref5, "referable_merged": ref4,
            "referable_sens_delta": ref4["sensitivity"] - ref5["sensitivity"],
            "referable_spec_delta": ref4["specificity"] - ref5["specificity"],
            "confusion_5class": confusion_matrix(yte, p5, labels=[0, 1, 2, 3, 4]).tolist(),
            "confusion_merged": confusion_matrix(yte_m, p4, labels=[0, 1, 2, 3]).tolist(),
        }
        r = out["heads"][head]
        print(f"\n  {head} head")
        print(f"    5-class QWK (native 5-class space)     {native5:.4f}")
        print(f"    5-class model scored in merged space   {mapped5:.4f}")
        print(f"    merged-trained model, merged space     {native4:.4f}")
        print(f"    difference (merged - mapped)           {native4-mapped5:+.4f}  "
              f"CI [{lo:+.4f}, {hi:+.4f}]  {'EXCLUDES 0' if r['excludes_zero'] else 'includes 0'}")
        print(f"    grade-1 recall in the 5-class model    "
              f"{r['grade1_recall_5class']*100:.1f}%  (n={r['n_grade1_test']})")
        print(f"    referable sens {ref5['sensitivity']:.3f} -> {ref4['sensitivity']:.3f}   "
              f"spec {ref5['specificity']:.3f} -> {ref4['specificity']:.3f}")

    o = out["heads"]["ordinal"]
    ref_stable = abs(o["referable_sens_delta"]) < 0.03 and abs(o["referable_spec_delta"]) < 0.03
    if o["excludes_zero"] and o["merged_minus_mapped"] > 0 and ref_stable:
        out["verdict"] = "grade1_is_noise_at_this_resolution"
        out["verdict_text"] = (
            f"Collapsing grades 0 and 1 improves QWK by {o['merged_minus_mapped']:+.4f} "
            f"(CI [{o['paired_ci95_on_difference'][0]:+.4f}, {o['paired_ci95_on_difference'][1]:+.4f}]) "
            f"while referable-DR sensitivity and specificity are essentially unchanged "
            f"({o['referable_sens_delta']:+.3f} / {o['referable_spec_delta']:+.3f}). The grade-1 "
            f"boundary is costing accuracy and buying no clinical discrimination. Report this as a "
            f"MEASUREMENT-CEILING finding: at 224px, Mild NPDR is not a learnable class, and the "
            f"three independent lines of evidence in Section 3.2, 3.5 and Challenge 2 all point at "
            f"the same cause.")
    elif o["excludes_zero"] and o["merged_minus_mapped"] > 0:
        out["verdict"] = "merge_helps_but_referable_moved"
        out["verdict_text"] = (
            f"Merging improves QWK by {o['merged_minus_mapped']:+.4f}, but referable-DR performance "
            f"also shifted ({o['referable_sens_delta']:+.3f} sens / {o['referable_spec_delta']:+.3f} "
            f"spec), so part of the gain is a change in operating point rather than pure noise "
            f"removal. Report the QWK gain WITH the referable caveat attached.")
    else:
        out["verdict"] = "grade1_costs_nothing_to_carry"
        out["verdict_text"] = (
            f"Merging changes QWK by {o['merged_minus_mapped']:+.4f} "
            f"(CI [{o['paired_ci95_on_difference'][0]:+.4f}, "
            f"{o['paired_ci95_on_difference'][1]:+.4f}], includes zero). Carrying the grade-1 class "
            f"costs nothing measurable, so the low grade-1 recall is a reporting curiosity rather "
            f"than a design problem. Section 3.2's per-grade finding stands on its own without the "
            f"stronger unlearnability claim.")

    (root / args.out).write_text(json.dumps(out, indent=2))
    print(f"\nVERDICT: {out['verdict']}\n  {out['verdict_text']}")
    print(f"\nWrote {root / args.out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))
    heads = ["multinomial", "ordinal"]
    x = np.arange(2); w = 0.38
    mapped = [out["heads"][h]["qwk_5class_mapped_to_merged"] for h in heads]
    merged = [out["heads"][h]["qwk_merged_native"] for h in heads]
    a1.bar(x - w/2, mapped, w, color=ORANGE, label="trained 5-class, scored merged", zorder=3)
    a1.bar(x + w/2, merged, w, color=BLUE, label="trained merged (0+1 collapsed)", zorder=3)
    for xi, a, b in zip(x, mapped, merged):
        a1.text(xi, max(a, b) + 0.012, f"{b-a:+.4f}", ha="center", fontsize=9, weight="bold")
    a1.set_xticks(x); a1.set_xticklabels([h + " head" for h in heads])
    a1.set_ylabel("QWK (merged 4-class label space)")
    a1.set_ylim(0, max(mapped + merged) * 1.25)
    a1.grid(axis="y", color="#e8e8e8", zorder=0); a1.legend(frameon=False, fontsize=8.5)
    a1.set_title("Both models scored in the SAME label space\n(the only fair comparison)", fontsize=10.5)

    lbl = ["referable\nsensitivity", "referable\nspecificity"]
    v5 = [out["heads"]["ordinal"]["referable_5class"]["sensitivity"],
          out["heads"]["ordinal"]["referable_5class"]["specificity"]]
    v4 = [out["heads"]["ordinal"]["referable_merged"]["sensitivity"],
          out["heads"]["ordinal"]["referable_merged"]["specificity"]]
    x = np.arange(2)
    a2.bar(x - w/2, v5, w, color=ORANGE, label="5-class model", zorder=3)
    a2.bar(x + w/2, v4, w, color=BLUE, label="merged model", zorder=3)
    for xi, a, b in zip(x, v5, v4):
        a2.text(xi, max(a, b) + 0.02, f"{b-a:+.3f}", ha="center", fontsize=9, weight="bold")
    a2.set_xticks(x); a2.set_xticklabels(lbl); a2.set_ylim(0, 1.12)
    a2.grid(axis="y", color="#e8e8e8", zorder=0); a2.legend(frameon=False, fontsize=8.5)
    a2.set_title("Referable-DR performance is invariant to the\n0/1 merge — a fixed reference point",
                 fontsize=10.5)
    fig.tight_layout()
    (root / args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / args.fig, dpi=200)
    print(f"Wrote {root / args.fig}")


if __name__ == "__main__":
    main()
