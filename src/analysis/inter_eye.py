"""
Inter-eye correlation analysis for diabetic retinopathy.
Measures how much one eye's DR grade tells you about the other eye's grade.

This single number decides the direction of the research paper:
  - HIGH correlation  -> image-level splits leak badly -> the "Two Eyes, One Patient"
                         paper is viable and interesting.
  - LOW correlation   -> fall back to the plain evaluation-protocol audit.

Requires only the EyePACS label CSV. No GPU, no images.

Usage:  python src/analysis/inter_eye.py --csv path/to/trainLabels15.csv
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

RNG = np.random.default_rng(42)
GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative"]


# ----------------------------------------------------------------------------
# 1. Load and reshape
# ----------------------------------------------------------------------------
def load_pairs(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    """Read the EyePACS CSV and pivot to one row per patient with both eyes."""
    df = pd.read_csv(csv_path)

    # Column names vary between mirrors. Find them positionally if needed.
    img_col = next((c for c in df.columns
                    if c.lower() in ("image", "id_code", "filename", "image_id")), df.columns[0])
    lbl_col = next((c for c in df.columns
                    if c.lower() in ("level", "diagnosis", "grade", "label")), df.columns[1])

    df = df[[img_col, lbl_col]].rename(columns={img_col: "image", lbl_col: "grade"})
    df["grade"] = df["grade"].astype(int)

    # Split '4521_left' -> patient 4521, eye 'left'
    parsed = df["image"].astype(str).str.extract(r"^(?P<patient>\d+)_(?P<eye>left|right)$")
    df = pd.concat([df, parsed], axis=1)
    n_unparsed = df["patient"].isna().sum()
    df = df.dropna(subset=["patient", "eye"])

    wide = df.pivot_table(index="patient", columns="eye", values="grade", aggfunc="first")
    complete = wide.dropna(subset=["left", "right"]).astype(int)

    meta = {
        "csv_rows": int(len(df) + n_unparsed),
        "rows_unparsed": int(n_unparsed),
        "unique_patients": int(wide.shape[0]),
        "patients_with_both_eyes": int(len(complete)),
        "patients_with_one_eye": int(wide.shape[0] - len(complete)),
    }
    return complete, meta


# ----------------------------------------------------------------------------
# 2. Bootstrap helper — resample PATIENTS, never images
# ----------------------------------------------------------------------------
def bootstrap_ci(fn, left, right, n=2000, alpha=0.05):
    idx = np.arange(len(left))
    stats = []
    for _ in range(n):
        s = RNG.choice(idx, size=len(idx), replace=True)
        try:
            stats.append(fn(left[s], right[s]))
        except Exception:
            continue
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ----------------------------------------------------------------------------
# 3. Core correlation metrics
# ----------------------------------------------------------------------------
def correlation_metrics(left: np.ndarray, right: np.ndarray) -> dict:
    qwk = cohen_kappa_score(left, right, weights="quadratic")
    lin = cohen_kappa_score(left, right, weights="linear")
    raw = cohen_kappa_score(left, right)
    rho, pval = spearmanr(left, right)

    diff = np.abs(left - right)
    exact = float((diff == 0).mean())
    within1 = float((diff <= 1).mean())

    ref_l, ref_r = (left >= 2), (right >= 2)
    ref_agree = float((ref_l == ref_r).mean())
    ref_kappa = cohen_kappa_score(ref_l, ref_r)

    qwk_lo, qwk_hi = bootstrap_ci(
        lambda a, b: cohen_kappa_score(a, b, weights="quadratic"), left, right)

    return {
        "quadratic_weighted_kappa": float(qwk),
        "qwk_ci95": [qwk_lo, qwk_hi],
        "linear_weighted_kappa": float(lin),
        "unweighted_kappa": float(raw),
        "spearman_rho": float(rho),
        "spearman_p": float(pval),
        "exact_agreement": exact,
        "agreement_within_1_grade": within1,
        "mean_abs_grade_difference": float(diff.mean()),
        "diff_distribution": {int(k): int(v) for k, v in
                              pd.Series(diff).value_counts().sort_index().items()},
        "referable_agreement": ref_agree,
        "referable_kappa": float(ref_kappa),
    }


# ----------------------------------------------------------------------------
# 4. The killer experiment:
#    predict eye B from eye A's GRADE ALONE — no image, no pixels, no network.
#    Fitted on a train split of patients, evaluated on held-out patients.
# ----------------------------------------------------------------------------
def label_only_baseline(left: np.ndarray, right: np.ndarray, test_frac=0.3) -> dict:
    n = len(left)
    perm = RNG.permutation(n)
    cut = int(n * (1 - test_frac))
    tr, te = perm[:cut], perm[cut:]

    # Lookup table: for each grade of the left eye, the most common right-eye grade.
    lookup = {}
    for g in range(5):
        mask = left[tr] == g
        lookup[g] = int(pd.Series(right[tr][mask]).mode()[0]) if mask.sum() else 0

    pred = np.array([lookup[g] for g in left[te]])
    truth = right[te]

    # Reference point: always predict the single most common grade overall.
    majority = int(pd.Series(right[tr]).mode()[0])
    maj_pred = np.full_like(truth, majority)

    return {
        "lookup_table": {f"left_grade_{k}": v for k, v in lookup.items()},
        "test_patients": int(len(te)),
        "lookup_qwk": float(cohen_kappa_score(truth, pred, weights="quadratic")),
        "lookup_accuracy": float((pred == truth).mean()),
        "lookup_referable_accuracy": float(((pred >= 2) == (truth >= 2)).mean()),
        "majority_baseline_qwk": float(cohen_kappa_score(truth, maj_pred, weights="quadratic")),
        "majority_baseline_accuracy": float((maj_pred == truth).mean()),
    }


# ----------------------------------------------------------------------------
# 5. Contingency table
# ----------------------------------------------------------------------------
def contingency(left: np.ndarray, right: np.ndarray) -> pd.DataFrame:
    tab = pd.crosstab(pd.Series(left, name="left_eye"),
                      pd.Series(right, name="right_eye"))
    return tab.reindex(index=range(5), columns=range(5), fill_value=0)


# ----------------------------------------------------------------------------
# 6. Verdict
# ----------------------------------------------------------------------------
def verdict(qwk: float, lookup_qwk: float) -> tuple[str, str]:
    if qwk >= 0.60:
        v = "STRONG"
        msg = ("Strong inter-eye correlation. Image-level random splitting leaks "
               "substantially. The 'Two Eyes, One Patient' paper is viable — proceed "
               "with all three claims.")
    elif qwk >= 0.40:
        v = "MODERATE"
        msg = ("Moderate correlation. The leakage effect is real but smaller than "
               "hoped. Keep the paper, but lead with Claim 2 (the evaluation audit) "
               "and present Claims 1 and 3 as supporting results.")
    else:
        v = "WEAK"
        msg = ("Weak correlation. The two-eye angle is not strong enough to carry a "
               "paper. Fall back to the plain evaluation-protocol audit "
               "(image-level vs patient-level vs cross-dataset). Still publishable.")

    if lookup_qwk >= 0.45:
        msg += (f"\n  NOTABLE: a label-only lookup table scores QWK {lookup_qwk:.3f} "
                "using no pixels at all. If a trained CNN scores near this on an "
                "image-level split, that is a headline finding — report it prominently.")

    return v, msg


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to the EyePACS label CSV")
    ap.add_argument("--out", default="results/inter_eye_correlation.json")
    args = ap.parse_args()

    pairs, meta = load_pairs(Path(args.csv))
    left = pairs["left"].to_numpy()
    right = pairs["right"].to_numpy()

    corr = correlation_metrics(left, right)
    base = label_only_baseline(left, right)
    tab = contingency(left, right)
    v, msg = verdict(corr["quadratic_weighted_kappa"], base["lookup_qwk"])

    # ---- report -------------------------------------------------------------
    print("\n" + "=" * 68)
    print("INTER-EYE CORRELATION ANALYSIS — EyePACS")
    print("=" * 68)

    print("\n--- DATA ---")
    for k, val in meta.items():
        print(f"  {k:32s} {val:,}")

    print("\n--- GRADE DISTRIBUTION (left eye) ---")
    for g, c in pd.Series(left).value_counts().sort_index().items():
        print(f"  {g} {GRADE_NAMES[g]:<16s} {c:6,}  ({c/len(left)*100:5.1f}%)")

    print("\n--- CORRELATION BETWEEN EYES ---")
    print(f"  Quadratic weighted kappa   {corr['quadratic_weighted_kappa']:.4f}  "
          f"95% CI [{corr['qwk_ci95'][0]:.4f}, {corr['qwk_ci95'][1]:.4f}]")
    print(f"  Linear weighted kappa      {corr['linear_weighted_kappa']:.4f}")
    print(f"  Unweighted kappa           {corr['unweighted_kappa']:.4f}")
    print(f"  Spearman rho               {corr['spearman_rho']:.4f}  (p={corr['spearman_p']:.2e})")
    print(f"  Exact agreement            {corr['exact_agreement']*100:.1f}%")
    print(f"  Within 1 grade             {corr['agreement_within_1_grade']*100:.1f}%")
    print(f"  Mean |grade difference|    {corr['mean_abs_grade_difference']:.3f}")

    print("\n--- REFERABLE DR (grade >= 2) ---")
    print(f"  Both eyes agree            {corr['referable_agreement']*100:.1f}%")
    print(f"  Cohen kappa                {corr['referable_kappa']:.4f}")

    print("\n--- GRADE DIFFERENCE DISTRIBUTION ---")
    tot = len(left)
    for d, c in corr["diff_distribution"].items():
        print(f"  |diff| = {d}   {c:6,}  ({c/tot*100:5.1f}%)")

    print("\n--- CONTINGENCY TABLE (rows = left, cols = right) ---")
    print(tab.to_string())

    print("\n--- LABEL-ONLY BASELINE (predict right eye from left eye's GRADE, no image) ---")
    print(f"  Lookup table               {base['lookup_table']}")
    print(f"  Held-out patients          {base['test_patients']:,}")
    print(f"  Lookup QWK                 {base['lookup_qwk']:.4f}")
    print(f"  Lookup accuracy            {base['lookup_accuracy']*100:.1f}%")
    print(f"  Lookup referable accuracy  {base['lookup_referable_accuracy']*100:.1f}%")
    print(f"  Majority-class QWK         {base['majority_baseline_qwk']:.4f}")
    print(f"  Majority-class accuracy    {base['majority_baseline_accuracy']*100:.1f}%")

    print("\n" + "=" * 68)
    print(f"VERDICT: {v}")
    print("=" * 68)
    print(msg)
    print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meta": meta,
        "correlation": corr,
        "label_only_baseline": base,
        "contingency_table": tab.to_dict(),
        "verdict": v,
    }, indent=2))
    print(f"Saved -> {out}\n")


if __name__ == "__main__":
    main()
