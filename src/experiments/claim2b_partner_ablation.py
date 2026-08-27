"""
Claim 2b -- the partner-eye ablation (MASTER_PLAN.md S3.4, Part 8.3).
The paper's decisive experiment and headline figure (Figure 3).

Design:
  1. Train ONE head on frozen features under the P1 (image-level) split's
     train fold.
  2. Partition P1's TEST set into partner-present (the image's fellow eye
     WAS in the training set -- recorded per-image by src/data/splits.py as
     `partner_in_train`) vs partner-absent. APTOS rows have no partner
     concept (`partner_in_train` is null) and are excluded from this
     ablation -- it is specifically about EyePACS's paired-eye leakage.
  3. Evaluate the SAME model, SAME weights, on both groups separately.

Two heads are analyzed with IDENTICAL downstream statistics
(run_ablation_analysis()), not just one: multinomial logistic regression
(the primary, headline result -- Figure 3 is built from this) and an
ordinal head (ridge + thresholds, from src/experiments/claim2_protocols.py)
as a robustness check. This exists because Claim 2's own protocol
comparison found the multinomial head under-shoots MASTER_PLAN.md's
expected QWK range and an ordinal head recovers meaningful QWK on the same
features/split -- so a null result here under multinomial alone would be
worth re-checking under a head that actually respects grade order before
reporting "no leakage detected" as a finding.

Critical detail from the plan: the two groups must be matched on grade
distribution or a QWK difference could reflect case difficulty rather than
leakage. This script does BOTH of the plan's suggested remedies, for each
head:
  (a) reports QWK broken out per grade for each group (stratified view), and
  (b) reports a grade-matched comparison: the larger group is subsampled
      WITHOUT replacement to match the smaller group's per-grade counts,
      and QWK is recomputed on that matched subsample. Both raw and
      grade-matched numbers are reported side by side -- if they tell the
      same story, grade imbalance was not driving the raw difference.

Also reports, for each head: a bootstrap CI on the difference (resampling
PATIENTS independently within each group, not images) and a label-shuffle
permutation test (shuffle the partner-present/absent assignment across the
P1 test set, holding true grades and model predictions fixed, to get a null
distribution for the QWK difference).

Standalone, CPU-only (frozen features). Run locally.

Writes results/claim2b_partner_ablation.json (both heads) and
figures/figure3_claim2b_partner_ablation.png (multinomial -- the headline
claim; the ordinal robustness numbers are annotated in the subtitle).

Usage:
    python src\\experiments\\claim2b_partner_ablation.py --backbone tf_efficientnet_b0 --size 224
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import cohen_kappa_score

from src.experiments.claim2_protocols import load_active_manifest, build_xy, fit_and_eval, ordinal_qwk

GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative"]


def qwk(y_true, y_pred):
    if len(y_true) == 0:
        return None
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def per_grade_breakdown(y_true, y_pred):
    out = {}
    for g in range(5):
        mask = y_true == g
        n_g = int(mask.sum())
        if n_g == 0:
            out[g] = {"n": 0, "accuracy": None}
            continue
        out[g] = {"n": n_g, "accuracy": float((y_pred[mask] == y_true[mask]).mean())}
    return out


def grade_matched_subsample(y_true_larger, rng, target_grade_counts):
    """Subsample the larger group WITHOUT replacement so its per-grade
    counts match target_grade_counts (the smaller group's). Returns the
    selected row indices (into the larger group's arrays) and any shortfall
    (grades where the larger group didn't have enough to match exactly)."""
    idx_by_grade = {g: np.where(y_true_larger == g)[0] for g in range(5)}
    selected, shortfall = [], {}
    for g, target_n in target_grade_counts.items():
        avail = idx_by_grade.get(g, np.array([], dtype=int))
        if len(avail) >= target_n:
            sel = rng.choice(avail, size=target_n, replace=False)
        else:
            sel = avail
            if target_n > 0:
                shortfall[g] = int(target_n - len(avail))
        selected.extend(sel.tolist())
    return np.array(selected, dtype=int), shortfall


def bootstrap_ci_diff(patient_ids_a, y_true_a, y_pred_a, patient_ids_b, y_true_b, y_pred_b,
                       n_boot=1000, seed=42):
    """Bootstrap the DIFFERENCE in QWK (present - absent), resampling
    patients independently within each group."""
    rng = np.random.RandomState(seed)

    def group_arrays(patient_ids, y_true, y_pred):
        df = pd.DataFrame({"patient_id": patient_ids, "y_true": y_true, "y_pred": y_pred})
        groups = df.groupby("patient_id").indices
        return np.array(list(groups.keys())), groups, df["y_true"].to_numpy(), df["y_pred"].to_numpy()

    pats_a, groups_a, yt_a, yp_a = group_arrays(patient_ids_a, y_true_a, y_pred_a)
    pats_b, groups_b, yt_b, yp_b = group_arrays(patient_ids_b, y_true_b, y_pred_b)

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        sampled_a = rng.choice(pats_a, size=len(pats_a), replace=True)
        idx_a = np.concatenate([groups_a[p] for p in sampled_a])
        sampled_b = rng.choice(pats_b, size=len(pats_b), replace=True)
        idx_b = np.concatenate([groups_b[p] for p in sampled_b])
        diffs[b] = (cohen_kappa_score(yt_a[idx_a], yp_a[idx_a], weights="quadratic")
                    - cohen_kappa_score(yt_b[idx_b], yp_b[idx_b], weights="quadratic"))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def permutation_test_group_diff(y_true, y_pred, is_present, n_perm=2000, seed=42):
    """Shuffle the partner-present/absent LABEL across the P1 test set
    (holding true grades and this model's predictions fixed), recompute the
    QWK difference each time."""
    rng = np.random.RandomState(seed)
    observed = qwk(y_true[is_present], y_pred[is_present]) - qwk(y_true[~is_present], y_pred[~is_present])
    n_present, n = int(is_present.sum()), len(is_present)
    null_diffs = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = np.zeros(n, dtype=bool)
        shuffled[rng.choice(n, size=n_present, replace=False)] = True
        null_diffs[i] = (qwk(y_true[shuffled], y_pred[shuffled])
                          - qwk(y_true[~shuffled], y_pred[~shuffled]))
    p_value = float((np.abs(null_diffs) >= abs(observed) - 1e-12).mean())
    return {"observed_diff": float(observed), "null_mean": float(null_diffs.mean()),
            "null_std": float(null_diffs.std()), "p_value_two_sided": p_value, "n_permutations": n_perm}


def run_ablation_analysis(head_name, y_test, y_pred, is_present, ordered_test_ids, patient_by_id,
                           seed, n_bootstrap, n_permutations):
    """Everything downstream of 'we have a fitted model's predictions on the
    full P1 test set' -- identical statistics for whichever head produced
    y_pred. Prints a report section and returns the JSON-able result dict."""
    y_true_p, y_pred_p = y_test[is_present], y_pred[is_present]
    y_true_a, y_pred_a = y_test[~is_present], y_pred[~is_present]
    patients_p = [patient_by_id[i] for i in np.array(ordered_test_ids)[is_present]]
    patients_a = [patient_by_id[i] for i in np.array(ordered_test_ids)[~is_present]]

    qwk_present, qwk_absent = qwk(y_true_p, y_pred_p), qwk(y_true_a, y_pred_a)
    print(f"\n=== HEAD: {head_name} ===")
    print(f"--- RAW QWK (same model, same weights) ---")
    print(f"  partner-present  n={len(y_true_p):5d}  QWK={qwk_present:.4f}")
    print(f"  partner-absent   n={len(y_true_a):5d}  QWK={qwk_absent:.4f}")
    print(f"  difference (present - absent) = {qwk_present - qwk_absent:+.4f}")

    grade_dist_p = pd.Series(y_true_p).value_counts(normalize=True).reindex(range(5), fill_value=0.0)
    grade_dist_a = pd.Series(y_true_a).value_counts(normalize=True).reindex(range(5), fill_value=0.0)

    breakdown_p = per_grade_breakdown(y_true_p, y_pred_p)
    breakdown_a = per_grade_breakdown(y_true_a, y_pred_a)
    print("--- PER-GRADE ACCURACY, STRATIFIED ---")
    for g in range(5):
        print(f"  grade {g} ({GRADE_NAMES[g]:<14s})  present: n={breakdown_p[g]['n']:4d} "
              f"acc={breakdown_p[g]['accuracy']}   absent: n={breakdown_a[g]['n']:4d} acc={breakdown_a[g]['accuracy']}")

    rng = np.random.RandomState(seed)
    if len(y_true_p) >= len(y_true_a):
        larger_true, larger_pred, smaller_true, smaller_pred = y_true_p, y_pred_p, y_true_a, y_pred_a
        larger_name, smaller_name = "present", "absent"
    else:
        larger_true, larger_pred, smaller_true, smaller_pred = y_true_a, y_pred_a, y_true_p, y_pred_p
        larger_name, smaller_name = "absent", "present"

    target_counts = {g: int((smaller_true == g).sum()) for g in range(5)}
    sel_idx, shortfall = grade_matched_subsample(larger_true, rng, target_counts)
    qwk_matched_larger = qwk(larger_true[sel_idx], larger_pred[sel_idx])
    qwk_matched_smaller = qwk(smaller_true, smaller_pred)
    qwk_matched_present = qwk_matched_larger if larger_name == "present" else qwk_matched_smaller
    qwk_matched_absent = qwk_matched_larger if larger_name == "absent" else qwk_matched_smaller

    print(f"--- GRADE-MATCHED (subsampled '{larger_name}' to match '{smaller_name}') ---")
    if shortfall:
        print(f"  NOTE: shortfall: {shortfall}")
    print(f"  matched present={qwk_matched_present:.4f}  matched absent={qwk_matched_absent:.4f}  "
          f"matched diff={qwk_matched_present - qwk_matched_absent:+.4f}  (raw diff was {qwk_present - qwk_absent:+.4f})")

    ci_lo, ci_hi = bootstrap_ci_diff(patients_p, y_true_p, y_pred_p, patients_a, y_true_a, y_pred_a,
                                      n_boot=n_bootstrap, seed=seed)
    perm_result = permutation_test_group_diff(y_test, y_pred, is_present, n_perm=n_permutations, seed=seed)
    print(f"--- BOOTSTRAP CI on difference: [{ci_lo:.4f}, {ci_hi:.4f}]   "
          f"PERMUTATION p={perm_result['p_value_two_sided']:.4g} ---")

    if ci_lo > 0 and perm_result["p_value_two_sided"] < 0.05:
        interpretation = ("Partner-present significantly outperforms partner-absent (CI excludes 0, "
                           "permutation p<0.05) -- DIRECT EVIDENCE OF LEAKAGE from image-level splitting.")
    elif ci_hi < 0 and perm_result["p_value_two_sided"] < 0.05:
        interpretation = ("Partner-absent significantly outperforms partner-present -- unexpected; "
                           "inspect before reporting, this contradicts the leakage hypothesis.")
    else:
        interpretation = ("No significant difference detected (CI includes 0, or p>=0.05) -- report "
                           "honestly as a negative/null result on this ablation for this head.")
    print(f"--- INTERPRETATION ({head_name}) ---\n{interpretation}")

    return {
        "raw_qwk_present": qwk_present, "raw_qwk_absent": qwk_absent,
        "raw_difference": qwk_present - qwk_absent,
        "grade_distribution_present": {str(g): float(v) for g, v in grade_dist_p.items()},
        "grade_distribution_absent": {str(g): float(v) for g, v in grade_dist_a.items()},
        "per_grade_accuracy_present": {str(g): v for g, v in breakdown_p.items()},
        "per_grade_accuracy_absent": {str(g): v for g, v in breakdown_a.items()},
        "grade_matched_comparison": {
            "larger_group_subsampled": larger_name, "shortfall": shortfall,
            "qwk_present": qwk_matched_present, "qwk_absent": qwk_matched_absent,
            "difference": qwk_matched_present - qwk_matched_absent,
        },
        "bootstrap_ci95_on_difference": [ci_lo, ci_hi],
        "permutation_test": perm_result,
        "interpretation": interpretation,
    }, grade_dist_p, grade_dist_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--manifest", default="data/manifests/manifest.csv")
    ap.add_argument("--p1-split", default="data/splits/p1.json")
    ap.add_argument("--features-dir", default="features")
    ap.add_argument("--backbone", default="tf_efficientnet_b0")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--n-permutations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/claim2b_partner_ablation.json")
    ap.add_argument("--fig", default="figures/figure3_claim2b_partner_ablation.png")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    npz_path = root / args.features_dir / f"{args.backbone}_{args.size}.npz"
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found. Run src/features/extract.py for this config first.", file=sys.stderr)
        sys.exit(1)

    active_manifest = load_active_manifest(root / args.manifest)
    grade_by_id = dict(zip(active_manifest["image_id"], active_manifest["grade"]))
    patient_by_id = dict(zip(active_manifest["image_id"], active_manifest["patient_id"]))

    data = np.load(npz_path)
    feats, ids = data["features"], data["image_id"]
    row_of = {i: r for r, i in enumerate(ids.tolist())}

    p1_split = json.loads((root / args.p1_split).read_text())
    train_ids_all = [i for i, v in p1_split.items() if v["fold"] == "train"]
    test_items = [(i, v) for i, v in p1_split.items() if v["fold"] == "test"]

    print("=" * 72)
    print(f"CLAIM 2b -- PARTNER-EYE ABLATION  ({args.backbone}@{args.size}, P1 split)")
    print("=" * 72)

    n_test_total = len(test_items)
    n_no_partner_concept = sum(1 for _, v in test_items if v.get("partner_in_train") is None)
    present_ids = [i for i, v in test_items if v.get("partner_in_train") is True]
    absent_ids = [i for i, v in test_items if v.get("partner_in_train") is False]
    print(f"\nP1 test set: {n_test_total} images total, {n_no_partner_concept} excluded "
          f"(no partner concept, i.e. APTOS) -> {len(present_ids)} partner-present, "
          f"{len(absent_ids)} partner-absent")

    X_train, y_train, _ = build_xy(train_ids_all, feats, row_of, grade_by_id)
    all_test_ids = present_ids + absent_ids
    X_test, y_test, ordered_test_ids = build_xy(all_test_ids, feats, row_of, grade_by_id)
    is_present = np.array([i in set(present_ids) for i in ordered_test_ids])

    _, y_pred_multi = fit_and_eval(X_train, y_train, X_test, y_test, seed=args.seed, bootstrap_train=False)
    _, _, y_pred_ord = ordinal_qwk(X_train, y_train, X_test, y_test, seed=args.seed)

    results_multi, grade_dist_p, grade_dist_a = run_ablation_analysis(
        "multinomial (headline)", y_test, y_pred_multi, is_present, ordered_test_ids, patient_by_id,
        args.seed, args.n_bootstrap, args.n_permutations)
    results_ord, _, _ = run_ablation_analysis(
        "ordinal (robustness check)", y_test, y_pred_ord, is_present, ordered_test_ids, patient_by_id,
        args.seed, args.n_bootstrap, args.n_permutations)

    agree = ((results_multi["bootstrap_ci95_on_difference"][0] > 0) ==
             (results_ord["bootstrap_ci95_on_difference"][0] > 0)) and \
            ((results_multi["permutation_test"]["p_value_two_sided"] < 0.05) ==
             (results_ord["permutation_test"]["p_value_two_sided"] < 0.05))
    print(f"\n--- ROBUSTNESS ACROSS HEADS ---")
    print(f"  multinomial and ordinal heads {'AGREE' if agree else 'DISAGREE'} on significance/direction "
          f"-- {'the conclusion is not a head-choice artifact' if agree else 'inspect further before trusting either head alone'}")

    # ---- Figure 3 (headline -- multinomial) --------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    x = np.arange(5)
    width = 0.35

    ax = axes[0]
    ax.bar(x - width / 2, grade_dist_p.to_numpy(), width, label="partner-present")
    ax.bar(x + width / 2, grade_dist_a.to_numpy(), width, label="partner-absent")
    ax.set_xticks(x); ax.set_xlabel("grade"); ax.set_ylabel("fraction of group")
    ax.set_title("Grade distribution by group\n(motivates grade-matching)")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    labels = ["raw", "grade-matched"]
    present_vals = [results_multi["raw_qwk_present"], results_multi["grade_matched_comparison"]["qwk_present"]]
    absent_vals = [results_multi["raw_qwk_absent"], results_multi["grade_matched_comparison"]["qwk_absent"]]
    xx = np.arange(len(labels))
    ax2.bar(xx - width / 2, present_vals, width, label="partner-present")
    ax2.bar(xx + width / 2, absent_vals, width, label="partner-absent")
    ax2.set_xticks(xx); ax2.set_xticklabels(labels)
    ax2.set_ylabel("QWK")
    ax2.set_ylim(0, 1)
    ci_lo, ci_hi = results_multi["bootstrap_ci95_on_difference"]
    ord_diff = results_ord["raw_difference"]
    ax2.set_title(f"Multinomial head (headline): diff 95% CI [{ci_lo:.3f}, {ci_hi:.3f}], "
                  f"p={results_multi['permutation_test']['p_value_two_sided']:.3g}\n"
                  f"Ordinal robustness check: raw diff {ord_diff:+.3f}, "
                  f"p={results_ord['permutation_test']['p_value_two_sided']:.3g}", fontsize=9)
    ax2.legend(fontsize=8)

    fig.suptitle("Claim 2b -- partner-eye ablation (Figure 3, headline)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig_path = root / args.fig
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nWrote {fig_path}")

    out = {
        "backbone": args.backbone, "size": args.size, "seed": args.seed,
        "n_test_total": n_test_total, "n_excluded_no_partner_concept": n_no_partner_concept,
        "n_partner_present": len(present_ids), "n_partner_absent": len(absent_ids),
        "heads_agree_on_significance_and_direction": agree,
        "multinomial": results_multi,
        "ordinal": results_ord,
    }
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
