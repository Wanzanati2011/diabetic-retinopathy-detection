"""
Claim 2 -- protocol comparison (MASTER_PLAN.md Part 8.2).
Trains the same linear head (multinomial logistic regression) on cached,
frozen features under each split protocol (P1 image-level, P2 patient-level,
P3 both cross-dataset directions) and compares QWK. This is the core
"evaluation leaks under image-level splitting" measurement.

Standalone, CPU-only (frozen features -> a few seconds per fit). Run locally
per the project's established pattern.

Methodology, written down for audit:
  - Head: StandardScaler + multinomial LogisticRegression (lbfgs), fit on the
    'train' fold only (the 'val' fold is reserved for future hyperparameter
    search and is unused here -- there are no hyperparameters to tune with a
    single plain linear head).
  - "5 seeds, nearly free" (MASTER_PLAN.md S8.2): a plain lbfgs
    LogisticRegression is otherwise deterministic given fixed data, so the
    5 seeds are realized as 5 independent BOOTSTRAP RESAMPLES of the training
    set (sampling training rows with replacement, per seed) -- this is what
    actually varies run-to-run and is the standard way to get a seed-style
    variance estimate when the base fit has no other stochastic component.
    Reported per protocol as mean +/- std QWK across the 5 bootstrap-trained
    models, all evaluated on the SAME (unperturbed) test fold.
  - Separately, a 95% QWK confidence interval per protocol is computed by
    bootstrap-resampling the TEST PATIENTS (not images -- a patient's two
    eyes must move together or the CI understates variance), applied to the
    unperturbed (seed=42, no train-resampling) reference model's test
    predictions.
  - Paired permutation test, P1 vs P2 only (as specified): pairs the 5
    seed-level QWKs by seed index and does an EXACT sign-flip permutation
    test over all 2^5=32 sign combinations (exact, not Monte Carlo, since
    there are only 5 pairs).

Sanity ranges from MASTER_PLAN.md S8.2 (frozen features, expected to sit
BELOW fine-tuned numbers -- the gap is the paper's subject, not a bug):
  P1 image-level:    0.80-0.88
  P2 patient-level:  0.70-0.80
  P3 cross-dataset:  0.50-0.65

Writes results/claim2_protocols.json and figures/figure1_claim2_protocols.png.

Usage:
    python src\\experiments\\claim2_protocols.py --backbone tf_efficientnet_b0 --size 224
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler

SEEDS = [42, 43, 44, 45, 46]
EXPECTED_RANGES = {
    "P1": (0.80, 0.88),
    "P2": (0.70, 0.80),
    "P3_eyepacs_to_aptos": (0.50, 0.65),
    "P3_aptos_to_eyepacs": (0.50, 0.65),
}
PROTOCOL_FILES = {
    "P1": "p1.json",
    "P2": "p2.json",
    "P3_eyepacs_to_aptos": "p3_eyepacs_to_aptos.json",
    "P3_aptos_to_eyepacs": "p3_aptos_to_eyepacs.json",
}


def load_active_manifest(manifest_path):
    df = pd.read_csv(manifest_path)
    if "excluded" in df.columns:
        df = df[~df["excluded"].fillna(False).astype(bool)].reset_index(drop=True)
    return df


def build_xy(ids, feats, row_of, grade_by_id):
    ids = [i for i in ids if i in row_of]
    X = np.stack([feats[row_of[i]] for i in ids])
    y = np.array([grade_by_id[i] for i in ids])
    return X, y, ids


def fit_and_eval(X_train, y_train, X_test, y_test, seed, bootstrap_train):
    rng = np.random.RandomState(seed)
    if bootstrap_train:
        idx = rng.randint(0, len(X_train), size=len(X_train))
        X_tr, y_tr = X_train[idx], y_train[idx]
    else:
        X_tr, y_tr = X_train, y_train
    scaler = StandardScaler().fit(X_tr)
    Xtr, Xte = scaler.transform(X_tr), scaler.transform(X_test)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(Xtr, y_tr)
    y_pred = clf.predict(Xte)
    qwk = float(cohen_kappa_score(y_test, y_pred, weights="quadratic"))
    return qwk, y_pred


def patient_bootstrap_ci(patient_ids, y_true, y_pred, n_boot=1000, seed=0):
    df = pd.DataFrame({"patient_id": patient_ids, "y_true": y_true, "y_pred": y_pred})
    groups = df.groupby("patient_id").indices
    patients = np.array(list(groups.keys()))
    rng = np.random.RandomState(seed)
    stats = np.empty(n_boot)
    y_true_arr, y_pred_arr = df["y_true"].to_numpy(), df["y_pred"].to_numpy()
    for b in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([groups[p] for p in sampled])
        stats[b] = cohen_kappa_score(y_true_arr[idx], y_pred_arr[idx], weights="quadratic")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def exact_paired_permutation_test(qwk_a, qwk_b):
    diffs = np.array(qwk_a) - np.array(qwk_b)
    observed = float(diffs.mean())
    n = len(diffs)
    null_means = np.array([np.mean(np.array(signs) * diffs)
                            for signs in itertools.product([1, -1], repeat=n)])
    p_value = float((np.abs(null_means) >= abs(observed) - 1e-12).mean())
    return {"observed_mean_diff": observed, "p_value_two_sided": p_value,
            "n_seeds": n, "n_permutations": len(null_means)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--manifest", default="data/manifests/manifest.csv")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--features-dir", default="features")
    ap.add_argument("--backbone", default="tf_efficientnet_b0")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--n-bootstrap-ci", type=int, default=1000)
    ap.add_argument("--out", default="results/claim2_protocols.json")
    ap.add_argument("--fig", default="figures/figure1_claim2_protocols.png")
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

    print("=" * 72)
    print(f"CLAIM 2 -- PROTOCOL COMPARISON  ({args.backbone}@{args.size})")
    print("=" * 72)

    results = {}
    per_protocol_seed_qwks = {}
    for name, fname in PROTOCOL_FILES.items():
        split_path = root / args.splits_dir / fname
        split_map = json.loads(split_path.read_text())
        train_ids_all = [i for i, v in split_map.items() if v["fold"] == "train"]
        test_ids_all = [i for i, v in split_map.items() if v["fold"] == "test"]

        X_train, y_train, train_ids = build_xy(train_ids_all, feats, row_of, grade_by_id)
        X_test, y_test, test_ids = build_xy(test_ids_all, feats, row_of, grade_by_id)

        seed_qwks = []
        reference_pred = None
        for seed in SEEDS:
            bootstrap = seed != SEEDS[0]  # first seed = unperturbed reference fit
            qwk, y_pred = fit_and_eval(X_train, y_train, X_test, y_test, seed, bootstrap_train=bootstrap)
            seed_qwks.append(qwk)
            if not bootstrap:
                reference_pred = y_pred

        patient_ids_test = [patient_by_id[i] for i in test_ids]
        ci_lo, ci_hi = patient_bootstrap_ci(patient_ids_test, y_test, reference_pred,
                                             n_boot=args.n_bootstrap_ci, seed=42)

        lo_exp, hi_exp = EXPECTED_RANGES[name]
        mean_qwk = float(np.mean(seed_qwks))
        in_range = lo_exp <= mean_qwk <= hi_exp
        results[name] = {
            "n_train": len(train_ids), "n_test": len(test_ids),
            "seed_qwks": seed_qwks, "mean_qwk": mean_qwk, "std_qwk": float(np.std(seed_qwks)),
            "reference_qwk_seed42": seed_qwks[0],
            "patient_bootstrap_ci95": [ci_lo, ci_hi],
            "expected_range": [lo_exp, hi_exp],
            "within_expected_range": in_range,
        }
        per_protocol_seed_qwks[name] = seed_qwks

        flag = "" if in_range else "  <-- OUTSIDE expected range, inspect before trusting"
        print(f"\n--- {name} ---  (train={len(train_ids)}, test={len(test_ids)})")
        print(f"  seed QWKs: {[round(q,4) for q in seed_qwks]}")
        print(f"  mean={mean_qwk:.4f}  std={np.std(seed_qwks):.4f}  "
              f"95% patient-bootstrap CI [{ci_lo:.4f}, {ci_hi:.4f}]  "
              f"expected [{lo_exp},{hi_exp}]{flag}")

    print("\n--- PAIRED PERMUTATION TEST: P1 vs P2 (exact, over 5 seeds) ---")
    perm_result = exact_paired_permutation_test(per_protocol_seed_qwks["P1"], per_protocol_seed_qwks["P2"])
    print(f"  mean(P1) - mean(P2) = {perm_result['observed_mean_diff']:.4f}   "
          f"p-value (two-sided, exact) = {perm_result['p_value_two_sided']:.4g}")

    # ---- Figure 1 -------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(PROTOCOL_FILES.keys())
    means = [results[n]["mean_qwk"] for n in names]
    cis = [results[n]["patient_bootstrap_ci95"] for n in names]
    errs = [[means[i] - cis[i][0] for i in range(len(names))],
            [cis[i][1] - means[i] for i in range(len(names))]]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=errs, capsize=4, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"])
    for i, n in enumerate(names):
        lo_exp, hi_exp = EXPECTED_RANGES[n]
        ax.plot([i - 0.4, i + 0.4], [lo_exp, lo_exp], "k--", linewidth=0.8)
        ax.plot([i - 0.4, i + 0.4], [hi_exp, hi_exp], "k--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("QWK (frozen features, mean of 5 bootstrap-trained heads)")
    ax.set_title(f"Claim 2 -- protocol comparison, {args.backbone}@{args.size} (Figure 1)\n"
                 f"error bars: 95% patient-level bootstrap CI; dashed: MASTER_PLAN.md expected range")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig_path = root / args.fig
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nWrote {fig_path}")

    out = {
        "backbone": args.backbone, "size": args.size,
        "seeds": SEEDS,
        "methodology_note": ("Seeds realized as bootstrap-resampled training sets, not solver "
                              "randomness (lbfgs is otherwise deterministic). First seed (42) is "
                              "the unperturbed reference fit used for the patient-bootstrap CI. "
                              "See module docstring for full methodology."),
        "protocols": results,
        "p1_vs_p2_paired_permutation_test": perm_result,
    }
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
