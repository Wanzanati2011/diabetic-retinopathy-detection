"""
Diagnostic for the resnet50@224 Acceptance Test 7.1 sanity-check failure
(multinomial-LR QWK=0.4701 on P2, vs. the 0.5 threshold in tests/test_features.py).

Answers three questions BEFORE anyone touches QWK_THRESHOLD:
  1. Side-by-side sanity QWK + per-class recall + 5x5 confusion matrix,
     for all three extracted configs.
  2. resnet50@224 specifically: rDR (grade>=2) AUROC, and an ordinal QWK
     (ridge regression on the grade + thresholds optimized on train, rather
     than a multinomial softmax head) -- does the signal recover if the
     classifier respects grade ORDER instead of treating grades as
     unordered categories?
  3. Are the resnet50@224 P2 train/test image_id sets EXACTLY identical to
     the effnetb0 ones? (Rules out a wiring/alignment bug as the cause.)

CPU-only -- run this locally (same machine as extract.py; no GPU needed,
just real cores/RAM the sandbox doesn't have for a 27k x 2048 LR fit).

Usage:
    python src\\experiments\\diagnose_resnet50_qwk.py
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

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import cohen_kappa_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

CONFIGS = [("tf_efficientnet_b0", 224), ("tf_efficientnet_b0", 384), ("resnet50", 224)]


def load_active_manifest(root, manifest_rel):
    df = pd.read_csv(root / manifest_rel)
    if "excluded" in df.columns:
        df = df[~df["excluded"].fillna(False).astype(bool)].reset_index(drop=True)
    return df


def get_split_arrays(backbone, size, features_dir, active_manifest, p2_split):
    npz_path = features_dir / f"{backbone}_{size}.npz"
    data = np.load(npz_path)
    feats, ids = data["features"], data["image_id"]
    row_of = {i: r for r, i in enumerate(ids.tolist())}
    grade_by_id = dict(zip(active_manifest["image_id"], active_manifest["grade"]))

    train_ids = sorted(i for i, v in p2_split.items() if v["fold"] == "train" and i in row_of)
    test_ids = sorted(i for i, v in p2_split.items() if v["fold"] == "test" and i in row_of)

    X_train = np.stack([feats[row_of[i]] for i in train_ids])
    y_train = np.array([grade_by_id[i] for i in train_ids])
    X_test = np.stack([feats[row_of[i]] for i in test_ids])
    y_test = np.array([grade_by_id[i] for i in test_ids])
    return X_train, y_train, X_test, y_test, train_ids, test_ids


def multinomial_qwk(X_train, y_train, X_test, y_test):
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)
    qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")
    cm = confusion_matrix(y_test, y_pred, labels=range(5))
    recall = recall_score(y_test, y_pred, average=None, labels=range(5), zero_division=0)
    return qwk, cm, recall


def rdr_auroc(X_train, y_train, X_test, y_test):
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
    y_train_bin = (y_train >= 2).astype(int)
    y_test_bin = (y_test >= 2).astype(int)
    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(Xtr, y_train_bin)
    proba = clf.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(y_test_bin, proba))


def ordinal_qwk(X_train, y_train, X_test, y_test):
    """Ridge regression on the integer grade (treated as continuous), then
    thresholds optimized on TRAIN ONLY to maximize QWK, applied to test.
    Standard 'regress-then-round' ordinal trick (e.g. Kaggle APTOS 2019
    winning solutions) -- tests whether resnet50 features carry ordinal DR
    signal a discrete multinomial softmax head is throwing away."""
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
    reg = Ridge(alpha=1.0, random_state=42)
    reg.fit(Xtr, y_train.astype(float))
    train_score = reg.predict(Xtr)
    test_score = reg.predict(Xte)

    def neg_qwk(thresholds, y_true, scores):
        t = np.sort(thresholds)
        preds = np.clip(np.digitize(scores, t), 0, 4)
        return -cohen_kappa_score(y_true, preds, weights="quadratic")

    x0 = np.array([0.5, 1.5, 2.5, 3.5])
    res = minimize(neg_qwk, x0, args=(y_train, train_score), method="Nelder-Mead")
    thresholds = np.sort(res.x)
    test_preds = np.clip(np.digitize(test_score, thresholds), 0, 4)
    qwk = cohen_kappa_score(y_test, test_preds, weights="quadratic")
    return float(qwk), thresholds.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--manifest", default="data/manifests/manifest.csv")
    ap.add_argument("--p2-split", default="data/splits/p2.json")
    ap.add_argument("--features-dir", default="features")
    ap.add_argument("--out", default="results/phase3_sanity.json")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    features_dir = root / args.features_dir
    active_manifest = load_active_manifest(root, args.manifest)
    p2_split = json.loads((root / args.p2_split).read_text())

    print("=" * 72)
    print("1. Side-by-side P2 sanity QWK, all 3 configs")
    print("=" * 72)

    per_config = {}
    id_sets = {}
    for backbone, size in CONFIGS:
        npz_path = features_dir / f"{backbone}_{size}.npz"
        if not npz_path.exists():
            print(f"SKIP {backbone}@{size}: {npz_path} not found (extract.py hasn't run for this config yet)")
            continue
        X_train, y_train, X_test, y_test, train_ids, test_ids = get_split_arrays(
            backbone, size, features_dir, active_manifest, p2_split)
        id_sets[f"{backbone}_{size}"] = (train_ids, test_ids)
        qwk, cm, recall = multinomial_qwk(X_train, y_train, X_test, y_test)
        per_config[f"{backbone}_{size}"] = {
            "multinomial_qwk": float(qwk),
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "confusion_matrix": cm.tolist(),
            "per_class_recall": {str(g): float(r) for g, r in enumerate(recall)},
        }
        print(f"\n--- {backbone}@{size} ---  QWK={qwk:.4f}  (train={len(train_ids)}, test={len(test_ids)})")
        print("  per-class recall:", {g: round(float(r), 3) for g, r in enumerate(recall)})
        print("  confusion matrix (rows=true, cols=pred, 0..4):")
        print(pd.DataFrame(cm, index=[f"true_{i}" for i in range(5)],
                            columns=[f"pred_{i}" for i in range(5)]).to_string())

    if "resnet50_224" not in per_config:
        print("\nresnet50_224.npz not found -- run extract.py --backbone resnet50 --size 224 first.")
        return

    print("\n" + "=" * 72)
    print("2. resnet50@224 -- rDR AUROC and ordinal (ridge+thresholds) QWK")
    print("=" * 72)
    X_train, y_train, X_test, y_test, _, _ = get_split_arrays(
        "resnet50", 224, features_dir, active_manifest, p2_split)
    auroc = rdr_auroc(X_train, y_train, X_test, y_test)
    ord_qwk, thresholds = ordinal_qwk(X_train, y_train, X_test, y_test)
    print(f"  rDR (grade>=2) AUROC:        {auroc:.4f}")
    print(f"  Ordinal QWK (ridge+thresh):  {ord_qwk:.4f}   thresholds={[round(t, 3) for t in thresholds]}")
    print(f"  Multinomial QWK (for ref):   {per_config['resnet50_224']['multinomial_qwk']:.4f}")

    print("\n" + "=" * 72)
    print("3. Are the P2 train/test image_id sets IDENTICAL across all configs?")
    print("=" * 72)
    ref_key = next(iter(id_sets))
    ref_train_set, ref_test_set = set(id_sets[ref_key][0]), set(id_sets[ref_key][1])
    all_identical = True
    for key, (tr, te) in id_sets.items():
        same = (set(tr) == ref_train_set) and (set(te) == ref_test_set)
        all_identical = all_identical and same
        print(f"  {key:28s} n_train={len(tr):6d} n_test={len(te):5d}  identical to {ref_key}: {same}")

    verdict = {
        "resnet50_beats_chance": per_config["resnet50_224"]["multinomial_qwk"] > 0.0,
        "resnet50_ordinal_beats_threshold": ord_qwk > 0.5,
        "resnet50_ordinal_beats_multinomial": ord_qwk > per_config["resnet50_224"]["multinomial_qwk"],
        "rdr_auroc_strong": auroc > 0.85,
        "id_sets_identical_across_configs": all_identical,
    }
    print("\n--- VERDICT INPUTS ---")
    for k, v in verdict.items():
        print(f"  {k:38s} {v}")

    out = {
        "purpose": "Diagnostic for Acceptance Test 7.1 resnet50@224 QWK=0.4701 < 0.5 threshold "
                   "-- run BEFORE lowering QWK_THRESHOLD, to distinguish a wiring bug from a "
                   "genuinely weaker frozen ImageNet backbone for this task.",
        "per_config_p2_sanity": per_config,
        "resnet50_rdr_auroc": auroc,
        "resnet50_ordinal_qwk": ord_qwk,
        "resnet50_ordinal_thresholds": thresholds,
        "id_sets_identical_across_configs": all_identical,
        "verdict_inputs": verdict,
    }
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
