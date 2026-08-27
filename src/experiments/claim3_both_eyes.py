"""
Phase 5 / Claim 3 -- the both-eyes model (MASTER_PLAN.md Part 9).

On cached (frozen) features, compares four ways of producing a PATIENT-level
DR grade:
  A. Per-eye grading (the same P2 image-level classifier as Claim 2), then
     take max(left_pred, right_pred) -- the standard clinical approach.
  B. Concatenate the two eyes' feature vectors (dim*2) and train directly on
     the patient-level label.
  C. Mean-pool the two eyes' feature vectors (same dim as one eye) and train
     directly on the patient-level label.
  D. Label-only lookup baseline -- CITED from results/inter_eye_correlation.json,
     not recomputed here (it needs no pixels at all, so it isn't a "model").

Patient label = max(left_grade, right_grade) -- clinical management follows
the worse eye. This is MASTER_PLAN.md Part 9's explicit, required definition;
do not change it without updating the plan.

Scope: EyePACS patients with BOTH eyes active only. APTOS has no eye-pairing
information (single unpaired eye per patient), so "both eyes" is undefined
there and those patients are excluded from this analysis entirely.

Evaluated under P2 (patient-level split) ONLY -- per the plan, "P1 is
meaningless here" (image-level split does not even guarantee a stable
per-patient train/test assignment for both eyes together).

Referable-DR (grade>=2) sensitivity/specificity is reported at the nearest
achievable operating point to 90% sensitivity. IMPORTANT CAVEAT: our shared
fit_and_eval()/ordinal_qwk() helpers return discretized class predictions
(grades 0-4), not a continuous risk score, so there are only ~5 achievable
threshold levels -- the reported point is the closest one reaching >=90%
sensitivity, not an exact interpolated 90% point. Flagged in the output JSON
via "note", not hidden.

Heavy CPU (LogisticRegression/Ridge fits on several thousand patient rows x
1280-2560 cols) -- per this project's standing convention, this is NOT run
in the sandbox. Run it on your machine. tests/test_claim3_both_eyes.py
validates the pure pairing/aggregation logic (no model fitting) in the
sandbox with synthetic data.

Outputs: results/claim3_both_eyes.json, figures/figure4_claim3_both_eyes.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.experiments.claim2_protocols import (
    load_active_manifest, build_xy, fit_and_eval, ordinal_qwk, patient_bootstrap_ci,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED = 42


def build_eyepacs_pairs(active_manifest):
    """Pure function: from the active manifest, return one row per EyePACS
    patient who has BOTH eyes active (not soft-excluded), with the
    patient-level label = max(left_grade, right_grade). Patients missing
    either eye are dropped -- never fabricated."""
    ep = active_manifest[active_manifest["dataset"] == "eyepacs"]
    by_patient = {}
    for _, row in ep.iterrows():
        by_patient.setdefault(row["patient_id"], {})[row["eye"]] = row

    records = []
    for patient_id, eyes in by_patient.items():
        if "left" not in eyes or "right" not in eyes:
            continue
        l, r = eyes["left"], eyes["right"]
        records.append({
            "patient_id": patient_id,
            "left_id": l["image_id"], "right_id": r["image_id"],
            "left_grade": int(l["grade"]), "right_grade": int(r["grade"]),
            "patient_grade": max(int(l["grade"]), int(r["grade"])),
        })
    cols = ["patient_id", "left_id", "right_id", "left_grade", "right_grade", "patient_grade"]
    return pd.DataFrame.from_records(records, columns=cols)


def assign_p2_fold(pairs, split_map):
    """Pure function: attach each patient's P2 fold ('train'/'test'). Verifies
    (does not assume) that both eyes of a patient land in the SAME fold --
    P2 is supposed to guarantee this by construction, but we check rather
    than trust. Patients with a fold mismatch or a missing split entry are
    dropped and counted, not silently kept on one side."""
    folds = []
    mismatches = 0
    for _, row in pairs.iterrows():
        l_entry = split_map.get(row["left_id"])
        r_entry = split_map.get(row["right_id"])
        if l_entry is None or r_entry is None:
            folds.append(None)
            continue
        l_fold, r_fold = l_entry["fold"], r_entry["fold"]
        if l_fold != r_fold:
            mismatches += 1
            folds.append(None)
            continue
        folds.append(l_fold)
    pairs = pairs.copy()
    pairs["fold"] = folds
    return pairs, mismatches


def referable_sens_spec_at_90_sens(y_true, y_score):
    """Sensitivity/specificity for the rDR (grade>=2) binary task at the
    nearest achievable operating point to 90% sensitivity. See module
    docstring CAVEAT: y_score here is a discretized grade prediction
    (0-4), not a continuous risk score, so there are only ~5 achievable
    thresholds -- this is NOT an exact interpolated 90% point."""
    y_bin = (np.asarray(y_true) >= 2).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return {"sensitivity": None, "specificity": None, "threshold": None,
                "note": "degenerate: all-positive or all-negative rDR labels in this set"}
    fpr, tpr, thresholds = roc_curve(y_bin, y_score)
    idx = np.where(tpr >= 0.90)[0]
    if len(idx) == 0:
        return {"sensitivity": None, "specificity": None, "threshold": None,
                "note": "90% sensitivity not reachable with this discrete-grade score"}
    i = idx[0]
    return {
        "sensitivity": float(tpr[i]), "specificity": float(1 - fpr[i]), "threshold": float(thresholds[i]),
        "note": "nearest achievable operating point (score is a discrete 0-4 grade, not continuous -- "
                "see module docstring)",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--manifest", default="data/manifests/manifest.csv")
    ap.add_argument("--p2-split", default="data/splits/p2.json")
    ap.add_argument("--features-dir", default="features")
    ap.add_argument("--backbone", default="tf_efficientnet_b0")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--inter-eye-json", default="results/inter_eye_correlation.json")
    ap.add_argument("--n-bootstrap-ci", type=int, default=1000)
    ap.add_argument("--out", default="results/claim3_both_eyes.json")
    ap.add_argument("--fig", default="figures/figure4_claim3_both_eyes.png")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    npz_path = root / args.features_dir / f"{args.backbone}_{args.size}.npz"
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found. Run src/features/extract.py for this config first.", file=sys.stderr)
        sys.exit(1)

    active_manifest = load_active_manifest(root / args.manifest)
    grade_by_id = dict(zip(active_manifest["image_id"], active_manifest["grade"]))

    data = np.load(npz_path)
    feats, ids = data["features"], data["image_id"]
    row_of = {i: r for r, i in enumerate(ids.tolist())}

    def feat_of(image_id):
        return feats[row_of[image_id]]

    pairs = build_eyepacs_pairs(active_manifest)
    print(f"EyePACS patients with both eyes active: {len(pairs)}")

    split_map = json.loads((root / args.p2_split).read_text())
    pairs, mismatches = assign_p2_fold(pairs, split_map)
    if mismatches:
        print(f"WARNING: {mismatches} patients had left/right eyes in DIFFERENT P2 folds -- "
              f"dropped. This should be 0 -- investigate P2 construction if not.", file=sys.stderr)
    pairs = pairs[pairs["fold"].notna()].reset_index(drop=True)

    train_pairs = pairs[pairs["fold"] == "train"].reset_index(drop=True)
    test_pairs = pairs[pairs["fold"] == "test"].reset_index(drop=True)
    n_train, n_test = len(train_pairs), len(test_pairs)
    print(f"After P2 fold assignment: {n_train} train patients, {n_test} test patients "
          f"(both eyes each, EyePACS only)")

    # ---- Arm A: per-eye grading (P2 image classifier), then max ----
    # Train on ALL P2-train images (both eyes, independent rows -- this IS
    # the P2 model from Claim 2). One fit call: evaluate on left+right test
    # images stacked together, then split the predictions back apart.
    p2_train_ids = [i for i, v in split_map.items() if v["fold"] == "train"]
    X_train_img, y_train_img, _ = build_xy(p2_train_ids, feats, row_of, grade_by_id)

    y_test_left = test_pairs["left_grade"].to_numpy()
    y_test_right = test_pairs["right_grade"].to_numpy()
    y_test_patient = test_pairs["patient_grade"].to_numpy()
    X_test_left = np.stack([feat_of(i) for i in test_pairs["left_id"]])
    X_test_right = np.stack([feat_of(i) for i in test_pairs["right_id"]])
    X_test_both = np.vstack([X_test_left, X_test_right])
    y_test_both = np.concatenate([y_test_left, y_test_right])

    _, pred_both = fit_and_eval(X_train_img, y_train_img, X_test_both, y_test_both, SEED, bootstrap_train=False)
    pred_left, pred_right = pred_both[:n_test], pred_both[n_test:]
    pred_A = np.maximum(pred_left, pred_right)
    qwk_A = float(cohen_kappa_score(y_test_patient, pred_A, weights="quadratic"))

    # ---- Arm B: concatenated both-eye features -> patient grade ----
    X_train_concat = np.stack([np.concatenate([feat_of(l), feat_of(r)])
                                for l, r in zip(train_pairs["left_id"], train_pairs["right_id"])])
    y_train_patient = train_pairs["patient_grade"].to_numpy()
    X_test_concat = np.stack([np.concatenate([feat_of(l), feat_of(r)])
                               for l, r in zip(test_pairs["left_id"], test_pairs["right_id"])])
    qwk_B, pred_B = fit_and_eval(X_train_concat, y_train_patient, X_test_concat, y_test_patient, SEED, bootstrap_train=False)
    ord_qwk_B, _, ord_pred_B = ordinal_qwk(X_train_concat, y_train_patient, X_test_concat, y_test_patient, seed=SEED)

    # ---- Arm C: mean-pooled both-eye features -> patient grade ----
    X_train_pool = np.stack([(feat_of(l) + feat_of(r)) / 2.0
                              for l, r in zip(train_pairs["left_id"], train_pairs["right_id"])])
    X_test_pool = np.stack([(feat_of(l) + feat_of(r)) / 2.0
                             for l, r in zip(test_pairs["left_id"], test_pairs["right_id"])])
    qwk_C, pred_C = fit_and_eval(X_train_pool, y_train_patient, X_test_pool, y_test_patient, SEED, bootstrap_train=False)
    ord_qwk_C, _, ord_pred_C = ordinal_qwk(X_train_pool, y_train_patient, X_test_pool, y_test_patient, seed=SEED)

    # ---- Arm D: label-only lookup, CITED not recomputed ----
    inter_eye_path = root / args.inter_eye_json
    lookup_qwk = None
    if inter_eye_path.exists():
        lookup_qwk = json.loads(inter_eye_path.read_text())["label_only_baseline"]["lookup_qwk"]
    else:
        print(f"WARNING: {inter_eye_path} not found -- Arm D omitted.", file=sys.stderr)

    patient_ids_test = test_pairs["patient_id"].tolist()
    ci_A = patient_bootstrap_ci(patient_ids_test, y_test_patient, pred_A, n_boot=args.n_bootstrap_ci, seed=42)
    ci_B = patient_bootstrap_ci(patient_ids_test, y_test_patient, pred_B, n_boot=args.n_bootstrap_ci, seed=42)
    ci_C = patient_bootstrap_ci(patient_ids_test, y_test_patient, pred_C, n_boot=args.n_bootstrap_ci, seed=42)

    rss_A = referable_sens_spec_at_90_sens(y_test_patient, pred_A)
    rss_B = referable_sens_spec_at_90_sens(y_test_patient, pred_B)
    rss_C = referable_sens_spec_at_90_sens(y_test_patient, pred_C)

    print("=" * 72)
    print(f"CLAIM 3 -- BOTH-EYES MODEL  ({args.backbone}@{args.size}, P2 only, EyePACS n_test={n_test})")
    print("=" * 72)
    print(f"  Arm A (per-eye, then max):      QWK={qwk_A:.4f}  95% CI [{ci_A[0]:.4f}, {ci_A[1]:.4f}]")
    print(f"  Arm B (concat features):        QWK={qwk_B:.4f}  95% CI [{ci_B[0]:.4f}, {ci_B[1]:.4f}]  ordinal={ord_qwk_B:.4f}")
    print(f"  Arm C (mean-pooled features):   QWK={qwk_C:.4f}  95% CI [{ci_C[0]:.4f}, {ci_C[1]:.4f}]  ordinal={ord_qwk_C:.4f}")
    if lookup_qwk is not None:
        print(f"  Arm D (label-only lookup, cited): QWK={lookup_qwk:.4f}")

    beats = (qwk_B > qwk_A) or (qwk_C > qwk_A) or (ord_qwk_B > qwk_A) or (ord_qwk_C > qwk_A)
    print()
    print("USABLE METHOD FOUND (B or C beats A)." if beats else
          "NEGATIVE RESULT: neither B nor C beats A under frozen features -- report honestly, "
          "per MASTER_PLAN.md Part 9 ('a negative result on Claim 3 doesn't hurt a paper whose "
          "headline is Claims 1 and 2').")

    # ---- Figure 4 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["A: per-eye\nthen max", "B: concat\n(multinomial)", "B: concat\n(ordinal)",
              "C: mean-pool\n(multinomial)", "C: mean-pool\n(ordinal)"]
    heights = [qwk_A, qwk_B, ord_qwk_B, qwk_C, ord_qwk_C]
    cis = [ci_A, ci_B, None, ci_C, None]
    colors = ["#888888", "#d62728", "#d62728", "#1f77b4", "#1f77b4"]
    if lookup_qwk is not None:
        labels.append("D: label-only\nlookup (cited)")
        heights.append(lookup_qwk)
        cis.append(None)
        colors.append("#2ca02c")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    err_lo = [max(0.0, h - c[0]) if c else 0.0 for h, c in zip(heights, cis)]
    err_hi = [max(0.0, c[1] - h) if c else 0.0 for h, c in zip(heights, cis)]
    ax.bar(x, heights, color=colors, yerr=[err_lo, err_hi], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Patient-level QWK")
    ax.set_ylim(0, 1)
    ax.set_title(f"Figure 4 -- Claim 3, both-eyes model, {args.backbone}@{args.size} (P2, EyePACS only)\n"
                 f"error bars: 95% patient-bootstrap CI (multinomial arms only)")
    for i, h in enumerate(heights):
        ax.text(i, h + 0.02, f"{h:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig_path = root / args.fig
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nWrote {fig_path}")

    out = {
        "backbone": args.backbone, "size": args.size,
        "scope_note": "EyePACS patients with both eyes active only; APTOS excluded (no eye pairing). P2 split only.",
        "n_train_patients": int(n_train), "n_test_patients": int(n_test),
        "n_fold_mismatches_dropped": int(mismatches),
        "patient_label_definition": "max(left_grade, right_grade)",
        "arm_A_per_eye_then_max": {
            "qwk": qwk_A, "patient_bootstrap_ci95": list(ci_A),
            "referable_sens_spec_near_90pct_sens": rss_A,
        },
        "arm_B_concat_features": {
            "qwk_multinomial": qwk_B, "patient_bootstrap_ci95": list(ci_B),
            "qwk_ordinal": ord_qwk_B,
            "referable_sens_spec_near_90pct_sens": rss_B,
        },
        "arm_C_mean_pooled_features": {
            "qwk_multinomial": qwk_C, "patient_bootstrap_ci95": list(ci_C),
            "qwk_ordinal": ord_qwk_C,
            "referable_sens_spec_near_90pct_sens": rss_C,
        },
        "arm_D_label_only_lookup_cited": lookup_qwk,
        "b_or_c_beats_a": bool(beats),
        "interpretation": ("Usable both-eyes method found: B or C beats per-eye-then-max." if beats else
                            "Negative result: concatenation/pooling does not beat per-eye-then-max under "
                            "frozen features. Report honestly -- does not undermine Claims 1/1b/2."),
    }
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
