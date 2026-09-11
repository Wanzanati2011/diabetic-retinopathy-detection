"""
Phase 5b -- decomposing Claim 3's baseline (MASTER_PLAN.md Part 9, follow-up).

claim3_both_eyes.py compared Arm A (per-eye, multinomial head, then MAX) against
Arm C (mean-pooled features, ORDINAL head) and reported +0.075 QWK. That changes
the classifier head AND the fusion method AND implicitly bakes in one aggregation
rule (max) at the same time -- three confounded variables in one comparison. Since
the ordinal head alone is worth ~+0.10 on the single-eye P2 comparison (Claim 2:
0.524 multinomial -> 0.621 ordinal), the +0.075 gain could be mostly, or entirely,
the head switch rather than the bilateral fusion.

This script isolates all three variables:
  HEAD:        multinomial vs ordinal (same as Claim 2/3)
  AGGREGATION: for per-eye predictions only -- max (clinical/original), mean
               (round-half-up), min. Isolates whether MAX itself is a biased
               patient-level estimator, independent of any feature fusion.
  FUSION:      per-eye (no fusion) vs concat (Arm B) vs mean-pooled features (Arm C).

Produces a full method x head matrix:
  per-eye-max, per-eye-mean, per-eye-min   x   multinomial, ordinal   (6 cells)
  concat, mean-pooled features             x   multinomial, ordinal   (4 cells)
  label-only lookup (Arm D, cited)                                    (1 cell)

...and the paired comparisons that actually decompose the original +0.075:
  1. HEAD EFFECT       = per-eye-max-ordinal      - per-eye-max-multinomial
  2. FUSION EFFECT (C) = mean-pooled-ordinal       - per-eye-max-ordinal
  3. FUSION EFFECT (B) = concat-ordinal            - per-eye-max-ordinal
  4. AGGREGATION (mean)= per-eye-mean-ordinal      - per-eye-max-ordinal
  5. AGGREGATION (min) = per-eye-min-ordinal       - per-eye-max-ordinal
  6. AGGREGATION (mean, multinomial) = per-eye-mean-multinomial - per-eye-max-multinomial
  7. AGGREGATION (min,  multinomial) = per-eye-min-multinomial  - per-eye-max-multinomial
  8. ORIGINAL TOTAL     = mean-pooled-ordinal      - per-eye-max-multinomial  (the +0.075 as first reported)

Every comparison is a PAIRED patient-bootstrap CI on the difference (not two
independent point estimates), same method as claim3_both_eyes.py's
paired_bootstrap_ci_diff -- a point estimate alone is not evidence.

Same scope/protocol as claim3_both_eyes.py: EyePACS patients with both eyes,
P2 split only, patient label = max(left_grade, right_grade) (ground truth
definition is unchanged -- only how we AGGREGATE PREDICTIONS is being tested).

Outputs: results/claim3_decomposed.json (+ backbone/size suffix for non-default
configs), figures/figure4_claim3_both_eyes.png (regenerated, full matrix),
figures/figure5_claim3_decomposition.png (new -- the decomposition itself).
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
from src.experiments.claim3_both_eyes import (
    build_eyepacs_pairs, assign_p2_fold, referable_sens_spec_at_90_sens, paired_bootstrap_ci_diff,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED = 42


def aggregate_predictions(pred_left, pred_right, rule):
    """Pure function. rule in {'max','mean','min'}. 'mean' rounds half-up
    (np.floor(x+0.5)), NOT banker's rounding -- banker's rounding would round
    x.5 to the nearest EVEN grade, which is an arbitrary and undocumented
    tie-break for a clinical grade; round-half-up is the explicit, stated
    choice here. Result clipped to [0,4] (should never be needed given
    inputs are already in that range, but asserted defensively)."""
    pred_left = np.asarray(pred_left, dtype=float)
    pred_right = np.asarray(pred_right, dtype=float)
    if rule == "max":
        out = np.maximum(pred_left, pred_right)
    elif rule == "min":
        out = np.minimum(pred_left, pred_right)
    elif rule == "mean":
        out = np.floor((pred_left + pred_right) / 2.0 + 0.5)
    else:
        raise ValueError(f"unknown aggregation rule: {rule}")
    out = np.clip(out, 0, 4).astype(int)
    return out


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
    ap.add_argument("--out", default=None)
    ap.add_argument("--fig4", default=None)
    ap.add_argument("--fig5", default=None)
    ap.add_argument("--cache-dir", default="/tmp/claim3_cache",
                     help="Checkpoint directory for the expensive per-eye/concat/pool fits. "
                          "Each stage is cached separately so a slow environment can resume "
                          "across multiple invocations instead of refitting everything each time.")
    args = ap.parse_args()
    import time as _time
    _t0 = _time.time()
    def _log(msg):
        print(f"[{_time.time()-_t0:7.1f}s] {msg}", flush=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_tag = f"{args.backbone}_{args.size}"

    root = Path(args.project_root).resolve()
    npz_path = root / args.features_dir / f"{args.backbone}_{args.size}.npz"
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found.", file=sys.stderr)
        sys.exit(1)

    suffix = "" if (args.backbone == "tf_efficientnet_b0" and args.size == 224) else f"_{args.backbone}_{args.size}"
    out_path = root / (args.out or f"results/claim3_decomposed{suffix}.json")
    fig4_path = root / (args.fig4 or "figures/figure4_claim3_both_eyes.png")
    fig5_path = root / (args.fig5 or f"figures/figure5_claim3_decomposition{suffix}.png")

    active_manifest = load_active_manifest(root / args.manifest)
    grade_by_id = dict(zip(active_manifest["image_id"], active_manifest["grade"]))

    data = np.load(npz_path)
    feats, ids = data["features"], data["image_id"]
    row_of = {i: r for r, i in enumerate(ids.tolist())}

    def feat_of(image_id):
        return feats[row_of[image_id]]

    pairs = build_eyepacs_pairs(active_manifest)
    split_map = json.loads((root / args.p2_split).read_text())
    pairs, mismatches = assign_p2_fold(pairs, split_map)
    if mismatches:
        print(f"WARNING: {mismatches} fold mismatches dropped.", file=sys.stderr)
    pairs = pairs[pairs["fold"].notna()].reset_index(drop=True)
    train_pairs = pairs[pairs["fold"] == "train"].reset_index(drop=True)
    test_pairs = pairs[pairs["fold"] == "test"].reset_index(drop=True)
    n_train, n_test = len(train_pairs), len(test_pairs)
    print(f"{args.backbone}@{args.size}: {n_train} train / {n_test} test patients (both eyes, EyePACS)")

    y_test_left = test_pairs["left_grade"].to_numpy()
    y_test_right = test_pairs["right_grade"].to_numpy()
    y_test_patient = test_pairs["patient_grade"].to_numpy()
    X_test_left = np.stack([feat_of(i) for i in test_pairs["left_id"]])
    X_test_right = np.stack([feat_of(i) for i in test_pairs["right_id"]])
    X_test_both = np.vstack([X_test_left, X_test_right])
    y_test_both = np.concatenate([y_test_left, y_test_right])

    # ---- per-eye P2 classifier: BOTH heads, same train set as Claim 2/3 ----
    per_eye_cache = cache_dir / f"per_eye_{cache_tag}.npz"
    if per_eye_cache.exists():
        _log(f"per-eye fits: loading cache {per_eye_cache}")
        c = np.load(per_eye_cache)
        pred_mult_both, pred_ord_both = c["pred_mult_both"], c["pred_ord_both"]
    else:
        p2_train_ids = [i for i, v in split_map.items() if v["fold"] == "train"]
        X_train_img, y_train_img, _ = build_xy(p2_train_ids, feats, row_of, grade_by_id)
        _log("per-eye multinomial fit starting (slowest step, ~1 min)...")
        _, pred_mult_both = fit_and_eval(X_train_img, y_train_img, X_test_both, y_test_both, SEED, bootstrap_train=False)
        _log("per-eye multinomial fit done")
        _, _, pred_ord_both = ordinal_qwk(X_train_img, y_train_img, X_test_both, y_test_both, seed=SEED)
        _log("per-eye ordinal fit done")
        np.savez(per_eye_cache, pred_mult_both=pred_mult_both, pred_ord_both=pred_ord_both)
    pred_mult_left, pred_mult_right = pred_mult_both[:n_test], pred_mult_both[n_test:]
    pred_ord_left, pred_ord_right = pred_ord_both[:n_test], pred_ord_both[n_test:]

    patient_ids_test = test_pairs["patient_id"].tolist()

    def cell(pred, label):
        qwk = float(cohen_kappa_score(y_test_patient, pred, weights="quadratic"))
        ci = patient_bootstrap_ci(patient_ids_test, y_test_patient, pred, n_boot=args.n_bootstrap_ci, seed=42)
        rss = referable_sens_spec_at_90_sens(y_test_patient, pred)
        return {"label": label, "qwk": qwk, "patient_bootstrap_ci95": list(ci),
                "referable_sens_spec_near_90pct_sens": rss}, pred

    cells = {}
    preds = {}
    for rule in ("max", "mean", "min"):
        for head, (pl, pr) in (("multinomial", (pred_mult_left, pred_mult_right)),
                                 ("ordinal", (pred_ord_left, pred_ord_right))):
            key = f"per_eye_{rule}_{head}"
            agg_pred = aggregate_predictions(pl, pr, rule)
            cells[key], preds[key] = cell(agg_pred, key)

    y_train_patient = train_pairs["patient_grade"].to_numpy()

    # ---- Arm B: concat features -> patient grade (both heads) ----
    concat_cache = cache_dir / f"concat_{cache_tag}.npz"
    if concat_cache.exists():
        _log(f"concat fits: loading cache {concat_cache}")
        c = np.load(concat_cache)
        pred_concat_mult, pred_concat_ord = c["pred_concat_mult"], c["pred_concat_ord"]
    else:
        X_train_concat = np.stack([np.concatenate([feat_of(l), feat_of(r)])
                                    for l, r in zip(train_pairs["left_id"], train_pairs["right_id"])])
        X_test_concat = np.stack([np.concatenate([feat_of(l), feat_of(r)])
                                   for l, r in zip(test_pairs["left_id"], test_pairs["right_id"])])
        _log("concat multinomial fit starting (slowest step, double-width features)...")
        _, pred_concat_mult = fit_and_eval(X_train_concat, y_train_patient, X_test_concat, y_test_patient, SEED, bootstrap_train=False)
        _log("concat multinomial fit done")
        _, _, pred_concat_ord = ordinal_qwk(X_train_concat, y_train_patient, X_test_concat, y_test_patient, seed=SEED)
        _log("concat ordinal fit done")
        np.savez(concat_cache, pred_concat_mult=pred_concat_mult, pred_concat_ord=pred_concat_ord)
    cells["concat_multinomial"], preds["concat_multinomial"] = cell(pred_concat_mult, "concat_multinomial")
    cells["concat_ordinal"], preds["concat_ordinal"] = cell(pred_concat_ord, "concat_ordinal")

    # ---- Arm C: mean-pooled features -> patient grade (both heads) ----
    pool_cache = cache_dir / f"pool_{cache_tag}.npz"
    if pool_cache.exists():
        _log(f"pool fits: loading cache {pool_cache}")
        c = np.load(pool_cache)
        pred_pool_mult, pred_pool_ord = c["pred_pool_mult"], c["pred_pool_ord"]
    else:
        X_train_pool = np.stack([(feat_of(l) + feat_of(r)) / 2.0
                                  for l, r in zip(train_pairs["left_id"], train_pairs["right_id"])])
        X_test_pool = np.stack([(feat_of(l) + feat_of(r)) / 2.0
                                 for l, r in zip(test_pairs["left_id"], test_pairs["right_id"])])
        _log("pool multinomial fit starting...")
        _, pred_pool_mult = fit_and_eval(X_train_pool, y_train_patient, X_test_pool, y_test_patient, SEED, bootstrap_train=False)
        _log("pool multinomial fit done")
        _, _, pred_pool_ord = ordinal_qwk(X_train_pool, y_train_patient, X_test_pool, y_test_patient, seed=SEED)
        _log("pool ordinal fit done")
        np.savez(pool_cache, pred_pool_mult=pred_pool_mult, pred_pool_ord=pred_pool_ord)
    cells["pool_multinomial"], preds["pool_multinomial"] = cell(pred_pool_mult, "pool_multinomial")
    cells["pool_ordinal"], preds["pool_ordinal"] = cell(pred_pool_ord, "pool_ordinal")

    # ---- Arm D: label-only lookup, cited ----
    inter_eye_path = root / args.inter_eye_json
    lookup_qwk = None
    if inter_eye_path.exists():
        lookup_qwk = json.loads(inter_eye_path.read_text())["label_only_baseline"]["lookup_qwk"]

    # ---- the decomposition: paired bootstrap CIs on the differences that matter ----
    def diff(key_a, key_b, seed=43):
        return paired_bootstrap_ci_diff(patient_ids_test, y_test_patient, preds[key_a], preds[key_b],
                                         n_boot=args.n_bootstrap_ci, seed=seed)

    decomposition = {
        "1_head_effect_ordinal_minus_multinomial_at_max": diff("per_eye_max_multinomial", "per_eye_max_ordinal"),
        "2_fusion_effect_pool_minus_per_eye_max_ordinal": diff("per_eye_max_ordinal", "pool_ordinal"),
        "3_fusion_effect_concat_minus_per_eye_max_ordinal": diff("per_eye_max_ordinal", "concat_ordinal"),
        "4_aggregation_effect_mean_minus_max_ordinal": diff("per_eye_max_ordinal", "per_eye_mean_ordinal"),
        "5_aggregation_effect_min_minus_max_ordinal": diff("per_eye_max_ordinal", "per_eye_min_ordinal"),
        "6_aggregation_effect_mean_minus_max_multinomial": diff("per_eye_max_multinomial", "per_eye_mean_multinomial"),
        "7_aggregation_effect_min_minus_max_multinomial": diff("per_eye_max_multinomial", "per_eye_min_multinomial"),
        "8_original_total_gain_pool_ordinal_minus_per_eye_max_multinomial": diff("per_eye_max_multinomial", "pool_ordinal"),
    }

    print("=" * 78)
    print(f"CLAIM 3 DECOMPOSED  ({args.backbone}@{args.size}, P2, EyePACS n_test={n_test})")
    print("=" * 78)
    print(f"{'cell':<28} {'QWK':>8}  95% CI")
    for k, v in cells.items():
        print(f"  {k:<26} {v['qwk']:.4f}  {v['patient_bootstrap_ci95']}")
    if lookup_qwk is not None:
        print(f"  {'label_only_lookup(cited)':<26} {lookup_qwk:.4f}  (no pixels)")
    print()
    print("--- decomposition (paired patient-bootstrap 95% CI on each difference) ---")
    for k, v in decomposition.items():
        print(f"  {k:<58} diff={v['mean_diff']:+.4f}  CI {v['ci95']}  sig={v['significantly_better']}")

    # ---- verdict logic ----
    head_sig = decomposition["1_head_effect_ordinal_minus_multinomial_at_max"]["significantly_better"]
    head_mag = decomposition["1_head_effect_ordinal_minus_multinomial_at_max"]["mean_diff"]
    fusion_pool_sig = decomposition["2_fusion_effect_pool_minus_per_eye_max_ordinal"]["significantly_better"]
    fusion_pool_mag = decomposition["2_fusion_effect_pool_minus_per_eye_max_ordinal"]["mean_diff"]
    fusion_concat_sig = decomposition["3_fusion_effect_concat_minus_per_eye_max_ordinal"]["significantly_better"]
    agg_mean_sig = decomposition["4_aggregation_effect_mean_minus_max_ordinal"]["significantly_better"]
    agg_mean_mag = decomposition["4_aggregation_effect_mean_minus_max_ordinal"]["mean_diff"]
    agg_min_sig = decomposition["5_aggregation_effect_min_minus_max_ordinal"]["significantly_better"]

    fusion_beats_a = fusion_pool_sig or fusion_concat_sig
    agg_only_explains = (agg_mean_sig or agg_min_sig) and not fusion_beats_a

    if fusion_beats_a:
        verdict = "a"
        verdict_text = ("(a) Bilateral feature fusion (mean-pooling) helps BEYOND the ordinal head and "
                         "beyond the aggregation rule -- Claim 3 stands as originally written: fusion "
                         f"contributes {fusion_pool_mag:+.4f} QWK on top of per-eye-max-ordinal "
                         f"(CI {decomposition['2_fusion_effect_pool_minus_per_eye_max_ordinal']['ci95']}), "
                         f"significant on its own.")
    elif agg_only_explains:
        verdict = "c"
        verdict_text = ("(c) The apparent gain is mostly MAX-aggregation bias, not feature fusion: "
                         f"switching from max to mean aggregation at a FIXED ordinal head alone moves QWK by "
                         f"{agg_mean_mag:+.4f} (CI {decomposition['4_aggregation_effect_mean_minus_max_ordinal']['ci95']}), "
                         "while feature fusion on top of per-eye-max-ordinal does not significantly help. "
                         "Reframe as: 'max is a biased patient-level estimator of the worse-eye grade "
                         "under classifier noise' -- still a real, useful, and separately reportable finding.")
    else:
        verdict = "b"
        verdict_text = ("(b) The gain is mostly the ordinal head, not fusion or aggregation: switching heads "
                         f"alone (still per-eye, still max) moves QWK by {head_mag:+.4f} "
                         f"(CI {decomposition['1_head_effect_ordinal_minus_multinomial_at_max']['ci95']}), "
                         "while neither feature fusion nor a different aggregation rule adds a further "
                         "significant gain on top of the ordinal head. Reframe Claim 3 as an ordinal-head "
                         "finding, not a both-eyes-fusion finding.")

    print()
    print("VERDICT:", verdict_text)

    # ---- Figure 4 (regenerated, full matrix) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["per_eye_max_multinomial", "per_eye_max_ordinal", "per_eye_mean_multinomial", "per_eye_mean_ordinal",
             "per_eye_min_multinomial", "per_eye_min_ordinal", "concat_multinomial", "concat_ordinal",
             "pool_multinomial", "pool_ordinal"]
    disp_labels = {
        "per_eye_max_multinomial": "per-eye\nmax\n(mult)", "per_eye_max_ordinal": "per-eye\nmax\n(ord)",
        "per_eye_mean_multinomial": "per-eye\nmean\n(mult)", "per_eye_mean_ordinal": "per-eye\nmean\n(ord)",
        "per_eye_min_multinomial": "per-eye\nmin\n(mult)", "per_eye_min_ordinal": "per-eye\nmin\n(ord)",
        "concat_multinomial": "concat\n(mult)", "concat_ordinal": "concat\n(ord)",
        "pool_multinomial": "pool\n(mult)", "pool_ordinal": "pool\n(ord)",
    }
    colors = {"multinomial": "#d62728", "ordinal": "#1f77b4"}
    bar_colors = [colors["ordinal"] if k.endswith("ordinal") else colors["multinomial"] for k in order]
    heights = [cells[k]["qwk"] for k in order]
    cis = [cells[k]["patient_bootstrap_ci95"] for k in order]
    labels = [disp_labels[k] for k in order]
    if lookup_qwk is not None:
        labels.append("label-only\nlookup (cited)")
        heights.append(lookup_qwk)
        cis.append(None)
        bar_colors.append("#2ca02c")

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(labels))
    err_lo = [max(0.0, h - c[0]) if c else 0.0 for h, c in zip(heights, cis)]
    err_hi = [max(0.0, c[1] - h) if c else 0.0 for h, c in zip(heights, cis)]
    ax.bar(x, heights, color=bar_colors, yerr=[err_lo, err_hi], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Patient-level QWK")
    ax.set_ylim(0, 1)
    ax.set_title(f"Figure 4 -- Claim 3 decomposed, {args.backbone}@{args.size} (P2, EyePACS only)\n"
                 f"red=multinomial, blue=ordinal, green=zero-pixel shortcut ceiling. "
                 f"Full {{aggregation/fusion}} x {{head}} matrix.", fontsize=10)
    for i, h in enumerate(heights):
        ax.text(i, h + 0.02, f"{h:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig4_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig4_path, dpi=150)
    plt.close(fig)
    print(f"\nWrote {fig4_path}")

    # ---- Figure 5 (the decomposition) ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    dorder = ["1_head_effect_ordinal_minus_multinomial_at_max",
              "4_aggregation_effect_mean_minus_max_ordinal",
              "5_aggregation_effect_min_minus_max_ordinal",
              "2_fusion_effect_pool_minus_per_eye_max_ordinal",
              "3_fusion_effect_concat_minus_per_eye_max_ordinal",
              "8_original_total_gain_pool_ordinal_minus_per_eye_max_multinomial"]
    dlabels = ["HEAD\n(ordinal-mult, at max)", "AGGREGATION\n(mean-max, ordinal)",
               "AGGREGATION\n(min-max, ordinal)", "FUSION\n(pool-per_eye_max, ordinal)",
               "FUSION\n(concat-per_eye_max, ordinal)", "ORIGINAL TOTAL\n(pool_ord - per_eye_max_mult)"]
    dvals = [decomposition[k]["mean_diff"] for k in dorder]
    dcis = [decomposition[k]["ci95"] for k in dorder]
    dsig = [decomposition[k]["significantly_better"] for k in dorder]
    dcolors = ["#2ca02c" if s else "#888888" for s in dsig]
    xx = np.arange(len(dorder))
    elo = [v - c[0] for v, c in zip(dvals, dcis)]
    ehi = [c[1] - v for v, c in zip(dvals, dcis)]
    ax.bar(xx, dvals, color=dcolors, yerr=[elo, ehi], capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xx)
    ax.set_xticklabels(dlabels, fontsize=7.5)
    ax.set_ylabel("QWK difference (paired patient-bootstrap)")
    ax.set_title(f"Figure 5 -- decomposing Claim 3's gain, {args.backbone}@{args.size}\n"
                 f"green = 95% CI excludes 0. Which bar is 'original total' made of?", fontsize=10)
    for i, (v, c) in enumerate(zip(dvals, dcis)):
        ax.text(i, v + ehi[i] + 0.004, f"{v:+.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig5_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig5_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {fig5_path}")

    out = {
        "backbone": args.backbone, "size": args.size,
        "n_train_patients": int(n_train), "n_test_patients": int(n_test),
        "n_fold_mismatches_dropped": int(mismatches),
        "patient_label_definition": "max(left_grade, right_grade) -- ground truth definition unchanged; "
                                     "only prediction-aggregation is varied here",
        "cells": cells,
        "arm_D_label_only_lookup_cited": lookup_qwk,
        "decomposition": decomposition,
        "verdict": verdict,
        "verdict_text": verdict_text,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
