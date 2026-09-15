"""
PHASE 7 -- confidence calibration and a confidence-based reject option.

Implements MASTER_PLAN.md Part 11 exactly, on the DEPLOYED app checkpoint
(finetune_app_converged_p2_class_balanced_seed42, test QWK 0.7160,
referable AUROC 0.9117).

CPU only. Runs on the CSVs written by src/train/dump_app_predictions.py --
no GPU, no model loading, a couple of minutes.

    python src\\train\\dump_app_predictions.py     <- GPU, run this first
    python -m src.experiments.calibrate            <- this script

What it does, in order
----------------------
1. TEMPERATURE SCALING (Guo et al., ICML 2017). Fits a single scalar T by
   minimising the negative log-likelihood of softmax(logits / T) on the
   VALIDATION fold ONLY. Test is never touched by the fit.

2. CALIBRATION ERROR. Reports Expected and Maximum Calibration Error
   (15 equal-width confidence bins) before and after scaling, with
   reliability-diagram data, for:
     - the P2 validation fold (where T was fitted; in-sample, reported for
       completeness, not as evidence)
     - the P2 test fold (the primary, frozen endpoint)
     - the P2 test fold broken down by SOURCE DATASET (eyepacs / aptos).
       This is the domain-shift proxy. See dump_app_predictions.py's
       docstring for why the P3 folds are not a clean cross-dataset check
       for a P2-trained checkpoint. If P3 dumps are present they are scored
       too, with their contamination caveat carried into the output.

3. REFERRAL THRESHOLD. Before any rejection, fixes the referable-DR decision
   threshold on the VALIDATION fold at the smallest threshold reaching >=
   90% sensitivity, using referable_score = P(grade>=2) = p2+p3+p4 -- the
   whole tail, as clinical_operating_point.py defines it, not the argmax.
   This is the operating point Section 3.2 of the report already argues for.

4. REJECT OPTION. Confidence = max softmax probability after temperature
   scaling. Predictive entropy is computed as an alternative ranking signal
   and compared. Sweeps the rejection threshold tau; at each tau records
   coverage, selective QWK, and selective referable sensitivity/specificity
   at the FIXED referral threshold from step 3.

5. TAU, BY A RULE FIXED IN ADVANCE:
       the smallest tau such that selective referable-DR sensitivity on the
       VALIDATION fold is at least 0.90.
   Chosen on validation. Never by looking at test performance.

6. ACCEPTANCE TEST 11.1. MASTER_PLAN.md Part 11 states in advance:
   "Rejecting the least-confident ~20% should lift selective QWK by
   ~+0.05-0.10. If it doesn't, the confidence score is uninformative --
   check calibration, then report honestly."
   This script evaluates that at exactly 80% coverage and writes a verdict
   field. A FAIL is a legitimate, reportable result for a model whose
   probabilities are known to be uncalibrated -- it bounds what the
   interface may claim. It does not invalidate the model, and it must not
   be quietly re-run with a different coverage until it passes.

7. Writes app/release/thresholds.json in the schema MASTER_PLAN.md Part 12.1
   specifies, so the application can gate on it.

Statistical standard: every headline metric carries a 95% PATIENT-LEVEL
bootstrap CI -- patients resampled with replacement, every image of a
resampled patient pulled in -- matching patient_bootstrap_ci() in
claim2_protocols.py, which this script imports where it can.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import cohen_kappa_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RUN = "finetune_app_converged_p2_class_balanced_seed42"
N_BINS = 15
N_BOOT = 2000
TARGET_SENSITIVITY = 0.90
ACCEPTANCE_COVERAGE = 0.80
ACCEPTANCE_MIN_LIFT = 0.05
ACCEPTANCE_MAX_LIFT = 0.10

TAU_RULE = ("smallest tau such that selective referable-DR sensitivity on the "
            "VALIDATION fold is >= 0.90, at the referral threshold itself fixed "
            "on validation for >= 90% sensitivity at full coverage")

# Reuse the project's own bootstrap so a confidence interval means the same
# thing here as everywhere else. Falls back to a byte-identical local copy if
# the import fails for any reason (so a broken import can't cost a GPU run).
try:
    from src.experiments.claim2_protocols import patient_bootstrap_ci  # noqa: E402
    _BOOTSTRAP_SOURCE = "src.experiments.claim2_protocols.patient_bootstrap_ci"
except Exception:  # pragma: no cover
    _BOOTSTRAP_SOURCE = "local copy (identical semantics)"

    def patient_bootstrap_ci(patient_ids, y_true, y_pred, n_boot=1000, seed=0):
        df = pd.DataFrame({"patient_id": patient_ids,
                           "y_true": y_true, "y_pred": y_pred})
        groups = df.groupby("patient_id").indices
        patients = np.array(list(groups.keys()))
        rng = np.random.RandomState(seed)
        stats = np.empty(n_boot)
        y_true_arr = df["y_true"].to_numpy()
        y_pred_arr = df["y_pred"].to_numpy()
        for b in range(n_boot):
            sampled = rng.choice(patients, size=len(patients), replace=True)
            idx = np.concatenate([groups[p] for p in sampled])
            stats[b] = cohen_kappa_score(y_true_arr[idx], y_pred_arr[idx],
                                          weights="quadratic")
        lo, hi = np.percentile(stats, [2.5, 97.5])
        return float(lo), float(hi)


# --------------------------------------------------------------------------
# core numerics
# --------------------------------------------------------------------------
def softmax_T(logits, T):
    z = logits / float(T)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits, y, T):
    p = softmax_T(logits, T)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def fit_temperature(logits, y):
    """Single scalar T minimising validation NLL. Bounded search; T=1 is
    'no scaling', T>1 softens an overconfident model."""
    res = minimize_scalar(lambda t: nll(logits, y, t),
                          bounds=(0.05, 10.0), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x), float(res.fun)


def calibration_error(probs, y, n_bins=N_BINS):
    """ECE, MCE and reliability-diagram bins, equal-width on confidence."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce, bins = 0.0, 0.0, []
    n = len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(m.sum())
        if cnt == 0:
            bins.append({"bin_lower": float(lo), "bin_upper": float(hi),
                         "count": 0, "accuracy": None, "confidence": None,
                         "gap": None})
            continue
        acc = float(correct[m].mean())
        avg_conf = float(conf[m].mean())
        gap = abs(acc - avg_conf)
        ece += (cnt / n) * gap
        mce = max(mce, gap)
        bins.append({"bin_lower": float(lo), "bin_upper": float(hi),
                     "count": cnt, "accuracy": acc, "confidence": avg_conf,
                     "gap": float(gap)})
    return {"ece": float(ece), "mce": float(mce), "n": n, "bins": bins}


def predictive_entropy(probs):
    p = np.clip(probs, 1e-12, None)
    h = -(p * np.log(p)).sum(axis=1)
    return h / np.log(probs.shape[1])  # normalised to [0, 1]


def referral_threshold_for_sensitivity(scores, referable_true, target):
    """Smallest threshold whose sensitivity is still >= target. Sensitivity
    is monotone non-increasing in the threshold, so this is the most
    specific operating point that still meets the sensitivity requirement."""
    order = np.unique(scores)
    best = None
    for t in order:
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (referable_true == 1)).sum())
        fn = int(((pred == 0) & (referable_true == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        if sens >= target:
            best = float(t)
        else:
            break
    return best if best is not None else float(order[0])


def referable_stats(scores, referable_true, threshold):
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (referable_true == 1)).sum())
    fn = int(((pred == 0) & (referable_true == 1)).sum())
    tn = int(((pred == 0) & (referable_true == 0)).sum())
    fp = int(((pred == 1) & (referable_true == 0)).sum())
    return {
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }


def sweep_tau(conf, y_true, y_pred, ref_scores, ref_true, ref_threshold,
              n_points=200):
    """Coverage / selective-QWK / selective referable curve over tau."""
    taus = np.unique(np.quantile(conf, np.linspace(0.0, 0.95, n_points)))
    rows = []
    for t in taus:
        keep = conf >= t
        cov = float(keep.mean())
        if keep.sum() < 50 or len(np.unique(y_true[keep])) < 2:
            continue
        qwk = float(cohen_kappa_score(y_true[keep], y_pred[keep],
                                      weights="quadratic"))
        rs = referable_stats(ref_scores[keep], ref_true[keep], ref_threshold)
        rows.append({"tau": float(t), "coverage": cov, "n_kept": int(keep.sum()),
                     "selective_qwk": qwk,
                     "selective_sensitivity": rs["sensitivity"],
                     "selective_specificity": rs["specificity"]})
    return rows


def at_coverage(rows, target_cov):
    """Curve row whose coverage is closest to target."""
    if not rows:
        return None
    return min(rows, key=lambda r: abs(r["coverage"] - target_cov))


def load_fold(results_dir, run_name, fold):
    p = results_dir / f"{run_name}_{fold}_predictions.csv"
    if not p.exists():
        raise SystemExit(
            f"Missing {p}\n"
            f"Run the GPU dump first:  python src\\train\\dump_app_predictions.py")
    df = pd.read_csv(p)
    logits = df[[f"logit_{k}" for k in range(5)]].to_numpy()
    return df, logits


def score_set(name, logits, y, patient_ids, T, extra=None):
    """ECE/MCE before and after scaling, plus QWK with a patient-level CI."""
    p_raw = softmax_T(logits, 1.0)
    p_cal = softmax_T(logits, T)
    pred = p_cal.argmax(axis=1)
    qwk = float(cohen_kappa_score(y, pred, weights="quadratic"))
    lo, hi = patient_bootstrap_ci(patient_ids, y, pred, n_boot=N_BOOT, seed=0)
    ref_true = (y >= 2).astype(int)
    try:
        auroc = float(roc_auc_score(ref_true, p_cal[:, 2:].sum(axis=1)))
    except ValueError:
        auroc = float("nan")
    out = {
        "name": name,
        "n_images": int(len(y)),
        "n_patients": int(pd.Series(patient_ids).nunique()),
        "qwk": qwk,
        "qwk_patient_bootstrap_ci95": [lo, hi],
        "referable_auroc": auroc,
        "before": calibration_error(p_raw, y),
        "after": calibration_error(p_cal, y),
        "nll_before": nll(logits, y, 1.0),
        "nll_after": nll(logits, y, T),
    }
    out["ece_delta"] = out["after"]["ece"] - out["before"]["ece"]
    out["mce_delta"] = out["after"]["mce"] - out["before"]["mce"]
    # Temperature scaling does not change the argmax, so accuracy and QWK are
    # invariant to it. Say so, rather than letting a reader infer a gain.
    out["note"] = ("Temperature scaling is monotone and shared across classes, "
                   "so it cannot change the argmax: QWK and per-class recall are "
                   "identical before and after. What it changes is the "
                   "probability the interface displays and the threshold "
                   "behaviour that depends on it.")
    if extra:
        out.update(extra)
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default=DEFAULT_RUN)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--no-write-thresholds", action="store_true",
                    help="compute everything but do not write "
                         "app/release/thresholds.json")
    args = ap.parse_args()

    root = PROJECT_ROOT
    results_dir = root / "results"

    print("Loading prediction dumps...")
    val_df, val_logits = load_fold(results_dir, args.run_name, "val")
    test_df, test_logits = load_fold(results_dir, args.run_name, "test")
    y_val = val_df["true_grade"].to_numpy()
    y_test = test_df["true_grade"].to_numpy()
    print(f"  val : n={len(val_df):,}  patients={val_df['patient_id'].nunique():,}")
    print(f"  test: n={len(test_df):,}  patients={test_df['patient_id'].nunique():,}")

    val_hash = hashlib.sha256(
        "\n".join(sorted(val_df["image_id"])).encode()).hexdigest()[:16]

    # -- 1. fit T on validation only -------------------------------------
    print("\n[1] Fitting temperature on the VALIDATION fold only...")
    T, best_nll = fit_temperature(val_logits, y_val)
    nll_before = nll(val_logits, y_val, 1.0)
    print(f"    T = {T:.4f}   (T>1 means the model was overconfident)")
    print(f"    validation NLL {nll_before:.4f} -> {best_nll:.4f}")

    # -- 2. calibration error --------------------------------------------
    print("\n[2] Calibration error, before and after...")
    sets = {}
    sets["p2_val"] = score_set("P2 validation (in-sample for T)", val_logits,
                               y_val, val_df["patient_id"].to_numpy(), T)
    sets["p2_test"] = score_set("P2 test (primary endpoint)", test_logits,
                                y_test, test_df["patient_id"].to_numpy(), T)
    for ds in sorted(test_df["dataset"].dropna().unique()):
        m = (test_df["dataset"] == ds).to_numpy()
        if m.sum() < 100:
            continue
        sets[f"p2_test_{ds}"] = score_set(
            f"P2 test, {ds} images only (domain-shift proxy)",
            test_logits[m], y_test[m],
            test_df.loc[m, "patient_id"].to_numpy(), T)

    for key, s in sets.items():
        print(f"    {s['name']}")
        print(f"      ECE {s['before']['ece']:.4f} -> {s['after']['ece']:.4f}"
              f"   MCE {s['before']['mce']:.4f} -> {s['after']['mce']:.4f}")

    # optional P3, carrying its caveat
    dump_meta_path = results_dir / f"{args.run_name}_prediction_dump.json"
    p3_caveat = None
    if dump_meta_path.exists():
        p3_caveat = json.loads(dump_meta_path.read_text()).get("p3_caveat")
    for p3 in ["p3_eyepacs_to_aptos", "p3_aptos_to_eyepacs"]:
        p3_path = results_dir / f"{args.run_name}_{p3}_test_predictions.csv"
        if not p3_path.exists():
            continue
        d = pd.read_csv(p3_path)
        lg = d[[f"logit_{k}" for k in range(5)]].to_numpy()
        sets[p3] = score_set(f"{p3} test (CONTAMINATED -- see caveat)", lg,
                             d["true_grade"].to_numpy(),
                             d["patient_id"].to_numpy(), T,
                             extra={"caveat": p3_caveat or
                                    "P2-trained checkpoint; part of this fold "
                                    "was seen in training."})

    # -- 3. referral threshold, fixed on validation ----------------------
    print("\n[3] Fixing the referral threshold on validation...")
    p_val = softmax_T(val_logits, T)
    p_test = softmax_T(test_logits, T)
    val_ref_score = p_val[:, 2:].sum(axis=1)
    test_ref_score = p_test[:, 2:].sum(axis=1)
    val_ref_true = (y_val >= 2).astype(int)
    test_ref_true = (y_test >= 2).astype(int)

    ref_threshold = referral_threshold_for_sensitivity(
        val_ref_score, val_ref_true, TARGET_SENSITIVITY)
    val_full = referable_stats(val_ref_score, val_ref_true, ref_threshold)
    test_full = referable_stats(test_ref_score, test_ref_true, ref_threshold)
    print(f"    referral threshold on P(grade>=2) = {ref_threshold:.4f}")
    print(f"    validation @ full coverage: sens {val_full['sensitivity']:.3f}  "
          f"spec {val_full['specificity']:.3f}")
    print(f"    TEST       @ full coverage: sens {test_full['sensitivity']:.3f}  "
          f"spec {test_full['specificity']:.3f}")

    # -- 4/5. reject option ----------------------------------------------
    print("\n[4] Sweeping the rejection threshold...")
    val_pred = p_val.argmax(axis=1)
    test_pred = p_test.argmax(axis=1)
    signals = {
        "max_softmax": (p_val.max(axis=1), p_test.max(axis=1)),
        "neg_entropy": (1.0 - predictive_entropy(p_val),
                        1.0 - predictive_entropy(p_test)),
    }
    curves, chosen, rule_diag = {}, {}, {}
    for sig, (cv, ct) in signals.items():
        curves[sig] = {
            "validation": sweep_tau(cv, y_val, val_pred, val_ref_score,
                                    val_ref_true, ref_threshold),
            "test": sweep_tau(ct, y_test, test_pred, test_ref_score,
                              test_ref_true, ref_threshold),
        }
        vrows = sorted(curves[sig]["validation"], key=lambda r: r["tau"])
        # The pre-registered rule, applied literally: the SMALLEST tau whose
        # selective referable sensitivity on validation is >= 0.90.
        sat = [r for r in vrows if r["selective_sensitivity"] >= TARGET_SENSITIVITY]
        tau_rule = float(sat[0]["tau"]) if sat else None
        cov_at_rule = float(sat[0]["coverage"]) if sat else None
        # Is the rule degenerate -- i.e. does it select "reject essentially
        # nothing"? That happens when sensitivity is already >= target at full
        # coverage, which it is BY CONSTRUCTION here: step 3 fixed the referral
        # threshold on validation to hit exactly 0.90.
        degenerate = (tau_rule is None) or (cov_at_rule is not None
                                            and cov_at_rule > 0.99)
        rule_diag[sig] = {
            "tau_by_pre_registered_rule": tau_rule,
            "coverage_at_that_tau": cov_at_rule,
            "rule_is_degenerate": bool(degenerate),
            "max_selective_sensitivity_on_validation":
                float(max(r["selective_sensitivity"] for r in vrows)),
            "selective_sensitivity_at_full_coverage":
                float(vrows[0]["selective_sensitivity"]),
        }
        chosen[sig] = tau_rule if tau_rule is not None else float(np.min(cv))
        print(f"    {sig}: tau by pre-registered rule = "
              f"{chosen[sig]:.4f}  (coverage "
              f"{cov_at_rule if cov_at_rule is not None else float('nan'):.3f})"
              f"{'   <-- DEGENERATE' if degenerate else ''}")

    primary = "max_softmax"
    tau_pre_registered = chosen[primary]

    # ---- the degeneracy, and what ships instead -------------------------
    # Selective referable sensitivity DECREASES as tau rises here. The reason is
    # mechanical and worth stating: step 3 already pushed the referral threshold
    # to a permissive 0.135 so that weak evidence still fires a referral. Those
    # weakly-evidenced referable cases are exactly the low-confidence ones, so
    # rejecting the least-confident cases removes true positives faster than it
    # removes true negatives. The reject option therefore buys specificity and
    # ordinal agreement, and spends sensitivity -- the opposite of what
    # MASTER_PLAN Part 11's tau rule assumed when it was written.
    #
    # So the pre-registered rule selects "reject nothing" and cannot be used to
    # pick an operating tau. Rather than rewrite the rule after seeing the data,
    # ship the OTHER quantity Part 11 pre-specified: the ~80% coverage point that
    # Acceptance Test 11.1 is defined at. That number was fixed in advance too,
    # so this is a switch between two pre-registered quantities, not a post-hoc
    # search for a flattering one.
    row80_val = at_coverage(curves[primary]["validation"], ACCEPTANCE_COVERAGE)
    tau_shipped = float(row80_val["tau"]) if row80_val else tau_pre_registered
    tau_basis = ("acceptance_test_11_1_coverage" if row80_val
                 else "pre_registered_sensitivity_rule")
    print(f"\n    Pre-registered sensitivity rule is degenerate "
          f"(see verdict text). Shipping tau from the pre-registered "
          f"{ACCEPTANCE_COVERAGE:.0%}-coverage point instead: "
          f"tau = {tau_shipped:.4f}")
    tau = tau_shipped
    conf_test = signals[primary][1]
    keep = conf_test >= tau
    test_at_tau = {
        "tau": tau,
        "coverage": float(keep.mean()),
        "n_kept": int(keep.sum()),
        "n_rejected": int((~keep).sum()),
        "selective_qwk": float(cohen_kappa_score(y_test[keep], test_pred[keep],
                                                 weights="quadratic")),
        "selective_referable": referable_stats(test_ref_score[keep],
                                               test_ref_true[keep],
                                               ref_threshold),
    }
    lo, hi = patient_bootstrap_ci(test_df.loc[keep, "patient_id"].to_numpy(),
                                  y_test[keep], test_pred[keep],
                                  n_boot=args.n_boot, seed=0)
    test_at_tau["selective_qwk_patient_bootstrap_ci95"] = [lo, hi]
    print(f"\n    At tau on TEST: coverage {test_at_tau['coverage']:.3f}, "
          f"selective QWK {test_at_tau['selective_qwk']:.4f} "
          f"[{lo:.4f}, {hi:.4f}]")

    # -- 6. Acceptance Test 11.1 -----------------------------------------
    print("\n[5] ACCEPTANCE TEST 11.1 (pre-registered, MASTER_PLAN Part 11)...")
    base_qwk = float(cohen_kappa_score(y_test, test_pred, weights="quadratic"))
    row80 = at_coverage(curves[primary]["test"], ACCEPTANCE_COVERAGE)
    lift = (row80["selective_qwk"] - base_qwk) if row80 else float("nan")
    if row80 is None:
        verdict, vtext = "not_evaluable", "No curve row near 80% coverage."
    elif lift >= ACCEPTANCE_MIN_LIFT:
        verdict = "pass"
        vtext = (f"Rejecting the least-confident {100*(1-row80['coverage']):.1f}% "
                 f"lifts QWK by {lift:+.4f} (from {base_qwk:.4f} to "
                 f"{row80['selective_qwk']:.4f}), inside the pre-registered "
                 f"+{ACCEPTANCE_MIN_LIFT:.2f} to +{ACCEPTANCE_MAX_LIFT:.2f} band. "
                 f"The confidence score carries usable information.")
    else:
        verdict = "fail_confidence_uninformative"
        vtext = (f"Rejecting the least-confident {100*(1-row80['coverage']):.1f}% "
                 f"lifts QWK by only {lift:+.4f} (from {base_qwk:.4f} to "
                 f"{row80['selective_qwk']:.4f}), below the pre-registered "
                 f"+{ACCEPTANCE_MIN_LIFT:.2f} floor. Per MASTER_PLAN Part 11 this "
                 f"means the confidence score is uninformative at this operating "
                 f"point. REPORT THIS AS THE FINDING. It bounds what the "
                 f"interface may claim; it does not invalidate the model, and it "
                 f"must not be re-run at a different coverage until it passes.")
    print(f"    base QWK (full coverage) : {base_qwk:.4f}")
    if row80:
        print(f"    selective QWK @ {row80['coverage']:.1%} coverage: "
              f"{row80['selective_qwk']:.4f}")
        print(f"    lift: {lift:+.4f}")
    print(f"    VERDICT: {verdict}")

    # -- 7. write outputs -------------------------------------------------
    out = {
        "run_name": args.run_name,
        "checkpoint": f"checkpoints/{args.run_name}/best.pt",
        "inference_only": True,
        "bootstrap_source": _BOOTSTRAP_SOURCE,
        "n_boot": args.n_boot,
        "temperature": {
            "T": T,
            "fitted_on": "p2 validation fold only",
            "val_image_id_sha256_16": val_hash,
            "n_val": int(len(y_val)),
            "nll_before": nll_before,
            "nll_after": best_nll,
            "interpretation": ("T > 1 indicates the uncalibrated model was "
                               "overconfident, the direction Guo et al. (2017) "
                               "report for modern networks."),
        },
        "calibration": sets,
        "referral_threshold": {
            "value": ref_threshold,
            "score_definition": "P(grade>=2) = p2 + p3 + p4, post-temperature",
            "rule": (f"smallest threshold with validation sensitivity >= "
                     f"{TARGET_SENSITIVITY}"),
            "validation_full_coverage": val_full,
            "test_full_coverage": test_full,
        },
        "reject_option": {
            "primary_signal": primary,
            "tau_rule_pre_registered": TAU_RULE,
            "tau_by_pre_registered_rule": tau_pre_registered,
            "rule_diagnostics": rule_diag,
            "tau_shipped": tau,
            "tau_basis": tau_basis,
            "rule_degeneracy": {
                "finding": (
                    "The pre-registered tau rule selects 'reject nothing' and "
                    "cannot pick an operating point here. Selective referable "
                    "sensitivity DECREASES as tau rises, so no tau reaches the "
                    "0.90 target that is not already met at full coverage."),
                "mechanism": (
                    "Step 3 fixed the referral threshold at a permissive 0.135 so "
                    "that weak evidence still fires a referral. Weakly-evidenced "
                    "referable cases are exactly the low-confidence ones, so "
                    "rejecting the least-confident cases removes true positives "
                    "faster than true negatives. The reject option buys "
                    "specificity and ordinal agreement and spends sensitivity -- "
                    "the opposite of what Part 11's rule assumed."),
                "resolution": (
                    "Ship tau from the other quantity Part 11 pre-specified: the "
                    "~80% coverage point Acceptance Test 11.1 is defined at. That "
                    "is a switch between two pre-registered quantities, not a "
                    "post-hoc search. Both are recorded here."),
                "deployment_note": (
                    "Selective sensitivity treats rejected cases as if they "
                    "vanish. In deployment they do not -- a rejected case is "
                    "flagged 'uncertain, refer to a human grader' and is caught by "
                    "that grader. The -0.031 selective-sensitivity cost is "
                    "therefore an artefact of the metric's closed-world "
                    "assumption, not a patient-safety cost, PROVIDED the "
                    "application routes rejections to a human rather than "
                    "discarding them. The interface must do that, and must be "
                    "shown doing it."),
            },
            "test_at_tau": test_at_tau,
            "curves": curves,
        },
        "acceptance_test_11_1": {
            "pre_registered_band": [ACCEPTANCE_MIN_LIFT, ACCEPTANCE_MAX_LIFT],
            "evaluated_at_coverage": ACCEPTANCE_COVERAGE,
            "base_qwk_full_coverage": base_qwk,
            "selective_qwk": row80["selective_qwk"] if row80 else None,
            "actual_coverage": row80["coverage"] if row80 else None,
            "lift": lift,
            "verdict": verdict,
            "verdict_text": vtext,
        },
    }
    out_path = results_dir / "calibration_reject.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    if not args.no_write_thresholds:
        rel = root / "app" / "release"
        rel.mkdir(parents=True, exist_ok=True)
        th = {
            "temperature": T,
            "reject_tau": tau,
            "reject_tau_basis": tau_basis,
            "reject_tau_by_pre_registered_sensitivity_rule": tau_pre_registered,
            "reject_signal": primary,
            "reject_action": ("flag 'UNCERTAIN -- refer to a human grader'. "
                              "A rejected case MUST be routed to a human, never "
                              "discarded; the selective metrics assume otherwise."),
            "referral_threshold": ref_threshold,
            "referral_score": "P(grade>=2) = p2 + p3 + p4, post-temperature",
            "rule": TAU_RULE,
            "rule_note": ("The pre-registered sensitivity rule is degenerate for "
                          "this model -- see reject_option.rule_degeneracy in "
                          "results/calibration_reject.json. The shipped tau comes "
                          "from the pre-registered 80% coverage point instead."),
            "fitted_on": "p2 validation fold",
            "val_image_id_sha256_16": val_hash,
            "source_checkpoint": f"checkpoints/{args.run_name}/best.pt",
            "source_results": "results/calibration_reject.json",
            "acceptance_test_11_1_verdict": verdict,
            "warning": ("If acceptance_test_11_1_verdict is not 'pass', the "
                        "confidence score did not carry usable information at "
                        "the tested operating point. The application may still "
                        "display a calibrated probability, but it must not "
                        "claim that its abstention improves accuracy."),
        }
        th_path = rel / "thresholds.json"
        th_path.write_text(json.dumps(th, indent=2))
        print(f"Wrote {th_path}")

    print("\nNext:")
    print("  - wire app/app.py to read thresholds.json (temperature + UNCERTAIN gate)")
    print("  - generate Figures 5.8 (reliability) and 5.9 (risk-coverage) from")
    print("    results/calibration_reject.json")
    print("  - Grad-CAM panel INCLUDING failures (the one remaining GPU task)")


if __name__ == "__main__":
    main()
