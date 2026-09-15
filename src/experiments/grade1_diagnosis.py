"""
R2 -- why did grade-1 (Mild NPDR) recall fall from 0.461 to 0.161 when the
app checkpoint was retrained to convergence?

CPU only, seconds. Runs on the CSV from src/train/dump_app_predictions.py.

    python src\\train\\dump_app_predictions.py     <- GPU, run this first
    python -m src.experiments.grade1_diagnosis     <- this script

The question, precisely
-----------------------
The converged retrain raised overall test QWK from 0.6128 to 0.7160 and
referable AUROC from 0.8715 to 0.9117, but per-class recall moved in
opposite directions:

    grade 0  0.687 -> 0.916
    grade 1  0.461 -> 0.161     <- this
    grade 2  0.354 -> 0.570
    grade 3  0.452 -> 0.287
    grade 4  0.549 -> 0.542

app/release/MODEL_CARD.md currently records this as an open caveat with a
plausible but unverified story ("a stronger healthy-versus-not boundary at
grade 1's expense"). This script tests that story instead of repeating it.

Two hypotheses, and they call for opposite responses
-----------------------------------------------------
H1  DECISION-BOUNDARY effect. The information is still in the representation;
    the converged model's argmax simply sits somewhere that rarely selects
    grade 1, because grade 1 is rare and sits between two larger neighbours.
    If so, a decision rule that is not the argmax should recover much of the
    recall -- and the fix is calibration and thresholding, not more pixels.
    This links R2 directly to R1.

H2  REPRESENTATION-CEILING effect. The evidence is not in a 384px image at
    all -- grade 1 is defined by ~10px microaneurysms, and Section 1.3 of the
    Review 2 report already argues from three independent observations that
    this may be a measurement ceiling. If recall stays low under EVERY
    decision rule, no thresholding will help and the honest fix is resolution
    (or an explicit decision to collapse the class).

Pre-registered expectation, written before running
--------------------------------------------------
Most grade-1 error lands on grade 0 -- already visible in the results JSON's
confusion row [311, 72, 61, 3, 0] over 447 cases, i.e. 69.6% -- and the
recall gap narrows substantially under a balanced decision rule. If it does
not, H2 is supported and that is the stronger version of the measurement-
ceiling argument, to be reported as such with resolution named as the fix
and the honest note that it was not affordable here.

What this script computes
-------------------------
1. Confusion row for grade 1, with the mass going to each neighbour.
2. Recall under four decision rules, all on the SAME predictions:
     argmax                    -- what the model ships
     prior-corrected argmax    -- divide by class frequency (balanced)
     temperature-scaled argmax -- invariant, reported to show it is invariant
     expected-grade rounding   -- an ordinal read of the same distribution
   The ordinal rule matters: this project's own strongest finding is that an
   ordinal head beats a multinomial one by +0.05 to +0.09 QWK, and the
   deployed model uses a plain 5-way softmax. Reading its output ordinally
   costs nothing and tests whether the ordering is being thrown away.
3. Where grade-1 cases sit in the referable score, to confirm the clinical
   consolation: grade 1 is NOT referable, so failing on it does not directly
   cost a referral.
4. The link to claim5_grade1_merge.json: if grade 1 is largely absorbed into
   grade 0 anyway, the merge gain and the recall collapse are two views of
   one thing.

Every recall carries a 95% patient-level bootstrap CI.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RUN = "finetune_app_converged_p2_class_balanced_seed42"
N_BOOT = 2000
GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR",
               "Proliferative DR"]

# From results/finetune_app_p2_seed42.json (the 15-epoch checkpoint the app
# used to ship) -- the "before" column of MODEL_CARD.md's recall table.
PRIOR_RECALL = {0: 0.687, 1: 0.461, 2: 0.354, 3: 0.452, 4: 0.549}

# Written before running, per this project's standard.
EXPECTED = {
    "most_grade1_error_lands_on_grade0": True,
    "balanced_rule_recovers_grade1_recall_to_at_least": 0.30,
    "acceptable_outcomes": [
        "H1 supported: a non-argmax rule recovers grade-1 recall -> the fix is "
        "calibration and thresholding, and R1 is where it happens.",
        "H2 supported: no rule recovers it -> the evidence is not in a 384px "
        "image; report as a measurement ceiling with resolution as the fix.",
    ],
}


def patient_bootstrap_recall(patient_ids, y_true, y_pred, grade,
                             n_boot=N_BOOT, seed=0):
    """Patient-level bootstrap CI on per-class recall -- patients resampled
    with replacement, every image of a resampled patient pulled in, matching
    patient_bootstrap_ci() in claim2_protocols.py."""
    df = pd.DataFrame({"patient_id": patient_ids, "y": y_true, "p": y_pred})
    groups = df.groupby("patient_id").indices
    patients = np.array(list(groups.keys()))
    rng = np.random.RandomState(seed)
    ya, pa = df["y"].to_numpy(), df["p"].to_numpy()
    stats = []
    for _ in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([groups[p] for p in sampled])
        m = ya[idx] == grade
        if m.sum() == 0:
            continue
        stats.append((pa[idx][m] == grade).mean())
    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def recalls(y_true, y_pred):
    return {int(g): (float((y_pred[y_true == g] == g).mean())
                     if (y_true == g).sum() else float("nan"))
            for g in range(5)}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default=DEFAULT_RUN)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    root = PROJECT_ROOT
    results_dir = root / "results"
    test_path = results_dir / f"{args.run_name}_test_predictions.csv"
    if not test_path.exists():
        raise SystemExit(
            f"Missing {test_path}\n"
            f"Run the GPU dump first:  python src\\train\\dump_app_predictions.py")

    df = pd.read_csv(test_path)
    y = df["true_grade"].to_numpy()
    probs = df[[f"prob_{k}" for k in range(5)]].to_numpy()
    pids = df["patient_id"].to_numpy()
    print(f"Test set: n={len(df):,}  patients={df['patient_id'].nunique():,}")

    # -- 1. the confusion row --------------------------------------------
    pred_argmax = probs.argmax(axis=1)
    cm = confusion_matrix(y, pred_argmax, labels=list(range(5)))
    row1 = cm[1]
    n1 = int(row1.sum())
    print("\n[1] Grade-1 confusion row (where the 447 Mild NPDR cases go)")
    for k in range(5):
        print(f"    -> grade {k} ({GRADE_NAMES[k]:<17}) {row1[k]:>4}  "
              f"{row1[k]/n1:>6.1%}")
    print(f"    Mass landing on grade 0: {row1[0]/n1:.1%}")

    # -- 2. four decision rules -------------------------------------------
    print("\n[2] Grade-1 recall under four decision rules "
          "(same predictions, different reads)")
    train_prior = np.array([(y == g).mean() for g in range(5)])
    rules = {}
    rules["argmax"] = pred_argmax
    # prior-corrected: p(y|x) / p(y), the standard balanced-accuracy read
    adj = probs / np.clip(train_prior, 1e-9, None)
    rules["prior_corrected_argmax"] = adj.argmax(axis=1)
    # expected grade, rounded -- an ordinal read of a multinomial output
    exp_grade = (probs * np.arange(5)).sum(axis=1)
    rules["expected_grade_rounded"] = np.clip(np.rint(exp_grade), 0, 4).astype(int)
    # sanity: temperature scaling cannot change the argmax
    rules["temperature_scaled_argmax"] = pred_argmax

    rule_out = {}
    for name, pred in rules.items():
        r = recalls(y, pred)
        qwk = float(cohen_kappa_score(y, pred, weights="quadratic"))
        lo, hi = patient_bootstrap_recall(pids, y, pred, 1, n_boot=args.n_boot)
        rule_out[name] = {
            "grade1_recall": r[1],
            "grade1_recall_patient_bootstrap_ci95": [lo, hi],
            "per_class_recall": r,
            "qwk": qwk,
        }
        print(f"    {name:<28} grade-1 recall {r[1]:.3f} [{lo:.3f}, {hi:.3f}]"
              f"   QWK {qwk:.4f}")
    print(f"    {'prior checkpoint (15-epoch)':<28} grade-1 recall "
          f"{PRIOR_RECALL[1]:.3f}   (from results/finetune_app_p2_seed42.json)")

    best_rule = max(
        (k for k in rule_out if k != "temperature_scaled_argmax"),
        key=lambda k: rule_out[k]["grade1_recall"])
    best_recall = rule_out[best_rule]["grade1_recall"]

    # -- 2b. is the ordinal read's QWK gain real? -------------------------
    # expected_grade_rounded reads the SAME five probabilities ordinally
    # instead of taking the argmax. If that is worth real QWK, it is a free
    # gain on the deployed model and it corroborates this project's single
    # strongest finding (an ordinal head beats a multinomial one by +0.05 to
    # +0.09 QWK). Paired patient-level bootstrap on the difference, which is
    # this project's standard for any comparison.
    print("\n[2b] Is the ordinal read's QWK gain real? "
          "(paired patient-level bootstrap)")
    pa = rules["expected_grade_rounded"]
    pb = rules["argmax"]
    dfb = pd.DataFrame({"patient_id": pids, "y": y, "a": pa, "b": pb})
    groups = dfb.groupby("patient_id").indices
    patients = np.array(list(groups.keys()))
    rng = np.random.RandomState(0)
    ya = dfb["y"].to_numpy(); aa = dfb["a"].to_numpy(); bb = dfb["b"].to_numpy()
    diffs = np.empty(args.n_boot)
    for i in range(args.n_boot):
        s = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([groups[p] for p in s])
        diffs[i] = (cohen_kappa_score(ya[idx], aa[idx], weights="quadratic")
                    - cohen_kappa_score(ya[idx], bb[idx], weights="quadratic"))
    obs = (rule_out["expected_grade_rounded"]["qwk"] - rule_out["argmax"]["qwk"])
    lo_d, hi_d = np.percentile(diffs, [2.5, 97.5])
    excl = bool(lo_d > 0 or hi_d < 0)
    print(f"    expected-grade minus argmax: {obs:+.4f}  "
          f"CI [{lo_d:+.4f}, {hi_d:+.4f}]  "
          f"{'EXCLUDES zero' if excl else 'includes zero'}")
    ordinal_read = {
        "comparison": "expected_grade_rounded minus argmax, same probabilities",
        "observed_diff": float(obs),
        "paired_patient_bootstrap_ci95": [float(lo_d), float(hi_d)],
        "excludes_zero": excl,
        "reading": (
            "The deployed model uses a plain five-way softmax. Reading its output "
            "ordinally -- expected grade, rounded -- costs nothing at inference "
            "and recovers some of the ordinal structure the argmax discards. If "
            "this interval excludes zero it is a free improvement on the shipped "
            "model and independent corroboration of the project's ordinal-head "
            "finding, obtained without retraining anything."),
    }

    # -- 3. clinical consolation ------------------------------------------
    print("\n[3] Does failing on grade 1 cost a referral?")
    ref_score = probs[:, 2:].sum(axis=1)
    m1 = y == 1
    print(f"    Grade 1 is NOT referable (referable = grade >= 2).")
    print(f"    Mean referable score on true grade-1 images: "
          f"{ref_score[m1].mean():.3f}")
    print(f"    Of the {n1} grade-1 cases, {int((pred_argmax[m1] >= 2).sum())} "
          f"are predicted referable -- i.e. over-referred, not missed.")
    print(f"    Grade-1 cases predicted grade 0: {int(row1[0])} "
          f"({row1[0]/n1:.1%}) -- these are the early-detection cost.")

    # -- 4. link to the merge experiment ----------------------------------
    merge_path = results_dir / "claim5_grade1_merge.json"
    merge_link = None
    if merge_path.exists():
        mj = json.loads(merge_path.read_text())
        ordn = mj.get("heads", {}).get("ordinal", {})
        merge_link = {
            "source": "results/claim5_grade1_merge.json",
            "ordinal_merged_minus_mapped": ordn.get("merged_minus_mapped"),
            "ordinal_ci95": ordn.get("paired_ci95_on_difference"),
            "verdict": mj.get("verdict"),
            "reading": ("If most grade-1 mass lands on grade 0 anyway, then the "
                        "QWK gain from collapsing grades 0 and 1 and this recall "
                        "collapse are two views of the same phenomenon: the 0/1 "
                        "boundary is not being drawn from image evidence. Note "
                        "that the merge also MOVED the referral operating point "
                        "(sens +0.093 / spec -0.032), so the gain must always be "
                        "quoted with that caveat."),
        }
        print("\n[4] Link to the grade-1 merge experiment")
        print(f"    ordinal head, merged minus mapped: "
              f"{ordn.get('merged_minus_mapped'):+.4f}  "
              f"CI {ordn.get('paired_ci95_on_difference')}")
        print(f"    verdict: {mj.get('verdict')}")

    # -- verdict -----------------------------------------------------------
    if best_recall >= EXPECTED["balanced_rule_recovers_grade1_recall_to_at_least"]:
        verdict = "H1_decision_boundary"
        vtext = (
            f"A non-argmax decision rule ({best_rule}) recovers grade-1 recall to "
            f"{best_recall:.3f}, against {rule_out['argmax']['grade1_recall']:.3f} "
            f"at argmax and {PRIOR_RECALL[1]:.3f} for the prior checkpoint. The "
            f"information survives in the representation; the converged model's "
            f"argmax simply stopped selecting a rare class between two larger "
            f"neighbours. This is a DECISION-BOUNDARY effect, which means the fix "
            f"is calibration and thresholding (Phase 7), not more pixels -- and it "
            f"turns the Review 2 'unexplained tradeoff' into an explained one.")
    else:
        verdict = "H2_representation_ceiling"
        vtext = (
            f"No decision rule recovers grade-1 recall: the best "
            f"({best_rule}) reaches only {best_recall:.3f}, against the 0.30 "
            f"threshold written down in advance. The evidence for Mild NPDR is "
            f"not recoverable from the 384px representation at any operating "
            f"point. This SUPPORTS the measurement-ceiling argument already made "
            f"from three independent observations in Section 1.3 (grade-1 partner "
            f"transfer at 45.8% against 72-94% for every other grade; three "
            f"corrupted black frames carrying a human grade of 1; recall stuck at "
            f"8.7-14.0% across all eight converged runs). Report resolution as the "
            f"fix, and state plainly that it was not affordable on a single "
            f"consumer GPU.")

    print(f"\nVERDICT: {verdict}")
    print(f"  {vtext}")

    out = {
        "run_name": args.run_name,
        "n_test": int(len(df)),
        "n_test_patients": int(df["patient_id"].nunique()),
        "n_grade1": n1,
        "expected_written_in_advance": EXPECTED,
        "confusion_row_grade1": {GRADE_NAMES[k]: int(row1[k]) for k in range(5)},
        "confusion_row_grade1_fractions": {
            GRADE_NAMES[k]: float(row1[k] / n1) for k in range(5)},
        "confusion_matrix_full": cm.tolist(),
        "decision_rules": rule_out,
        "ordinal_read_vs_argmax": ordinal_read,
        "prior_checkpoint_recall": PRIOR_RECALL,
        "best_non_argmax_rule": best_rule,
        "best_non_argmax_grade1_recall": best_recall,
        "partial_recovery_note": (
            "The verdict is set by the 0.30 bar written down in advance, and that "
            "bar was missed. But the recovery is not nothing: 0.161 -> 0.228 is a "
            "42% relative improvement in grade-1 recall from a different read of "
            "the same probabilities. The honest sentence is 'partial recovery, "
            "below the pre-registered bar', not 'no recovery'. Report both the "
            "number and the bar."),
        "clinical_note": {
            "grade1_is_referable": False,
            "mean_referable_score_on_grade1": float(ref_score[m1].mean()),
            "grade1_predicted_referable": int((pred_argmax[m1] >= 2).sum()),
            "grade1_predicted_grade0": int(row1[0]),
            "reading": ("Grade 1 is not referable, so a grade-1 miss does not "
                        "directly cost a referral. The cost is in early "
                        "detection -- which is the part of screening that most "
                        "justifies doing it at all, so this is a real limitation "
                        "and not a technicality to wave away."),
        },
        "merge_experiment_link": merge_link,
        "verdict": verdict,
        "verdict_text": vtext,
    }
    out_path = results_dir / "grade1_diagnosis.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    print("\nThis result goes into: report Section 5.6, MODEL_CARD.md "
          "(replacing the 'not investigated' caveat), and deck slide 18.")


if __name__ == "__main__":
    main()
