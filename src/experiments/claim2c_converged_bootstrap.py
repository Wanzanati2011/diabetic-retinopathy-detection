"""
Claim 2c -- P1 vs P2 under FULL CONVERGENCE, with the project's own
patient-level paired bootstrap (MASTER_PLAN.md Part 10 follow-up).

Section 3.5 of the report states the Phase 6b P1-vs-P2 gap as a raw average
of two seeds and explicitly refuses to call it real until it has been through
the same standard every other claim in this project is held to. This script
is that test.

Why this is not just "subtract two numbers":
  P1 and P2 do NOT share a test set. P1 is split image-level (n_test=5819),
  P2 patient-level (n_test=5814); they overlap heavily but are not identical,
  so a raw P1-minus-P2 difference confounds "leakage" with "different test
  images". Two comparisons are therefore reported:

    (a) PAIRED, on the intersection of the two test sets (the primary
        result). Both models scored the same images, so resampling patients
        with replacement and recomputing both QWKs on the same resample
        isolates the training-protocol effect. This is the number that
        answers the question.

    (b) UNPAIRED, each model on its own FULL test set (5819 vs 5814), the
        two sides resampled independently. This is the PRIMARY result,
        because the intersection in (a) turns out to be only ~963 images:
        P1 and P2 assign folds independently, so their test sets overlap far
        less than their sizes suggest. Restricting to the intersection
        controls the composition confound but costs ~83% of the data and
        badly widens the interval. Both are reported, and the gap in
        precision between them is itself worth stating in the paper.

  POWER. A null is only as good as the effect it could have ruled out, so
  the minimum detectable effect (half-width of the CI on the difference) is
  computed for every condition. Tampu et al. measured 0.07-0.43 MCC
  inflation; a test that cannot resolve 0.07 cannot claim to contradict the
  small end of that range, and the report must say so rather than present an
  underpowered null as a strong one.

  A patient's two eyes always move together inside a bootstrap resample
  (resample patient IDs, take every image belonging to a resampled patient),
  which is the whole point -- resampling images independently would
  understate variance exactly where the correlation lives.

EyePACS-only is reported alongside all-images: APTOS carries no eye pairing
(one image = one "patient"), so it cannot express the leakage mechanism and
only dilutes the effect. The EyePACS-only row is the mechanistically
meaningful one.

Pre-registered outcome, written before running (per this project's standing
practice of stating an acceptable null in advance):
  - If the paired CI on the P1-P2 difference EXCLUDES zero and is positive,
    the Section 3.3/3.6 null result does NOT survive convergence and the
    report must say so -- leakage inflation appears once the model is
    trained to completion.
  - If the CI INCLUDES zero, the null holds under convergence and the
    provisional caveat in Section 3.5 can be discharged.
  Either outcome is reportable. Neither is a failure.

Inputs : results/finetune_converged_{p1,p2}_{class_balanced,none}_seed{42,43}
         _test_predictions.csv  (written by src/train/dump_test_predictions.py)
Writes : results/claim2c_converged_bootstrap.json
         figures/report_v2/figI_converged_bootstrap.png

Usage:
  python src/experiments/claim2c_converged_bootstrap.py
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLERS = ["class_balanced", "none"]
SEEDS = [42, 43]
BLUE, ORANGE, INK, MUTED = "#0072B2", "#D55E00", "#222222", "#888888"


def qwk(y_true, y_pred):
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def load_run(root, split, sampler, seed):
    p = root / "results" / f"finetune_converged_{split}_{sampler}_seed{seed}_test_predictions.csv"
    if not p.exists():
        raise SystemExit(f"ERROR: {p} not found. Run src/train/dump_test_predictions.py first.")
    df = pd.read_csv(p)
    return df.set_index("image_id")


def paired_patient_bootstrap(pid, y_true, pred_a, pred_b, n_boot=2000, seed=0):
    """Resample PATIENTS with replacement; recompute both QWKs on the same
    resample; CI on the difference (a - b). Returns dict."""
    df = pd.DataFrame({"pid": pid, "y": y_true, "a": pred_a, "b": pred_b})
    groups = df.groupby("pid").indices
    patients = np.array(list(groups.keys()))
    y = df["y"].to_numpy(); A = df["a"].to_numpy(); B = df["b"].to_numpy()
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([groups[p] for p in s])
        diffs[i] = qwk(y[idx], A[idx]) - qwk(y[idx], B[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    obs = qwk(y, A) - qwk(y, B)
    return {
        "qwk_a": qwk(y, A), "qwk_b": qwk(y, B),
        "observed_diff": float(obs),
        "ci95": [float(lo), float(hi)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "minimum_detectable_effect": float((hi - lo) / 2),
        "n_images": int(len(df)), "n_patients": int(len(patients)),
    }


def patient_bootstrap_ci(pid, y_true, y_pred, n_boot=2000, seed=0):
    df = pd.DataFrame({"pid": pid, "y": y_true, "p": y_pred})
    groups = df.groupby("pid").indices
    patients = np.array(list(groups.keys()))
    y = df["y"].to_numpy(); P = df["p"].to_numpy()
    rng = np.random.RandomState(seed)
    st = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([groups[p] for p in s])
        st[i] = qwk(y[idx], P[idx])
    lo, hi = np.percentile(st, [2.5, 97.5])
    return [float(lo), float(hi)]


def unpaired_patient_bootstrap_diff(df_a, df_b, n_boot=2000, seed=0):
    """Two disjoint test sets -> resample each side's patients independently,
    take the difference of the resampled QWKs. Uses ALL of both test sets, at
    the cost of not controlling for test-set composition."""
    ga = df_a.groupby("patient_id").indices
    gb = df_b.groupby("patient_id").indices
    pa, pb = np.array(list(ga.keys())), np.array(list(gb.keys()))
    ya, qa = df_a["true_grade"].to_numpy(), df_a["pred_grade"].to_numpy()
    yb, qb = df_b["true_grade"].to_numpy(), df_b["pred_grade"].to_numpy()
    rng = np.random.RandomState(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        ia = np.concatenate([ga[p] for p in rng.choice(pa, len(pa), replace=True)])
        ib = np.concatenate([gb[p] for p in rng.choice(pb, len(pb), replace=True)])
        d[i] = qwk(ya[ia], qa[ia]) - qwk(yb[ib], qb[ib])
    lo, hi = np.percentile(d, [2.5, 97.5])
    obs = qwk(ya, qa) - qwk(yb, qb)
    return {"qwk_p1": qwk(ya, qa), "qwk_p2": qwk(yb, qb),
            "observed_diff": float(obs), "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "minimum_detectable_effect": float((hi - lo) / 2),
            "n_images_p1": int(len(df_a)), "n_images_p2": int(len(df_b)),
            "n_patients_p1": int(len(pa)), "n_patients_p2": int(len(pb))}


def exact_sign_flip(diffs):
    d = np.asarray(diffs, dtype=float)
    obs = float(d.mean())
    nulls = np.array([np.mean(np.array(s) * d) for s in itertools.product([1, -1], repeat=len(d))])
    return {"observed_mean_diff": obs,
            "p_value_two_sided": float((np.abs(nulls) >= abs(obs) - 1e-12).mean()),
            "n_pairs": len(d), "n_permutations": len(nulls)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/claim2c_converged_bootstrap.json")
    ap.add_argument("--fig", default="figures/report_v2/figI_converged_bootstrap.png")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()

    out = {"n_boot": args.n_boot, "scopes": {}, "per_run_unpaired": {}}
    print("=" * 74)
    print("CLAIM 2c -- P1 vs P2 UNDER CONVERGENCE, PATIENT-LEVEL PAIRED BOOTSTRAP")
    print("=" * 74)

    for scope in ["all_images", "eyepacs_only"]:
        out["scopes"][scope] = {}
        print(f"\n--- scope: {scope} ---")
        for sampler in SAMPLERS:
            per_seed = []
            for seed in SEEDS:
                p1 = load_run(root, "p1", sampler, seed)
                p2 = load_run(root, "p2", sampler, seed)
                common = p1.index.intersection(p2.index)
                if scope == "eyepacs_only":
                    common = pd.Index([i for i in common if str(i).startswith("eyepacs_")])
                a, b = p1.loc[common], p2.loc[common]
                assert (a["true_grade"].to_numpy() == b["true_grade"].to_numpy()).all(), \
                    "true labels disagree across protocols -- manifest mismatch"
                r = paired_patient_bootstrap(
                    a["patient_id"].to_numpy(), a["true_grade"].to_numpy(),
                    a["pred_grade"].to_numpy(), b["pred_grade"].to_numpy(),
                    n_boot=args.n_boot, seed=seed)
                r["seed"] = seed
                per_seed.append(r)
                print(f"  {sampler:15s} seed {seed}: P1 {r['qwk_a']:.4f}  P2 {r['qwk_b']:.4f}  "
                      f"diff {r['observed_diff']:+.4f}  CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
                      f"  {'EXCLUDES 0' if r['excludes_zero'] else 'includes 0'}"
                      f"   (n={r['n_images']} imgs / {r['n_patients']} pts)")
            out["scopes"][scope][sampler] = {
                "per_seed": per_seed,
                "mean_diff_across_seeds": float(np.mean([r["observed_diff"] for r in per_seed])),
                "both_seeds_exclude_zero": all(r["excludes_zero"] for r in per_seed),
                "any_seed_excludes_zero": any(r["excludes_zero"] for r in per_seed),
            }
        alld = [r["observed_diff"] for s in SAMPLERS for r in out["scopes"][scope][s]["per_seed"]]
        out["scopes"][scope]["sign_flip_over_4_conditions"] = exact_sign_flip(alld)

    # (b) PRIMARY: unpaired, full test sets, all the data
    out["unpaired_full"] = {}
    print("\n--- PRIMARY: unpaired, FULL test sets (all the data) ---")
    for scope in ["all_images", "eyepacs_only"]:
        out["unpaired_full"][scope] = {}
        for sampler in SAMPLERS:
            per_seed = []
            for seed in SEEDS:
                a = load_run(root, "p1", sampler, seed).reset_index()
                b = load_run(root, "p2", sampler, seed).reset_index()
                if scope == "eyepacs_only":
                    a = a[a["image_id"].astype(str).str.startswith("eyepacs_")]
                    b = b[b["image_id"].astype(str).str.startswith("eyepacs_")]
                r = unpaired_patient_bootstrap_diff(a, b, args.n_boot, seed)
                r["seed"] = seed
                per_seed.append(r)
                if scope == "eyepacs_only":
                    print(f"  {sampler:15s} seed {seed}: P1 {r['qwk_p1']:.4f}  P2 {r['qwk_p2']:.4f}  "
                          f"diff {r['observed_diff']:+.4f}  CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
                          f"  MDE {r['minimum_detectable_effect']:.4f}"
                          f"  {'EXCLUDES 0' if r['excludes_zero'] else 'includes 0'}")
            out["unpaired_full"][scope][sampler] = {
                "per_seed": per_seed,
                "mean_diff_across_seeds": float(np.mean([r["observed_diff"] for r in per_seed])),
                "both_seeds_exclude_zero": all(r["excludes_zero"] for r in per_seed),
                "any_seed_excludes_zero": any(r["excludes_zero"] for r in per_seed),
            }

    # (c) unpaired, each run on its own full test set -- continuity with Table 3
    for split in ["p1", "p2"]:
        for sampler in SAMPLERS:
            for seed in SEEDS:
                d = load_run(root, split, sampler, seed)
                key = f"{split}_{sampler}_seed{seed}"
                out["per_run_unpaired"][key] = {
                    "qwk_full_test": qwk(d["true_grade"], d["pred_grade"]),
                    "patient_bootstrap_ci95": patient_bootstrap_ci(
                        d["patient_id"].to_numpy(), d["true_grade"].to_numpy(),
                        d["pred_grade"].to_numpy(), n_boot=args.n_boot, seed=seed),
                    "n_test": int(len(d)),
                }

    # verdict
    ey = out["unpaired_full"]["eyepacs_only"]
    any_sig = any(ey[s]["any_seed_excludes_zero"] for s in SAMPLERS)
    all_sig = all(ey[s]["both_seeds_exclude_zero"] for s in SAMPLERS)
    mean_all = float(np.mean([r["observed_diff"] for s in SAMPLERS for r in ey[s]["per_seed"]]))
    mde = float(np.mean([r["minimum_detectable_effect"] for s in SAMPLERS for r in ey[s]["per_seed"]]))
    mde_paired = float(np.mean([r["minimum_detectable_effect"]
                                for s in SAMPLERS for r in out["scopes"]["eyepacs_only"][s]["per_seed"]]))
    out["minimum_detectable_effect_unpaired"] = mde
    out["minimum_detectable_effect_paired"] = mde_paired
    if all_sig and mean_all > 0:
        verdict = "leakage_detected_under_convergence"
        text = (f"P1 exceeds P2 by {mean_all:+.4f} QWK on average and the paired patient-level CI "
                f"excludes zero in EVERY condition (EyePACS-only). The Section 3.3 null result does "
                f"NOT survive full convergence: report that leakage inflation appears once the model "
                f"is trained to completion, and revise Sections 3.5/3.6 accordingly.")
    elif any_sig:
        verdict = "mixed"
        text = (f"P1 exceeds P2 by {mean_all:+.4f} QWK on average, but the paired CI excludes zero in "
                f"only SOME conditions. Report as suggestive and condition-dependent, not established; "
                f"the honest statement is that the null is no longer clean but is not overturned either.")
    else:
        verdict = "null_holds_under_convergence"
        text = (f"P1 minus P2 is {mean_all:+.4f} QWK and the patient-level CI includes zero in every "
                f"condition (EyePACS-only, full test sets). The null reported in Section 3.3 survives "
                f"full model convergence, so the provisional caveat in Section 3.5 can be discharged "
                f"-- but report the power alongside it: this test resolves effects down to about "
                f"{mde:.3f} QWK, which rules out an inflation the size of Tampu et al.'s upper range "
                f"(0.43) comfortably and their lower range (0.07) only marginally. A null without a "
                f"minimum detectable effect is not interpretable.")
    out["verdict"] = verdict
    out["verdict_text"] = text
    out["primary_scope"] = "eyepacs_only"
    out["mean_diff_eyepacs_only"] = mean_all

    (root / args.out).parent.mkdir(parents=True, exist_ok=True)
    (root / args.out).write_text(json.dumps(out, indent=2))
    print(f"\nPOWER: unpaired MDE {mde:.4f} QWK | paired-on-common MDE {mde_paired:.4f} QWK")
    print(f"\nVERDICT: {verdict}\n  {text}")
    print(f"\nWrote {root / args.out}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    rows = []
    for scope in ["eyepacs_only", "all_images"]:
        for sampler in SAMPLERS:
            for r in out["scopes"][scope][sampler]["per_seed"]:
                rows.append((f"{'EyePACS only' if scope=='eyepacs_only' else 'All images'}\n"
                             f"{sampler.replace('_',' ')}, seed {r['seed']}",
                             r["observed_diff"], r["ci95"], r["excludes_zero"]))
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, ci, sig) in zip(y, rows):
        col = BLUE if v > 0 else ORANGE
        ax.plot(ci, [yi, yi], color=col, lw=2.4, solid_capstyle="round", zorder=3)
        ax.plot([v], [yi], "o", ms=8.5, zorder=4, color=col if sig else "white",
                markeredgecolor=col, markeredgewidth=1.9)
        ax.text(0.062, yi, f"{v:+.4f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
                           f"{'CI excludes 0' if sig else 'n.s.'}",
                va="center", fontsize=8.2, family="monospace")
    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.4)
    ax.set_xlim(-0.055, 0.175); ax.set_xticks(np.arange(-0.05, 0.06, 0.025))
    ax.set_xlabel("QWK difference, P1 (image-level) minus P2 (patient-level)")
    ax.grid(axis="x", color="#eeeeee", zorder=0); ax.spines["left"].set_visible(False)
    ax.set_title("Phase 6b converged models: P1 vs P2, paired patient-level bootstrap\n"
                 "on the common test images. Filled marker = 95% CI excludes zero.", fontsize=10.5)
    fig.tight_layout()
    (root / args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / args.fig, dpi=200)
    print(f"Wrote {root / args.fig}")


if __name__ == "__main__":
    main()
