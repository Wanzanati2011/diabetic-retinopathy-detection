"""
Claim 2b (converged) -- partner-eye ablation on the Phase 6b models.

The original Claim 2b (src/experiments/claim2b_partner_ablation.py) ran on
frozen features and, later, on the Phase 6 model -- which Section 3.5 shows
was undertrained. An undertrained model is precisely the configuration LEAST
able to memorise a patient, so a null there is weak evidence. This re-runs
the same ablation on the converged Phase 6b models, which is where the
mechanism has a real chance to appear.

The mechanism under test, stated plainly: under P1 (image-level split) a
test eye's PARTNER eye may sit in the training set. If the model is
exploiting patient identity rather than reading the retina, test eyes whose
partner was seen in training should score higher than test eyes whose
partner was not. Same model, same test set, split by a property of the data
rather than by anything the model was told.

Two confounds are handled, matching the original script so the numbers are
comparable:
  - GRADE MATCHING. The two groups' grade distributions are close but not
    identical, and QWK is sensitive to the label mix. A grade-matched
    comparison (subsample the larger group to the smaller group's per-grade
    counts) is reported next to the raw one.
  - PATIENT-LEVEL RESAMPLING. The bootstrap resamples patients, not images.

Only EyePACS images can participate: APTOS has no eye pairing, so APTOS rows
are excluded from both groups rather than silently counted as "partner
absent" (which would be a bug -- absent-by-design is not the same as
absent-by-split).

Pre-registered outcome:
  - CI on (present - absent) EXCLUDING zero and positive => the leakage
    mechanism is real in converged models; Section 3.3's null is overturned
    at the mechanism level and must be reported as such.
  - CI INCLUDING zero => the null holds where it matters most, and is now
    tested against a model that actually converged.

Writes results/claim2b_converged_ablation.json
       figures/report_v2/figJ_converged_ablation.png

Usage:
  python src/experiments/claim2b_converged_ablation.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLERS = ["class_balanced", "none"]
SEEDS = [42, 43]
BLUE, ORANGE, INK = "#0072B2", "#D55E00", "#222222"


def qwk(y, p):
    return float(cohen_kappa_score(y, p, weights="quadratic"))


def bootstrap_diff_ci(dfa, dfb, n_boot=2000, seed=0):
    """Independent patient-level resamples of the two disjoint groups; CI on
    the QWK difference. The groups are disjoint patients, so this is an
    unpaired difference -- resample each side independently."""
    ga = dfa.groupby("patient_id").indices
    gb = dfb.groupby("patient_id").indices
    pa, pb = np.array(list(ga.keys())), np.array(list(gb.keys()))
    ya, qa = dfa["true_grade"].to_numpy(), dfa["pred_grade"].to_numpy()
    yb, qb = dfb["true_grade"].to_numpy(), dfb["pred_grade"].to_numpy()
    rng = np.random.RandomState(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        ia = np.concatenate([ga[p] for p in rng.choice(pa, len(pa), replace=True)])
        ib = np.concatenate([gb[p] for p in rng.choice(pb, len(pb), replace=True)])
        d[i] = qwk(ya[ia], qa[ia]) - qwk(yb[ib], qb[ib])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def permutation_test(dfa, dfb, n_perm=2000, seed=0):
    """Shuffle the present/absent label across patients and recompute the gap."""
    both = pd.concat([dfa, dfb], ignore_index=True)
    n_a_pat = dfa["patient_id"].nunique()
    pats = both["patient_id"].unique()
    obs = qwk(dfa["true_grade"], dfa["pred_grade"]) - qwk(dfb["true_grade"], dfb["pred_grade"])
    rng = np.random.RandomState(seed)
    nulls = np.empty(n_perm)
    idx_by_pat = both.groupby("patient_id").indices
    y, p = both["true_grade"].to_numpy(), both["pred_grade"].to_numpy()
    for i in range(n_perm):
        perm = rng.permutation(pats)
        sa = np.concatenate([idx_by_pat[q] for q in perm[:n_a_pat]])
        sb = np.concatenate([idx_by_pat[q] for q in perm[n_a_pat:]])
        nulls[i] = qwk(y[sa], p[sa]) - qwk(y[sb], p[sb])
    return {"observed_diff": float(obs),
            "null_mean": float(nulls.mean()), "null_std": float(nulls.std()),
            "p_value_two_sided": float((np.abs(nulls) >= abs(obs) - 1e-12).mean()),
            "n_permutations": n_perm}


def grade_match(dfa, dfb, seed=0):
    """Subsample the larger group per grade to the smaller group's counts."""
    rng = np.random.RandomState(seed)
    ca, cb = dfa["true_grade"].value_counts(), dfb["true_grade"].value_counts()
    keep_a, keep_b, shortfall = [], [], {}
    for g in sorted(set(ca.index) | set(cb.index)):
        na, nb = int(ca.get(g, 0)), int(cb.get(g, 0))
        n = min(na, nb)
        ia = dfa.index[dfa["true_grade"] == g].to_numpy()
        ib = dfb.index[dfb["true_grade"] == g].to_numpy()
        if n == 0:
            shortfall[int(g)] = {"present": na, "absent": nb}
            continue
        keep_a.append(rng.choice(ia, n, replace=False))
        keep_b.append(rng.choice(ib, n, replace=False))
    ma, mb = dfa.loc[np.concatenate(keep_a)], dfb.loc[np.concatenate(keep_b)]
    return ma, mb, shortfall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--manifest", default="data/manifests/manifest.csv")
    ap.add_argument("--split", default="data/splits/p1.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/claim2b_converged_ablation.json")
    ap.add_argument("--fig", default="figures/report_v2/figJ_converged_ablation.png")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()

    man = pd.read_csv(root / args.manifest)
    if "excluded" in man.columns:
        man = man[~man["excluded"].fillna(False).astype(bool)]
    partner_of = dict(zip(man["image_id"], man["partner_id"]))

    split = json.loads((root / args.split).read_text())
    train_ids = {i for i, v in split.items() if v["fold"] == "train"}

    print("=" * 74)
    print("CLAIM 2b (CONVERGED) -- PARTNER-EYE ABLATION ON PHASE 6b MODELS")
    print("=" * 74)

    out = {"split": args.split, "n_boot": args.n_boot, "runs": {}}
    for sampler in SAMPLERS:
        for seed in SEEDS:
            csv = root / "results" / f"finetune_converged_p1_{sampler}_seed{seed}_test_predictions.csv"
            if not csv.exists():
                raise SystemExit(f"ERROR: {csv} not found. Run dump_test_predictions.py first.")
            df = pd.read_csv(csv)
            df = df[df["image_id"].astype(str).str.startswith("eyepacs_")].copy()
            df["partner_id"] = df["image_id"].map(partner_of)
            df = df[df["partner_id"].notna()].copy()
            df["partner_in_train"] = df["partner_id"].isin(train_ids)
            df = df.reset_index(drop=True)

            pres = df[df["partner_in_train"]].copy()
            absent = df[~df["partner_in_train"]].copy()
            raw_p, raw_a = qwk(pres["true_grade"], pres["pred_grade"]), qwk(absent["true_grade"], absent["pred_grade"])
            lo, hi = bootstrap_diff_ci(pres, absent, args.n_boot, seed)
            perm = permutation_test(pres, absent, args.n_boot, seed)
            ma, mb, short = grade_match(pres, absent, seed)
            gm_p, gm_a = qwk(ma["true_grade"], ma["pred_grade"]), qwk(mb["true_grade"], mb["pred_grade"])

            key = f"{sampler}_seed{seed}"
            out["runs"][key] = {
                "sampler": sampler, "seed": seed,
                "n_partner_present": int(len(pres)), "n_partner_absent": int(len(absent)),
                "raw_qwk_present": raw_p, "raw_qwk_absent": raw_a,
                "raw_difference": float(raw_p - raw_a),
                "bootstrap_ci95_on_difference": [lo, hi],
                "excludes_zero": bool(lo > 0 or hi < 0),
                "permutation_test": perm,
                "grade_matched": {"qwk_present": gm_p, "qwk_absent": gm_a,
                                  "difference": float(gm_p - gm_a), "shortfall": short},
            }
            print(f"  {sampler:15s} seed {seed}: present {raw_p:.4f} (n={len(pres)})  "
                  f"absent {raw_a:.4f} (n={len(absent)})  diff {raw_p-raw_a:+.4f}  "
                  f"CI [{lo:+.4f}, {hi:+.4f}]  p={perm['p_value_two_sided']:.4f}  "
                  f"grade-matched {gm_p-gm_a:+.4f}  {'EXCLUDES 0' if (lo>0 or hi<0) else 'includes 0'}")

    diffs = [r["raw_difference"] for r in out["runs"].values()]
    sig = [r["excludes_zero"] for r in out["runs"].values()]
    out["mean_raw_difference"] = float(np.mean(diffs))
    out["n_conditions_excluding_zero"] = int(sum(sig))
    if all(sig) and np.mean(diffs) > 0:
        out["verdict"] = "mechanism_confirmed_under_convergence"
        out["verdict_text"] = (
            f"Test eyes whose partner was in training outscore those whose partner was not by "
            f"{np.mean(diffs):+.4f} QWK on average, and the CI excludes zero in all "
            f"{len(diffs)} converged conditions. The leakage MECHANISM is real once the model "
            f"converges, even if the whole-protocol P1-vs-P2 gap is small -- report this as the "
            f"decisive result and revise Sections 3.3 and 3.6.")
    elif any(sig):
        out["verdict"] = "mixed"
        out["verdict_text"] = (
            f"Mean gap {np.mean(diffs):+.4f} QWK; the CI excludes zero in {sum(sig)} of {len(diffs)} "
            f"converged conditions. Suggestive but not consistent -- report as condition-dependent.")
    else:
        out["verdict"] = "null_holds_at_mechanism_level"
        out["verdict_text"] = (
            f"Mean gap {np.mean(diffs):+.4f} QWK and every CI includes zero. The ablation that most "
            f"directly targets the leakage mechanism returns null on CONVERGED models, which is a "
            f"materially stronger negative result than the Phase 6 version reported in Section 3.3.")
    (root / args.out).write_text(json.dumps(out, indent=2))
    print(f"\nVERDICT: {out['verdict']}\n  {out['verdict_text']}")
    print(f"\nWrote {root / args.out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    keys = list(out["runs"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))
    x = np.arange(len(keys)); w = 0.38
    pv = [out["runs"][k]["raw_qwk_present"] for k in keys]
    av = [out["runs"][k]["raw_qwk_absent"] for k in keys]
    a1.bar(x - w/2, pv, w, color=BLUE, label="partner eye in training", zorder=3)
    a1.bar(x + w/2, av, w, color=ORANGE, label="partner eye absent", zorder=3)
    for xi, p_, a_ in zip(x, pv, av):
        a1.text(xi, max(p_, a_) + 0.012, f"{p_-a_:+.4f}", ha="center", fontsize=9, weight="bold")
    a1.set_xticks(x)
    a1.set_xticklabels([k.replace("_seed", "\nseed ").replace("_", " ") for k in keys], fontsize=8.6)
    a1.set_ylabel("QWK"); a1.set_ylim(0, max(pv + av) * 1.25)
    a1.grid(axis="y", color="#e8e8e8", zorder=0); a1.legend(frameon=False, fontsize=9)
    a1.set_title("Converged Phase 6b models, P1 test set\n(EyePACS only)", fontsize=10.5)

    y = np.arange(len(keys))[::-1]
    for yi, k in zip(y, keys):
        r = out["runs"][k]; ci = r["bootstrap_ci95_on_difference"]; v = r["raw_difference"]
        col = BLUE if v > 0 else ORANGE
        a2.plot(ci, [yi, yi], color=col, lw=2.4, solid_capstyle="round", zorder=3)
        a2.plot([v], [yi], "o", ms=8, zorder=4, color=col if r["excludes_zero"] else "white",
                markeredgecolor=col, markeredgewidth=1.8)
    a2.axvline(0, color=INK, lw=1.2, zorder=2)
    a2.set_yticks(y)
    a2.set_yticklabels([k.replace("_seed", "\nseed ").replace("_", " ") for k in keys], fontsize=8.6)
    a2.set_xlabel("QWK difference (partner present minus absent)")
    a2.grid(axis="x", color="#eeeeee", zorder=0)
    a2.set_title("Filled marker = 95% patient-bootstrap CI\nexcludes zero", fontsize=10.5)
    fig.tight_layout()
    (root / args.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / args.fig, dpi=200)
    print(f"Wrote {root / args.fig}")


if __name__ == "__main__":
    main()
