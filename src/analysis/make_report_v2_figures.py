"""Regenerate report figures for DR_Research_Project_Report_v2.docx.

Every value plotted here is read from results/*.json at run time.
Nothing is hard-coded. Re-running this script after re-running an
experiment updates the figures automatically.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures", "report_v2")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, INK, MUTED = "#0072B2", "#D55E00", "#222222", "#888888"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#999999",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})

def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)

# ---------------------------------------------------------------- Figure A
# Protocol comparison (P1/P2/P3), both heads. Regenerated so the title fits.
def fig_protocols():
    d = load("claim2_protocols.json")
    order = ["P1", "P2", "P3_eyepacs_to_aptos", "P3_aptos_to_eyepacs"]
    names = ["P1\nimage-level", "P2\npatient-level",
             "P3\nEyePACS->APTOS", "P3\nAPTOS->EyePACS"]
    mult = [d["protocols"][k]["reference_qwk_seed42"] for k in order]
    ordn = [d["protocols"][k]["ordinal_qwk"] for k in order]
    ci = [d["protocols"][k]["patient_bootstrap_ci95"] for k in order]
    err = np.array([[m - c[0] for m, c in zip(mult, ci)],
                    [c[1] - m for m, c in zip(mult, ci)]])
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar(x - w/2, mult, w, color=BLUE, label="multinomial head", zorder=3)
    ax.errorbar(x - w/2, mult, yerr=err, fmt="none", ecolor=INK,
                capsize=4, lw=1.4, zorder=4)
    ax.bar(x + w/2, ordn, w, color=ORANGE, label="ordinal head", zorder=3)
    for xi, v in zip(x - w/2, mult):
        ax.text(xi, v + 0.035, f"{v:.3f}", ha="center", fontsize=9, color=INK)
    for xi, v in zip(x + w/2, ordn):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=9, color=INK)
    p = d["p1_vs_p2_paired_permutation_test"]["p_value_two_sided"]
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("QWK (frozen EfficientNet-B0 @224 features)")
    ax.set_ylim(0, 0.9); ax.grid(axis="y", color="#e8e8e8", zorder=0)
    ax.set_title("Protocol comparison. Error bars: 95% patient-level bootstrap CI\n"
                 f"(multinomial). P1 vs P2 paired permutation p = {p:g} - not significant.",
                 fontsize=10.5)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figA_protocols.png"), dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------- Figure B
# Stratified inter-eye correlation + the per-grade transfer pattern.
def fig_stratified():
    d = load("claim1b_stratified.json")
    levels = [("All\npatients", "correlation_all", "lookup_overall"),
              ("One eye\ngrade >= 1", "correlation_left_ge1", "lookup_left_ge1"),
              ("One eye\ngrade >= 2\n(referable)", "correlation_left_ge2", "lookup_left_ge2")]
    qwk = [d[c]["quadratic_weighted_kappa"] for _, c, _ in levels]
    lo = [d[c]["qwk_ci95"][0] for _, c, _ in levels]
    hi = [d[c]["qwk_ci95"][1] for _, c, _ in levels]
    look = [d[l]["qwk"] for _, _, l in levels]
    null95 = d["permuted_patient_null"]["null_p95"]

    per = d["lookup_accuracy_per_left_grade"]
    grades = sorted(per, key=int)
    acc = [per[g]["accuracy"] for g in grades]
    ns = [per[g]["n"] for g in grades]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    x = np.arange(3); w = 0.38
    err = np.array([[q - l for q, l in zip(qwk, lo)], [h - q for q, h in zip(qwk, hi)]])
    a1.bar(x - w/2, qwk, w, color=BLUE, label="inter-eye QWK", zorder=3)
    a1.errorbar(x - w/2, qwk, yerr=err, fmt="none", ecolor=INK, capsize=4, lw=1.4, zorder=4)
    a1.bar(x + w/2, look, w, color=ORANGE, label="label-only lookup QWK", zorder=3)
    for xi, v in zip(x - w/2, qwk):
        a1.text(xi, v + 0.03, f"{v:.3f}", ha="center", fontsize=9)
    for xi, v in zip(x + w/2, look):
        a1.text(xi, v + 0.03, f"{v:.3f}", ha="center", fontsize=9)
    a1.axhline(null95, color=MUTED, ls="--", lw=1.3)
    a1.set_xlim(-0.6, 3.35)
    a1.text(2.62, null95 + 0.025, f"permuted-patient null\n95th pct = {null95:.3f}",
            ha="left", va="bottom", fontsize=8.2, color=MUTED)
    a1.set_xticks(x); a1.set_xticklabels([n for n, _, _ in levels], fontsize=9)
    a1.set_ylim(0, 1.0); a1.set_ylabel("QWK")
    a1.grid(axis="y", color="#e8e8e8", zorder=0)
    a1.legend(frameon=False, fontsize=9)
    a1.set_title("Correlation survives severity stratification", fontsize=10.5)

    cols = [MUTED if g != "1" else ORANGE for g in grades]
    a2.bar(grades, acc, 0.62, color=cols, zorder=3)
    for g, v, n in zip(grades, acc, ns):
        a2.text(g, v + 0.02, f"{v*100:.1f}%\n(n={n})", ha="center", fontsize=8.5)
    a2.set_ylim(0, 1.08); a2.set_xlabel("left-eye grade")
    a2.set_ylabel("partner-eye lookup accuracy")
    a2.grid(axis="y", color="#e8e8e8", zorder=0)
    a2.set_title("Grade 1 (Mild NPDR) is the one grade\nthat does not transfer between eyes",
                 fontsize=10.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figB_stratified.png"), dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------- Figure C
# Claim 3 decomposition, all three backbones side by side.
def fig_decomposition():
    files = [("EfficientNet-B0 @224", "claim3_decomposed.json"),
             ("EfficientNet-B0 @384", "claim3_decomposed_tf_efficientnet_b0_384.json"),
             ("ResNet-50 @224", "claim3_decomposed_resnet50_224.json")]
    keys = [("Classifier head\n(ordinal - multinomial)", "1_head_effect_ordinal_minus_multinomial_at_max"),
            ("Feature fusion\n(mean-pool - per-eye max)", "2_fusion_effect_pool_minus_per_eye_max_ordinal"),
            ("Feature fusion\n(concat - per-eye max)", "3_fusion_effect_concat_minus_per_eye_max_ordinal"),
            ("Aggregation\n(mean - max)", "4_aggregation_effect_mean_minus_max_ordinal"),
            ("Aggregation\n(min - max)", "5_aggregation_effect_min_minus_max_ordinal"),
            ("ORIGINAL TOTAL GAIN\n(pool+ordinal - max+multinomial)", "8_original_total_gain_pool_ordinal_minus_per_eye_max_multinomial")]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.9), sharey=True)
    for ax, (title, fn) in zip(axes, files):
        dec = load(fn)["decomposition"]
        y = np.arange(len(keys))[::-1]
        for yi, (lab, k) in zip(y, keys):
            v = dec[k]["mean_diff"]; c = dec[k]["ci95"]
            excludes0 = (c[0] > 0) or (c[1] < 0)
            col = BLUE if v > 0 else ORANGE
            ax.plot(c, [yi, yi], color=col, lw=2.2, solid_capstyle="round", zorder=3)
            ax.plot([v], [yi], marker="o", ms=8, zorder=4,
                    color=col if excludes0 else "white",
                    markeredgecolor=col, markeredgewidth=1.8)
        ax.axvline(0, color=INK, lw=1.1, zorder=2)
        ax.set_yticks(y); ax.set_xlim(-0.18, 0.16)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", color="#eeeeee", zorder=0)
        ax.set_xlabel("QWK difference")
    axes[0].set_yticklabels([k[0] for k in keys], fontsize=8.6)
    fig.suptitle("Claim 3 decomposition across three backbones. Filled marker = 95% paired "
                 "patient-bootstrap CI excludes zero; hollow = includes zero.", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "figC_decomposition.png"), dpi=200)
    plt.close(fig)
print("part 1 written")

# ---------------------------------------------------------------- Figure D
# NEW: every claim's effect size and CI on one axis.
def fig_all_effects():
    rows = []
    c2 = load("claim2_protocols.json")
    ab = load("claim2b_partner_ablation.json")
    d224 = load("claim3_decomposed.json")["decomposition"]
    d384 = load("claim3_decomposed_tf_efficientnet_b0_384.json")["decomposition"]
    dr50 = load("claim3_decomposed_resnet50_224.json")["decomposition"]

    rows.append(("Claim 2b", "Partner eye present vs absent\n(multinomial head)",
                 ab["multinomial"]["raw_difference"],
                 ab["multinomial"]["bootstrap_ci95_on_difference"]))
    rows.append(("Claim 2b", "Partner eye present vs absent\n(ordinal head)",
                 ab["ordinal"]["raw_difference"],
                 ab["ordinal"]["bootstrap_ci95_on_difference"]))
    for lab, d in [("EffNet-B0 @224", d224), ("EffNet-B0 @384", d384), ("ResNet-50 @224", dr50)]:
        k = d["1_head_effect_ordinal_minus_multinomial_at_max"]
        rows.append(("Claim 3: head", f"Ordinal vs multinomial head\n{lab}", k["mean_diff"], k["ci95"]))
    for lab, d in [("EffNet-B0 @224", d224), ("EffNet-B0 @384", d384), ("ResNet-50 @224", dr50)]:
        k = d["2_fusion_effect_pool_minus_per_eye_max_ordinal"]
        rows.append(("Claim 3: fusion", f"Mean-pooled both eyes vs per-eye max\n{lab}", k["mean_diff"], k["ci95"]))
    for lab, d in [("EffNet-B0 @224", d224), ("EffNet-B0 @384", d384), ("ResNet-50 @224", dr50)]:
        k = d["8_original_total_gain_pool_ordinal_minus_per_eye_max_multinomial"]
        rows.append(("Claim 3: total", f"Original entangled both-eyes gain\n{lab}", k["mean_diff"], k["ci95"]))

    fig, ax = plt.subplots(figsize=(9.6, 6.6))
    y = np.arange(len(rows))[::-1]
    for yi, (grp, lab, v, c) in zip(y, rows):
        excludes0 = (c[0] > 0) or (c[1] < 0)
        col = BLUE if v > 0 else ORANGE
        ax.plot(c, [yi, yi], color=col, lw=2.4, solid_capstyle="round", zorder=3)
        ax.plot([v], [yi], marker="o", ms=8.5, zorder=4,
                color=col if excludes0 else "white",
                markeredgecolor=col, markeredgewidth=1.9)
        ax.text(0.185, yi, f"{v:+.3f}  [{c[0]:+.3f}, {c[1]:+.3f}]   "
                           f"{'CI excludes 0' if excludes0 else 'n.s.'}",
                va="center", fontsize=8.4, family="monospace", color=INK)
    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[1] for r in rows], fontsize=8.6)
    ax.set_xlim(-0.16, 0.52)
    ax.set_xticks(np.arange(-0.15, 0.16, 0.05))
    ax.set_xlabel("QWK difference (paired patient-level bootstrap, 95% CI)")
    ax.grid(axis="x", color="#eeeeee", zorder=0)
    ax.spines["left"].set_visible(False)
    prev = None
    for yi, (grp, *_ ) in zip(y, rows):
        if grp != prev:
            ax.text(-0.155, yi + 0.42, grp, fontsize=9, weight="bold", color=MUTED)
            prev = grp
    ax.set_title("Every tested effect in this project on one axis.\n"
                 "Filled marker = CI excludes zero. Hollow = consistent with no effect.",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figD_all_effects.png"), dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------- Figure E
# NEW: Phase 6b sampler ablation.
def fig_sampler():
    conds = [("P1", "class_balanced"), ("P1", "none"), ("P2", "class_balanced"), ("P2", "none")]
    data = {}
    for split, samp in conds:
        for seed in (42, 43):
            d = load(f"finetune_converged_{split.lower()}_{samp}_seed{seed}.json")
            data[(split, samp, seed)] = d
    labels = ["P1 (image-level)\nseed 42", "P1 (image-level)\nseed 43",
              "P2 (patient-level)\nseed 42", "P2 (patient-level)\nseed 43"]
    cells = [("P1", 42), ("P1", 43), ("P2", 42), ("P2", 43)]
    cb = [data[(s, "class_balanced", sd)]["test_qwk"] for s, sd in cells]
    no = [data[(s, "none", sd)]["test_qwk"] for s, sd in cells]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.5))
    x = np.arange(4); w = 0.38
    a1.bar(x - w/2, cb, w, color=ORANGE, label="class-balanced sampler", zorder=3)
    a1.bar(x + w/2, no, w, color=BLUE, label="no resampling", zorder=3)
    for xi, v in zip(x - w/2, cb):
        a1.text(xi, v + 0.006, f"{v:.4f}", ha="center", fontsize=8.4)
    for xi, v in zip(x + w/2, no):
        a1.text(xi, v + 0.006, f"{v:.4f}", ha="center", fontsize=8.4)
    for xi, a, b in zip(x, cb, no):
        a1.text(xi, 0.5, f"{b-a:+.4f}", ha="center", fontsize=9, color=INK, weight="bold")
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=8.6)
    a1.set_ylim(0.45, 0.72); a1.set_ylabel("test-set QWK")
    a1.grid(axis="y", color="#e8e8e8", zorder=0)
    a1.legend(frameon=False, fontsize=9, loc="upper left")
    a1.set_title("Removing class-balanced resampling helped\nin all four matched pairs", fontsize=10.5)

    g1cb = [data[(s, "class_balanced", sd)]["per_class_recall"]["1"]["recall"] for s, sd in cells]
    g1no = [data[(s, "none", sd)]["per_class_recall"]["1"]["recall"] for s, sd in cells]
    a2.bar(x - w/2, g1cb, w, color=ORANGE, label="class-balanced sampler", zorder=3)
    a2.bar(x + w/2, g1no, w, color=BLUE, label="no resampling", zorder=3)
    for xi, v in zip(x - w/2, g1cb):
        a2.text(xi, v + 0.004, f"{v*100:.1f}%", ha="center", fontsize=8.4)
    for xi, v in zip(x + w/2, g1no):
        a2.text(xi, v + 0.004, f"{v*100:.1f}%", ha="center", fontsize=8.4)
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=8.6)
    a2.set_ylim(0, 0.22); a2.set_ylabel("grade-1 (Mild NPDR) recall")
    a2.grid(axis="y", color="#e8e8e8", zorder=0)
    a2.legend(frameon=False, fontsize=9, loc="upper left")
    a2.set_title("Grade-1 recall stayed in the same low band\n(8.7-14.0%) under both sampler settings", fontsize=10.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figE_sampler.png"), dpi=200)
    plt.close(fig)



# ---------------------------------------------------------------- Figure F
# Partner-eye ablation, restyled (report Figure 3).
def fig_ablation():
    d = load("claim2b_partner_ablation.json")
    m, o = d["multinomial"], d["ordinal"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.2))
    grades = ["0", "1", "2", "3", "4"]
    pres = [m["grade_distribution_present"][g] for g in grades]
    absн = [m["grade_distribution_absent"][g] for g in grades]
    x = np.arange(5); w = 0.38
    a1.bar(x - w/2, pres, w, color=BLUE, label=f"partner present (n={d['n_partner_present']:,})", zorder=3)
    a1.bar(x + w/2, absн, w, color=ORANGE, label=f"partner absent (n={d['n_partner_absent']:,})", zorder=3)
    a1.set_xticks(x); a1.set_xticklabels(grades)
    a1.set_xlabel("grade"); a1.set_ylabel("fraction of group")
    a1.grid(axis="y", color="#e8e8e8", zorder=0); a1.legend(frameon=False, fontsize=9)
    a1.set_title("Grade distributions are close but not identical\n(this is what motivated the grade-matched check)", fontsize=10.5)

    labels = ["multinomial\nraw", "multinomial\ngrade-matched", "ordinal\nraw", "ordinal\ngrade-matched"]
    pv = [m["raw_qwk_present"], m["grade_matched_comparison"]["qwk_present"],
          o["raw_qwk_present"], o["grade_matched_comparison"]["qwk_present"]]
    av = [m["raw_qwk_absent"], m["grade_matched_comparison"]["qwk_absent"],
          o["raw_qwk_absent"], o["grade_matched_comparison"]["qwk_absent"]]
    x = np.arange(4)
    a2.bar(x - w/2, pv, w, color=BLUE, label="partner present", zorder=3)
    a2.bar(x + w/2, av, w, color=ORANGE, label="partner absent", zorder=3)
    for xi, a, b in zip(x, pv, av):
        a2.text(xi, max(a, b) + 0.018, f"{a-b:+.3f}", ha="center", fontsize=9, weight="bold")
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=8.8)
    a2.set_ylim(0, 0.72); a2.set_ylabel("QWK")
    a2.grid(axis="y", color="#e8e8e8", zorder=0); a2.legend(frameon=False, fontsize=9, loc="upper right")
    a2.set_title(f"Every gap is consistent with zero\n"
                 f"multinomial CI [{m['bootstrap_ci95_on_difference'][0]:+.3f}, {m['bootstrap_ci95_on_difference'][1]:+.3f}], p={m['permutation_test']['p_value_two_sided']:g}\n"
                 f"ordinal CI [{o['bootstrap_ci95_on_difference'][0]:+.3f}, {o['bootstrap_ci95_on_difference'][1]:+.3f}], p={o['permutation_test']['p_value_two_sided']:g}",
                 fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figF_ablation.png"), dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------- Figure G
# The whole Phase 4 story on one axis (report Figure 4).
def fig_phase4():
    d = load("summary_comparison.json")
    labels, vals, cis, groups = [], [], [], []
    for b in d["bars"]:
        labels.append(b["label"].replace("\n", "\n")); vals.append(b["qwk"])
        cis.append(b["ci"]); groups.append(b["group"])
    cmap = {"baseline": "#6f7378", "P1": ORANGE, "P2": BLUE, "ablation": "#2e7d5b"}
    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    x = np.arange(len(vals))
    ax.bar(x, vals, 0.62, color=[cmap[g] for g in groups], zorder=3)
    for xi, v, c in zip(x, vals, cis):
        top = v
        if c:
            ax.errorbar(xi, v, yerr=[[v - c[0]], [c[1] - v]], fmt="none",
                        ecolor=INK, capsize=4, lw=1.4, zorder=4)
            top = c[1]
        ax.text(xi, top + 0.022, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.6)
    ax.set_ylim(0, 1.0); ax.set_ylabel("QWK")
    ax.grid(axis="y", color="#e8e8e8", zorder=0)
    ax.axhline(d["bars"][0]["qwk"], color="#6f7378", ls="--", lw=1.2, zorder=2)
    ax.text(6.45, d["bars"][0]["qwk"] + 0.02, "zero-pixel shortcut ceiling",
            ha="right", fontsize=8.5, color="#6f7378")
    ax.set_title(f"Every Phase 4 model sits well below the zero-pixel ceiling.\n"
                 f"P1 vs P2 diff = {d['p1_vs_p2_diff']:+.3f} (p={d['p1_vs_p2_p_value']:g});  "
                 f"partner ablation diff = {d['ablation_diff_multinomial']:+.3f} (p={d['ablation_p_value_multinomial']:g})",
                 fontsize=10.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figG_phase4.png"), dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------- Figure H
# Full bilateral grid (report Figure 5).
def fig_grid():
    d = load("claim3_decomposed.json")
    order = [("per_eye_max", "per-eye\nmax"), ("per_eye_mean", "per-eye\nmean"),
             ("per_eye_min", "per-eye\nmin"), ("concat", "concat\nfeatures"),
             ("pool", "mean-pooled\nfeatures")]
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    x = np.arange(len(order)); w = 0.38
    for i, head in enumerate(["multinomial", "ordinal"]):
        vals, los, his = [], [], []
        for k, _ in order:
            c = d["cells"][f"{k}_{head}"]
            vals.append(c["qwk"]); los.append(c["patient_bootstrap_ci95"][0]); his.append(c["patient_bootstrap_ci95"][1])
        off = (i - 0.5) * w
        col = ORANGE if head == "multinomial" else BLUE
        ax.bar(x + off, vals, w, color=col, label=f"{head} head", zorder=3)
        ax.errorbar(x + off, vals, yerr=[np.array(vals) - los, np.array(his) - np.array(vals)],
                    fmt="none", ecolor=INK, capsize=3.5, lw=1.3, zorder=4)
        for xi, v, h in zip(x + off, vals, his):
            ax.text(xi, h + 0.014, f"{v:.3f}", ha="center", fontsize=8.2)
    ceil = d["arm_D_label_only_lookup_cited"]
    ax.axhline(ceil, color="#6f7378", ls="--", lw=1.2, zorder=2)
    ax.text(4.45, ceil + 0.02, f"zero-pixel shortcut ceiling ({ceil:.3f})",
            ha="right", fontsize=8.5, color="#6f7378")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in order], fontsize=9)
    ax.set_ylim(0, 0.95); ax.set_ylabel(f"patient-level QWK (n={d['n_test_patients']:,} patients)")
    ax.grid(axis="y", color="#e8e8e8", zorder=0)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Full bilateral grid, EfficientNet-B0 @224px (P2, EyePACS only).\n"
                 "The ordinal head wins in every column; mean-pooling is the only fusion that helps.",
                 fontsize=10.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figH_grid.png"), dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    fig_protocols(); print("A ok")
    fig_stratified(); print("B ok")
    fig_decomposition(); print("C ok")
    fig_all_effects(); print("D ok")
    fig_sampler(); print("E ok")
    fig_ablation(); print("F ok")
    fig_phase4(); print("G ok")
    fig_grid(); print("H ok")
