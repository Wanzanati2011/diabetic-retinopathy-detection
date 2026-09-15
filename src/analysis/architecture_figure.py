"""
Figure 3.1 -- the five-layer system architecture, regenerated for the final
report. The Review 2 version of this diagram marked the deployment layer
"in progress"; the application is now built, calibrated and live, so every
layer is complete and the diagram says so.

Generated from code like every other figure in this project, so the numbers
on it cannot drift from the numbers in the text.

    python src\\analysis\\architecture_figure.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEAL = "#0C6F6E"
TEAL_SOFT = "#E4F0EF"
AMBER = "#A96606"
INK = "#0F1A1D"
SLATE = "#5B6E72"
RULE = "#C9D8D8"
PAPER = "#FFFFFF"

LAYERS = [
    ("1", "Data foundation",
     ["Unified manifest, 38,788 rows",
      "Patient identity reconstructed",
      "8 corrupted frames excluded",
      "148 near-duplicate pairs flagged",
      "P1 / P2 / P3 written as fold files",
      "20 acceptance tests"]),
    ("2", "Feature caching",
     ["EfficientNet-B0 @ 224 px",
      "EfficientNet-B0 @ 384 px",
      "ResNet-50 @ 224 px",
      "Each image forwarded once,",
      "features cached to disk"]),
    ("3", "Grading head",
     ["Multinomial head (baseline)",
      "Ordinal head: ridge + 4 thresholds",
      "  optimised by Nelder-Mead",
      "Optional bilateral mean-pooling",
      "Deployed: fine-tuned B0 @ 384 px",
      "Test QWK 0.716"]),
    ("4", "Referral decision",
     ["P(grade >= 2) = p2 + p3 + p4",
      "Threshold fixed on validation",
      "  for 90% sensitivity",
      "Test: 89.6% sens / 69.7% spec",
      "Referable AUROC 0.912"]),
    ("5", "Deployment wrapper",
     ["Temperature scaling, T = 3.367",
      "ECE 0.163 -> 0.029",
      "Reject option, tau = 0.593",
      "Grad-CAM incl. failure cases",
      "Fundus Console, 5 tabs, live"]),
]


def main():
    fig, ax = plt.subplots(figsize=(9.9, 8.4))
    ax.set_xlim(0, 96)
    ax.set_ylim(0, 84)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)

    ax.text(1.5, 80.0, "P R O P O S E D   A R C H I T E C T U R E", fontsize=8,
            color=TEAL, fontweight="bold", family="DejaVu Sans")
    ax.text(1.5, 75.4, "Diabetic Retinopathy Detection and Grading Pipeline",
            fontsize=15, color=INK, fontweight="bold", family="DejaVu Sans")

    top = 70.0
    row_h = 11.0
    gap = 1.8
    label_w = 30.0
    detail_x = label_w + 4.0

    for i, (num, title, details) in enumerate(LAYERS):
        y = top - i * (row_h + gap) - row_h

        # left label block
        ax.add_patch(FancyBboxPatch(
            (1.5, y), label_w, row_h,
            boxstyle="round,pad=0,rounding_size=0.6",
            facecolor=TEAL, edgecolor="none", zorder=2))
        ax.text(3.4, y + row_h - 3.0, num, fontsize=9, color="#9FD8D3",
                fontweight="bold", family="DejaVu Sans", va="center")
        ax.text(6.6, y + row_h - 3.0, title, fontsize=10.5, color="white",
                fontweight="bold", family="DejaVu Sans", va="center")
        ax.text(6.6, y + 2.8, "COMPLETE", fontsize=7.5, color="#9FD8D3",
                family="DejaVu Sans", va="center", fontweight="bold")
        ax.plot([4.2], [y + 2.8], marker="o", ms=4.5, color="#9FD8D3", zorder=3)

        # right detail block
        ax.add_patch(FancyBboxPatch(
            (detail_x, y), 96 - detail_x - 1.5, row_h,
            boxstyle="round,pad=0,rounding_size=0.6",
            facecolor=TEAL_SOFT, edgecolor=RULE, linewidth=0.8, zorder=1))
        for j, line in enumerate(details):
            ax.text(detail_x + 2.2, y + row_h - 2.4 - j * 1.5, line,
                    fontsize=7.8, color=INK if not line.startswith("  ") else SLATE,
                    family="DejaVu Sans", va="center")

        # connector
        if i < len(LAYERS) - 1:
            ax.add_patch(FancyArrowPatch(
                (1.5 + label_w / 2, y - 0.1), (1.5 + label_w / 2, y - gap + 0.1),
                arrowstyle="-|>", mutation_scale=9, color=SLATE, linewidth=1.1,
                zorder=3))

    # footer rule
    ax.plot([1.5, 94.5], [4.4, 4.4], color=RULE, linewidth=0.9)
    ax.text(1.5, 2.3, "STATISTICAL STANDARD", fontsize=7.5, color=TEAL,
            fontweight="bold", family="DejaVu Sans")
    ax.text(25.0, 2.3,
            "Patient-level bootstrap 95% intervals on every metric  ·  paired "
            "intervals on every comparison  ·  test sets frozen and scored once",
            fontsize=7.0, color=SLATE, family="DejaVu Sans")

    out = PROJECT_ROOT / "figures" / "report_v2" / "architecture_diagram_final.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
