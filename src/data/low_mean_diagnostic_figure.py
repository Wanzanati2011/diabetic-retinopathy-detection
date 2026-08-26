"""Builds figures/low_mean_diagnostic.png: original source vs 224px processed
crop, side by side, for each of the 8 flagged low-mean-pixel images."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    root = PROJECT_ROOT
    diag = json.loads((root / "results/low_mean_diagnostic.json").read_text())
    entries = diag["flagged_images"]

    n = len(entries)
    fig, axes = plt.subplots(n, 2, figsize=(6.5, n * 3.4))

    for row_i, e in enumerate(entries):
        image_id = e["image_id"]
        orig_path = root / "resized train 15" / f"{image_id.replace('eyepacs_', '').rsplit('_', 1)[0]}_{e['eye']}.jpg"
        proc_path = root / "data/processed/224" / f"{image_id}.jpg"

        ax_o, ax_p = axes[row_i]
        for ax in (ax_o, ax_p):
            ax.axis("off")

        if orig_path.exists():
            ax_o.imshow(Image.open(orig_path))
        else:
            ax_o.text(0.5, 0.5, "MISSING", ha="center", va="center")
        ax_o.set_title(f"ORIGINAL — {image_id}\nmean={e['original_stats']['mean']:.2f}  "
                        f"min={e['original_stats']['min']:.0f}  max={e['original_stats']['max']:.0f}",
                        fontsize=8)

        if proc_path.exists():
            ax_p.imshow(Image.open(proc_path))
        else:
            ax_p.text(0.5, 0.5, "MISSING", ha="center", va="center")
        ax_p.set_title(f"PROCESSED 224px\ngrade={e['grade']}  patient={e['patient_id']}",
                        fontsize=8)

    fig.suptitle("Acceptance Test 6.1 follow-up: 8 low-mean-pixel crops — original vs processed\n"
                  "(all 8: partner eye looks normal — single failed capture, not a bad patient)",
                  fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = root / "figures/low_mean_diagnostic.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
