"""
Figure 5.10 -- Grad-CAM panel, INCLUDING failure cases.

The last GPU task in this project. Minutes, not hours: it runs the model
over a small hand-picked set of test images, not the whole fold.

    python src\\train\\gradcam_panel.py

Why the panel must include failures
-----------------------------------
This project took a constraint from Selvaraju et al. (2017/2020) and wrote
it into MASTER_PLAN.md Part 11: the explanation panel has to include
misclassified images, not a curated set of convincing correct ones. An
explanation method applied only to successes cannot tell you anything you
did not already believe, and a panel of six confident hits reads as
marketing rather than evidence. So this script selects its images by a
FIXED RULE, stated here, rather than by looking at the maps and keeping the
pretty ones:

    row 1  confident and correct     -- highest-confidence correct referable
    row 2  confident and WRONG       -- highest-confidence misclassification
    row 3  rejected by the reject option -- lowest-confidence cases, i.e.
                                        exactly what the app would hand to a
                                        human grader
    row 4  grade-1 failures          -- true grade 1 predicted grade 0, the
                                        project's documented weak spot

Selection is deterministic (sorted by confidence, no sampling), so the panel
is reproducible and cannot be quietly re-rolled until it looks good.

Reads app/release/thresholds.json for the temperature and the reject
threshold, so the "rejected" row is genuinely the set the deployed
application would abstain on, not an illustration of one.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_converged import build_model  # noqa: E402

RUN = "finetune_app_converged_p2_class_balanced_seed42"
GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR",
               "Proliferative DR"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
N_PER_ROW = 4


def softmax_T(logits, T):
    z = logits / float(T)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def load_tensor(img_path, device):
    with Image.open(img_path) as im:
        arr = np.array(im.convert("RGB")).astype(np.float32) / 255.0
    norm = (arr - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(norm).permute(2, 0, 1).float().unsqueeze(0)
    return t.to(device), arr


def gradcam(model, x, target_class, target_layer):
    """Minimal Grad-CAM: no external dependency, so this cannot fail on a
    grad-cam version mismatch the way the app's optional import can."""
    acts, grads = {}, {}

    def fwd_hook(_m, _i, o):
        acts["v"] = o.detach()

    def bwd_hook(_m, _gi, go):
        grads["v"] = go[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[0, target_class].backward()
        a, g = acts["v"][0], grads["v"][0]
        weights = g.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * a).sum(dim=0))
        cam = cam - cam.min()
        if float(cam.max()) > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy()
    finally:
        h1.remove(); h2.remove()


def overlay(ax, rgb, cam, title, subtitle, colour):
    cam_img = np.array(Image.fromarray((cam * 255).astype(np.uint8))
                       .resize((rgb.shape[1], rgb.shape[0]), Image.BILINEAR))
    ax.imshow(rgb)
    ax.imshow(cam_img, cmap="inferno", alpha=0.42)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(colour); s.set_linewidth(2.0)
    ax.set_title(title, fontsize=7.6, color=colour, pad=3)
    ax.set_xlabel(subtitle, fontsize=6.8, color="#33484D", labelpad=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default=RUN)
    args = ap.parse_args()

    root = PROJECT_ROOT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pred_csv = root / "results" / f"{args.run_name}_test_predictions.csv"
    if not pred_csv.exists():
        raise SystemExit(f"Missing {pred_csv} -- run dump_app_predictions.py first.")
    df = pd.read_csv(pred_csv)
    logits = df[[f"logit_{k}" for k in range(5)]].to_numpy()

    th_path = root / "app" / "release" / "thresholds.json"
    if th_path.exists():
        th = json.loads(th_path.read_text())
        T, tau = float(th["temperature"]), float(th["reject_tau"])
        print(f"Using deployed thresholds: T={T:.4f}  tau={tau:.4f}")
    else:
        T, tau = 1.0, 0.0
        print("WARNING: no thresholds.json -- using T=1, tau=0 (uncalibrated).")

    probs = softmax_T(logits, T)
    df["conf"] = probs.max(axis=1)
    df["pred"] = probs.argmax(axis=1)
    df["correct"] = df["pred"] == df["true_grade"]

    rows = [
        ("Confident and correct", "#0C6F6E",
         df[(df["correct"]) & (df["true_grade"] >= 2)]
         .sort_values("conf", ascending=False)),
        ("Confident and WRONG", "#A96606",
         df[~df["correct"]].sort_values("conf", ascending=False)),
        (f"Rejected by the app (conf < {tau:.2f})", "#5B6E72",
         df[df["conf"] < tau].sort_values("conf")),
        ("Grade 1 called grade 0", "#A96606",
         df[(df["true_grade"] == 1) & (df["pred"] == 0)]
         .sort_values("conf", ascending=False)),
    ]

    ckpt = torch.load(root / "checkpoints" / args.run_name / "best.pt",
                      map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg["model"], device, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    target_layer = model.conv_head if hasattr(model, "conv_head") else \
        list(model.modules())[-6]
    print(f"Grad-CAM target layer: {type(target_layer).__name__}")

    img_dir = root / "data" / "processed" / str(cfg["image_size"])
    fig, axes = plt.subplots(len(rows), N_PER_ROW,
                             figsize=(2.35 * N_PER_ROW, 2.62 * len(rows)))
    for r, (label, colour, sub) in enumerate(rows):
        picks = sub.head(N_PER_ROW)
        print(f"  row {r+1}: {label} -- {len(sub)} candidates, showing {len(picks)}")
        for c in range(N_PER_ROW):
            ax = axes[r, c]
            if c >= len(picks):
                ax.axis("off")
                continue
            row = picks.iloc[c]
            p = img_dir / f"{row['image_id']}.jpg"
            if not p.exists():
                ax.axis("off"); continue
            x, rgb = load_tensor(p, device)
            cam = gradcam(model, x, int(row["pred"]), target_layer)
            overlay(ax, rgb, cam,
                    f"true {int(row['true_grade'])} → pred {int(row['pred'])}",
                    f"{GRADE_NAMES[int(row['pred'])]} · conf {row['conf']:.2f}",
                    colour)
        axes[r, 0].set_ylabel(label, fontsize=8.5, color=colour,
                              rotation=90, labelpad=10)

    fig.suptitle("Grad-CAM on the deployed checkpoint — successes, failures, "
                 "and the cases the application abstains on\n"
                 "Rows are selected by a fixed rule (sorted by confidence), "
                 "not curated.", fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = root / "figures" / "report_v2" / "figR_gradcam_panel.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nWrote {out}")
    print("This is report Figure 5.10 and deck slide 22.")


if __name__ == "__main__":
    main()
