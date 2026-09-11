"""
Phase 6b, step 2 -- dump per-image test predictions from the 8 converged
checkpoints so the Claim 2 (P1 vs P2) and Claim 2b (partner-eye ablation)
re-tests can be run with this project's actual standard: a PATIENT-LEVEL
bootstrap CI (resampling patients, not images), not a raw average of two
numbers. The aggregate JSONs finetune_converged.py already wrote
(test_qwk, confusion matrix, per-class recall) are NOT enough for that --
they don't carry per-image predictions or patient IDs. This script is pure
inference (loads best.pt, no training), so it's fast: a couple of minutes
per checkpoint even on CPU, well under a minute each on your GPU.

For each of the 8 run directories under checkpoints/, loads best.pt,
rebuilds the model from the checkpoint's own saved config (so this can't
silently mismatch architecture/image-size), re-runs it on that run's own
test set (same split it was trained under -- P1 checkpoints evaluate on
the P1 test set, P2 checkpoints on the P2 test set, exactly matching what
finetune_converged.py itself evaluated on), and writes one CSV per run:
    results/{run_name}_test_predictions.csv
Columns: image_id, patient_id, eye, true_grade, pred_grade,
         prob_0..prob_4, split, sampler, seed, run_name

Run once, no arguments, from the project root:
    python src\\train\\dump_test_predictions.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_converged import (  # noqa: E402  (same directory)
    FineTuneDataset, load_split_and_manifest, build_model,
)
from torch.utils.data import DataLoader

RUN_NAMES = [
    "finetune_converged_p1_class_balanced_seed42",
    "finetune_converged_p1_class_balanced_seed43",
    "finetune_converged_p2_class_balanced_seed42",
    "finetune_converged_p2_class_balanced_seed43",
    "finetune_converged_p1_none_seed42",
    "finetune_converged_p1_none_seed43",
    "finetune_converged_p2_none_seed42",
    "finetune_converged_p2_none_seed43",
]


def main():
    root = PROJECT_ROOT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    manifest_path = root / "data" / "manifests" / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    patient_of = dict(zip(manifest["image_id"], manifest["patient_id"]))
    eye_of = dict(zip(manifest["image_id"], manifest["eye"]))

    for run_name in RUN_NAMES:
        best_path = root / "checkpoints" / run_name / "best.pt"
        out_path = root / "results" / f"{run_name}_test_predictions.csv"
        if not best_path.exists():
            print(f"SKIP {run_name}: {best_path} not found")
            continue
        print(f"\n{run_name}")
        ckpt = torch.load(best_path, map_location=device)
        cfg = ckpt["config"]
        split = ckpt["split"]
        sampler_name = ckpt.get("sampler", "unknown")
        seed = ckpt["seed"]

        by_fold, grade_by_id = load_split_and_manifest(
            manifest_path, root / "data" / "splits" / f"{split}.json")
        img_dir = root / "data" / "processed" / str(cfg["image_size"])
        test_ids = [i for i in by_fold["test"] if (img_dir / f"{i}.jpg").exists()]
        test_grades = [grade_by_id[i] for i in test_ids]
        test_ds = FineTuneDataset(test_ids, test_grades, img_dir, augment=False)
        test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                                  num_workers=4, pin_memory=(device.type == "cuda"))

        model = build_model(cfg["model"], device, pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        all_ids, all_true, all_pred, all_probs = [], [], [], []
        autocast_enabled = device.type == "cuda"
        with torch.no_grad():
            for x, y, ids in test_loader:
                x = x.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
                    logits = model(x)
                probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
                all_ids.extend(list(ids))
                all_true.append(y.numpy())
                all_pred.append(probs.argmax(axis=1))
                all_probs.append(probs)

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        probs = np.concatenate(all_probs)
        from sklearn.metrics import cohen_kappa_score
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        print(f"  n_test={len(all_ids)}  qwk={qwk:.4f}  "
              f"(sanity check -- should match finetune_converged's own test_qwk for this run)")

        df = pd.DataFrame({
            "image_id": all_ids,
            "patient_id": [patient_of.get(i, "") for i in all_ids],
            "eye": [eye_of.get(i, "") for i in all_ids],
            "true_grade": y_true,
            "pred_grade": y_pred,
            "prob_0": probs[:, 0], "prob_1": probs[:, 1], "prob_2": probs[:, 2],
            "prob_3": probs[:, 3], "prob_4": probs[:, 4],
            "split": split, "sampler": sampler_name, "seed": seed, "run_name": run_name,
        })
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"  Wrote {out_path}")

    print("\nDone. Send the 8 *_test_predictions.csv files back for the patient-level "
          "Claim 2 / Claim 2b re-analysis.")


if __name__ == "__main__":
    main()
