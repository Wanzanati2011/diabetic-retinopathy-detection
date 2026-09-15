"""
Phase 7, step 0 -- dump per-image LOGITS and probabilities from the DEPLOYED
app checkpoint, for the folds calibration actually needs.

Why this script has to exist at all
-----------------------------------
The eight results/*_test_predictions.csv files that dump_test_predictions.py
already wrote are NOT usable for Phase 7, for two independent reasons:

  1. They are TEST folds only. Temperature scaling must be fitted on the
     VALIDATION set and never on test (MASTER_PLAN.md Part 11). No
     validation-fold predictions have ever been dumped by this project.
  2. They come from the 224px finetune_converged runs, not from the 384px
     checkpoint the application actually ships
     (checkpoints/finetune_app_converged_p2_class_balanced_seed42/best.pt,
     test QWK 0.7160, referable AUROC 0.9117).

This script is PURE INFERENCE. It loads best.pt and runs forward passes.
It does not train, fine-tune or modify any checkpoint. The deployed model
is final; this only records what it already predicts.

It also stores raw LOGITS, not just softmax probabilities. Temperature
scaling divides logits by T before the softmax, so logits are the natural
input. (Softmax is shift-invariant, so T could in principle be fitted from
stored probabilities via log(p) -- but there is no reason to lean on that
when the logits are right here.)

Numerical note: like evaluate() in finetune_converged.py, the forward pass
runs under fp16 autocast on CUDA and the logits are cast to float32 before
storage. That is deliberate -- it reproduces the exact pipeline that
produced this project's reported test_qwk, so the sanity check below can be
an equality check rather than an approximate one. Pass --fp32 to disable
autocast if you want pure fp32 logits instead (slower, and the QWK may
differ in the 4th decimal).

Folds dumped
------------
  p2 / val    <- temperature T and reject threshold tau are fitted here
  p2 / test   <- the primary, frozen endpoint; scored once
  p2 / train  <- optional (--with-train), only useful for diagnostics

Cross-dataset (P3) is NOT dumped by default, and that is a deliberate
methodological choice rather than an omission. MASTER_PLAN.md Part 11 asks
for calibration "separately for P2 test and each P3 test set", but that
instruction was written for models TRAINED under P3. This checkpoint was
trained under P2, whose training fold already contains APTOS images -- so
running it on the P3 APTOS test fold is not a clean out-of-domain check,
it is partly a training-set check. The honest substitute, which calibrate.py
computes automatically from the P2 test dump, is to break the P2 test set
down BY SOURCE DATASET (eyepacs vs aptos) using the manifest's own dataset
column. If you want the P3 folds anyway for reference, pass --with-p3 and
the contamination caveat will be written into the output JSON so it cannot
be quoted later without it.

Usage (from the project root, Windows):
    python src\\train\\dump_app_predictions.py

Runtime: roughly 20-30 minutes total on a laptop GPU for val + test at
384px (5,843 + 5,814 images), against the 59.3 minutes the retrain itself
took. Set --num-workers 0 (the default) -- this is the same machine and the
same DataLoader-worker pattern that produced the WinError 1455 paging-file
crash during the retrain's final test evaluation.
"""
import argparse
import hashlib
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
from torch.utils.data import DataLoader  # noqa: E402
from sklearn.metrics import cohen_kappa_score, roc_auc_score  # noqa: E402

DEFAULT_RUN = "finetune_app_converged_p2_class_balanced_seed42"

# The number finetune_converged.py itself recorded for this checkpoint, from
# results/finetune_app_converged_p2_class_balanced_seed42.json. The test-fold
# dump below must reproduce it. If it does not, something about the model
# rebuild, the split, or the processed-image directory has silently changed,
# and nothing downstream of this script should be trusted.
EXPECTED_TEST_QWK = 0.7159509675171527
EXPECTED_TEST_AUROC = 0.9117124008736891


def dump_fold(model, device, fold_ids, grade_by_id, img_dir, batch_size,
              num_workers, autocast_enabled, meta_by_id):
    """Run the model over one fold and return a tidy DataFrame."""
    ids = [i for i in fold_ids if (img_dir / f"{i}.jpg").exists()]
    n_missing = len(fold_ids) - len(ids)
    if n_missing:
        print(f"    NOTE: {n_missing} image(s) in this fold have no processed "
              f"file under {img_dir} and were skipped.")
    grades = [grade_by_id[i] for i in ids]
    ds = FineTuneDataset(ids, grades, img_dir, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        pin_memory=(device.type == "cuda"))

    all_ids, all_true, all_logits = [], [], []
    with torch.no_grad():
        for bi, (x, y, batch_ids) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=autocast_enabled):
                logits = model(x)
            all_logits.append(logits.float().cpu().numpy())
            all_true.append(y.numpy())
            all_ids.extend(list(batch_ids))
            if bi % 50 == 0:
                done = min((bi + 1) * batch_size, len(ids))
                print(f"      {done}/{len(ids)}", flush=True)

    logits = np.concatenate(all_logits)
    y_true = np.concatenate(all_true)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    y_pred = probs.argmax(axis=1)

    df = pd.DataFrame({
        "image_id": all_ids,
        "patient_id": [meta_by_id["patient_id"].get(i, "") for i in all_ids],
        "eye": [meta_by_id["eye"].get(i, "") for i in all_ids],
        "dataset": [meta_by_id["dataset"].get(i, "") for i in all_ids],
        "true_grade": y_true,
        "pred_grade": y_pred,
    })
    for k in range(5):
        df[f"logit_{k}"] = logits[:, k]
    for k in range(5):
        df[f"prob_{k}"] = probs[:, k]
    # Referable-DR score, the way clinical_operating_point.py defines it:
    # the whole tail, not the argmax.
    df["referable_score"] = probs[:, 2:].sum(axis=1)
    df["referable_true"] = (y_true >= 2).astype(int)
    return df


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default=DEFAULT_RUN,
                    help="checkpoint directory under checkpoints/")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="keep at 0 on this machine -- see the WinError 1455 note")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="default: the checkpoint's own training batch size")
    ap.add_argument("--fp32", action="store_true",
                    help="disable fp16 autocast (slower; QWK may differ slightly "
                         "from the reported value)")
    ap.add_argument("--with-train", action="store_true",
                    help="also dump the training fold (diagnostics only)")
    ap.add_argument("--with-p3", action="store_true",
                    help="also dump the two P3 test folds -- NOT a clean "
                         "out-of-domain check for this P2-trained checkpoint; "
                         "the caveat is recorded in the output JSON")
    args = ap.parse_args()

    root = PROJECT_ROOT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("  WARNING: no CUDA device found. This will work but will be slow.")

    best_path = root / "checkpoints" / args.run_name / "best.pt"
    if not best_path.exists():
        raise SystemExit(f"Checkpoint not found: {best_path}")

    print(f"Loading {best_path}")
    ckpt = torch.load(best_path, map_location=device)
    cfg = ckpt["config"]
    split = ckpt["split"]
    sampler_name = ckpt.get("sampler", "unknown")
    seed = ckpt["seed"]
    batch_size = args.batch_size or cfg["batch_size"]
    autocast_enabled = (device.type == "cuda") and not args.fp32

    print(f"  model={cfg['model']}  image_size={cfg['image_size']}  "
          f"split={split}  sampler={sampler_name}  seed={seed}")
    print(f"  checkpoint epoch={ckpt.get('epoch')}  "
          f"val_qwk={ckpt.get('val_qwk')}")
    print("  NOTE: best.pt stores the EMA weights, which is what the "
          "deployed app loads.")

    manifest_path = root / "data" / "manifests" / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    meta_by_id = {
        "patient_id": dict(zip(manifest["image_id"], manifest["patient_id"])),
        "eye": dict(zip(manifest["image_id"], manifest["eye"])),
        "dataset": dict(zip(manifest["image_id"], manifest["dataset"])),
    }

    model = build_model(cfg["model"], device, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    img_dir = root / "data" / "processed" / str(cfg["image_size"])
    if not img_dir.exists():
        raise SystemExit(f"Processed image directory not found: {img_dir}")

    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    summary = {
        "run_name": args.run_name,
        "checkpoint": str(best_path.relative_to(root)),
        "model": cfg["model"],
        "image_size": cfg["image_size"],
        "split": split,
        "sampler": sampler_name,
        "seed": seed,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_qwk": ckpt.get("val_qwk"),
        "autocast_fp16": autocast_enabled,
        "weights": "EMA (best.pt stores ema.state_dict())",
        "inference_only": True,
        "folds": {},
    }

    # ---- P2 folds -------------------------------------------------------
    by_fold, grade_by_id = load_split_and_manifest(
        manifest_path, root / "data" / "splits" / f"{split}.json")

    wanted = ["val", "test"] + (["train"] if args.with_train else [])
    for fold in wanted:
        print(f"\n  Fold: {split}/{fold}  ({len(by_fold[fold])} images)")
        df = dump_fold(model, device, by_fold[fold], grade_by_id, img_dir,
                       batch_size, args.num_workers, autocast_enabled,
                       meta_by_id)
        qwk = float(cohen_kappa_score(df["true_grade"], df["pred_grade"],
                                      weights="quadratic"))
        try:
            auroc = float(roc_auc_score(df["referable_true"],
                                        df["referable_score"]))
        except ValueError:
            auroc = float("nan")
        out_path = out_dir / f"{args.run_name}_{fold}_predictions.csv"
        df.to_csv(out_path, index=False)
        written[fold] = out_path

        id_hash = hashlib.sha256(
            "\n".join(sorted(df["image_id"])).encode()).hexdigest()[:16]
        summary["folds"][fold] = {
            "n_images": int(len(df)),
            "n_patients": int(df["patient_id"].nunique()),
            "qwk": qwk,
            "referable_auroc": auroc,
            "image_id_sha256_16": id_hash,
            "file": out_path.name,
        }
        print(f"    n={len(df)}  patients={df['patient_id'].nunique()}  "
              f"qwk={qwk:.4f}  referable_auroc={auroc:.4f}")
        print(f"    fold id hash: {id_hash}")
        print(f"    Wrote {out_path}")

        if fold == "test":
            dq = abs(qwk - EXPECTED_TEST_QWK)
            da = abs(auroc - EXPECTED_TEST_AUROC)
            tol = 5e-4 if autocast_enabled else 5e-3
            status = "MATCH" if (dq < tol and da < tol) else "MISMATCH"
            print(f"\n    SANITY CHECK vs results/{args.run_name}.json: {status}")
            print(f"      qwk   {qwk:.6f} vs expected {EXPECTED_TEST_QWK:.6f} "
                  f"(delta {dq:.2e})")
            print(f"      auroc {auroc:.6f} vs expected {EXPECTED_TEST_AUROC:.6f} "
                  f"(delta {da:.2e})")
            summary["sanity_check"] = {
                "expected_test_qwk": EXPECTED_TEST_QWK,
                "observed_test_qwk": qwk,
                "expected_test_auroc": EXPECTED_TEST_AUROC,
                "observed_test_auroc": auroc,
                "tolerance": tol,
                "status": status,
            }
            if status == "MISMATCH":
                print("      >>> STOP. Do not run calibrate.py on this dump.")
                print("      >>> Something in the model rebuild, the split file, or")
                print("      >>> data/processed/ has changed since the checkpoint was")
                print("      >>> trained. Investigate before going further.")

    # ---- optional P3 folds ---------------------------------------------
    if args.with_p3:
        caveat = ("This checkpoint was trained under P2, whose training fold "
                  "already contains APTOS images. Running it on a P3 test fold "
                  "is therefore NOT a clean out-of-domain measurement -- part of "
                  "that fold was seen in training. Use the by-dataset breakdown "
                  "of the P2 test set (which calibrate.py computes) as the "
                  "domain-shift proxy, and quote these P3 numbers only with this "
                  "caveat attached.")
        summary["p3_caveat"] = caveat
        print("\n  P3 folds requested. CAVEAT RECORDED:")
        print(f"    {caveat}")
        for p3 in ["p3_eyepacs_to_aptos", "p3_aptos_to_eyepacs"]:
            p3_path = root / "data" / "splits" / f"{p3}.json"
            if not p3_path.exists():
                print(f"    SKIP {p3}: {p3_path} not found")
                continue
            p3_by_fold, p3_grades = load_split_and_manifest(manifest_path, p3_path)
            print(f"\n  Fold: {p3}/test  ({len(p3_by_fold['test'])} images)")
            df = dump_fold(model, device, p3_by_fold["test"], p3_grades, img_dir,
                           batch_size, args.num_workers, autocast_enabled,
                           meta_by_id)
            qwk = float(cohen_kappa_score(df["true_grade"], df["pred_grade"],
                                          weights="quadratic"))
            out_path = out_dir / f"{args.run_name}_{p3}_test_predictions.csv"
            df.to_csv(out_path, index=False)
            summary["folds"][f"{p3}/test"] = {
                "n_images": int(len(df)),
                "n_patients": int(df["patient_id"].nunique()),
                "qwk": qwk,
                "file": out_path.name,
                "caveat": "contaminated -- see p3_caveat",
            }
            print(f"    n={len(df)}  qwk={qwk:.4f}")
            print(f"    Wrote {out_path}")

    summary_path = out_dir / f"{args.run_name}_prediction_dump.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")

    print("\nDone. Nothing was trained; no checkpoint was modified.")
    print("Next (CPU only, no GPU needed):")
    print("    python -m src.experiments.calibrate")
    print("    python -m src.experiments.grade1_diagnosis")


if __name__ == "__main__":
    main()
