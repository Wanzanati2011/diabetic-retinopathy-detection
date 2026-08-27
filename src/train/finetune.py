"""
Phase 6 -- real fine-tuning (MASTER_PLAN.md Part 10).

Frozen features (Phases 3-5) answer the paper's core claims cheaply; this
script does the two REAL fine-tunes that give the headline numbers
reviewers expect:
  Run 1 (headline): effnetb0@224, fast config, under BOTH P1 and P2,
                     2 seeds each = 4 runs total (~40 min each on your GPU).
  Run 2 (app model): effnetb0@384, P2 split only, best config, 1 run
                     (~2 hours).

Needs your CUDA GPU -- do NOT run this in the Cowork sandbox (no GPU there,
and torch/timm aren't even installed there). Wrapped in
if __name__ == "__main__": for Windows spawn-based multiprocessing (same
reason as src/features/extract.py).

IMPORTANT: this script could not be executed where it was written (no GPU,
no torch/timm installed in that sandbox) -- only syntax-checked and
carefully reviewed by hand. Run the cheap debug pass below BEFORE trusting
a real multi-hour run:
    python src\\train\\finetune.py --config configs\\finetune_headline.yaml --split p2 --seed 42 --epochs 1 --limit-train 200 --limit-val 64 --limit-test 64
That should finish in well under a minute and confirms the whole pipeline
(data loading, model, AMP, EMA, checkpointing, evaluation, JSON output)
actually runs before you commit real time to it.

Resumable: checkpoints every epoch to checkpoints/{run_name}/checkpoint.pt
(atomic write -- write to .tmp then rename). Re-running the exact same
command picks up where it left off. Separately saves
checkpoints/{run_name}/best.pt (the EMA weights from the best val_qwk
epoch, NOT the final epoch) -- later phases (Claim 2b rerun on the
fine-tuned model, calibration, the app) should load best.pt.

Config keys `optimizer: adamw` / `scheduler: cosine` in the YAML are
documentation of a fixed choice, not a dispatch table -- MASTER_PLAN.md
Part 4.3 specifies exactly one optimizer/scheduler, so there's no branching
logic for other values. If you want to try something else, edit this file.

Usage (from the project root -- see run.ps1 for the exact 5 commands):
    python src\\train\\finetune.py --config configs\\finetune_headline.yaml --split p1 --seed 42
    python src\\train\\finetune.py --config configs\\finetune_headline.yaml --split p1 --seed 43
    python src\\train\\finetune.py --config configs\\finetune_headline.yaml --split p2 --seed 42
    python src\\train\\finetune.py --config configs\\finetune_headline.yaml --split p2 --seed 43
    python src\\train\\finetune.py --config configs\\finetune_app.yaml      --split p2 --seed 42
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FineTuneDataset(Dataset):
    """Loads an already-preprocessed {size}px cached JPEG (same cache,
    same normalization as src/features/extract.py's FeatureDataset).
    Defined at module level -- Windows' spawn-based multiprocessing needs
    to pickle this class by reference; a locally-scoped class can't be
    pickled (this exact bug was hit and fixed in extract.py earlier).

    train=True applies light augmentation (random horizontal flip + small
    rotation + mild brightness jitter). This is NOT specified by
    MASTER_PLAN.md's fine-tune config -- added as standard practice for a
    15-epoch unfrozen-backbone fine-tune on ~27k images, to reduce
    overfitting. Set augment=False in the config to disable (e.g. for an
    ablation). Eval (augment=False) is always fully deterministic --
    MASTER_PLAN.md Part 14 lists "randomness in validation/test
    transforms" as leakage-symptom #3, so val/test datasets are always
    constructed with augment=False, never toggled at call time."""

    def __init__(self, ids, grades, img_dir, augment):
        self.ids = ids
        self.grades = grades
        self.img_dir = img_dir
        self.augment = augment
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        grade = self.grades[idx]
        with Image.open(self.img_dir / f"{image_id}.jpg") as im:
            im = im.convert("RGB")
            if self.augment:
                if np.random.rand() < 0.5:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                angle = float(np.random.uniform(-15, 15))
                im = im.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
            arr = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
        if self.augment:
            arr = (arr * float(np.random.uniform(0.9, 1.1))).clamp(0.0, 1.0)
        arr = (arr - self.mean) / self.std
        return arr, grade, image_id


def _worker_init_fn(worker_id):
    """Without this, DataLoader workers inherit numpy's RNG state from the
    parent process and produce CORRELATED augmentation across workers
    (a well-known PyTorch gotcha) -- reseed each worker independently."""
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)


def load_split_and_manifest(manifest_path, split_path):
    manifest = pd.read_csv(manifest_path)
    if "excluded" in manifest.columns:
        manifest = manifest[~manifest["excluded"].fillna(False).astype(bool)].reset_index(drop=True)
    grade_by_id = dict(zip(manifest["image_id"], manifest["grade"]))
    split_map = json.loads(split_path.read_text())
    by_fold = {"train": [], "val": [], "test": []}
    for image_id, entry in split_map.items():
        if image_id in grade_by_id:
            by_fold[entry["fold"]].append(image_id)
    for fold in by_fold:
        by_fold[fold].sort()  # deterministic order regardless of dict iteration order
    return by_fold, grade_by_id


def build_model(backbone_name, device, grad_checkpointing=False, pretrained=True):
    import timm
    model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=5)
    if grad_checkpointing and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    model.to(device)
    return model


def split_param_groups(model, lr_backbone, lr_head):
    """Two LR groups per MASTER_PLAN.md Part 4.3 (lr_head 3e-4, lr_backbone
    3e-5). Uses timm's standard get_classifier() API to identify head
    params generically across architectures (effnet's `classifier`,
    resnet's `fc`, etc.) rather than hardcoding attribute names."""
    head_params = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return [
        {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
        {"params": head_params, "lr": lr_head, "name": "head"},
    ]


def set_backbone_trainable(model, trainable):
    """freeze_backbone_epochs (MASTER_PLAN.md Part 4.3): freeze everything
    except the classifier head for the first N epochs, then unfreeze."""
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad = trainable


class ModelEMA:
    """Exponential moving average of ALL model state (params + buffers,
    e.g. BatchNorm running stats) -- MASTER_PLAN.md Part 4.3: ema_decay
    0.999. Standard practice: evaluate/checkpoint off the EMA weights,
    which are smoother epoch-to-epoch than the raw training weights, but
    keep training the raw weights every step -- this class never mutates
    the live model, callers swap state dicts in/out explicitly (see
    finetune()'s per-epoch eval block)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {
            k: (v.detach().clone().float() if v.dtype.is_floating_point else v.detach().clone())
            for k, v in model.state_dict().items()
        }

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
                else:
                    self.shadow[k] = v.detach().clone()

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}


def lr_lambda_factory(total_steps, warmup_steps):
    """Linear warmup for warmup_pct of total steps, then cosine decay to 0
    (MASTER_PLAN.md Part 4.3: scheduler: cosine, warmup_pct: 0.05). Applied
    identically to both param groups -- LambdaLR multiplies each group's
    OWN base lr by this factor, so lr_head and lr_backbone keep their
    3e-4/3e-5 ratio throughout, only the shared shape changes."""
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + float(np.cos(np.pi * progress)))
    return fn


@torch.no_grad()
def evaluate(model, loader, device):
    was_training = model.training
    model.eval()
    all_pred, all_true, all_probs = [], [], []
    autocast_enabled = device.type == "cuda"
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            logits = model(x)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        all_pred.append(probs.argmax(axis=1))
        all_true.append(y.numpy())
        all_probs.append(probs)
    if was_training:
        model.train()
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    probs = np.concatenate(all_probs)
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    return qwk, y_true, y_pred, probs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True, choices=["p1", "p2"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest", default="data/manifests/manifest.csv")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--checkpoints-dir", default="checkpoints")
    ap.add_argument("--out", default=None, help="default: results/{config-stem}_{split}_seed{seed}.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-train", type=int, default=0, help="debug: cap train set size")
    ap.add_argument("--limit-val", type=int, default=0, help="debug: cap val set size")
    ap.add_argument("--limit-test", type=int, default=0, help="debug: cap test set size")
    ap.add_argument("--epochs", type=int, default=0, help="debug: override config epochs")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    with open(root / args.config) as f:
        cfg = yaml.safe_load(f)
    if args.epochs:
        cfg["epochs"] = args.epochs

    run_name = f"{Path(args.config).stem}_{args.split}_seed{args.seed}"
    print(f"Run: {run_name}")
    print(f"Config: {cfg}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but not available -- falling back to CPU. "
              "Only sane for a --limit-* debug smoke test, never for a real run.", file=sys.stderr)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    by_fold, grade_by_id = load_split_and_manifest(
        root / args.manifest, root / args.splits_dir / f"{args.split}.json")
    img_dir = root / args.processed_dir / str(cfg["image_size"])
    if not img_dir.exists():
        print(f"ERROR: {img_dir} not found. Run src/data/build_cache.py for size={cfg['image_size']} first.",
              file=sys.stderr)
        sys.exit(1)

    def filter_present(ids):
        present = [i for i in ids if (img_dir / f"{i}.jpg").exists()]
        return present, len(ids) - len(present)

    train_ids, n_miss_train = filter_present(by_fold["train"])
    val_ids, n_miss_val = filter_present(by_fold["val"])
    test_ids, n_miss_test = filter_present(by_fold["test"])
    if n_miss_train or n_miss_val or n_miss_test:
        print(f"WARNING: missing cached images under {img_dir} -- "
              f"train:{n_miss_train} val:{n_miss_val} test:{n_miss_test}", file=sys.stderr)

    if args.limit_train:
        train_ids = train_ids[:args.limit_train]
    if args.limit_val:
        val_ids = val_ids[:args.limit_val]
    if args.limit_test:
        test_ids = test_ids[:args.limit_test]

    print(f"train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    train_grades = [grade_by_id[i] for i in train_ids]
    val_grades = [grade_by_id[i] for i in val_ids]
    test_grades = [grade_by_id[i] for i in test_ids]

    train_ds = FineTuneDataset(train_ids, train_grades, img_dir, augment=cfg.get("augment", True))
    val_ds = FineTuneDataset(val_ids, val_grades, img_dir, augment=False)
    test_ds = FineTuneDataset(test_ids, test_grades, img_dir, augment=False)

    # Class-balanced sampler (MASTER_PLAN.md Part 4.3: sampler: class_balanced) -- TRAIN ONLY.
    # val/test are never resampled -- that would be exactly the kind of eval-time
    # randomness MASTER_PLAN.md Part 14 flags as a leakage symptom.
    class_counts = np.bincount(train_grades, minlength=5).astype(np.float64)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[g] for g in train_grades]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ids), replacement=True)

    n_workers = cfg.get("num_workers", 6)
    persistent = cfg.get("persistent_workers", True) and n_workers > 0
    pin = cfg.get("pin_memory", True)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler,
                               num_workers=n_workers, pin_memory=pin, persistent_workers=persistent,
                               worker_init_fn=_worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=n_workers, pin_memory=pin, persistent_workers=persistent)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=n_workers, pin_memory=pin, persistent_workers=persistent)

    model = build_model(cfg["model"], device, grad_checkpointing=cfg.get("grad_checkpointing", False))
    param_groups = split_param_groups(model, cfg["lr_backbone"], cfg["lr_head"])
    optimizer = torch.optim.AdamW(param_groups)
    ema = ModelEMA(model, cfg.get("ema_decay", 0.999))

    grad_accum = cfg.get("grad_accum_steps", 1)
    steps_per_epoch = int(np.ceil(len(train_ids) / cfg["batch_size"])) if train_ids else 0
    optimizer_steps_per_epoch = max(1, int(np.ceil(steps_per_epoch / grad_accum)))
    total_optimizer_steps = optimizer_steps_per_epoch * cfg["epochs"]
    warmup_steps = max(1, int(total_optimizer_steps * cfg.get("warmup_pct", 0.05)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_factory(total_optimizer_steps, warmup_steps))
    amp_enabled = (cfg.get("amp", "fp16") == "fp16" and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)  # torch>=2.3 non-deprecated API
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)     # older torch fallback
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = root / args.checkpoints_dir / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "checkpoint.pt"
    best_path = ckpt_dir / "best.pt"

    start_epoch = 0
    best_val_qwk = -1.0
    best_epoch = -1
    patience_counter = 0
    freeze_epochs = cfg.get("freeze_backbone_epochs", 0)

    if ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        if ckpt.get("config", {}).get("model") != cfg["model"] or ckpt.get("config", {}).get("image_size") != cfg["image_size"]:
            print("ERROR: checkpoint's model/image_size doesn't match the current config -- "
                  "refusing to resume into a mismatched run. Delete the checkpoint dir if this "
                  "is intentional.", file=sys.stderr)
            sys.exit(1)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        ema.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_qwk = ckpt["best_val_qwk"]
        best_epoch = ckpt["best_epoch"]
        patience_counter = ckpt["patience_counter"]
        print(f"  Resuming at epoch {start_epoch}, best_val_qwk so far = {best_val_qwk:.4f}")

    set_backbone_trainable(model, trainable=(start_epoch >= freeze_epochs))
    if start_epoch < freeze_epochs:
        print(f"Backbone FROZEN for epochs 0-{freeze_epochs - 1} (currently at epoch {start_epoch})")

    def save_checkpoint(epoch_num):
        tmp_path = ckpt_path.with_suffix(".tmp.pt")
        torch.save({
            "epoch": epoch_num,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "best_val_qwk": best_val_qwk, "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "config": cfg, "split": args.split, "seed": args.seed,
        }, tmp_path)
        tmp_path.replace(ckpt_path)

    epoch = start_epoch - 1  # defined even if the loop body never executes (already-finished resume)
    t_start = time.time()
    for epoch in range(start_epoch, cfg["epochs"]):
        if epoch == freeze_epochs:
            print(f"Unfreezing backbone at epoch {epoch}")
            set_backbone_trainable(model, trainable=True)

        model.train()
        optimizer.zero_grad()
        running_loss, n_batches = 0.0, 0
        t_epoch = time.time()
        n_train_batches = len(train_loader)
        for i, (x, y, _) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y) / grad_accum
            scaler.scale(loss).backward()
            running_loss += loss.item() * grad_accum
            n_batches += 1
            if (i + 1) % grad_accum == 0 or (i + 1) == n_train_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                ema.update(model)

        # Evaluate on val using EMA weights: swap EMA state into the live model,
        # evaluate, then restore the raw training weights -- training must
        # continue from the RAW weights next epoch, not the smoothed EMA copy.
        raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict())
        val_qwk, _, _, _ = evaluate(model, val_loader, device)
        model.load_state_dict(raw_state)
        model.train()

        epoch_time = time.time() - t_epoch
        print(f"Epoch {epoch}: train_loss={running_loss / max(1, n_batches):.4f}  "
              f"val_qwk={val_qwk:.4f}  ({epoch_time:.0f}s)")

        if val_qwk > best_val_qwk:
            best_val_qwk, best_epoch, patience_counter = val_qwk, epoch, 0
            tmp_best = best_path.with_suffix(".tmp.pt")
            torch.save({"model_state_dict": ema.state_dict(), "epoch": epoch, "val_qwk": val_qwk,
                        "config": cfg, "split": args.split, "seed": args.seed}, tmp_best)
            tmp_best.replace(best_path)
            print(f"  New best (val_qwk={val_qwk:.4f}) -> {best_path}")
        else:
            patience_counter += 1

        save_checkpoint(epoch)

        if patience_counter >= cfg.get("early_stop_patience", 4):
            print(f"Early stopping at epoch {epoch} (no val_qwk improvement for {patience_counter} epochs)")
            break

    total_minutes = (time.time() - t_start) / 60
    print(f"\nTraining done in {total_minutes:.1f} min. Best val_qwk={best_val_qwk:.4f} at epoch {best_epoch}.")

    # ---- Final test-set evaluation, using the BEST checkpoint (never the final epoch) ----
    if not best_path.exists():
        print(f"ERROR: {best_path} was never written (0 epochs ran?) -- cannot evaluate on test.", file=sys.stderr)
        sys.exit(1)
    print(f"Loading best checkpoint from {best_path} for test evaluation...")
    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_qwk, test_true, test_pred, test_probs = evaluate(model, test_loader, device)

    cm = confusion_matrix(test_true, test_pred, labels=[0, 1, 2, 3, 4])
    per_class_recall = {}
    for g in range(5):
        mask = test_true == g
        per_class_recall[str(g)] = {
            "n": int(mask.sum()),
            "recall": float((test_pred[mask] == g).mean()) if mask.sum() else None,
        }

    y_bin = (test_true >= 2).astype(int)
    rdr_auroc = None
    if 0 < y_bin.sum() < len(y_bin):
        rdr_score = test_probs[:, 2:].sum(axis=1)  # P(grade >= 2)
        rdr_auroc = float(roc_auc_score(y_bin, rdr_score))

    if test_qwk < 0.40:
        verdict = "BROKEN -- check label alignment, LR, normalization"
    elif test_qwk < 0.70:
        verdict = "undertrained or preprocessing bug"
    elif test_qwk <= 0.85:
        verdict = "correct -- proceed"
    elif test_qwk <= 0.95:
        verdict = "SUSPICIOUS -- run MASTER_PLAN.md Part 14 leakage checklist before trusting this"
    else:
        verdict = "DEFINITELY A LEAK -- stop"

    g1 = per_class_recall["1"]["recall"]
    grade1_note = (
        "expected: grade-1 recall 0.20-0.45 is normal (microaneurysms destroyed by "
        "downsampling) -- do not chase this, per MASTER_PLAN.md Part 10"
        if g1 is not None and 0.20 <= g1 <= 0.45 else
        "grade-1 recall outside the MASTER_PLAN.md 0.20-0.45 expected band -- note but this "
        "band is a guideline, not a hard pass/fail test"
    )

    out = {
        "run_name": run_name, "config_file": args.config, "config": cfg,
        "split": args.split, "seed": args.seed,
        "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids),
        "best_epoch": best_epoch, "best_val_qwk": float(best_val_qwk),
        "total_epochs_run": epoch + 1, "training_minutes": total_minutes,
        "test_qwk": test_qwk,
        "test_confusion_matrix": cm.tolist(),
        "per_class_recall": per_class_recall,
        "grade1_recall_note": grade1_note,
        "referable_dr_auroc": rdr_auroc,
        "acceptance_test_10_1_verdict": verdict,
    }
    is_debug_run = bool(args.limit_train or args.limit_val or args.limit_test or args.epochs)
    out["is_debug_run"] = is_debug_run
    out_name = args.out or f"results/{run_name}.json"
    out_path = root / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    if is_debug_run:
        print(f"DEBUG RUN (--limit-*/--epochs override was used) -- test_qwk={test_qwk:.4f} is NOT "
              f"meaningful and the Acceptance Test 10.1 verdict table does not apply here. This run "
              f"only checks that the pipeline executes end-to-end without crashing (data loading, "
              f"model, AMP, EMA, checkpointing, JSON output) -- it does not check quality.")
    else:
        print(f"ACCEPTANCE TEST 10.1 VERDICT: test_qwk={test_qwk:.4f} -> {verdict}")
    if out["split"] == "p1" and test_qwk > 0.95:
        print("Reminder (MASTER_PLAN.md Part 10): P1 is SUPPOSED to look inflated relative to P2 "
              "-- that's expected. Never 'improve' P1's split to bring it down.")


if __name__ == "__main__":
    main()
