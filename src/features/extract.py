"""
Phase 3 -- frozen feature extraction (MASTER_PLAN.md Part 7).

Runs a frozen, pretrained timm backbone (num_classes=0, i.e. global-pooled
features with no classifier head) over every cached image in
data/processed/{size}/ and caches the result to features/{backbone}_{size}.npz.

This is the "freeze the backbone once, then everything downstream is
CPU-only logistic regression" step -- Claims 1b/2/2b/3 (Phase 4/5) all read
from these .npz files instead of re-running any CNN forward pass.

Meant to run on your local machine (Windows, CUDA GPU) -- NOT in the Cowork
sandbox, which has no GPU. See run.ps1 for the exact commands.

Resumable: checkpoints partial progress to
features/_checkpoint_{backbone}_{size}.npz every --checkpoint-every images.
Safe to Ctrl+C and restart -- already-embedded image_ids are skipped and the
checkpoint is merged into the final output. The checkpoint file is deleted
once the run completes and the final .npz is written.

Skips excluded rows (manifest.csv `excluded` column -- see
apply_exclusions.py) and any image_id missing from data/processed/{size}/
(reported as a warning; run build_cache.py first if this number is large).

Output .npz contains BOTH the feature matrix and the aligned image_id array
(sorted image_id order), so downstream code never has to worry about row
order drifting between configs or reruns:
    data = np.load("features/tf_efficientnet_b0_224.npz")
    feats = data["features"]      # (N, num_features) float32
    ids   = data["image_id"]      # (N,) <U40, feats[i] is embedding of ids[i]

Usage (from the project root):
    python src\\features\\extract.py --backbone tf_efficientnet_b0 --size 224
    python src\\features\\extract.py --backbone tf_efficientnet_b0 --size 384
    python src\\features\\extract.py --backbone resnet50 --size 224
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Default batch sizes tuned for a 6GB VRAM card (MASTER_PLAN.md Part 7).
DEFAULT_BATCH_SIZE = {224: 64, 384: 32}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FeatureDataset(Dataset):
    """Loads an already-preprocessed {size}px cached JPEG and normalizes it
    with standard ImageNet stats. No resize/crop here on purpose --
    build_cache.py already produced a square crop at exactly `size`, and for
    effnetb0@384 we deliberately feed a larger-than-pretrained input into a
    fully-convolutional/global-pooled backbone rather than resizing back
    down to 224.

    Defined at module level (not nested in a function) because Windows'
    spawn-based multiprocessing needs to pickle this class by reference to
    hand it to DataLoader worker processes -- a locally-scoped class can't
    be pickled and raises AttributeError: Can't pickle local object."""

    def __init__(self, ids, img_dir):
        self.ids = ids
        self.img_dir = img_dir
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        with Image.open(self.img_dir / f"{image_id}.jpg") as im:
            im = im.convert("RGB")
            arr = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
        arr = (arr - self.mean) / self.std
        return arr, image_id


def build_dataset(image_ids, img_dir):
    return FeatureDataset(image_ids, img_dir)


def load_checkpoint(ckpt_path):
    if not ckpt_path.exists():
        return [], np.empty((0,), dtype="<U40")
    data = np.load(ckpt_path, allow_pickle=False)
    feats = data["features"]
    ids = data["image_id"]
    print(f"  Resuming from checkpoint: {len(ids)} images already embedded ({ckpt_path.name})")
    return [feats], ids


def save_checkpoint(ckpt_path, feats_list, ids_list):
    """Write-to-temp-then-replace so a Ctrl+C mid-write never corrupts the checkpoint."""
    feats = np.concatenate(feats_list, axis=0)
    ids = np.array(ids_list, dtype="<U40")
    tmp_path = ckpt_path.with_suffix(".tmp.npz")
    np.savez(tmp_path, features=feats, image_id=ids)
    tmp_path.replace(ckpt_path)
    return feats, ids


def _finalize(feats_list, ids_list, expected_ids, final_path, ckpt_path):
    feats = np.concatenate(feats_list, axis=0) if feats_list else np.empty((0, 0), dtype=np.float32)
    ids = list(ids_list)

    # Align to the canonical sorted order of expected_ids -- this is what
    # guarantees the saved image_id array can never drift out of sync with
    # the feature rows, no matter what order batches finished in or how
    # many resumes it took.
    id_to_row = {i: r for i, r in zip(ids, feats)}
    missing = [i for i in expected_ids if i not in id_to_row]
    if missing:
        raise RuntimeError(
            f"{len(missing)} expected image_ids have no embedded feature row after extraction "
            f"(first 5: {missing[:5]}) -- this should not happen; re-run to retry."
        )
    ordered_feats = np.stack([id_to_row[i] for i in expected_ids], axis=0).astype(np.float32)
    ordered_ids = np.array(expected_ids, dtype="<U40")

    n_nan = int(np.isnan(ordered_feats).sum())
    if n_nan:
        print(f"WARNING: {n_nan} NaN values in the final feature matrix.", file=sys.stderr)

    final_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(final_path, features=ordered_feats, image_id=ordered_ids)
    print(f"Wrote {final_path}  (features shape: {ordered_feats.shape}, dtype: {ordered_feats.dtype})")

    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"Removed checkpoint {ckpt_path.name}")


def _report_existing(final_path, expected_ids):
    data = np.load(final_path)
    feats, ids = data["features"], data["image_id"]
    print(f"Existing file: {final_path}  shape={feats.shape}  n_ids={len(ids)}  "
          f"matches expected active-image count: {len(ids) == len(expected_ids)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--manifest", default="data/manifests/manifest.csv")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--out-dir", default="features")
    ap.add_argument("--backbone", required=True, help="timm model name, e.g. tf_efficientnet_b0, resnet50")
    ap.add_argument("--size", type=int, required=True, choices=[224, 384])
    ap.add_argument("--batch-size", type=int, default=0, help="0 = size-based default (64@224, 32@384)")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint-every", type=int, default=2000, help="save a resume checkpoint every N images")
    ap.add_argument("--limit", type=int, default=0, help="debug: only embed the first N active images")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    batch_size = args.batch_size or DEFAULT_BATCH_SIZE.get(args.size, 64)

    print(f"Project root: {root}")
    print(f"Backbone: {args.backbone}   size: {args.size}   batch_size: {batch_size}   "
          f"num_workers: {args.num_workers}   device: {args.device}")

    import timm
    from torch.utils.data import DataLoader

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but torch.cuda.is_available() is False. "
              "Falling back to CPU -- this will be very slow and is only meant for a tiny "
              "--limit smoke test, not a real extraction run.", file=sys.stderr)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    manifest = pd.read_csv(root / args.manifest)
    if "excluded" in manifest.columns:
        n_excluded = int(manifest["excluded"].fillna(False).astype(bool).sum())
        manifest = manifest[~manifest["excluded"].fillna(False).astype(bool)].reset_index(drop=True)
        print(f"Manifest: {len(manifest) + n_excluded} rows, {n_excluded} excluded "
              f"(apply_exclusions.py) -> {len(manifest)} active rows")
    else:
        print(f"Manifest: {len(manifest)} rows (no 'excluded' column found)")

    img_dir = root / args.processed_dir / str(args.size)
    all_ids = sorted(manifest["image_id"].tolist())  # sorted -> deterministic, resumable order

    present_ids = [i for i in all_ids if (img_dir / f"{i}.jpg").exists()]
    n_missing = len(all_ids) - len(present_ids)
    if n_missing:
        print(f"WARNING: {n_missing} active manifest rows have no cached {args.size}px file under "
              f"{img_dir} -- skipping them. Run build_cache.py first if this number looks large.",
              file=sys.stderr)

    if args.limit:
        present_ids = present_ids[: args.limit]

    print(f"Images to embed (this size, active, on disk): {len(present_ids)}")

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{args.backbone}_{args.size}.npz"
    ckpt_path = out_dir / f"_checkpoint_{args.backbone}_{args.size}.npz"

    if final_path.exists() and not ckpt_path.exists():
        print(f"{final_path} already exists and there's no in-progress checkpoint -- "
              f"assuming this config is already complete. Delete the .npz to force a re-run.")
        _report_existing(final_path, present_ids)
        return

    ckpt_feats_list, ckpt_ids_arr = load_checkpoint(ckpt_path)
    ckpt_ids_list = ckpt_ids_arr.tolist()
    done_ids = set(ckpt_ids_list)

    remaining_ids = [i for i in present_ids if i not in done_ids]
    print(f"Already embedded (checkpoint): {len(done_ids)}   Remaining: {len(remaining_ids)}")

    if not remaining_ids:
        print("Nothing left to embed -- finalizing from checkpoint.")
        _finalize(ckpt_feats_list, ckpt_ids_list, present_ids, final_path, ckpt_path)
        return

    print(f"Loading backbone '{args.backbone}' (pretrained, num_classes=0)...")
    model = timm.create_model(args.backbone, pretrained=True, num_classes=0)
    model.eval()
    model.to(device)
    print(f"Backbone output feature dim: {model.num_features}")

    dataset = build_dataset(remaining_ids, img_dir)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    use_amp = device.type == "cuda"
    new_feats_list = []
    new_ids_list = []
    since_ckpt = 0
    n_done = 0
    n_batches = 0
    t_start = time.time()

    with torch.no_grad():
        for batch_imgs, batch_ids in loader:
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(batch_imgs)
            else:
                out = model(batch_imgs)
            feats = out.float().cpu().numpy()

            if not np.isfinite(feats).all():
                bad = [bid for bid, row in zip(batch_ids, feats) if not np.isfinite(row).all()]
                print(f"WARNING: non-finite features for {len(bad)} images in this batch: {bad[:5]}",
                      file=sys.stderr)

            new_feats_list.append(feats)
            new_ids_list.extend(batch_ids)
            n_done += len(batch_ids)
            since_ckpt += len(batch_ids)
            n_batches += 1

            if n_batches % 10 == 0 or n_done == len(remaining_ids):
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(remaining_ids) - n_done) / rate if rate > 0 else float("inf")
                print(f"  {n_done}/{len(remaining_ids)}  ({rate:.1f} img/s, {elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

            if since_ckpt >= args.checkpoint_every:
                merged_feats = ckpt_feats_list + new_feats_list
                merged_ids = ckpt_ids_list + new_ids_list
                feats_arr, ids_arr = save_checkpoint(ckpt_path, merged_feats, merged_ids)
                ckpt_feats_list = [feats_arr]
                ckpt_ids_list = ids_arr.tolist()
                new_feats_list = []
                new_ids_list = []
                since_ckpt = 0
                print(f"  [checkpoint saved: {len(ckpt_ids_list)} images]")

    total_elapsed = time.time() - t_start
    rate = n_done / total_elapsed if total_elapsed > 0 else 0
    print(f"\nExtraction pass complete: {n_done} images in {total_elapsed:.0f}s ({rate:.1f} img/s)")

    all_feats_list = ckpt_feats_list + new_feats_list
    all_ids_list = ckpt_ids_list + new_ids_list
    _finalize(all_feats_list, all_ids_list, present_ids, final_path, ckpt_path)


if __name__ == "__main__":
    main()
