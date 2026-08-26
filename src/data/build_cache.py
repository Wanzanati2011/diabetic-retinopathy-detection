"""
Phase 2 — preprocessing cache (MASTER_PLAN.md Part 6).

Runs preprocess() from src/data/preprocess.py (the single shared function
also used by the app) over every image in the manifest, at each requested
size, and caches the result to disk as JPEG (quality=95 by default — chosen
over the plan's original PNG suggestion to cut cache size and I/O time;
95-quality JPEG is visually lossless for this purpose and the training
signal that matters here is macro lesion structure, not per-pixel fidelity).

Meant to run on your local machine (Windows, real CPU cores) — NOT in the
Cowork sandbox. See run.ps1 for the exact command.

Resumable: re-running skips any (image_id, size) whose output file already
exists on disk. Safe to Ctrl+C and restart.

Usage (from the project root, e.g. C:\\dr-project or wherever this repo lives):
    python src\\data\\build_cache.py --workers 12
"""
import argparse
import json
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import preprocess, ben_graham  # noqa: E402


def process_one(task):
    """Runs in a worker process. task = (image_id, src_path, out_root_str, sizes, quality, use_ben_graham)."""
    image_id, src_path, out_root_str, sizes, quality, use_ben_graham = task
    out_root = Path(out_root_str)
    result = {"image_id": image_id, "error": None, "mean_pixel": {}, "elapsed_sec": 0.0, "wrote": []}
    t0 = time.perf_counter()
    try:
        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img is None:
            result["error"] = f"cv2.imread returned None for {src_path}"
            return result
        for size in sizes:
            out_path = out_root / str(size) / f"{image_id}.jpg"
            if out_path.exists():
                continue
            rgb = preprocess(img, size=size)
            if use_ben_graham:
                rgb = ben_graham(rgb)
            result["mean_pixel"][size] = float(rgb.mean())
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(str(out_path), "JPEG", quality=quality)
            result["wrote"].append(size)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    result["elapsed_sec"] = time.perf_counter() - t0
    return result


def build_contact_sheet(manifest: pd.DataFrame, out_root: Path, size: int, fig_path: Path, n_per_dataset: int = 20, seed: int = 42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = random.Random(seed)
    rows = []
    for dataset, grp in manifest.groupby("dataset"):
        ids = grp["image_id"].tolist()
        sample = rng.sample(ids, min(n_per_dataset, len(ids)))
        for image_id in sample:
            rows.append((dataset, image_id))

    n = len(rows)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.9))
    axes = np.array(axes).reshape(-1)
    grade_by_id = dict(zip(manifest["image_id"], manifest["grade"]))

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        dataset, image_id = rows[i]
        img_path = out_root / str(size) / f"{image_id}.jpg"
        if img_path.exists():
            im = Image.open(img_path)
            ax.imshow(im)
        else:
            ax.text(0.5, 0.5, "MISSING", ha="center", va="center")
        ax.set_title(f"{dataset} g{grade_by_id.get(image_id, '?')}\n{image_id}", fontsize=6)

    fig.suptitle(f"Phase 2 sanity check — {size}px crops, {n_per_dataset} random per dataset "
                  f"(Acceptance Test 6.1 — look at these personally: did the crop keep the whole retina?)",
                  fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(PROJECT_ROOT))
    ap.add_argument("--manifest", default="data/manifests/manifest.csv")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--sizes", default="224,384")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ben-graham", action="store_true", help="apply Ben Graham local-contrast ablation (not default)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only process first N manifest rows")
    ap.add_argument("--contact-sheet-size", type=int, default=224, help="which cached size to sample for sanity_crops.png")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    sizes = [int(s) for s in args.sizes.split(",")]
    manifest = pd.read_csv(root / args.manifest)
    if args.limit:
        manifest = manifest.head(args.limit)

    out_root = root / args.out_dir
    print(f"Project root: {root}")
    print(f"Manifest rows: {len(manifest)}   sizes: {sizes}   quality: {args.quality}   workers: {args.workers}")
    print(f"Output: {out_root}")
    if args.ben_graham:
        print("Ben Graham ablation: ON (non-default)")

    tasks = []
    n_already_done = 0
    for row in manifest.itertuples():
        missing = [s for s in sizes if not (out_root / str(s) / f"{row.image_id}.jpg").exists()]
        if not missing:
            n_already_done += 1
            continue
        src_path = str(root / row.filepath)
        tasks.append((row.image_id, src_path, str(out_root), missing, args.quality, args.ben_graham))

    print(f"Already cached (all sizes): {n_already_done}   To process: {len(tasks)}")

    errors = []
    low_mean_warnings = []
    elapsed_times = []
    n_done = 0
    t_start = time.time()

    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_one, t) for t in tasks]
            for fut in as_completed(futures):
                r = fut.result()
                n_done += 1
                if r["error"]:
                    errors.append(r)
                else:
                    elapsed_times.append(r["elapsed_sec"])
                    for size, mean_val in r["mean_pixel"].items():
                        if mean_val < 5:
                            low_mean_warnings.append({"image_id": r["image_id"], "size": size, "mean_pixel": mean_val})
                if n_done % 1000 == 0 or n_done == len(tasks):
                    elapsed = time.time() - t_start
                    rate = n_done / elapsed if elapsed > 0 else 0
                    eta = (len(tasks) - n_done) / rate if rate > 0 else float("inf")
                    print(f"  {n_done}/{len(tasks)}  ({rate:.1f} img/s, {elapsed:.0f}s elapsed, ETA {eta:.0f}s, "
                          f"{len(errors)} errors, {len(low_mean_warnings)} low-mean warnings)")

    total_elapsed = time.time() - t_start

    # cache size on disk
    cache_bytes = 0
    per_size_counts = {}
    for size in sizes:
        size_dir = out_root / str(size)
        files = list(size_dir.glob("*.jpg")) if size_dir.exists() else []
        per_size_counts[size] = len(files)
        cache_bytes += sum(f.stat().st_size for f in files)

    summary = {
        "phase": "Phase 2 — preprocessing cache",
        "manifest_rows": int(len(manifest)),
        "already_cached_at_start": n_already_done,
        "processed_this_run": len(tasks),
        "errors": len(errors),
        "error_examples": errors[:10],
        "low_mean_pixel_warnings": low_mean_warnings,
        "n_low_mean_pixel_warnings": len(low_mean_warnings),
        "sizes": sizes,
        "quality": args.quality,
        "ben_graham_ablation": args.ben_graham,
        "per_size_file_counts": per_size_counts,
        "cache_size_bytes": cache_bytes,
        "cache_size_gb": round(cache_bytes / 1e9, 2),
        "median_processing_time_sec_per_image": round(statistics.median(elapsed_times), 4) if elapsed_times else None,
        "mean_processing_time_sec_per_image": round(statistics.mean(elapsed_times), 4) if elapsed_times else None,
        "total_run_time_sec": round(total_elapsed, 1),
        "workers": args.workers,
    }

    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "phase2_preprocess.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {results_dir / 'phase2_preprocess.json'}")
    print(f"Cache size: {summary['cache_size_gb']} GB   "
          f"median {summary['median_processing_time_sec_per_image']}s/image   "
          f"{len(errors)} errors   {len(low_mean_warnings)} low-mean warnings")

    if errors:
        print(f"WARNING: {len(errors)} images failed to process. See results/phase2_preprocess.json for details.", file=sys.stderr)
    if low_mean_warnings:
        print(f"WARNING: {len(low_mean_warnings)} processed crops have mean pixel value < 5 "
              f"(possibly blank/black — see results/phase2_preprocess.json).", file=sys.stderr)

    print(f"\nBuilding sanity contact sheet from {args.contact_sheet_size}px cache...")
    fig_path = root / "figures" / "sanity_crops.png"
    build_contact_sheet(manifest, out_root, args.contact_sheet_size, fig_path)
    print(f"Wrote {fig_path}")
    print("\n>>> ACCEPTANCE TEST 6.1: open figures/sanity_crops.png and look at it yourself.")
    print(">>> Check that no crop sliced off part of the retina. This step cannot be automated.")


if __name__ == "__main__":
    main()
