"""
Resumable, time-budgeted perceptual-hash pass (phash, hash_size=16 -> 256-bit
hash) for every image in the manifest. Caches to data/manifests/_phash_cache.csv.
Same resumable pattern as probe_cache.py — call repeatedly until DONE.
"""
import argparse
import csv
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait

import pandas as pd
import imagehash
from PIL import Image

CACHE_COLS = ["filepath", "phash", "error"]


def phash_one(filepath):
    try:
        with Image.open(filepath) as im:
            h = imagehash.phash(im, hash_size=16)
        return (filepath, str(h), "")
    except Exception as e:
        return (filepath, None, f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--manifest", default="data/manifests/manifest.csv")
    ap.add_argument("--cache", default="data/manifests/_phash_cache.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--time-budget", type=float, default=30.0)
    ap.add_argument("--max-inflight-factor", type=int, default=3)
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    manifest = pd.read_csv(project_root / args.manifest)
    all_paths = [str(project_root / fp) for fp in manifest["filepath"]]

    cache_path = project_root / args.cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    already = set()
    if cache_path.exists():
        cdf = pd.read_csv(cache_path)
        already = set(cdf["filepath"].astype(str))

    remaining = [p for p in all_paths if p not in already]
    print(f"Total files: {len(all_paths)}  cached: {len(already)}  remaining: {len(remaining)}")

    if not remaining:
        print("DONE — nothing left to hash.")
        return

    file_exists = cache_path.exists()
    f = open(cache_path, "a", newline="")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(CACHE_COLS)

    t0 = time.time()
    n_done = 0
    max_inflight = args.workers * args.max_inflight_factor
    it = iter(remaining)
    pending = set()

    ex = ProcessPoolExecutor(max_workers=args.workers)
    try:
        for _ in range(min(max_inflight, len(remaining))):
            pending.add(ex.submit(phash_one, next(it)))

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                filepath, h, err = fut.result()
                writer.writerow([filepath, h or "", err])
                n_done += 1
            if time.time() - t0 > args.time_budget:
                break
            for _ in range(len(done)):
                nxt = next(it, None)
                if nxt is None:
                    break
                pending.add(ex.submit(phash_one, nxt))
    finally:
        f.flush()
        f.close()
        ex.shutdown(wait=False, cancel_futures=True)

    elapsed = time.time() - t0
    rate = n_done / elapsed if elapsed > 0 else 0
    still_remaining = len(remaining) - n_done
    print(f"Hashed {n_done} files this call ({rate:.0f} img/s, {elapsed:.1f}s). "
          f"{still_remaining} remaining out of {len(all_paths)} total.")
    if still_remaining > 0:
        print("RESUME: call this script again to continue.")
    else:
        print("DONE — all files hashed. Run dedup.py to find near-duplicates.")


if __name__ == "__main__":
    main()
