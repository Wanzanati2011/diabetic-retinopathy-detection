"""
Resumable, time-budgeted probe pass: reads (width, height, sha256) for every
image referenced by the EyePACS + APTOS train label CSVs and caches results
to data/manifests/_probe_cache.csv.

Each invocation runs for at most --time-budget seconds (kept under the
device_bash 45s tool timeout) and skips files already in the cache, so it is
safe to call repeatedly until it reports 0 remaining. manifest.py then reads
this cache to build the final manifest without re-hashing anything.
"""
import argparse
import csv
import hashlib
import re
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, FIRST_COMPLETED, wait

import pandas as pd
from PIL import Image

EYEPACS_RE = re.compile(r"^(?P<patient>\d+)_(?P<eye>left|right)$")
CACHE_COLS = ["filepath", "width", "height", "sha256", "error"]


def sha256_of_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_one(filepath):
    try:
        with Image.open(filepath) as im:
            w, h = im.size
        digest = sha256_of_file(filepath)
        return (filepath, w, h, digest, "")
    except Exception as e:
        return (filepath, None, None, None, f"{type(e).__name__}: {e}")


def list_all_filepaths(project_root: Path):
    ep_csv = project_root / "labels" / "trainLabels15.csv"
    ap_csv = project_root / "labels" / "trainLabels19.csv"
    ep_dir = project_root / "resized train 15"
    ap_dir = project_root / "resized train 19"

    ep = pd.read_csv(ep_csv)
    img_col = next((c for c in ep.columns if c.lower() in ("image", "id_code", "filename", "image_id")), ep.columns[0])
    ep_paths = [str(ep_dir / f"{s}.jpg") for s in ep[img_col].astype(str)]

    ap = pd.read_csv(ap_csv)
    id_col = next((c for c in ap.columns if c.lower() in ("id_code", "image", "filename", "image_id")), ap.columns[0])
    ap_paths = [str(ap_dir / f"{s}.jpg") for s in ap[id_col].astype(str)]

    return ep_paths + ap_paths


def load_cache(cache_path: Path) -> set:
    if not cache_path.exists():
        return set()
    df = pd.read_csv(cache_path)
    return set(df["filepath"].astype(str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--cache", default="data/manifests/_probe_cache.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--time-budget", type=float, default=28.0)
    ap.add_argument("--max-inflight-factor", type=int, default=3,
                     help="cap on submitted-but-unfinished tasks, as a multiple of --workers")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    cache_path = project_root / args.cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    all_paths = list_all_filepaths(project_root)
    already = load_cache(cache_path)
    remaining = [p for p in all_paths if p not in already]

    print(f"Total files: {len(all_paths)}  cached: {len(already)}  remaining: {len(remaining)}")

    if not remaining:
        print("DONE — nothing left to probe.")
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
        # prime the pipeline
        for _ in range(min(max_inflight, len(remaining))):
            pending.add(ex.submit(probe_one, next(it)))

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                filepath, w, h, digest, err = fut.result()
                writer.writerow([filepath, w if w is not None else "", h if h is not None else "", digest or "", err])
                n_done += 1
            if time.time() - t0 > args.time_budget:
                break
            for _ in range(len(done)):
                nxt = next(it, None)
                if nxt is None:
                    break
                pending.add(ex.submit(probe_one, nxt))
    finally:
        f.flush()
        f.close()
        ex.shutdown(wait=False, cancel_futures=True)

    elapsed = time.time() - t0
    rate = n_done / elapsed if elapsed > 0 else 0
    still_remaining = len(remaining) - n_done
    print(f"Probed {n_done} files this call ({rate:.0f} img/s, {elapsed:.1f}s). "
          f"{still_remaining} remaining out of {len(all_paths)} total.")
    if still_remaining > 0:
        print("RESUME: call this script again to continue.")
    else:
        print("DONE — all files probed. Run manifest.py to assemble.")


if __name__ == "__main__":
    main()
