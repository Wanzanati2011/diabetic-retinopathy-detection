"""
Perceptual-hash near-duplicate detection (MASTER_PLAN.md S5.2).

Uses phash (hash_size=16, 256-bit) computed by probe_phash.py. Flags pairs at
Hamming distance <= 5 as duplicates, over the UNION of both datasets (so
cross-dataset duplicates are caught too). Adds a dup_group column to the
manifest — a group ID shared by every image that is mutually reachable
through a chain of <=5-distance pairs. Splitters must keep a whole dup_group
on one side of any split.

Method: brute-force O(n^2) Hamming comparison is ~1.5e9 pairs for ~38.8k
images — too slow at this CPU budget. Instead we use LSH banding: split the
256-bit hash into 6 disjoint bands. Any pair with total Hamming distance <=5
must match EXACTLY on at least one band (pigeonhole: 5 differing bits spread
over 6 bands leaves at least one band with 0 differences). So we only need to
group by each band and verify true distance among same-band candidates —
correctness matches brute force, at a fraction of the cost, because the
dataset has no meaningful chance collisions (each band is >=40 bits, i.e.
>10^12 possible values, versus ~38.8k images).
"""
import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

N_BANDS = 6
HASH_HEX_LEN = 64  # 256 bits


def band_slices(hex_len=HASH_HEX_LEN, n_bands=N_BANDS):
    base = hex_len // n_bands
    extra = hex_len % n_bands
    slices = []
    start = 0
    for i in range(n_bands):
        length = base + (1 if i < extra else 0)
        slices.append((start, start + length))
        start += length
    assert start == hex_len
    return slices


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--manifest", default="data/manifests/manifest.csv")
    ap.add_argument("--phash-cache", default="data/manifests/_phash_cache.csv")
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--out-manifest", default="data/manifests/manifest.csv")
    ap.add_argument("--out-json", default="results/duplicates.json")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    manifest = pd.read_csv(root / args.manifest)
    phash_df = pd.read_csv(root / args.phash_cache)

    manifest["filepath_abs"] = manifest["filepath"].apply(lambda p: str(root / p))
    ph_map = dict(zip(phash_df["filepath"], phash_df["phash"]))
    manifest["phash"] = manifest["filepath_abs"].map(ph_map)

    n_missing = manifest["phash"].isna().sum()
    if n_missing:
        raise RuntimeError(f"{n_missing} manifest rows have no phash — run probe_phash.py to DONE first.")

    slices = band_slices()
    print(f"Bands: {slices}")

    candidates = set()
    for (lo, hi) in slices:
        buckets = defaultdict(list)
        for image_id, h in zip(manifest["image_id"], manifest["phash"]):
            buckets[h[lo:hi]].append(image_id)
        n_multi = sum(1 for v in buckets.values() if len(v) > 1)
        print(f"  band[{lo}:{hi}] -> {len(buckets)} distinct values, {n_multi} groups with >1 member")
        for members in buckets.values():
            if len(members) > 1:
                for a, b in combinations(sorted(members), 2):
                    candidates.add((a, b))

    print(f"Candidate pairs to verify: {len(candidates)}")

    phash_by_id = dict(zip(manifest["image_id"], manifest["phash"]))
    confirmed = []
    for a, b in candidates:
        d = hamming_hex(phash_by_id[a], phash_by_id[b])
        if d <= args.threshold:
            confirmed.append((a, b, d))

    print(f"Confirmed near-duplicate pairs (Hamming <= {args.threshold}): {len(confirmed)}")

    # sanity: verify the hashing pipeline itself works, per acceptance test 5.2 —
    # hash a deliberately copied file and confirm distance 0 from its source.
    sample_row = manifest.iloc[0]
    sample_path = Path(sample_row["filepath_abs"])
    copy_path = sample_path.parent / f"_dedup_selftest_copy_{sample_path.name}"
    import shutil
    selftest_ok = False
    try:
        shutil.copyfile(sample_path, copy_path)
        import imagehash
        from PIL import Image
        with Image.open(copy_path) as im:
            copy_hash = str(imagehash.phash(im, hash_size=16))
        d = hamming_hex(sample_row["phash"], copy_hash)
        selftest_ok = (d == 0)
        print(f"Self-test (hash a byte-identical copy of {sample_row['image_id']}): Hamming distance = {d} "
              f"({'OK' if selftest_ok else 'FAIL — hashing pipeline may be broken'})")
    finally:
        try:
            copy_path.unlink()
        except Exception:
            pass

    # union-find over confirmed pairs
    uf = UnionFind(manifest["image_id"].tolist())
    for a, b, _ in confirmed:
        uf.union(a, b)

    manifest["dup_group"] = manifest["image_id"].map(uf.find)

    # cross-dataset pair count
    ds_by_id = dict(zip(manifest["image_id"], manifest["dataset"]))
    cross_dataset_pairs = sum(1 for a, b, _ in confirmed if ds_by_id[a] != ds_by_id[b])

    group_sizes = manifest.groupby("dup_group").size()
    multi_groups = group_sizes[group_sizes > 1]

    out = {
        "threshold_hamming_distance": args.threshold,
        "hash_size": 16,
        "n_bands": N_BANDS,
        "total_images": int(len(manifest)),
        "candidate_pairs_checked": len(candidates),
        "confirmed_duplicate_pairs": len(confirmed),
        "cross_dataset_duplicate_pairs": int(cross_dataset_pairs),
        "n_dup_groups_with_multiple_images": int(len(multi_groups)),
        "n_images_involved_in_a_dup_group": int(multi_groups.sum()) if len(multi_groups) else 0,
        "dup_group_size_distribution": {int(k): int(v) for k, v in group_sizes.value_counts().sort_index().items()},
        "selftest_copy_hamming_distance_is_zero": selftest_ok,
        "example_pairs": [
            {"a": a, "b": b, "hamming_distance": d} for a, b, d in confirmed[:20]
        ],
    }

    out_json_path = root / args.out_json
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote duplicate report -> {out_json_path}")

    manifest_out = manifest.drop(columns=["filepath_abs", "phash"])
    manifest_out.to_csv(root / args.out_manifest, index=False)
    print(f"Updated manifest with dup_group column -> {root / args.out_manifest}")

    if len(confirmed) == 0:
        print("\nWARNING: zero duplicate pairs found — per MASTER_PLAN.md S5.2, this can mean the hash "
              "is broken. The self-test above (byte-identical copy) is the check for that: "
              f"{'it PASSED (distance 0), so zero real duplicates in this dataset is plausible.' if selftest_ok else 'it FAILED — investigate before trusting this.'}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
