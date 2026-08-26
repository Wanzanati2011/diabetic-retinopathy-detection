"""
Applies documented image exclusions to the manifest -- SOFT exclude, not
delete. Rows stay in data/manifests/manifest.csv for auditability; excluded
rows get `excluded=True` and a human-readable `excluded_reason`. Downstream
split building (splits.py) filters these out before assigning folds; the
image cache on disk (data/processed/) is left untouched, and
src/features/extract.py skips excluded rows explicitly.

Current exclusion list: 8 EyePACS images identified by the Acceptance Test
6.1 follow-up (results/low_mean_diagnostic.json) as flat-black/underexposed
source captures -- original-image mean pixel value < 5, min=0, near-uniform.
Every excluded image's partner eye was confirmed normal (single failed
capture, not a bad patient); 3 of the 8 carried a nonzero grade despite
being visually blank. Full evidence: results/excluded_images.json.

This is idempotent -- safe to re-run.
"""
import argparse
from pathlib import Path

import pandas as pd

EXCLUSIONS = {
    "eyepacs_1557_left": "flat-black capture, mean pixel < 5",
    "eyepacs_1986_left": "flat-black capture, mean pixel < 5",
    "eyepacs_21720_left": "flat-black capture, mean pixel < 5",
    "eyepacs_26064_right": "flat-black capture, mean pixel < 5",
    "eyepacs_32253_right": "flat-black capture, mean pixel < 5",
    "eyepacs_34689_left": "flat-black capture, mean pixel < 5",
    "eyepacs_42130_left": "flat-black capture, mean pixel < 5",
    "eyepacs_43457_left": "flat-black capture, mean pixel < 5",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/manifest.csv")
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    manifest_path = root / args.manifest
    df = pd.read_csv(manifest_path)

    missing = set(EXCLUSIONS) - set(df["image_id"])
    if missing:
        raise RuntimeError(f"exclusion list references image_ids not in the manifest: {missing}")

    df["excluded"] = df["image_id"].isin(EXCLUSIONS)
    df["excluded_reason"] = df["image_id"].map(EXCLUSIONS)  # NaN for non-excluded rows, by design

    n_excluded = int(df["excluded"].sum())
    df.to_csv(manifest_path, index=False)
    print(f"Applied exclusions: {n_excluded} rows marked excluded=True out of {len(df)} total.")
    print(f"Wrote {manifest_path}")
    for image_id, reason in EXCLUSIONS.items():
        print(f"  {image_id}: {reason}")


if __name__ == "__main__":
    main()
