"""
Split protocols P1 / P2 / P3 (MASTER_PLAN.md S5.3).

P1 -- image-level random, 70/15/15, stratified by grade. DELIBERATELY FLAWED:
      the split unit is the individual image (a patient's left and right eye
      can land in different folds). This is the leaky protocol the whole
      paper is about -- never "fix" it. The only integrity constraint kept
      even here is dup_group (near-duplicate images must not straddle a
      fold split -- that would just be a bug, not the leakage under study).
      For every P1 test image we additionally record whether its partner eye
      (fellow eye of the same EyePACS patient) ended up in train -- this
      drives the Claim 2b partner-eye ablation in Phase 4.

P2 -- patient-level, 70/15/15, stratified by each patient's MAX grade (worst
      eye), CORRECT protocol. Split unit = "patient_group": patient_id
      merged with dup_group via union-find, so an APTOS near-duplicate that
      happens to carry a different patient_id than its duplicate is still
      forced onto the same side of the split.

P3 -- cross-dataset, both directions. Train+val (85/15) on dataset A at the
      patient_group level; test = ALL of dataset B. No image from B is ever
      used for anything but testing in that direction.

All splits are seeded (seed=42) and written as JSON: data/splits/{name}.json
mapping image_id -> {"fold": "train"|"val"|"test", ...extra fields}.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
RATIOS = (0.70, 0.15, 0.15)  # train, val, test


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


def build_patient_group(df: pd.DataFrame) -> pd.Series:
    """Union patient_id-sharing images AND dup_group-sharing images into one
    'must stay together' unit. Returns a Series aligned to df.index."""
    uf = UnionFind(df["image_id"].tolist())
    for _, grp in df.groupby("patient_id"):
        ids = grp["image_id"].tolist()
        for i in range(1, len(ids)):
            uf.union(ids[0], ids[i])
    for _, grp in df.groupby("dup_group"):
        ids = grp["image_id"].tolist()
        for i in range(1, len(ids)):
            uf.union(ids[0], ids[i])
    return df["image_id"].map(uf.find)


def stratified_group_split(units: pd.DataFrame, unit_col: str, strat_col: str,
                            ratios=RATIOS, seed=SEED):
    """units: one row per split-unit (group), with a stratification key.
    Returns dict: unit_id -> fold."""
    rng = np.random.default_rng(seed)
    fold_of_unit = {}
    for strat_value, grp in units.groupby(strat_col):
        ids = grp[unit_col].tolist()
        ids = sorted(ids)  # deterministic order before shuffling
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        # clamp so rounding never overflows n, and give val/test at least 0 (not forced >0 for tiny strata)
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        train_ids = ids[:n_train]
        val_ids = ids[n_train:n_train + n_val]
        test_ids = ids[n_train + n_val:]
        for i in train_ids:
            fold_of_unit[i] = "train"
        for i in val_ids:
            fold_of_unit[i] = "val"
        for i in test_ids:
            fold_of_unit[i] = "test"
    return fold_of_unit


def two_way_split(units: pd.DataFrame, unit_col: str, strat_col: str,
                   ratios=(0.85, 0.15), seed=SEED):
    rng = np.random.default_rng(seed)
    fold_of_unit = {}
    for strat_value, grp in units.groupby(strat_col):
        ids = sorted(grp[unit_col].tolist())
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * ratios[0]))
        n_train = min(n_train, n)
        train_ids = ids[:n_train]
        val_ids = ids[n_train:]
        for i in train_ids:
            fold_of_unit[i] = "train"
        for i in val_ids:
            fold_of_unit[i] = "val"
    return fold_of_unit


def make_p1(df: pd.DataFrame) -> dict:
    """Image-level random split, stratified by grade, dup_group-safe only."""
    units = df.groupby("dup_group")["grade"].agg(lambda s: s.max()).reset_index()
    units.columns = ["dup_group", "strat_grade"]
    fold_of_group = stratified_group_split(units, "dup_group", "strat_grade")

    df = df.copy()
    df["fold"] = df["dup_group"].map(fold_of_group)

    train_ids = set(df.loc[df["fold"] == "train", "image_id"])
    result = {}
    for _, row in df.iterrows():
        rec = {"fold": row["fold"]}
        if row["dataset"] == "eyepacs" and pd.notna(row["partner_id"]):
            rec["partner_in_train"] = bool(row["partner_id"] in train_ids) if row["fold"] == "test" else None
        else:
            rec["partner_in_train"] = None
        result[row["image_id"]] = rec
    return result


def make_p2(df: pd.DataFrame) -> dict:
    """Patient-level split, stratified by each patient_group's max grade."""
    df = df.copy()
    df["patient_group"] = build_patient_group(df)
    units = df.groupby("patient_group")["grade"].agg(lambda s: s.max()).reset_index()
    units.columns = ["patient_group", "strat_grade"]
    fold_of_group = stratified_group_split(units, "patient_group", "strat_grade")
    df["fold"] = df["patient_group"].map(fold_of_group)
    return {row["image_id"]: {"fold": row["fold"]} for _, row in df.iterrows()}


def make_p3_direction(df: pd.DataFrame, source_dataset: str, target_dataset: str) -> dict:
    """Train+val (85/15, patient-level) on source_dataset; test = all of target_dataset."""
    df = df.copy()
    df["patient_group"] = build_patient_group(df)

    src = df[df["dataset"] == source_dataset]
    tgt = df[df["dataset"] == target_dataset]

    units = src.groupby("patient_group")["grade"].agg(lambda s: s.max()).reset_index()
    units.columns = ["patient_group", "strat_grade"]
    fold_of_group = two_way_split(units, "patient_group", "strat_grade")

    result = {}
    for _, row in src.iterrows():
        result[row["image_id"]] = {"fold": fold_of_group[row["patient_group"]]}
    for _, row in tgt.iterrows():
        result[row["image_id"]] = {"fold": "test"}
    return result


def grade_distribution_report(df, fold_map):
    df = df.copy()
    df["fold"] = df["image_id"].map(lambda i: fold_map[i]["fold"])
    tab = pd.crosstab(df["fold"], df["grade"], normalize="index")
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--manifest", default="data/manifests/manifest.csv")
    ap.add_argument("--out-dir", default="data/splits")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    df = pd.read_csv(root / args.manifest)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded manifest: {len(df)} rows")

    if "excluded" in df.columns:
        n_excluded = int(df["excluded"].fillna(False).astype(bool).sum())
        df = df[~df["excluded"].fillna(False).astype(bool)].reset_index(drop=True)
        print(f"Filtered out {n_excluded} excluded rows (see apply_exclusions.py) -> {len(df)} active rows")

    print("\n=== P1 (image-level, deliberately flawed) ===")
    p1 = make_p1(df)
    (out_dir / "p1.json").write_text(json.dumps(p1, indent=2, sort_keys=True))
    print(f"Wrote {out_dir / 'p1.json'}")
    print(grade_distribution_report(df, p1))
    n_test_eyepacs = sum(1 for i, r in p1.items() if r["fold"] == "test" and df.set_index("image_id").loc[i, "dataset"] == "eyepacs")

    print("\n=== P2 (patient-level, correct) ===")
    p2 = make_p2(df)
    (out_dir / "p2.json").write_text(json.dumps(p2, indent=2, sort_keys=True))
    print(f"Wrote {out_dir / 'p2.json'}")
    print(grade_distribution_report(df, p2))

    print("\n=== P3 direction A: train/val=eyepacs, test=aptos ===")
    p3a = make_p3_direction(df, "eyepacs", "aptos")
    (out_dir / "p3_eyepacs_to_aptos.json").write_text(json.dumps(p3a, indent=2, sort_keys=True))
    print(f"Wrote {out_dir / 'p3_eyepacs_to_aptos.json'}")

    print("\n=== P3 direction B: train/val=aptos, test=eyepacs ===")
    p3b = make_p3_direction(df, "aptos", "eyepacs")
    (out_dir / "p3_aptos_to_eyepacs.json").write_text(json.dumps(p3b, indent=2, sort_keys=True))
    print(f"Wrote {out_dir / 'p3_aptos_to_eyepacs.json'}")

    print("\nAll splits written.")


if __name__ == "__main__":
    main()
