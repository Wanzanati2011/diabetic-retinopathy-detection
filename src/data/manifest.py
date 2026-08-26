"""
Build the master manifest for the "Two Eyes, One Patient" project.

Scope (per MASTER_PLAN.md Phase 1, S5.1): EyePACS train (trainLabels15.csv,
35,126 images) + APTOS train (trainLabels19.csv, 3,662 images). The Kaggle
*test* CSVs (testLabels15.csv, testImages19.csv) are out of scope -- we build
our own P1/P2/P3 splits rather than reusing Kaggle's original split.

Requires the probe cache built by probe_cache.py (data/manifests/_probe_cache.csv,
columns: filepath,width,height,sha256,error) to already cover every image --
run probe_cache.py repeatedly until it reports "DONE" before running this.

Writes data/manifests/manifest.csv with columns:
  image_id, filepath, patient_id, eye, partner_id, grade, dataset, width, height, sha256
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

EYEPACS_RE = re.compile(r"^(?P<patient>\d+)_(?P<eye>left|right)$")


def load_eyepacs(csv_path: Path, img_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    img_col = next((c for c in df.columns if c.lower() in ("image", "id_code", "filename", "image_id")), df.columns[0])
    lbl_col = next((c for c in df.columns if c.lower() in ("level", "diagnosis", "grade", "label")), df.columns[1])
    df = df[[img_col, lbl_col]].rename(columns={img_col: "image", lbl_col: "grade"})
    df["grade"] = df["grade"].astype(int)

    parsed = df["image"].astype(str).str.extract(EYEPACS_RE)
    n_unparsed = parsed["patient"].isna().sum()
    if n_unparsed:
        print(f"  WARNING: {n_unparsed} EyePACS rows did not match the "
              f"'<patient>_<left|right>' filename pattern -- inspect before trusting the manifest.",
              file=sys.stderr)

    df = pd.concat([df, parsed], axis=1)
    df["dataset"] = "eyepacs"
    df["patient_id"] = "eyepacs_" + df["patient"].astype(str)
    df["image_id"] = df["patient_id"] + "_" + df["eye"].astype(str)
    df["filepath_abs"] = df["image"].apply(lambda s: str(img_dir / f"{s}.jpg"))
    return df[["image_id", "filepath_abs", "patient_id", "eye", "grade", "dataset"]]


def load_aptos(csv_path: Path, img_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    id_col = next((c for c in df.columns if c.lower() in ("id_code", "image", "filename", "image_id")), df.columns[0])
    lbl_col = next((c for c in df.columns if c.lower() in ("diagnosis", "level", "grade", "label")), df.columns[1])
    df = df[[id_col, lbl_col]].rename(columns={id_col: "id_code", lbl_col: "grade"})
    df["grade"] = df["grade"].astype(int)
    df["dataset"] = "aptos"
    df["patient_id"] = "aptos_" + df["id_code"].astype(str)
    df["eye"] = "unknown"
    df["image_id"] = "aptos_" + df["id_code"].astype(str)
    df["filepath_abs"] = df["id_code"].apply(lambda s: str(img_dir / f"{s}.jpg"))
    return df[["image_id", "filepath_abs", "patient_id", "eye", "grade", "dataset"]]


def assign_partner_ids(df: pd.DataFrame) -> pd.DataFrame:
    """For EyePACS rows, partner_id = image_id of the fellow eye of the same patient. Null otherwise."""
    partner_map = {}
    eyepacs = df[df["dataset"] == "eyepacs"]
    for patient_id, grp in eyepacs.groupby("patient_id"):
        by_eye = dict(zip(grp["eye"], grp["image_id"]))
        left_id = by_eye.get("left")
        right_id = by_eye.get("right")
        if left_id is not None and right_id is not None:
            partner_map[left_id] = right_id
            partner_map[right_id] = left_id
    df["partner_id"] = df["image_id"].map(partner_map)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/manifests/manifest.csv")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--cache", default="data/manifests/_probe_cache.csv")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    eyepacs_csv = project_root / "labels" / "trainLabels15.csv"
    aptos_csv = project_root / "labels" / "trainLabels19.csv"
    eyepacs_dir = project_root / "resized train 15"
    aptos_dir = project_root / "resized train 19"
    cache_path = project_root / args.cache

    for p in (eyepacs_csv, aptos_csv, eyepacs_dir, aptos_dir, cache_path):
        if not p.exists():
            raise FileNotFoundError(f"Expected path not found: {p}")

    print("Loading label CSVs...")
    ep = load_eyepacs(eyepacs_csv, eyepacs_dir)
    ap_ = load_aptos(aptos_csv, aptos_dir)
    df = pd.concat([ep, ap_], ignore_index=True)
    print(f"  eyepacs rows: {len(ep)}   aptos rows: {len(ap_)}   total: {len(df)}")

    df = assign_partner_ids(df)

    print("Loading probe cache...")
    cache = pd.read_csv(cache_path, dtype={"filepath": str})
    cache = cache.drop_duplicates(subset=["filepath"], keep="last").set_index("filepath")

    missing_from_cache = set(df["filepath_abs"]) - set(cache.index)
    if missing_from_cache:
        raise RuntimeError(
            f"{len(missing_from_cache)} images are not yet in the probe cache. "
            f"Run probe_cache.py repeatedly until it reports DONE before running manifest.py. "
            f"Example missing: {list(missing_from_cache)[:3]}"
        )

    joined = cache.loc[df["filepath_abs"]]
    df["width"] = joined["width"].to_numpy()
    df["height"] = joined["height"].to_numpy()
    df["sha256"] = joined["sha256"].to_numpy()
    err = joined["error"].fillna("").to_numpy()

    n_errors = (err != "").sum()
    if n_errors:
        print(f"WARNING: {n_errors} images failed to open/hash during probing. Dropping them.", file=sys.stderr)
        bad_mask = err != ""
        print(df.loc[bad_mask, ["image_id"]].head(10).to_string(), file=sys.stderr)
        df = df.loc[~bad_mask].reset_index(drop=True)

    df["filepath"] = df["filepath_abs"].apply(lambda p: str(Path(p).relative_to(project_root)).replace("\\", "/"))
    df = df.drop(columns=["filepath_abs"])

    out_cols = ["image_id", "filepath", "patient_id", "eye", "partner_id", "grade", "dataset", "width", "height", "sha256"]
    df = df[out_cols].sort_values(["dataset", "patient_id", "eye"]).reset_index(drop=True)
    df["width"] = df["width"].astype(int)
    df["height"] = df["height"].astype(int)

    out_path = project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows -> {out_path}")

    print("\n--- quick summary ---")
    print(df.groupby(["dataset", "grade"]).size().unstack(fill_value=0))
    ep_df = df[df["dataset"] == "eyepacs"]
    print(f"EyePACS unique patients: {ep_df['patient_id'].nunique()}")
    print(f"EyePACS rows with null partner_id: {ep_df['partner_id'].isna().sum()}")
    print(f"Null patient_id: {df['patient_id'].isna().sum()}   Null grade: {df['grade'].isna().sum()}")
    print(f"Distinct grades present: {sorted(df['grade'].unique().tolist())}")


if __name__ == "__main__":
    main()
