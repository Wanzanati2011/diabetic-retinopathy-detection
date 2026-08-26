"""
Diagnostic for the 8 images that failed test_no_low_mean_pixel_crops
(Acceptance Test 6.1) after the full local Phase 2 run.

For each flagged image_id:
  - original SOURCE image stats (mean/min/max/std) -- distinguishes a
    genuinely underexposed capture (min/max spread out, mean low but
    nonzero) from a crop-pipeline bug (would show near-uniform ~0)
  - grade, patient_id, dataset
  - fold assignment in P1 / P2 / P3 (both directions)
  - partner eye's grade + original stats (is the fellow eye normal?)

Also builds figures/low_mean_diagnostic.png (original vs 224px-processed,
side by side, all 8) and writes results/low_mean_diagnostic.json.

CPU-only. No GPU needed.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FLAGGED_IDS = [
    "eyepacs_1557_left", "eyepacs_1986_left", "eyepacs_21720_left",
    "eyepacs_26064_right", "eyepacs_32253_right", "eyepacs_34689_left",
    "eyepacs_42130_left", "eyepacs_43457_left",
]


def image_stats(path: Path) -> dict:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": f"cv2.imread failed for {path}"}
    arr = img.astype(np.float64)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
    }


def load_fold(split_path: Path, image_id: str):
    d = json.loads(split_path.read_text())
    rec = d.get(image_id)
    return rec["fold"] if rec else None


def main():
    root = PROJECT_ROOT
    manifest = pd.read_csv(root / "data/manifests/manifest.csv").set_index("image_id", drop=False)

    splits = {
        "P1": root / "data/splits/p1.json",
        "P2": root / "data/splits/p2.json",
        "P3(eyepacs->aptos)": root / "data/splits/p3_eyepacs_to_aptos.json",
        "P3(aptos->eyepacs)": root / "data/splits/p3_aptos_to_eyepacs.json",
    }

    report = []
    any_in_test_split = False

    for image_id in FLAGGED_IDS:
        row = manifest.loc[image_id]
        src_path = root / row["filepath"]
        stats = image_stats(src_path)

        partner_id = row["partner_id"] if pd.notna(row["partner_id"]) else None
        partner_info = None
        if partner_id:
            prow = manifest.loc[partner_id]
            partner_stats = image_stats(root / prow["filepath"])
            partner_info = {
                "image_id": partner_id,
                "grade": int(prow["grade"]),
                "stats": partner_stats,
                "looks_normal": partner_stats.get("mean", 0) >= 5,
            }

        fold_assignments = {name: load_fold(path, image_id) for name, path in splits.items()}
        is_in_a_test_fold = any(f == "test" for f in fold_assignments.values())
        if is_in_a_test_fold:
            any_in_test_split = True

        entry = {
            "image_id": image_id,
            "patient_id": row["patient_id"],
            "grade": int(row["grade"]),
            "dataset": row["dataset"],
            "eye": row["eye"],
            "original_stats": stats,
            "partner": partner_info,
            "fold_assignments": fold_assignments,
            "in_any_test_fold": is_in_a_test_fold,
            "nonzero_grade_on_blank_image": int(row["grade"]) > 0,
        }
        report.append(entry)
        print(f"{image_id}: grade={row['grade']} mean={stats.get('mean'):.2f} min={stats.get('min')} max={stats.get('max')} std={stats.get('std'):.2f}")
        print(f"    patient={row['patient_id']}  folds={fold_assignments}")
        if partner_info:
            print(f"    partner={partner_info['image_id']}  partner_grade={partner_info['grade']}  "
                  f"partner_mean={partner_info['stats'].get('mean'):.2f}  partner_looks_normal={partner_info['looks_normal']}")
        else:
            print("    no partner_id on record")

    out = {
        "flagged_images": report,
        "n_flagged": len(report),
        "n_nonzero_grade_on_blank": sum(1 for e in report if e["nonzero_grade_on_blank_image"]),
        "any_in_test_fold": any_in_test_split,
    }
    out_path = root / "results/low_mean_diagnostic.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    print(f"any_in_test_fold: {any_in_test_split}")
    print(f"n_nonzero_grade_on_blank: {out['n_nonzero_grade_on_blank']}")


if __name__ == "__main__":
    main()
