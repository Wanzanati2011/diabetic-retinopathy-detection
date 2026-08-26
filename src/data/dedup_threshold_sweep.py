"""
Threshold sweep + calibration for the perceptual-hash dedup threshold
(follow-up review of dedup.py's Hamming <= 5 choice).

Context: dedup.py used hash_size=16 (256-bit phash) with Hamming <= 5. That
threshold (~5) is the common convention for the DEFAULT hash_size=8 (64-bit)
phash. Kept as an absolute bit count on a 256-bit hash it is 4x stricter
than intended: 5/256 = 2.0% of bits vs 5/64 = 7.8% for the convention it was
borrowed from. This script checks empirically whether that mismatch cost us
any real EyePACS duplicates, using three independent checks rather than
trusting the scaling arithmetic alone:

  1. Threshold sweep: EXACT (not LSH-approximated) pairwise Hamming distance
     counts at 5/10/20/30/40, for EyePACS and APTOS separately, via the
     popcount-via-matmul identity H(a,b) = sum(a)+sum(b)-2*dot(a,b) (BLAS-
     accelerated, chunked by row-block to bound memory).
  2. Calibration: perturb one real EyePACS image (resize +/-2px, re-encode
     JPEG q90) and measure its phash distance from the original — this is
     what a genuine near-duplicate photo looks like in this metric.
  3. Diagnostic on the swept candidates: for EyePACS pairs found at a looser
     threshold, check (a) same-patient rate and (b) grade-agreement rate.
     A TRUE duplicate photo must carry the same grade (same underlying image
     -> same diagnosis). Grade agreement near the dataset's chance baseline
     is evidence the pair is a coincidental structural match, not a
     duplicate.
"""
import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

THRESHOLDS = [5, 10, 20, 30, 40]
DIAGNOSTIC_THRESHOLD = 25  # inspect candidates up to this distance for same-patient / grade-match


def hex_to_bits_matrix(hex_series: pd.Series) -> np.ndarray:
    n = len(hex_series)
    out = np.empty((n, 256), dtype=np.float32)
    for i, h in enumerate(hex_series):
        out[i] = np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8))
    return out


def scan_pairs(bits: np.ndarray, max_threshold: int, block: int = 3000):
    """Exact pairwise Hamming distances <= max_threshold (upper triangle, no
    self-pairs). Returns (i_idx, j_idx, dist) arrays plus per-threshold
    cumulative counts for THRESHOLDS <= max_threshold."""
    n = bits.shape[0]
    rowsum = bits.sum(axis=1)
    all_i, all_j, all_d = [], [], []
    counts = {t: 0 for t in THRESHOLDS if t <= max_threshold}

    t0 = time.time()
    for start in range(0, n, block):
        end = min(start + block, n)
        dot = bits[start:end] @ bits.T
        dist = rowsum[start:end, None] + rowsum[None, :] - 2 * dot
        col_idx = np.arange(n)[None, :]
        row_global = np.arange(start, end)[:, None]
        mask = (dist <= max_threshold) & (col_idx > row_global)
        li, gj = np.nonzero(mask)
        gi = li + start
        all_i.append(gi); all_j.append(gj); all_d.append(dist[li, gj])
        for t in counts:
            counts[t] += int(((dist <= t) & (col_idx > row_global)).sum())
    elapsed = time.time() - t0
    gi = np.concatenate(all_i) if all_i else np.array([], dtype=int)
    gj = np.concatenate(all_j) if all_j else np.array([], dtype=int)
    dd = np.concatenate(all_d) if all_d else np.array([], dtype=float)
    return gi, gj, dd, counts, elapsed


def calibration_check(project_root: Path, manifest: pd.DataFrame, phash_cache: pd.DataFrame):
    import imagehash

    ep_row = manifest[manifest["dataset"] == "eyepacs"].iloc[0]
    src_path = project_root / ep_row["filepath"]
    ph_map = dict(zip(phash_cache["filepath"], phash_cache["phash"]))
    original_hash = ph_map[str(src_path)]

    im = Image.open(src_path)
    w, h = im.size
    perturbed = im.resize((w - 2, h + 2), Image.BILINEAR)
    buf = io.BytesIO()
    perturbed.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    perturbed_hash = str(imagehash.phash(Image.open(buf), hash_size=16))
    dist = bin(int(original_hash, 16) ^ int(perturbed_hash, 16)).count("1")

    return {
        "image_id": ep_row["image_id"],
        "original_size": [w, h],
        "perturbed_size": [w - 2, h + 2],
        "perturbation": "resize (w-2, h+2), re-encoded JPEG quality=90",
        "hamming_distance": dist,
        "exceeds_threshold_5": dist > 5,
    }


def eyepacs_diagnostics(sub: pd.DataFrame, gi, gj, dd, threshold: int):
    rows = []
    for i, j, d in zip(gi.tolist(), gj.tolist(), dd.tolist()):
        if d > threshold:
            continue
        ri, rj = sub.iloc[i], sub.iloc[j]
        rows.append({
            "dist": d, "a": ri["image_id"], "b": rj["image_id"],
            "same_patient": bool(ri["patient_id"] == rj["patient_id"]),
            "a_grade": int(ri["grade"]), "b_grade": int(rj["grade"]),
            "grade_match": bool(ri["grade"] == rj["grade"]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {"n_pairs": 0}

    grade_counts = sub["grade"].value_counts(normalize=True)
    chance_agreement = float((grade_counts ** 2).sum())  # P(two random images share a grade)

    summary = {
        "n_pairs": int(len(df)),
        "n_same_patient": int(df["same_patient"].sum()),
        "n_different_patient": int((~df["same_patient"]).sum()),
        "grade_agreement_rate_overall": float(df["grade_match"].mean()),
        "grade_agreement_rate_different_patient_only": float(df.loc[~df["same_patient"], "grade_match"].mean()),
        "chance_grade_agreement_rate": chance_agreement,
        "interpretation": (
            "Grade agreement near the chance baseline means these candidate pairs are coincidental "
            "structural matches (similar illumination/framing), not duplicate photographs -- a true "
            "duplicate must carry the same grade, since it's the same underlying image."
        ),
    }
    return df, summary


def make_plot(sweep_results: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for dataset, style in [("eyepacs", "o-"), ("aptos", "s-")]:
        d = sweep_results["by_dataset"][dataset]["counts_by_threshold"]
        xs = sorted(int(k) for k in d.keys())
        ys = [d[str(x)] if str(x) in d else d[x] for x in xs]
        ax.plot(xs, [max(y, 0.5) for y in ys], style, label=dataset, linewidth=2, markersize=7)

    ax.axvline(5, color="gray", linestyle="--", linewidth=1, label="threshold used (5)")
    ax.set_yscale("log")
    ax.set_xlabel("Hamming distance threshold (out of 256 bits)")
    ax.set_ylabel("Confirmed pairs at or below threshold (log scale)")
    ax.set_title("Near-duplicate pairs vs Hamming threshold\nEyePACS: no plateau, near-zero until false positives flood in")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--manifest", default="data/manifests/manifest.csv")
    ap.add_argument("--phash-cache", default="data/manifests/_phash_cache.csv")
    ap.add_argument("--block", type=int, default=3000)
    ap.add_argument("--out-json", default="results/dedup_threshold_sweep.json")
    ap.add_argument("--out-plot", default="figures/dedup_threshold_sweep.png")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    manifest = pd.read_csv(root / args.manifest)
    phash_cache = pd.read_csv(root / args.phash_cache)

    manifest = manifest.copy()
    manifest["filepath_abs"] = manifest["filepath"].apply(lambda p: str(root / p))
    ph_map = dict(zip(phash_cache["filepath"], phash_cache["phash"]))
    manifest["phash"] = manifest["filepath_abs"].map(ph_map)
    assert manifest["phash"].isna().sum() == 0

    results = {"thresholds": THRESHOLDS, "diagnostic_threshold": DIAGNOSTIC_THRESHOLD, "by_dataset": {}}
    eyepacs_sub = None
    eyepacs_gi = eyepacs_gj = eyepacs_dd = None

    for dataset in ["eyepacs", "aptos"]:
        sub = manifest[manifest["dataset"] == dataset].reset_index(drop=True)
        print(f"\n=== {dataset}: {len(sub)} images ===")
        bits = hex_to_bits_matrix(sub["phash"])
        gi, gj, dd, counts, elapsed = scan_pairs(bits, max(THRESHOLDS), block=args.block)
        n_pairs_total = len(sub) * (len(sub) - 1) // 2
        print(f"  computed in {elapsed:.1f}s over {n_pairs_total:,} total pairs")
        for t in THRESHOLDS:
            print(f"    Hamming <= {t:2d}: {counts[t]:,} pairs")
        results["by_dataset"][dataset] = {
            "n_images": len(sub),
            "n_total_pairs": n_pairs_total,
            "counts_by_threshold": counts,
            "compute_seconds": round(elapsed, 1),
        }
        if dataset == "eyepacs":
            eyepacs_sub, eyepacs_gi, eyepacs_gj, eyepacs_dd = sub, gi, gj, dd

    print(f"\n=== EyePACS diagnostic: candidates at Hamming <= {DIAGNOSTIC_THRESHOLD} ===")
    diag_df, diag_summary = eyepacs_diagnostics(eyepacs_sub, eyepacs_gi, eyepacs_gj, eyepacs_dd, DIAGNOSTIC_THRESHOLD)
    for k, v in diag_summary.items():
        print(f"  {k}: {v}")
    results["eyepacs_diagnostic"] = diag_summary
    if not diag_df.empty:
        diag_df.sort_values("dist").to_csv(root / "results" / "eyepacs_near_dup_candidates.csv", index=False)
        print(f"  wrote results/eyepacs_near_dup_candidates.csv ({len(diag_df)} rows)")

    print("\n=== calibration check ===")
    calib = calibration_check(root, manifest, phash_cache)
    for k, v in calib.items():
        print(f"  {k}: {v}")
    results["calibration"] = calib

    out_json_path = root / args.out_json
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_json_path}")

    make_plot(results, root / args.out_plot)
    print(f"Wrote {root / args.out_plot}")


if __name__ == "__main__":
    main()
