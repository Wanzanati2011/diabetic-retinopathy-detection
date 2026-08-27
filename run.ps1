# Two Eyes, One Patient — per-phase commands.
# Run individual lines from here, or `. .\run.ps1` won't auto-run anything —
# these are reference commands, copy the one you need into your terminal.
# All commands assume you're in the project root (this script cd's there).

Set-Location $PSScriptRoot

# ---- one-time setup ----
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# pip install timm albumentations opencv-python-headless pandas numpy scikit-learn scipy imagehash pillow pyyaml tqdm matplotlib gradio grad-cam pytest
# python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# ---- Phase 1 (already done in the sandbox — CPU only, re-run here only to verify) ----
# python -m pytest tests/test_manifest.py tests/test_dedup.py tests/test_splits.py -v

# ---- Phase 2: preprocessing cache ----
python src\data\build_cache.py --workers 12
# then LOOK AT figures\sanity_crops.png yourself before continuing.
# python -m pytest tests\test_preprocess.py -v

# ---- Phase 3: frozen feature extraction (needs your CUDA GPU) ----
# One-time (if not already installed): pip install timm
# Three configs, ~45 min total per MASTER_PLAN.md Part 7. Each is independently
# resumable (Ctrl+C-safe) -- rerun the same command to pick up where it left off.
python src\features\extract.py --backbone tf_efficientnet_b0 --size 224
python src\features\extract.py --backbone tf_efficientnet_b0 --size 384
python src\features\extract.py --backbone resnet50 --size 224
python -m pytest tests\test_features.py -v   # Acceptance Test 7.1
# If test_p2_logistic_regression_qwk_sanity[resnet50-224] fails again, run the
# diagnostic below and paste the output before touching QWK_THRESHOLD.
python src\experiments\diagnose_resnet50_qwk.py

# ---- Phase 4: Claims 1b / 2 / 2b on cached features (MASTER_PLAN.md Part 8) ----
# 1b is pure label analysis (no GPU/features needed, ran once already in the sandbox).
python src\experiments\claim1b_stratified.py
# 2 and 2b need a features/*.npz -- effnetb0@224 is the default backbone/size;
# rerun with --backbone/--size to compare across configs.
python src\experiments\claim2_protocols.py --backbone tf_efficientnet_b0 --size 224
python src\experiments\claim2b_partner_ablation.py --backbone tf_efficientnet_b0 --size 224

# Part 8.4: three-way comparison figure (label-only lookup vs P1 vs P2 vs partner
# ablation). Pure aggregation over the JSONs above -- no GPU/heavy compute, just
# reads results/*.json and plots. Rerun any time those change.
python src\experiments\summary_comparison.py

# Optional: rerun Claim 2 / 2b for the other two feature configs, to check
# whether the "no leakage under frozen features" finding holds across
# backbones. Distinct --out/--fig so these don't overwrite the effnetb0@224
# results above.
python src\experiments\claim2_protocols.py --backbone tf_efficientnet_b0 --size 384 --out results\claim2_protocols_effnetb0_384.json --fig figures\figure1_claim2_protocols_effnetb0_384.png
python src\experiments\claim2b_partner_ablation.py --backbone tf_efficientnet_b0 --size 384 --out results\claim2b_partner_ablation_effnetb0_384.json --fig figures\figure3_claim2b_partner_ablation_effnetb0_384.png
python src\experiments\claim2_protocols.py --backbone resnet50 --size 224 --out results\claim2_protocols_resnet50_224.json --fig figures\figure1_claim2_protocols_resnet50_224.png
python src\experiments\claim2b_partner_ablation.py --backbone resnet50 --size 224 --out results\claim2b_partner_ablation_resnet50_224.json --fig figures\figure3_claim2b_partner_ablation_resnet50_224.png
# Then, to compare: rerun the summary figure pointing at each config's files, e.g.
# python src\experiments\summary_comparison.py --backbone tf_efficientnet_b0 --size 384 --claim2-json results\claim2_protocols_effnetb0_384.json --claim2b-json results\claim2b_partner_ablation_effnetb0_384.json --out results\summary_comparison_effnetb0_384.json --fig figures\figure1_summary_comparison_effnetb0_384.png

# ---- Phase 5: Claim 3 both-eyes model (MASTER_PLAN.md Part 9) ----
# Heavy CPU (several LogisticRegression/Ridge fits) -- run on your machine,
# not in a sandbox. Needs a P2 split + a features/*.npz + results/inter_eye_correlation.json
# (all already present). Compares per-eye-then-max vs concatenated vs mean-pooled
# both-eye features for predicting the PATIENT-level grade (= max of the two eyes).
python src\experiments\claim3_both_eyes.py
python -m pytest tests\test_claim3_both_eyes.py -v

# Optional: check the Claim 3 finding (mean-pool + ordinal beats per-eye-then-max)
# holds across the other two feature configs, not just effnetb0@224.
python src\experiments\claim3_both_eyes.py --backbone tf_efficientnet_b0 --size 384 --out results\claim3_both_eyes_effnetb0_384.json --fig figures\figure4_claim3_both_eyes_effnetb0_384.png
python src\experiments\claim3_both_eyes.py --backbone resnet50 --size 224 --out results\claim3_both_eyes_resnet50_224.json --fig figures\figure4_claim3_both_eyes_resnet50_224.png

# ---- Phase 6: real fine-tunes (MASTER_PLAN.md Part 10) ----
# One-time (if not already installed):
#   pip install pyyaml
#
# STEP 1 -- cheap smoke test first (well under a minute, confirms the whole
# pipeline runs before you commit real GPU time). Do this before Run 1.
python src\train\finetune.py --config configs\finetune_headline.yaml --split p2 --seed 42 --epochs 1 --limit-train 200 --limit-val 64 --limit-test 64
#
# STEP 2 -- Run 1 (headline): effnetb0@224, P1 and P2, 2 seeds each = 4 runs,
# ~40 min each. Each is independently resumable (Ctrl+C-safe, rerun the same
# command to pick up where it left off).
python src\train\finetune.py --config configs\finetune_headline.yaml --split p1 --seed 42
python src\train\finetune.py --config configs\finetune_headline.yaml --split p1 --seed 43
python src\train\finetune.py --config configs\finetune_headline.yaml --split p2 --seed 42
python src\train\finetune.py --config configs\finetune_headline.yaml --split p2 --seed 43
#
# STEP 3 -- Run 2 (the app model): effnetb0@384, P2 only, ~2 hours.
python src\train\finetune.py --config configs\finetune_app.yaml --split p2 --seed 42
#
# After Run 1 finishes: rerun the Claim 2b partner ablation on the
# fine-tuned P1 model (checkpoints\finetune_headline_p1_seed42\best.pt) --
# this is the version that goes in the paper (MASTER_PLAN.md Part 10).
# Script for this comes next, once Run 1's checkpoint actually exists.
