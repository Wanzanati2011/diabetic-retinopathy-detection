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
# python -m pytest tests\test_features.py -v   # Acceptance Test 7.1

# ---- Phase 4: Claims 1b / 2 / 2b on cached features (writes here later) ----
# python src\experiments\claim1b_stratified.py
# python src\experiments\claim2_protocols.py
# python src\experiments\claim2b_partner_ablation.py

# ---- Phase 5: Claim 3 both-eyes model (writes here later) ----
# python src\experiments\claim3_both_eyes.py

# ---- Phase 6: fine-tunes (writes here later) ----
# python src\train\finetune.py --config configs\finetune_headline.yaml
# python src\train\finetune.py --config configs\finetune_app.yaml
