# V1: EMA weights check

## Finding

**The app loads the correct EMA weights.** No STOP condition triggered.

## Evidence

1. **File identity.** `app/release/best_model.pt` and
   `checkpoints/finetune_app_converged_p2_class_balanced_seed42/best.pt` are
   byte-identical (SHA-256
   `8d78cea5c1677a2ba406caf4f73f10f9dc84eb010b0a2b26b642c19d617ecbe4` for both).

2. **Save code.** `src/train/finetune_converged.py:472-475` saves
   `best.pt` as `{"model_state_dict": ema.state_dict(), ...}` — i.e. `best.pt`
   stores the EMA-smoothed weights, never the raw model weights. This matches
   `results/finetune_app_converged_p2_class_balanced_seed42_prediction_dump.json`'s
   `"weights": "EMA (best.pt stores ema.state_dict())"` provenance note.

3. **App load code.** `app/app.py:480-486` (`load_model()`) does
   `ckpt = torch.load(CHECKPOINT_PATH); model.load_state_dict(ckpt["model_state_dict"])`
   — i.e. it loads the same `model_state_dict` key, so it gets the EMA weights.

4. **Numerical parity.** Ran the app's model-loading path (timm
   `tf_efficientnet_b0`, `load_state_dict` from `best_model.pt`) on 20 images
   sampled from `results/finetune_app_converged_p2_class_balanced_seed42_test_predictions.csv`
   (`random_state=42`), reading the same pre-processed 384px cached images
   (`data/processed/384/{image_id}.jpg`) the dump script used, with identical
   ImageNet normalization and no augmentation. Compared the 5-way logits
   against the CSV's `logit_0..4` columns.

   - **Max |Δlogit| across all 20 images: 0.188741** (well under the STOP
     threshold of 0.5).
   - This residual is consistent with fp16-autocast (the CSV was produced
     with CUDA autocast fp16) vs. fp32 CPU forward pass here, not a weights
     mismatch — per-image deltas cluster around 0.05–0.19 with no outliers.

## Script

Scratch script used (not committed, per V2's instruction to keep such
scripts under `scripts/verify/` or uncommitted):
`v1_check.py`, run via `.venv/Scripts/python.exe`. Logic: load
`app/release/best_model.pt` exactly as `app/app.py:load_model()` does,
run on 20 processed test-fold images, diff against the CSV logits.

## Conclusion

No STOP condition. Proceed to V2.
