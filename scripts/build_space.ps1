<#
.SYNOPSIS
    Copies an explicit allowlist of files into a target folder for deploying
    to the Hugging Face Space (AGENT_EXECUTION_PLAN.md Task B8).

.DESCRIPTION
    This script NEVER runs `git push` or touches the Space itself -- per R1,
    only the owner deploys. It only assembles a clean copy of what should be
    pushed, into -TargetDir, so the owner can review it before pushing.

    Allowlist:
      - app/  (everything EXCEPT app/samples_local/, which is local-only
        until the Kaggle licence is confirmed -- R6/H1)
      - src/data/preprocess.py and the other src/ modules the app imports
        (src/data/dedup.py, for the Both Eyes identical-image guard)
      - the results/*.json and results/*.csv files the app reads at runtime
        (listed explicitly below -- revisit this list once Task B2's
        METRICS loader lands and can enumerate its own sources)
      - app/release/ (checkpoint, thresholds, config, model card)
      - requirements-lock.txt

.PARAMETER TargetDir
    Destination folder. Created if it doesn't exist. Existing contents are
    left alone except for files this script overwrites.

.EXAMPLE
    .\scripts\build_space.ps1 -TargetDir ..\fundus-console-space
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$TargetDir
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
$TargetDir = (Resolve-Path $TargetDir).Path

# Results files the app currently reads at runtime. Keep this in sync with
# app/app.py, app/core/decision.py, app/data/context.py, app/report/pdf.py --
# grep those for `PROJECT_ROOT / "results"` / `"app" / "release"` if this
# list goes stale, or better: revisit once Task B2's METRICS loader can
# enumerate its own sources programmatically.
$ResultsFiles = @(
    "results\finetune_app_converged_p2_class_balanced_seed42.json",
    "results\finetune_app_converged_p2_class_balanced_seed42_curve.json",
    "results\finetune_app_converged_p2_class_balanced_seed42_test_predictions.csv",
    "results\finetune_app_converged_p2_class_balanced_seed42_val_predictions.csv",
    "results\calibration_reject.json",
    "results\grade1_diagnosis.json",
    "results\claim3_decomposed_tf_efficientnet_b0_384.json",
    "results\qml_pqc.json",
    "results\qcnn_no_cnn_pixels.json"
)

$SrcFiles = @(
    "src\data\preprocess.py",
    "src\data\dedup.py"
)

$copied = @()
$skipped = @()

function Copy-OneFile($relativePath) {
    $source = Join-Path $ProjectRoot $relativePath
    $dest = Join-Path $TargetDir $relativePath
    if (-not (Test-Path $source)) {
        $script:skipped += $relativePath
        return
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
    Copy-Item -Path $source -Destination $dest -Force
    $script:copied += $relativePath
}

# app/ (everything except app/samples_local/, __pycache__/, and byte-code)
$AppSource = Join-Path $ProjectRoot "app"
Get-ChildItem -Path $AppSource -Recurse -File | Where-Object {
    $_.FullName -notmatch [regex]::Escape((Join-Path $AppSource "samples_local")) -and
    $_.FullName -notmatch '\\__pycache__\\' -and
    $_.Extension -ne ".pyc"
} | ForEach-Object {
    $rel = $_.FullName.Substring($ProjectRoot.Length + 1)
    Copy-OneFile $rel
}

foreach ($f in $ResultsFiles) { Copy-OneFile $f }
foreach ($f in $SrcFiles) { Copy-OneFile $f }
Copy-OneFile "requirements-lock.txt"

Write-Host ""
Write-Host "Copied $($copied.Count) file(s) to $TargetDir :"
$copied | Sort-Object | ForEach-Object { Write-Host "  $_" }

if ($skipped.Count -gt 0) {
    Write-Host ""
    Write-Host "SKIPPED (not found -- check these before deploying):" -ForegroundColor Yellow
    $skipped | Sort-Object | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
}

Write-Host ""
Write-Host "Nothing was pushed anywhere. Review $TargetDir, then push it yourself (R1: the owner deploys)." -ForegroundColor Cyan
