"""
Single source of numbers (AGENT_EXECUTION_PLAN.md Task B2). No Gradio
import. Every JSON/CSV the app displays is loaded ONCE here, at import
time, into a typed, read-only structure. A missing file leaves its field
None (or an empty container); callers render "data unavailable", never
crash and never fall back to a hand-typed number (R3).

app/data/context.py and app/report/pdf.py's provenance-reading pieces will
be folded into this loader over time; kept separate where they already
work, to avoid a disruptive rename mid-project.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
APP_RELEASE_DIR = PROJECT_ROOT / "app" / "release"

DEPLOYED_MODEL_JSON = RESULTS_DIR / "finetune_app_converged_p2_class_balanced_seed42.json"
CALIBRATION_JSON = RESULTS_DIR / "calibration_reject.json"
THRESHOLDS_JSON = APP_RELEASE_DIR / "thresholds.json"
GRADE1_JSON = RESULTS_DIR / "grade1_diagnosis.json"
CLAIM3_DECOMPOSED_384_JSON = RESULTS_DIR / "claim3_decomposed_tf_efficientnet_b0_384.json"
QML_JSON = RESULTS_DIR / "qml_pqc.json"
QCNN_JSON = RESULTS_DIR / "qcnn_no_cnn_pixels.json"
ACCEPTANCE_12_1_JSON = RESULTS_DIR / "acceptance_12_1.json"
INTER_EYE_JSON = RESULTS_DIR / "inter_eye_correlation.json"



def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


@dataclass(frozen=True)
class DeployedModel:
    """From results/finetune_app_converged_p2_class_balanced_seed42.json."""
    test_qwk: Optional[float] = None
    referable_auroc: Optional[float] = None
    best_epoch: Optional[int] = None
    total_epochs_run: Optional[int] = None
    n_train: Optional[int] = None
    n_val: Optional[int] = None
    n_test: Optional[int] = None
    per_class_recall: dict = field(default_factory=dict)  # {"0": {"n":.., "recall":..}, ...}
    acceptance_test_10_1_verdict: Optional[str] = None
    available: bool = False


def _load_deployed_model() -> DeployedModel:
    data = _load_json(DEPLOYED_MODEL_JSON)
    if data is None:
        return DeployedModel()
    return DeployedModel(
        test_qwk=data.get("test_qwk"),
        referable_auroc=data.get("referable_dr_auroc"),
        best_epoch=data.get("best_epoch"),
        total_epochs_run=data.get("total_epochs_run"),
        n_train=data.get("n_train"), n_val=data.get("n_val"), n_test=data.get("n_test"),
        per_class_recall=data.get("per_class_recall", {}),
        acceptance_test_10_1_verdict=data.get("acceptance_test_10_1_verdict"),
        available=True,
    )


@dataclass(frozen=True)
class Calibration:
    """From app/release/thresholds.json + results/calibration_reject.json."""
    temperature: Optional[float] = None
    referral_threshold: Optional[float] = None
    reject_tau: Optional[float] = None
    acceptance_test_11_1_verdict: Optional[str] = None
    test_ece_before: Optional[float] = None
    test_ece_after: Optional[float] = None
    selective_qwk_at_tau: Optional[float] = None
    referral_sens_full: Optional[float] = None
    referral_spec_full: Optional[float] = None
    available: bool = False


def _load_calibration() -> Calibration:
    th = _load_json(THRESHOLDS_JSON)
    cal = _load_json(CALIBRATION_JSON)
    if th is None and cal is None:
        return Calibration()
    test_set = (cal or {}).get("calibration", {}).get("p2_test", {})
    full_cov = (cal or {}).get("referral_threshold", {}).get("test_full_coverage", {})
    return Calibration(
        temperature=(th or {}).get("temperature"),
        referral_threshold=(th or {}).get("referral_threshold"),
        reject_tau=(th or {}).get("reject_tau"),
        acceptance_test_11_1_verdict=(th or {}).get("acceptance_test_11_1_verdict"),
        test_ece_before=test_set.get("before", {}).get("ece"),
        test_ece_after=test_set.get("after", {}).get("ece"),
        selective_qwk_at_tau=(cal or {}).get("reject_option", {}).get("test_at_tau", {}).get("selective_qwk"),
        referral_sens_full=full_cov.get("sensitivity"),
        referral_spec_full=full_cov.get("specificity"),
        available=th is not None or cal is not None,
    )


@dataclass(frozen=True)
class Grade1:
    """From results/grade1_diagnosis.json."""
    recall_argmax: Optional[float] = None  # confusion_row_grade1_fractions["Mild NPDR"]
    verdict: Optional[str] = None
    ordinal_read_qwk_delta: Optional[float] = None
    ordinal_read_ci95: Optional[list] = None
    available: bool = False


def _load_grade1() -> Grade1:
    data = _load_json(GRADE1_JSON)
    if data is None:
        return Grade1()
    ordinal = data.get("ordinal_read_vs_argmax", {})
    return Grade1(
        recall_argmax=data.get("confusion_row_grade1_fractions", {}).get("Mild NPDR"),
        verdict=data.get("verdict"),
        ordinal_read_qwk_delta=ordinal.get("observed_diff"),
        ordinal_read_ci95=ordinal.get("paired_patient_bootstrap_ci95"),
        available=True,
    )


@dataclass(frozen=True)
class Hero:
    """St stat tile data for the hero section (U3)."""
    lookup_qwk: Optional[float] = None  # label-only baseline from inter_eye_correlation.json
    referral_sens_in_10: Optional[int] = None  # e.g. 9
    referral_spec_in_10: Optional[int] = None  # e.g. 7
    n_models: Optional[int] = None  # count of finetune_*.json + frozen-feature configs


def _load_hero() -> Hero:
    # 1. lookup_qwk from inter_eye_correlation.json
    ie = _load_json(INTER_EYE_JSON)
    lookup_qwk = None
    if ie and "label_only_baseline" in ie:
        lookup_qwk = ie["label_only_baseline"].get("lookup_qwk")

    # 2. referral sens/spec → in-10 numbers
    cal = _load_json(CALIBRATION_JSON)
    full_cov = (cal or {}).get("referral_threshold", {}).get("test_full_coverage", {})
    sens = full_cov.get("sensitivity")
    spec = full_cov.get("specificity")
    sens_in_10 = round(sens * 10) if sens is not None else None
    spec_in_10 = round(spec * 10) if spec is not None else None

    # 3. n_models: count finetune_*.json plus frozen-feature configs
    import glob
    finetune_files = glob.glob(str(RESULTS_DIR / "finetune_*.json"))
    frozen_files = glob.glob(str(RESULTS_DIR / "*frozen*.json"))
    n_models = len(finetune_files) + len(frozen_files) if finetune_files else None

    return Hero(
        lookup_qwk=lookup_qwk,
        referral_sens_in_10=sens_in_10,
        referral_spec_in_10=spec_in_10,
        n_models=n_models,
    )


@dataclass(frozen=True)
class BothEyesDecomposition:
    """From results/claim3_decomposed_tf_efficientnet_b0_384.json."""
    head_effect: Optional[float] = None
    fusion_effect: Optional[float] = None
    total_gain: Optional[float] = None
    available: bool = False


def _load_both_eyes() -> BothEyesDecomposition:
    data = _load_json(CLAIM3_DECOMPOSED_384_JSON)
    if data is None:
        return BothEyesDecomposition()
    dec = data.get("decomposition", {})
    return BothEyesDecomposition(
        head_effect=dec.get("1_head_effect_ordinal_minus_multinomial_at_max", {}).get("mean_diff"),
        fusion_effect=dec.get("2_fusion_effect_pool_minus_per_eye_max_ordinal", {}).get("mean_diff"),
        total_gain=dec.get("8_original_total_gain_pool_ordinal_minus_per_eye_max_multinomial", {}).get("mean_diff"),
        available=True,
    )


@dataclass(frozen=True)
class Metrics:
    """The single object app.py reads for every displayed number."""
    deployed_model: DeployedModel
    calibration: Calibration
    grade1: Grade1
    both_eyes: BothEyesDecomposition
    hero: Hero
    qml: Optional[dict]
    qcnn: Optional[dict]
    acceptance_12_1: Optional[dict]
    missing_files: list


def load_metrics() -> Metrics:
    sources = {
        "deployed model results": DEPLOYED_MODEL_JSON,
        "calibration thresholds": THRESHOLDS_JSON,
        "calibration reject": CALIBRATION_JSON,
        "grade1 diagnosis": GRADE1_JSON,
        "claim3 decomposed (384px)": CLAIM3_DECOMPOSED_384_JSON,
        "qml_pqc": QML_JSON,
        "qcnn_no_cnn_pixels": QCNN_JSON,
        "acceptance_12_1": ACCEPTANCE_12_1_JSON,
    }
    missing = [name for name, path in sources.items() if not path.exists()]
    return Metrics(
        deployed_model=_load_deployed_model(),
        calibration=_load_calibration(),
        grade1=_load_grade1(),
        both_eyes=_load_both_eyes(),
        hero=_load_hero(),
        qml=_load_json(QML_JSON),
        qcnn=_load_json(QCNN_JSON),
        acceptance_12_1=_load_json(ACCEPTANCE_12_1_JSON),
        missing_files=missing,
    )


def fmt_pct(value: Optional[float], decimals: int = 1) -> str:
    """'data unavailable' instead of crashing on None -- the B2 rule,
    applied at render time too."""
    if value is None:
        return "data unavailable"
    return f"{value * 100:.{decimals}f}%"


def fmt_num(value: Optional[Any], decimals: int = 3) -> str:
    if value is None:
        return "data unavailable"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)
