"""
Session log entry construction (AGENT_EXECUTION_PLAN.md Task A7). No Gradio
import. Will move into app/render/console.py when Task B3's module split
lands; kept here for now alongside decision.py so A7 doesn't have to wait
on the Phase 2 file layout.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Optional

from app.core.decision import Outcome


@dataclass(frozen=True)
class LogEntry:
    time: str
    mode: str
    outcome: str
    grade: int
    grade_name: str
    expected_grade: float
    referral_score: float
    calibrated_conf: float
    raw_conf: float
    calibration_active: bool
    model_sha12: str

    def as_dict(self) -> dict:
        return {
            "time": self.time, "mode": self.mode, "outcome": self.outcome,
            "grade": self.grade, "grade_name": self.grade_name,
            "expected_grade": self.expected_grade,
            "referral_score": self.referral_score,
            "calibrated_conf": self.calibrated_conf, "raw_conf": self.raw_conf,
            "calibration_active": self.calibration_active,
            "model_sha12": self.model_sha12,
        }


def make_log_entry(mode: str, grade: int, grade_name: str, outcome: Outcome,
                    expected_grade: float, referral_score: float,
                    calibrated_conf: float, raw_conf: float,
                    calibration_active: bool, model_sha12: str,
                    now: Optional[datetime.datetime] = None) -> LogEntry:
    now = now or datetime.datetime.now()
    return LogEntry(
        time=now.strftime("%H:%M:%S"),
        mode=mode,
        outcome=outcome.value,
        grade=int(grade),
        grade_name=grade_name,
        expected_grade=float(expected_grade),
        referral_score=float(referral_score),
        calibrated_conf=float(calibrated_conf),
        raw_conf=float(raw_conf),
        calibration_active=bool(calibration_active),
        model_sha12=model_sha12,
    )
