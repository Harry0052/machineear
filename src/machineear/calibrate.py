"""Quantile thresholds and the recall gate. Scores stay frozen; only the cut can move."""

from __future__ import annotations

import math
from dataclasses import dataclass

from machineear.constants import (
    MIN_TARGET_CALIBRATION,
    PROPOSED_THRESHOLD_VERSION,
    QUANTILE,
    RECALL_DROP_TOLERANCE,
    SOURCE_THRESHOLD_VERSION,
)


@dataclass(frozen=True)
class Rate:
    alerts: int
    count: int

    @property
    def rate(self) -> float | None:
        if self.count == 0:
            return None
        return self.alerts / self.count

    def to_public(self) -> dict[str, float | int | None]:
        return {"alerts": self.alerts, "count": self.count, "rate": _round(self.rate)}


@dataclass(frozen=True)
class Threshold:
    version: str
    quantile: float
    value: float
    n_calibration: int
    accepted: bool | None
    reason: str

    def to_public(self) -> dict[str, object]:
        return {
            "version": self.version,
            "quantile": self.quantile,
            "value": round(self.value, 6),
            "n_calibration": self.n_calibration,
            "accepted": self.accepted,
            "reason": self.reason,
        }


def empirical_quantile(scores: list[float], quantile: float = QUANTILE) -> float:
    """Smallest score t such that at least fraction `quantile` of scores are <= t.

    Hand check: 1.0, 1.1, ..., 1.9 at 0.90 returns 1.8 (9th of 10).
    A clip alerts when its score is strictly greater than t.
    """
    if not scores:
        raise ValueError("quantile of an empty score list is undefined")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    ordered = sorted(scores)
    need = math.ceil(quantile * len(ordered) - 1e-12)
    need = max(1, min(len(ordered), need))
    return float(ordered[need - 1])


def count_alerts(scores: list[float], threshold: float) -> int:
    return sum(score > threshold for score in scores)


def evaluate_threshold(normal_scores: list[float], anomaly_scores: list[float], threshold: float) -> dict[str, Rate]:
    return {
        "target_normal_false_alerts": Rate(count_alerts(normal_scores, threshold), len(normal_scores)),
        "target_anomaly_recall": Rate(count_alerts(anomaly_scores, threshold), len(anomaly_scores)),
    }


def source_threshold(scores: list[float]) -> Threshold:
    value = empirical_quantile(scores, QUANTILE)
    return Threshold(
        version=SOURCE_THRESHOLD_VERSION,
        quantile=QUANTILE,
        value=value,
        n_calibration=len(scores),
        accepted=True,
        reason="frozen_source_calibration",
    )


@dataclass(frozen=True)
class Proposal:
    threshold: Threshold | None
    reason: str
    recall_drop: float | None


def propose_target_threshold(
    calibration_scores: list[float],
    *,
    approved_count: int,
    calibration_count: int,
    labels_ok: bool,
    source_threshold_value: float,
    target_anomaly_scores: list[float],
) -> Proposal:
    """Return a versioned threshold only after size and label checks.

    The recall gate can still mark that threshold unsuccessful. The value remains
    visible so the recall cost can be read next to the false-alert change.
    """
    if calibration_count < MIN_TARGET_CALIBRATION or approved_count < MIN_TARGET_CALIBRATION:
        return Proposal(None, "insufficient_normals", None)
    if not labels_ok or approved_count != calibration_count:
        return Proposal(None, "label_check_failed", None)
    value = empirical_quantile(calibration_scores, QUANTILE)
    anomaly_count = len(target_anomaly_scores)
    if anomaly_count == 0:
        threshold = Threshold(
            PROPOSED_THRESHOLD_VERSION,
            QUANTILE,
            value,
            len(calibration_scores),
            False,
            "recall_unavailable",
        )
        return Proposal(threshold, "recall_unavailable", None)
    source_hits = count_alerts(target_anomaly_scores, source_threshold_value)
    proposed_hits = count_alerts(target_anomaly_scores, value)
    source_recall = source_hits / anomaly_count
    proposed_recall = proposed_hits / anomaly_count
    drop = source_recall - proposed_recall
    if drop > RECALL_DROP_TOLERANCE:
        reason = "recall_drop_exceeds_tolerance"
        accepted = False
    else:
        reason = "within_tolerance"
        accepted = True
    threshold = Threshold(
        PROPOSED_THRESHOLD_VERSION,
        QUANTILE,
        value,
        len(calibration_scores),
        accepted,
        reason,
    )
    return Proposal(threshold, reason, drop)


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)
