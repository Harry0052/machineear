import math

import pytest

from machineear.calibrate import empirical_quantile, propose_target_threshold


def test_hand_quantile_matches_the_overlap_probe() -> None:
    scores = [1.0 + 0.1 * index for index in range(10)]
    # 9th of 10 ordered values. ceil(0.90 * 10) = 9, value 1.8.
    assert math.isclose(empirical_quantile(scores, 0.90), 1.8)


def test_recall_drop_rejects_the_proposal() -> None:
    calibration = [2.0 + 0.1 * index for index in range(10)]
    anomalies = [2.1, 2.3, 2.5, 2.7, 3.5, 4.0, 4.5, 5.0]
    proposal = propose_target_threshold(
        calibration,
        approved_count=10,
        calibration_count=10,
        labels_ok=True,
        source_threshold_value=1.8,
        target_anomaly_scores=anomalies,
    )
    assert proposal.threshold is not None
    assert proposal.threshold.accepted is False
    assert proposal.reason == "recall_drop_exceeds_tolerance"
    assert proposal.recall_drop is not None
    assert proposal.recall_drop > 0.25


def test_short_target_calibration_does_not_propose() -> None:
    proposal = propose_target_threshold(
        [1.0, 1.2, 1.4],
        approved_count=3,
        calibration_count=3,
        labels_ok=True,
        source_threshold_value=0.5,
        target_anomaly_scores=[2.0, 3.0],
    )
    assert proposal.threshold is None
    assert proposal.reason == "insufficient_normals"


def test_quantile_rejects_empty_scores() -> None:
    with pytest.raises(ValueError):
        empirical_quantile([], 0.9)
