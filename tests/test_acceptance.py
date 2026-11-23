import json
from pathlib import Path

from machineear.pipeline import analyze_manifest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "src" / "machineear" / "fixtures"
ORACLE = json.loads((FIXTURES / "oracle.json").read_text(encoding="utf-8"))


def _observed(case: str, domain: str) -> dict[str, object]:
    result = analyze_manifest(FIXTURES / case / "manifest.csv", domain=domain)
    reference = result["reference"]
    return {
        "status": result["status"],
        "anomaly_status_omitted": result["anomaly_status_omitted"],
        "domain": result["domain"],
        "issue_codes": [issue["code"] for issue in result["issues"]],
        "reference_clip_ids": None if reference is None else reference["clip_ids"],
        "thresholds": result["thresholds"],
        "metrics": result["metrics"],
        "recall_drop": result["recall_drop"],
        "splits": result["splits"],
    }


def test_showcase_matches_the_pinned_oracle() -> None:
    observed = _observed("showcase", "mic-b")
    assert observed == ORACLE["showcase"]
    metrics = observed["metrics"]
    source = metrics["source_threshold"]["target_normal_false_alerts"]
    proposed = metrics["proposed_threshold"]["target_normal_false_alerts"]
    recall = metrics["proposed_threshold"]["target_anomaly_recall"]
    assert source == {"alerts": 10, "count": 10, "rate": 1.0}
    assert proposed["alerts"] < source["alerts"]
    assert recall == {"alerts": 8, "count": 8, "rate": 1.0}
    assert observed["recall_drop"] <= 0.25
    fit_ids = observed["reference_clip_ids"]
    assert fit_ids is not None
    assert all("source_fit" in clip_id for clip_id in fit_ids)
    assert not any("source_validation" in clip_id for clip_id in fit_ids)


def test_recall_loss_rejects_the_proposal() -> None:
    observed = _observed("recall_loss", "mic-b")
    assert observed == ORACLE["recall_loss"]
    assert observed["status"] == "calibration_unsuccessful"
    assert observed["thresholds"]["proposed"]["accepted"] is False
    assert observed["recall_drop"] > 0.25
    assert observed["anomaly_status_omitted"] is False


def test_insufficient_normals_do_not_propose() -> None:
    observed = _observed("insufficient", "mic-b")
    assert observed == ORACLE["insufficient"]
    assert observed["thresholds"]["proposed"] is None


def test_invalid_overlap_and_missing_omit_anomaly_status() -> None:
    invalid = analyze_manifest(FIXTURES / "invalid" / "manifest.csv")
    overlap = analyze_manifest(FIXTURES / "overlap" / "manifest.csv")
    missing = analyze_manifest(FIXTURES / "missing_metadata" / "manifest.csv")
    assert [issue["code"] for issue in invalid["issues"]] == ORACLE["invalid_issue_codes"]
    assert [issue["code"] for issue in overlap["issues"]] == ORACLE["overlap_issue_codes"]
    assert [issue["code"] for issue in missing["issues"]] == ORACLE["missing_issue_codes"]
    assert invalid["anomaly_status_omitted"] is True
    assert overlap["anomaly_status_omitted"] is True
    assert missing["anomaly_status_omitted"] is True
    assert all(clip["score"] is None for clip in invalid["clips"])
    assert "before fitting" in overlap["issues"][0]["message"]
