"""Load a manifest, enforce the contracts, and score clips with a frozen reference."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from machineear.calibrate import (
    evaluate_threshold,
    propose_target_threshold,
    source_threshold,
)
from machineear.constants import (
    HOP_LENGTH,
    LOG_EPS,
    MIN_SOURCE_CALIBRATION,
    MIN_SOURCE_FIT,
    MIN_TARGET_CALIBRATION,
    N_FFT,
    QUANTILE,
    RECALL_DROP_TOLERANCE,
    SAMPLE_RATE_HZ,
    SCHEMA_VERSION,
)
from machineear.features import mean_log_spectrum
from machineear.manifest import ClipRecord, Issue, Manifest, load_manifest
from machineear.model import anomaly_scores, fit_reference
from machineear.util import canonical_sha256, file_sha256
from machineear.wavio import WavDiagnostic, read_wav

_HARD_CODES = {
    "schema_error",
    "duplicate_clip_id",
    "path_escape",
    "session_overlap",
    "unreadable",
    "too_many_clips",
    "unknown_domain",
    "oversized",
    "truncated",
    "unsupported_format",
    "unsupported_channels",
    "unsupported_sample_rate",
    "bad_duration",
    "all_zero",
    "clipped",
    "non_finite",
}

_LIMITATIONS = [
    "Clips are fixed-duration files. This is not a live alarm or a factory control.",
    "Anomaly means unusual relative to the frozen source-normal centroid, not a named mechanical fault.",
    "A new microphone or background can move scores the same way a fault can. Recalibration uses operator-reviewed normals only.",
    "Scores are dimensionless Euclidean distances. Rates are counts over labeled clips. The timeline is clip order.",
    "Default audio is synthetic. Optional external benchmarks are not loaded.",
]


def configuration() -> dict[str, object]:
    return {
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "log_eps": LOG_EPS,
        "quantile": QUANTILE,
        "recall_drop_tolerance": RECALL_DROP_TOLERANCE,
        "min_target_calibration": MIN_TARGET_CALIBRATION,
        "min_source_fit": MIN_SOURCE_FIT,
        "min_source_calibration": MIN_SOURCE_CALIBRATION,
        "alert_rule": "score_strictly_greater_than_threshold",
        "score_units": "euclidean_distance_in_standardized_mean_log_spectrum",
        "standardizer": "sklearn StandardScaler; per-bin scale floored at max(10th percentile, 1e-3)",
    }


def validate_manifest(path: Path) -> dict[str, object]:
    return analyze_manifest(path, score=False)


def analyze_manifest(path: Path, *, domain: str | None = None, score: bool = True) -> dict[str, object]:
    manifest = load_manifest(path)
    domains = sorted({clip.microphone_domain for clip in manifest.clips if clip.microphone_domain})
    issues = list(manifest.issues)
    if domain is not None and domain not in domains:
        issues.append(
            Issue(
                "unknown_domain",
                f"domain {domain!r} is not in this manifest; available: {', '.join(domains) or 'none'}",
            )
        )
    wav_notes: dict[str, WavDiagnostic] = {}
    digests: dict[str, str] = {}
    if not any(issue.code in _HARD_CODES or issue.code == "missing_metadata" for issue in issues):
        for clip in manifest.clips:
            if clip.absolute_wav is None or not clip.absolute_wav.is_file():
                issues.append(Issue("unreadable", f"{clip.clip_id} wav file is missing", clip.clip_id))
                continue
            digests[clip.clip_id] = file_sha256(clip.absolute_wav)
            _clip, diagnostic = read_wav(clip.absolute_wav)
            if diagnostic is not None:
                wav_notes[clip.clip_id] = diagnostic
                issues.append(Issue(diagnostic.code, diagnostic.message, clip.clip_id))
    blocked = [issue for issue in issues if issue.code in _HARD_CODES or issue.code == "missing_metadata"]
    if (not score) or blocked or not manifest.clips:
        status = _blocked_status(issues)
        return _envelope(
            manifest,
            domain=domain,
            domains=domains,
            issues=issues,
            status=status,
            anomaly_status_omitted=True,
            recovery=_recovery(status, issues),
            clips=_clip_rows(manifest.clips, digests, wav_notes, scores=None, alerts=None),
        )
    return _score(manifest, issues, domain, domains, digests)


def _score(
    manifest: Manifest,
    issues: list[Issue],
    domain: str | None,
    domains: list[str],
    digests: dict[str, str],
) -> dict[str, object]:
    spectra: dict[str, np.ndarray] = {}
    for clip in manifest.clips:
        assert clip.absolute_wav is not None
        loaded, diagnostic = read_wav(clip.absolute_wav)
        if diagnostic is not None or loaded is None:
            issues.append(
                Issue(
                    diagnostic.code if diagnostic else "unreadable",
                    "wav became unreadable",
                    clip.clip_id,
                )
            )
            return _envelope(
                manifest,
                domain=domain,
                domains=domains,
                issues=issues,
                status="rejected_input",
                anomaly_status_omitted=True,
                recovery=_recovery("rejected_input", issues),
                clips=_clip_rows(manifest.clips, digests, {}, scores=None, alerts=None),
            )
        spectra[clip.clip_id] = mean_log_spectrum(loaded.samples, loaded.sample_rate_hz)

    def rows(split: str, label: str | None = None) -> list[ClipRecord]:
        chosen = [clip for clip in manifest.clips if clip.split == split]
        if label is not None:
            chosen = [clip for clip in chosen if clip.label == label]
        return sorted(chosen, key=lambda clip: clip.clip_id)

    fit_rows = rows("source_fit", "normal")
    calibration_rows = rows("source_calibration", "normal")
    if len(fit_rows) < MIN_SOURCE_FIT or len(calibration_rows) < MIN_SOURCE_CALIBRATION:
        status = "insufficient_normals"
        issues.append(
            Issue(
                status,
                f"need at least {MIN_SOURCE_FIT} source-fit normals and "
                f"{MIN_SOURCE_CALIBRATION} source-calibration normals",
            )
        )
        return _envelope(
            manifest,
            domain=domain,
            domains=domains,
            issues=issues,
            status=status,
            anomaly_status_omitted=True,
            recovery=_recovery(status, issues),
            clips=_clip_rows(manifest.clips, digests, {}, scores=None, alerts=None),
        )

    reference = fit_reference(
        np.stack([spectra[clip.clip_id] for clip in fit_rows]),
        [clip.clip_id for clip in fit_rows],
    )
    score_map = {
        clip.clip_id: float(value)
        for clip, value in zip(
            sorted(manifest.clips, key=lambda item: item.clip_id),
            anomaly_scores(
                reference,
                np.stack([spectra[clip.clip_id] for clip in sorted(manifest.clips, key=lambda item: item.clip_id)]),
            ),
            strict=True,
        )
    }
    frozen = source_threshold([score_map[clip.clip_id] for clip in calibration_rows])
    target_cal = rows("target_calibration")
    approved = [clip for clip in target_cal if clip.approved_normal and clip.label == "normal"]
    labels_ok = all(clip.label == "normal" for clip in target_cal) and all(clip.approved_normal for clip in target_cal)
    target_test = _domain_filter(rows("target_test"), domain)
    anomaly_rows = [clip for clip in target_test if clip.label == "anomaly"]
    proposal = propose_target_threshold(
        [score_map[clip.clip_id] for clip in approved],
        approved_count=len(approved),
        calibration_count=len(target_cal),
        labels_ok=labels_ok and len(target_cal) > 0,
        source_threshold_value=frozen.value,
        target_anomaly_scores=[score_map[clip.clip_id] for clip in anomaly_rows],
    )
    proposed_value = None if proposal.threshold is None else proposal.threshold.value
    alerts = {
        clip.clip_id: _alerts(score_map[clip.clip_id], clip, frozen.value, proposed_value) for clip in manifest.clips
    }
    if proposal.reason == "insufficient_normals":
        status = "insufficient_normals"
    elif proposal.reason in {
        "label_check_failed",
        "recall_unavailable",
        "recall_drop_exceeds_tolerance",
    }:
        status = "calibration_unsuccessful"
    else:
        status = "ok"
    metrics = _metrics(target_test, score_map, frozen.value, proposed_value)
    by_domain = {
        name: _metrics(_domain_filter(rows("target_test"), name), score_map, frozen.value, proposed_value)
        for name in domains
    }
    return _envelope(
        manifest,
        domain=domain,
        domains=domains,
        issues=issues,
        status=status,
        anomaly_status_omitted=False,
        recovery=_recovery(status, issues),
        clips=_clip_rows(manifest.clips, digests, {}, scores=score_map, alerts=alerts),
        reference=reference.to_public(),
        thresholds={
            "source": frozen.to_public(),
            "proposed": None if proposal.threshold is None else proposal.threshold.to_public(),
        },
        metrics=metrics,
        metrics_by_domain=by_domain,
        sessions=_sessions(manifest.clips, score_map, frozen.value, proposed_value, domain),
        recall_drop=proposal.recall_drop,
    )


def _alerts(score: float, clip: ClipRecord, source_value: float, proposed_value: float | None) -> dict[str, object]:
    uncertain = clip.label == "unknown"
    source_alert = None if uncertain else score > source_value
    proposed_alert = None if uncertain or proposed_value is None else score > proposed_value
    changed = None
    if isinstance(source_alert, bool) and isinstance(proposed_alert, bool):
        changed = source_alert != proposed_alert
    return {
        "source_alert": source_alert,
        "proposed_alert": proposed_alert,
        "classification_changed": changed,
        "uncertain": uncertain,
    }


def _metrics(
    clips: list[ClipRecord],
    score_map: dict[str, float],
    source_value: float,
    proposed_value: float | None,
) -> dict[str, object]:
    normals = [score_map[clip.clip_id] for clip in clips if clip.label == "normal"]
    anomalies = [score_map[clip.clip_id] for clip in clips if clip.label == "anomaly"]
    source_rates = evaluate_threshold(normals, anomalies, source_value)
    proposed_rates = None if proposed_value is None else evaluate_threshold(normals, anomalies, proposed_value)
    source_public = {name: rate.to_public() for name, rate in source_rates.items()}
    proposed_public = (
        None if proposed_rates is None else {name: rate.to_public() for name, rate in proposed_rates.items()}
    )
    delta_far = _delta(source_public, proposed_public, "target_normal_false_alerts")
    delta_recall = _delta(source_public, proposed_public, "target_anomaly_recall")
    return {
        "split_sizes": {
            "target_test": len(clips),
            "target_test_normal": len(normals),
            "target_test_anomaly": len(anomalies),
            "target_test_unknown": sum(clip.label == "unknown" for clip in clips),
        },
        "source_threshold": source_public,
        "proposed_threshold": proposed_public,
        "delta_false_alert_rate": delta_far,
        "delta_anomaly_recall": delta_recall,
    }


def _delta(source: dict[str, dict], proposed: dict[str, dict] | None, key: str) -> float | None:
    if proposed is None:
        return None
    before = source[key]["rate"]
    after = proposed[key]["rate"]
    if before is None or after is None:
        return None
    return round(float(after) - float(before), 6)


def _sessions(
    clips: list[ClipRecord],
    score_map: dict[str, float],
    source_value: float,
    proposed_value: float | None,
    domain: str | None,
) -> list[dict[str, object]]:
    grouped: dict[str, list[ClipRecord]] = {}
    for clip in clips:
        if domain is not None and clip.split.startswith("target") and clip.microphone_domain != domain:
            continue
        grouped.setdefault(clip.session_id, []).append(clip)
    rows = []
    for session_id, members in sorted(grouped.items()):
        normals = [member for member in members if member.label == "normal"]
        source_alerts = sum(score_map[member.clip_id] > source_value for member in normals)
        proposed_alerts = None
        if proposed_value is not None:
            proposed_alerts = sum(score_map[member.clip_id] > proposed_value for member in normals)
        rows.append(
            {
                "session_id": session_id,
                "microphone_domain": members[0].microphone_domain,
                "splits": sorted({member.split for member in members}),
                "n_clips": len(members),
                "n_labeled_normal": len(normals),
                "source_threshold_normal_alerts": source_alerts,
                "proposed_threshold_normal_alerts": proposed_alerts,
            }
        )
    return rows


def _domain_filter(clips: list[ClipRecord], domain: str | None) -> list[ClipRecord]:
    if domain is None:
        return clips
    return [clip for clip in clips if clip.microphone_domain == domain]


def _clip_rows(
    clips: list[ClipRecord],
    digests: dict[str, str],
    wav_notes: dict[str, WavDiagnostic],
    scores: dict[str, float] | None,
    alerts: dict[str, dict[str, object]] | None,
) -> list[dict[str, object]]:
    rows = []
    for clip in sorted(clips, key=lambda item: item.clip_id):
        note = wav_notes.get(clip.clip_id)
        alert = None if alerts is None else alerts.get(clip.clip_id, {})
        score = None if scores is None else round(scores[clip.clip_id], 6)
        rows.append(
            {
                "clip_id": clip.clip_id,
                "session_id": clip.session_id,
                "microphone_domain": clip.microphone_domain,
                "wav_path": clip.wav_path,
                "sha256": digests.get(clip.clip_id, ""),
                "split": clip.split,
                "label": clip.label,
                "approved_normal": clip.approved_normal,
                "seed": clip.seed,
                "score": score,
                "source_alert": None if alert is None else alert.get("source_alert"),
                "proposed_alert": None if alert is None else alert.get("proposed_alert"),
                "classification_changed": None if alert is None else alert.get("classification_changed"),
                "uncertain": clip.label == "unknown",
                "diagnostic": None if note is None else {"code": note.code, "message": note.message},
            }
        )
    return rows


def _blocked_status(issues: list[Issue]) -> str:
    codes = {issue.code for issue in issues}
    if codes & (_HARD_CODES - {"unknown_domain"}):
        if "missing_metadata" in codes and not (codes & (_HARD_CODES - {"unknown_domain"})):
            return "missing_metadata"
        return "rejected_input"
    if "unknown_domain" in codes:
        return "rejected_input"
    if "missing_metadata" in codes:
        return "missing_metadata"
    if not codes:
        return "ok"
    return "rejected_input"


def _recovery(status: str, issues: list[Issue]) -> str:
    if status == "missing_metadata":
        return "Fill microphone_domain for every clip, keep sessions disjoint, and run the audit again."
    if status == "insufficient_normals":
        return (
            f"Collect at least {MIN_TARGET_CALIBRATION} operator-approved normal clips "
            "in a target session that is not used for testing, then rerun."
        )
    if status == "calibration_unsuccessful":
        return (
            "The reviewed target threshold was not accepted. Keep the frozen source threshold "
            "and collect a clearer held-out anomaly check before adopting a new cut."
        )
    if status == "rejected_input":
        detail = "; ".join(issue.message for issue in issues[:3])
        return f"Fix the input and rerun. Anomaly status was omitted. {detail}"
    return "Review the source and proposed thresholds before changing a plant setting."


def _envelope(
    manifest: Manifest,
    *,
    domain: str | None,
    domains: list[str],
    issues: list[Issue],
    status: str,
    anomaly_status_omitted: bool,
    recovery: str,
    clips: list[dict[str, object]],
    reference: dict[str, object] | None = None,
    thresholds: dict[str, object] | None = None,
    metrics: dict[str, object] | None = None,
    metrics_by_domain: dict[str, object] | None = None,
    sessions: list[dict[str, object]] | None = None,
    recall_drop: float | None = None,
) -> dict[str, object]:
    config = configuration()
    split_counts: dict[str, int] = {}
    for clip in manifest.clips:
        split_counts[clip.split] = split_counts.get(clip.split, 0) + 1
    diagnostic = None
    if issues:
        diagnostic = {"code": issues[0].code, "message": issues[0].message}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "anomaly_status_omitted": anomaly_status_omitted,
        "diagnostic": diagnostic,
        "issues": [{"code": issue.code, "message": issue.message, "clip_id": issue.clip_id} for issue in issues],
        "domain": domain or "all",
        "domains": domains,
        "config": config,
        "config_sha256": canonical_sha256(config),
        "manifest_path": manifest.path.name,
        "manifest_sha256": manifest.sha256,
        "reference": reference,
        "thresholds": thresholds or {"source": None, "proposed": None},
        "metrics": metrics,
        "metrics_by_domain": metrics_by_domain or {},
        "recall_drop": None if recall_drop is None else round(recall_drop, 6),
        "splits": split_counts,
        "sessions": sessions or [],
        "clips": clips,
        "limitations": list(_LIMITATIONS),
        "recovery": recovery,
    }
