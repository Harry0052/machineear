"""CSV manifest loading. Paths must stay inside the manifest directory."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from machineear.constants import MAX_CLIPS, REQUIRED_COLUMNS, SPLITS
from machineear.util import file_sha256


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    clip_id: str | None = None


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    session_id: str
    microphone_domain: str
    wav_path: str
    absolute_wav: Path | None
    split: str
    label: str
    approved_normal: bool
    seed: str
    row_number: int


@dataclass
class Manifest:
    path: Path
    root: Path
    clips: list[ClipRecord]
    issues: list[Issue]
    sha256: str


_TRUE = {"1", "true", "yes", "y"}
_FALSE = {"0", "false", "no", "n", ""}


def load_manifest(path: Path) -> Manifest:
    issues: list[Issue] = []
    clips: list[ClipRecord] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        issues.append(Issue("unreadable", f"manifest cannot be read ({exc.strerror})"))
        return Manifest(path, path.parent, [], issues, "")
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        issues.append(Issue("schema_error", "manifest is empty"))
        return Manifest(path, path.parent, [], issues, file_sha256(path))
    missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
    if missing:
        issues.append(Issue("schema_error", "manifest is missing columns: " + ", ".join(missing)))
        return Manifest(path, path.parent, [], issues, file_sha256(path))
    seen: set[str] = set()
    rows = list(reader)
    if len(rows) > MAX_CLIPS:
        issues.append(Issue("too_many_clips", f"manifest has {len(rows)} clips and the limit is {MAX_CLIPS}"))
        rows = rows[:MAX_CLIPS]
    for index, row in enumerate(rows, start=2):
        clip_id = (row.get("clip_id") or "").strip()
        session_id = (row.get("session_id") or "").strip()
        domain = (row.get("microphone_domain") or "").strip()
        wav_path = (row.get("wav_path") or "").strip()
        split = (row.get("split") or "").strip()
        label = (row.get("label") or "").strip().lower() or "unknown"
        approved_text = (row.get("approved_normal") or "").strip().lower()
        seed = (row.get("seed") or "").strip()
        if not clip_id or not session_id or not wav_path or not split:
            issues.append(
                Issue(
                    "schema_error",
                    f"row {index} needs clip_id, session_id, wav_path, and split",
                    clip_id or None,
                )
            )
            continue
        if clip_id in seen:
            issues.append(Issue("duplicate_clip_id", f"clip_id {clip_id} is repeated", clip_id))
            continue
        seen.add(clip_id)
        if not domain:
            issues.append(
                Issue(
                    "missing_metadata",
                    f"{clip_id} has no microphone_domain; fill it before scoring",
                    clip_id,
                )
            )
        if split not in SPLITS:
            issues.append(
                Issue(
                    "schema_error",
                    f"{clip_id} has split {split!r}; expected one of {', '.join(SPLITS)}",
                    clip_id,
                )
            )
        if label not in {"normal", "anomaly", "unknown"}:
            issues.append(
                Issue(
                    "schema_error",
                    f"{clip_id} has label {label!r}; use normal, anomaly, or unknown",
                    clip_id,
                )
            )
        if approved_text not in _TRUE and approved_text not in _FALSE:
            issues.append(
                Issue(
                    "schema_error",
                    f"{clip_id} has approved_normal {approved_text!r}; use true or false",
                    clip_id,
                )
            )
            approved = False
        else:
            approved = approved_text in _TRUE
        absolute = _resolve_inside(path.parent, wav_path)
        if absolute is None:
            issues.append(
                Issue(
                    "path_escape",
                    f"{clip_id} wav_path {wav_path!r} leaves the manifest directory",
                    clip_id,
                )
            )
        clips.append(
            ClipRecord(
                clip_id=clip_id,
                session_id=session_id,
                microphone_domain=domain,
                wav_path=wav_path,
                absolute_wav=absolute,
                split=split,
                label=label,
                approved_normal=approved,
                seed=seed,
                row_number=index,
            )
        )
    by_session: dict[str, set[str]] = {}
    for clip in clips:
        by_session.setdefault(clip.session_id, set()).add(clip.split)
    for session_id, splits in sorted(by_session.items()):
        if len(splits) > 1:
            joined = ", ".join(sorted(splits))
            issues.append(
                Issue(
                    "session_overlap",
                    f"session {session_id} appears in more than one split ({joined}) "
                    "and was rejected before fitting",
                    None,
                )
            )
    return Manifest(path, path.parent, clips, issues, file_sha256(path))


def _resolve_inside(root: Path, relative: str) -> Path | None:
    candidate_path = Path(relative)
    if candidate_path.is_absolute() or ".." in candidate_path.parts:
        return None
    root_resolved = root.resolve()
    candidate = (root_resolved / candidate_path).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        return None
    return candidate
