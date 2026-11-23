"""Generate the synthetic WAV fixtures and the provenance manifest.

The default demo does not download audio. Every clip is a sum of sines plus a
documented per-clip gain and phase. Seeds are integers and are stored on the
CSV row.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from machineear.constants import SAMPLE_RATE_HZ  # noqa: E402
from machineear.wavio import write_pcm16, write_truncated_wav  # noqa: E402

FIXTURE_ROOT = ROOT / "src" / "machineear" / "fixtures"
SAMPLE_RATE = SAMPLE_RATE_HZ
DURATION_S = 1.0
GENERATOR_VERSION = "1"

COLUMNS = [
    "clip_id",
    "session_id",
    "microphone_domain",
    "wav_path",
    "split",
    "label",
    "approved_normal",
    "seed",
]


def synthesize(seed: int, *, background: bool, anomaly: str) -> np.ndarray:
    frame = np.arange(SAMPLE_RATE * DURATION_S) / SAMPLE_RATE
    phase = (seed % 360) * np.pi / 180.0
    gain = 1.0 + 0.02 * (((seed % 5) - 2) / 2.0)
    signal = np.zeros_like(frame)
    for frequency, amplitude in ((120.0, 0.22), (240.0, 0.11), (360.0, 0.05)):
        signal += amplitude * np.sin(2 * np.pi * frequency * frame + phase)
    if background:
        signal += 0.16 * np.sin(2 * np.pi * 900.0 * frame + phase / 2.0)
    if anomaly == "strong":
        mask = (frame >= 0.30) & (frame < 0.70)
        signal[mask] += 0.28 * np.sin(2 * np.pi * 540.0 * frame[mask])
    elif anomaly == "weak":
        mask = (frame >= 0.48) & (frame < 0.52)
        signal[mask] += 0.004 * np.sin(2 * np.pi * 540.0 * frame[mask])
    signal *= gain
    return np.clip(signal, -0.95, 0.95)


def main() -> None:
    if FIXTURE_ROOT.exists():
        for path in FIXTURE_ROOT.rglob("*"):
            if path.is_file() and path.suffix in {".wav", ".csv", ".json"}:
                path.unlink()
    records: list[dict[str, object]] = []
    _write_population(
        "showcase",
        anomaly="strong",
        records=records,
        source_fit=8,
        source_validation=4,
        source_calibration=10,
        target_calibration=10,
        target_normals=10,
        target_anomalies=8,
        seed_base=1100,
    )
    _write_population(
        "recall_loss",
        anomaly="weak",
        records=records,
        source_fit=8,
        source_validation=4,
        source_calibration=10,
        target_calibration=10,
        target_normals=10,
        target_anomalies=8,
        seed_base=2100,
    )
    _write_population(
        "insufficient",
        anomaly="strong",
        records=records,
        source_fit=4,
        source_validation=0,
        source_calibration=4,
        target_calibration=3,
        target_normals=4,
        target_anomalies=4,
        seed_base=3100,
    )
    _write_missing()
    _write_invalid()
    _write_overlap()
    provenance = {
        "generator": "scripts/generate_fixtures.py",
        "generator_version": GENERATOR_VERSION,
        "license": "Synthetic audio generated in this repository. No third-party recording. MIT, same as the code.",
        "source_url": None,
        "sample_rate_hz": SAMPLE_RATE,
        "duration_s": DURATION_S,
        "channels": 1,
        "encoding": "PCM signed 16-bit",
        "motor_hz": [120, 240, 360],
        "motor_amplitude": [0.22, 0.11, 0.05],
        "background_hz": 900,
        "background_amplitude": 0.16,
        "strong_anomaly_hz": 540,
        "strong_anomaly_amplitude": 0.28,
        "strong_anomaly_window_s": [0.30, 0.70],
        "weak_anomaly_hz": 540,
        "weak_anomaly_amplitude": 0.004,
        "weak_anomaly_window_s": [0.48, 0.52],
        "gain": "1 + 0.02 * (((seed % 5) - 2) / 2)",
        "phase_rad": "(seed % 360) * pi / 180",
        "clips": records,
    }
    (FIXTURE_ROOT / "fixture-manifest.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"wrote fixtures under {FIXTURE_ROOT}")


def _write_population(
    case: str,
    *,
    anomaly: str,
    records: list[dict[str, object]],
    source_fit: int,
    source_validation: int,
    source_calibration: int,
    target_calibration: int,
    target_normals: int,
    target_anomalies: int,
    seed_base: int,
) -> None:
    rows: list[dict[str, str]] = []
    seed = seed_base
    plan = [
        ("src-fit", "mic-a", "source_fit", "normal", "true", False, "none", source_fit),
        (
            "src-val",
            "mic-a",
            "source_validation",
            "normal",
            "true",
            False,
            "none",
            source_validation,
        ),
        (
            "src-cal",
            "mic-a",
            "source_calibration",
            "normal",
            "true",
            False,
            "none",
            source_calibration,
        ),
        (
            "tgt-cal",
            "mic-b",
            "target_calibration",
            "normal",
            "true",
            True,
            "none",
            target_calibration,
        ),
        ("tgt-test", "mic-b", "target_test", "normal", "false", True, "none", target_normals),
        ("tgt-fault", "mic-b", "target_test", "anomaly", "false", True, anomaly, target_anomalies),
    ]
    wav_dir = FIXTURE_ROOT / case / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    for session, domain, split, label, approved, background, kind, count in plan:
        for index in range(count):
            clip_id = f"{case}-{split}-{label}-{index:02d}"
            relative = f"wav/{clip_id}.wav"
            samples = synthesize(seed, background=background, anomaly=kind)
            path = wav_dir / f"{clip_id}.wav"
            write_pcm16(path, samples, SAMPLE_RATE)
            rows.append(
                {
                    "clip_id": clip_id,
                    "session_id": f"{case}-{session}",
                    "microphone_domain": domain,
                    "wav_path": relative,
                    "split": split,
                    "label": label,
                    "approved_normal": approved,
                    "seed": str(seed),
                }
            )
            records.append(
                {
                    "case": case,
                    "clip_id": clip_id,
                    "seed": seed,
                    "sha256": _sha256(path),
                    "kind": kind,
                    "background": background,
                }
            )
            seed += 1
    _write_csv(FIXTURE_ROOT / case / "manifest.csv", rows)


def _write_missing() -> None:
    case = FIXTURE_ROOT / "missing_metadata"
    wav_dir = case / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    samples = synthesize(4100, background=False, anomaly="none")
    write_pcm16(wav_dir / "known.wav", samples, SAMPLE_RATE)
    write_pcm16(wav_dir / "blank-domain.wav", samples, SAMPLE_RATE)
    _write_csv(
        case / "manifest.csv",
        [
            {
                "clip_id": "missing-known",
                "session_id": "missing-src",
                "microphone_domain": "mic-a",
                "wav_path": "wav/known.wav",
                "split": "source_fit",
                "label": "normal",
                "approved_normal": "true",
                "seed": "4100",
            },
            {
                "clip_id": "missing-domain",
                "session_id": "missing-other",
                "microphone_domain": "",
                "wav_path": "wav/blank-domain.wav",
                "split": "source_calibration",
                "label": "normal",
                "approved_normal": "true",
                "seed": "4100",
            },
        ],
    )


def _write_invalid() -> None:
    case = FIXTURE_ROOT / "invalid"
    wav_dir = case / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    silence = np.zeros(SAMPLE_RATE)
    write_pcm16(wav_dir / "all-zero.wav", silence, SAMPLE_RATE)
    write_pcm16(wav_dir / "clipped.wav", np.ones(SAMPLE_RATE), SAMPLE_RATE)
    write_truncated_wav(wav_dir / "truncated.wav")
    stereo = np.column_stack([synthesize(5100, background=False, anomaly="none"), silence])
    write_pcm16(wav_dir / "stereo.wav", stereo, SAMPLE_RATE, channels=2)
    write_pcm16(wav_dir / "rate-8k.wav", synthesize(5101, background=False, anomaly="none")[:8000], 8000)
    rows = []
    specs = [
        ("invalid-zero", "wav/all-zero.wav", "zero"),
        ("invalid-clipped", "wav/clipped.wav", "clip"),
        ("invalid-truncated", "wav/truncated.wav", "trunc"),
        ("invalid-stereo", "wav/stereo.wav", "stereo"),
        ("invalid-rate", "wav/rate-8k.wav", "rate"),
    ]
    for index, (clip_id, relative, session) in enumerate(specs):
        rows.append(
            {
                "clip_id": clip_id,
                "session_id": f"invalid-{session}",
                "microphone_domain": "mic-a",
                "wav_path": relative,
                "split": "source_fit",
                "label": "normal",
                "approved_normal": "true",
                "seed": str(5000 + index),
            }
        )
    _write_csv(case / "manifest.csv", rows)


def _write_overlap() -> None:
    case = FIXTURE_ROOT / "overlap"
    wav_dir = case / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    write_pcm16(wav_dir / "one.wav", synthesize(6100, background=False, anomaly="none"), SAMPLE_RATE)
    write_pcm16(wav_dir / "two.wav", synthesize(6101, background=True, anomaly="none"), SAMPLE_RATE)
    _write_csv(
        case / "manifest.csv",
        [
            {
                "clip_id": "overlap-fit",
                "session_id": "shared-session",
                "microphone_domain": "mic-a",
                "wav_path": "wav/one.wav",
                "split": "source_fit",
                "label": "normal",
                "approved_normal": "true",
                "seed": "6100",
            },
            {
                "clip_id": "overlap-test",
                "session_id": "shared-session",
                "microphone_domain": "mic-b",
                "wav_path": "wav/two.wav",
                "split": "target_test",
                "label": "normal",
                "approved_normal": "false",
                "seed": "6101",
            },
        ],
    )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
