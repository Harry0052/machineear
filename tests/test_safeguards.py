import csv
import time
from pathlib import Path

import numpy as np

from machineear.cli import main
from machineear.constants import MAX_WAV_BYTES, SAMPLE_RATE_HZ
from machineear.pipeline import analyze_manifest
from machineear.wavio import read_wav, write_pcm16


def test_oversized_wav_is_rejected_before_scoring(tmp_path: Path) -> None:
    path = tmp_path / "big.wav"
    path.write_bytes(b"R" * (MAX_WAV_BYTES + 1))
    _clip, diagnostic = read_wav(path)
    assert diagnostic is not None
    assert diagnostic.code == "oversized"


def test_unknown_label_is_uncertain_and_excluded_from_rates(tmp_path: Path) -> None:
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    tone = 0.2 * np.sin(2 * np.pi * 120.0 * np.arange(SAMPLE_RATE_HZ) / SAMPLE_RATE_HZ)
    rows = []
    plan = [
        ("fit-a", "source_fit", "normal", "true"),
        ("fit-b", "source_fit", "normal", "true"),
        ("fit-c", "source_fit", "normal", "true"),
        ("fit-d", "source_fit", "normal", "true"),
        ("cal-a", "source_calibration", "normal", "true"),
        ("cal-b", "source_calibration", "normal", "true"),
        ("cal-c", "source_calibration", "normal", "true"),
        ("cal-d", "source_calibration", "normal", "true"),
        ("unk", "target_test", "unknown", "false"),
    ]
    for clip_id, split, label, approved in plan:
        write_pcm16(wav_dir / f"{clip_id}.wav", tone, SAMPLE_RATE_HZ)
        rows.append(
            {
                "clip_id": clip_id,
                "session_id": split,
                "microphone_domain": "mic-a",
                "wav_path": f"wav/{clip_id}.wav",
                "split": split,
                "label": label,
                "approved_normal": approved,
                "seed": "1",
            }
        )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = analyze_manifest(manifest, domain="mic-a")
    unknown = next(clip for clip in result["clips"] if clip["clip_id"] == "unk")
    assert unknown["uncertain"] is True
    assert unknown["source_alert"] is None
    sizes = result["metrics"]["split_sizes"]
    assert sizes["target_test_unknown"] == 1
    assert sizes["target_test_normal"] == 0
    assert sizes["target_test_anomaly"] == 0
    assert result["metrics"]["source_threshold"]["target_anomaly_recall"]["rate"] is None


def test_demo_finishes_inside_the_budget() -> None:
    output = Path("/tmp/machineear-budget")
    start = time.perf_counter()
    code = main(["demo", "--output", str(output)])
    elapsed = time.perf_counter() - start
    assert code == 0
    assert elapsed < 10.0


def test_report_files_stay_in_the_output_directory(tmp_path: Path) -> None:
    code = main(["demo", "--output", str(tmp_path / "out"), "--json"])
    assert code == 0
    written = sorted(path.name for path in (tmp_path / "out").iterdir())
    assert written == ["result.json", "scores.csv", "summary.html", "summary.txt"]
