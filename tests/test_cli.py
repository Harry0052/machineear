import csv
from pathlib import Path

import pytest

from machineear.cli import main
from machineear.constants import SAMPLE_RATE_HZ
from machineear.wavio import write_pcm16

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "src" / "machineear" / "fixtures"


def test_help_exits_zero() -> None:
    for argv in (["--help"], ["validate", "--help"], ["demo", "--help"]):
        with pytest.raises(SystemExit) as caught:
            main(argv)
        assert caught.value.code == 0


def test_validate_showcase_and_invalid() -> None:
    assert main(["validate", "--manifest", str(FIXTURES / "showcase" / "manifest.csv")]) == 0
    assert main(["validate", "--manifest", str(FIXTURES / "invalid" / "manifest.csv")]) == 2
    assert main(["validate", "--manifest", str(FIXTURES / "overlap" / "manifest.csv")]) == 2
    assert main(["validate", "--manifest", str(FIXTURES / "missing_metadata" / "manifest.csv")]) == 2


def test_path_escape_is_rejected(tmp_path: Path, capsys) -> None:
    wav = tmp_path / "tone.wav"
    write_pcm16(wav, [0.1] * SAMPLE_RATE_HZ, SAMPLE_RATE_HZ)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "clip_id",
                "session_id",
                "microphone_domain",
                "wav_path",
                "split",
                "label",
                "approved_normal",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "clip_id": "escape",
                "session_id": "s",
                "microphone_domain": "mic-a",
                "wav_path": "../tone.wav",
                "split": "source_fit",
                "label": "normal",
                "approved_normal": "true",
                "seed": "1",
            }
        )
    code = main(["validate", "--manifest", str(manifest)])
    captured = capsys.readouterr()
    assert code == 2
    assert "path_escape" in captured.err


def test_demo_writes_a_report(tmp_path: Path) -> None:
    output = tmp_path / "demo"
    code = main(["demo", "--output", str(output), "--case", "showcase"])
    assert code == 0
    assert (output / "summary.html").is_file()
    assert (output / "result.json").is_file()
    assert (output / "scores.csv").is_file()
    text = (output / "summary.txt").read_text(encoding="utf-8")
    assert "MachineEar calibration review" in text
    assert "anomaly_status_omitted: false" in text
