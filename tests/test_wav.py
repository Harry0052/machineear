import numpy as np

from machineear.constants import SAMPLE_RATE_HZ
from machineear.wavio import read_wav, write_pcm16, write_truncated_wav


def test_roundtrip_and_invalid_diagnostics(tmp_path) -> None:
    tone = 0.2 * np.sin(2 * np.pi * 120.0 * np.arange(SAMPLE_RATE_HZ) / SAMPLE_RATE_HZ)
    good = tmp_path / "good.wav"
    write_pcm16(good, tone, SAMPLE_RATE_HZ)
    clip, diagnostic = read_wav(good)
    assert diagnostic is None
    assert clip is not None
    assert clip.sample_rate_hz == SAMPLE_RATE_HZ
    assert clip.samples.shape == (SAMPLE_RATE_HZ,)

    zero = tmp_path / "zero.wav"
    write_pcm16(zero, np.zeros(SAMPLE_RATE_HZ), SAMPLE_RATE_HZ)
    _, zero_note = read_wav(zero)
    assert zero_note is not None and zero_note.code == "all_zero"

    clipped = tmp_path / "clipped.wav"
    write_pcm16(clipped, np.ones(SAMPLE_RATE_HZ), SAMPLE_RATE_HZ)
    _, clipped_note = read_wav(clipped)
    assert clipped_note is not None and clipped_note.code == "clipped"

    truncated = tmp_path / "truncated.wav"
    write_truncated_wav(truncated)
    _, truncated_note = read_wav(truncated)
    assert truncated_note is not None and truncated_note.code == "truncated"

    stereo = tmp_path / "stereo.wav"
    write_pcm16(stereo, np.column_stack([tone, tone]), SAMPLE_RATE_HZ, channels=2)
    _, stereo_note = read_wav(stereo)
    assert stereo_note is not None and stereo_note.code == "unsupported_channels"

    slow = tmp_path / "slow.wav"
    write_pcm16(slow, tone[:8000], 8000)
    _, slow_note = read_wav(slow)
    assert slow_note is not None and slow_note.code == "unsupported_sample_rate"

    long = tmp_path / "long.wav"
    write_pcm16(long, np.tile(tone, 3)[: SAMPLE_RATE_HZ * 2 + 100], SAMPLE_RATE_HZ)
    _, long_note = read_wav(long)
    assert long_note is not None and long_note.code == "bad_duration"


def test_missing_file(tmp_path) -> None:
    _, diagnostic = read_wav(tmp_path / "absent.wav")
    assert diagnostic is not None
    assert diagnostic.code == "unreadable"
