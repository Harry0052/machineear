import numpy as np

from machineear.constants import N_FFT, ORACLE_TONE_BIN, ORACLE_TONE_HZ, SAMPLE_RATE_HZ
from machineear.features import expected_peak_bin, mean_log_spectrum


def test_bin_centered_tone_peaks_on_the_hand_bin() -> None:
    assert expected_peak_bin(ORACLE_TONE_HZ) == ORACLE_TONE_BIN
    assert ORACLE_TONE_HZ * N_FFT / SAMPLE_RATE_HZ == ORACLE_TONE_BIN
    time = np.arange(SAMPLE_RATE_HZ) / SAMPLE_RATE_HZ
    tone = 0.5 * np.sin(2 * np.pi * ORACLE_TONE_HZ * time)
    spectrum = mean_log_spectrum(tone, SAMPLE_RATE_HZ)
    assert spectrum.shape == (N_FFT // 2 + 1,)
    assert np.isfinite(spectrum).all()
    assert int(np.argmax(spectrum)) == ORACLE_TONE_BIN


def test_same_tone_is_reproducible() -> None:
    time = np.arange(SAMPLE_RATE_HZ) / SAMPLE_RATE_HZ
    tone = 0.25 * np.sin(2 * np.pi * 120.0 * time)
    first = mean_log_spectrum(tone)
    second = mean_log_spectrum(tone)
    assert np.array_equal(first, second)
