"""Frame mean log-spectrum. The sampling rate is the declared 16 kHz contract."""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window

from machineear.constants import HOP_LENGTH, LOG_EPS, N_FFT, SAMPLE_RATE_HZ
from machineear.errors import MachineEarError


def expected_peak_bin(frequency_hz: float, sample_rate_hz: int = SAMPLE_RATE_HZ, n_fft: int = N_FFT) -> int:
    """Hand formula: bin k = round(frequency * n_fft / sample_rate) for a bin-centered tone."""
    return int(round(frequency_hz * n_fft / sample_rate_hz))


def mean_log_spectrum(samples: np.ndarray, sample_rate_hz: int = SAMPLE_RATE_HZ) -> np.ndarray:
    if sample_rate_hz != SAMPLE_RATE_HZ:
        raise MachineEarError(
            "unsupported_sample_rate",
            f"features are defined at {SAMPLE_RATE_HZ} Hz, received {sample_rate_hz} Hz",
        )
    signal = np.asarray(samples, dtype=np.float64)
    if signal.ndim != 1 or signal.size < N_FFT:
        raise MachineEarError(
            "bad_duration",
            f"need at least {N_FFT} samples to form one frame, received {signal.size}",
        )
    if not np.isfinite(signal).all():
        raise MachineEarError("non_finite", "clip samples are not finite")
    window = get_window("hann", N_FFT, fftbins=True)
    frames: list[np.ndarray] = []
    for start in range(0, signal.size - N_FFT + 1, HOP_LENGTH):
        frame = signal[start : start + N_FFT] * window
        spectrum = np.fft.rfft(frame)
        power = spectrum.real**2 + spectrum.imag**2
        frames.append(np.log(power + LOG_EPS))
    mean = np.mean(np.stack(frames, axis=0), axis=0)
    if not np.isfinite(mean).all():
        raise MachineEarError("non_finite", "log-spectrum was not finite")
    return mean
