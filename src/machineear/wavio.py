"""Bounded PCM WAV loading.

Anomaly status is never decided here. Callers omit it when a diagnostic is set.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from machineear.constants import (
    CLIPPED_FRACTION,
    INT16_RAIL,
    MAX_DURATION_S,
    MAX_WAV_BYTES,
    MIN_DURATION_S,
    SAMPLE_RATE_HZ,
)

_PCM = 1
_HEADER_FLOOR = 44


@dataclass(frozen=True)
class WavDiagnostic:
    code: str
    message: str


@dataclass(frozen=True)
class WavClip:
    samples: np.ndarray
    sample_rate_hz: int
    channels: int
    duration_s: float
    path: str


def write_pcm16(path: Path, samples: np.ndarray, sample_rate_hz: int, channels: int = 1) -> None:
    """Write float samples in [-1, 1] as 16-bit PCM. Values outside that range hit the rail."""
    array = np.asarray(samples, dtype=np.float64)
    if channels == 1:
        if array.ndim != 1:
            raise ValueError("mono samples must be one-dimensional")
        pcm = _quantize(array)
        payload = pcm.tobytes()
        channel_count = 1
    else:
        if array.ndim != 2 or array.shape[1] != channels:
            raise ValueError("expected samples shaped (frames, channels)")
        pcm = _quantize(array)
        payload = pcm.astype("<i2", copy=False).tobytes()
        channel_count = channels
    _write_container(path, payload, sample_rate_hz, channel_count, bits=16, claimed_data_bytes=None)


def read_wav(path: Path) -> tuple[WavClip | None, WavDiagnostic | None]:
    """Return a clip or one diagnostic. Hard limits are applied before any spectrum work."""
    try:
        size = path.stat().st_size
    except OSError as exc:
        return None, WavDiagnostic("unreadable", f"{path.name} cannot be read ({exc.strerror})")
    if size > MAX_WAV_BYTES:
        return None, WavDiagnostic(
            "oversized",
            f"{path.name} is {size} bytes and the limit is {MAX_WAV_BYTES} bytes",
        )
    if size < _HEADER_FLOOR:
        return None, WavDiagnostic(
            "truncated",
            f"{path.name} is {size} bytes, shorter than a WAV header",
        )
    try:
        blob = path.read_bytes()
    except OSError as exc:
        return None, WavDiagnostic("unreadable", f"{path.name} cannot be read ({exc.strerror})")
    return _parse(blob, path.name)


def _quantize(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(samples, -1.0, 1.0)
    return np.rint(clipped * INT16_RAIL).astype("<i2")


def _write_container(
    path: Path,
    payload: bytes,
    sample_rate_hz: int,
    channels: int,
    bits: int,
    claimed_data_bytes: int | None,
) -> None:
    block_align = channels * (bits // 8)
    byte_rate = sample_rate_hz * block_align
    fmt = struct.pack("<HHIIHH", _PCM, channels, sample_rate_hz, byte_rate, block_align, bits)
    data_size = len(payload) if claimed_data_bytes is None else claimed_data_bytes
    fmt_chunk = b"fmt " + struct.pack("<I", 16) + fmt
    data_chunk = b"data" + struct.pack("<I", data_size) + payload
    body = fmt_chunk + data_chunk
    blob = b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body
    path.write_bytes(blob)


def write_truncated_wav(path: Path) -> None:
    """Header claims 1.0 s at 16 kHz but the data chunk is only 100 frames."""
    payload = b"\x00\x00" * 100
    _write_container(
        path,
        payload,
        SAMPLE_RATE_HZ,
        channels=1,
        bits=16,
        claimed_data_bytes=SAMPLE_RATE_HZ * 2,
    )


def _parse(blob: bytes, name: str) -> tuple[WavClip | None, WavDiagnostic | None]:
    if blob[0:4] != b"RIFF" or blob[8:12] != b"WAVE":
        return None, WavDiagnostic("unsupported_format", f"{name} is not a RIFF WAVE file")
    riff_size = struct.unpack_from("<I", blob, 4)[0]
    if riff_size + 8 > len(blob):
        return None, WavDiagnostic(
            "truncated",
            f"{name} RIFF size claims {riff_size + 8} bytes but the file has {len(blob)}",
        )
    offset = 12
    fmt = None
    data = None
    data_claimed = None
    while offset + 8 <= len(blob):
        chunk_id = blob[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", blob, offset + 4)[0]
        start = offset + 8
        end = start + chunk_size
        if end > len(blob):
            if chunk_id == b"data":
                data = blob[start:]
                data_claimed = chunk_size
                break
            return None, WavDiagnostic(
                "truncated",
                f"{name} chunk {chunk_id!r} claims {chunk_size} bytes past the end of the file",
            )
        payload = blob[start:end]
        if chunk_id == b"fmt ":
            fmt = payload
        elif chunk_id == b"data" and data is None:
            data = payload
            data_claimed = chunk_size
        offset = end + (chunk_size % 2)
    if fmt is None or data is None or data_claimed is None:
        return None, WavDiagnostic("truncated", f"{name} is missing a fmt or data chunk")
    if len(fmt) < 16:
        return None, WavDiagnostic("unsupported_format", f"{name} has a truncated fmt chunk")
    audio_format, channels, sample_rate, _byte_rate, block_align, bits = struct.unpack_from("<HHIIHH", fmt, 0)
    if audio_format != _PCM or bits != 16:
        return None, WavDiagnostic(
            "unsupported_format",
            f"{name} is format {audio_format} at {bits} bits; only 16-bit PCM is supported",
        )
    if channels != 1:
        return None, WavDiagnostic(
            "unsupported_channels",
            f"{name} has {channels} channels; only mono clips are supported",
        )
    if sample_rate != SAMPLE_RATE_HZ:
        return None, WavDiagnostic(
            "unsupported_sample_rate",
            f"{name} is {sample_rate} Hz; only {SAMPLE_RATE_HZ} Hz clips are supported",
        )
    if block_align != 2:
        return None, WavDiagnostic(
            "unsupported_format",
            f"{name} has block align {block_align}; 16-bit mono requires 2",
        )
    if len(data) < data_claimed or len(data) % 2 != 0:
        return None, WavDiagnostic(
            "truncated",
            f"{name} declares {data_claimed} data bytes but {len(data)} bytes are present",
        )
    pcm = np.frombuffer(data, dtype="<i2")
    duration = float(pcm.size) / float(sample_rate)
    if duration < MIN_DURATION_S:
        return None, WavDiagnostic(
            "truncated",
            f"{name} lasts {duration:.3f} s, below the {MIN_DURATION_S:.1f} s minimum",
        )
    if duration > MAX_DURATION_S:
        return None, WavDiagnostic(
            "bad_duration",
            f"{name} lasts {duration:.3f} s, above the {MAX_DURATION_S:.1f} s maximum",
        )
    if pcm.size and int(np.max(np.abs(pcm.astype(np.int32)))) == 0:
        return None, WavDiagnostic(
            "all_zero",
            f"{name} is silent (every sample is 0); anomaly status was not computed",
        )
    rail = np.abs(pcm.astype(np.int32)) >= INT16_RAIL
    fraction = float(np.count_nonzero(rail)) / float(pcm.size)
    if fraction > CLIPPED_FRACTION:
        return None, WavDiagnostic(
            "clipped",
            f"{name} has {fraction:.1%} of samples at the 16-bit rail; anomaly status was not computed",
        )
    samples = pcm.astype(np.float64) / float(INT16_RAIL)
    return WavClip(samples, sample_rate, channels, duration, name), None
