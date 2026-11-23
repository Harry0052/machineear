"""Frozen normal reference: standardized mean log-spectrum and distance to its centroid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from machineear.errors import MachineEarError


@dataclass(frozen=True)
class FrozenReference:
    mean: np.ndarray
    scale: np.ndarray
    centroid: np.ndarray
    n_clips: int
    clip_ids: tuple[str, ...]

    def to_public(self) -> dict[str, object]:
        return {
            "n_clips": self.n_clips,
            "clip_ids": list(self.clip_ids),
            "n_bins": int(self.mean.shape[0]),
            "scale_floor": round(float(np.min(self.scale)), 6),
            "mean_sha256_prefix": _prefix(self.mean),
        }


def fit_reference(spectra: np.ndarray, clip_ids: list[str]) -> FrozenReference:
    matrix = np.asarray(spectra, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise MachineEarError(
            "insufficient_normals",
            "source fit needs at least two normal clips before a centroid can be frozen",
        )
    if not np.isfinite(matrix).all():
        raise MachineEarError("non_finite", "source-fit spectra are not finite")
    scaler = StandardScaler()
    scaler.fit(matrix)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    positive = scale[scale > 0]
    percentile_floor = float(np.percentile(positive, 10)) if positive.size else 1.0
    floor = max(percentile_floor, 1e-3)
    scale = np.maximum(scale, floor)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    standardized = (matrix - mean) / scale
    centroid = standardized.mean(axis=0)
    order = tuple(clip_ids)
    return FrozenReference(
        mean=mean,
        scale=scale,
        centroid=np.asarray(centroid, dtype=np.float64),
        n_clips=matrix.shape[0],
        clip_ids=order,
    )


def anomaly_scores(reference: FrozenReference, spectra: np.ndarray) -> np.ndarray:
    matrix = np.asarray(spectra, dtype=np.float64)
    standardized = (matrix - reference.mean) / reference.scale
    distances = np.linalg.norm(standardized - reference.centroid, axis=1)
    if not np.isfinite(distances).all():
        raise MachineEarError("non_finite", "anomaly scores are not finite")
    return distances


def _prefix(values: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()[:12]
