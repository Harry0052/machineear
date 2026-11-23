"""P01 shared-fixture probe. Research evidence only, not the product.

Holds one frozen score table constant and applies documented threshold rules.
The acceptance artifact is the change in target-normal false-alert rate, the
change in target anomaly recall, and the split sizes.

This does not execute the DCASE autoencoder. The gamma rule below is the
documented 90th-percentile decision, approximated with a Wilson–Hilferty
transform because that rule is what the baseline publishes for an already
computed score. audio-anomaly is imported only when it is installed.
"""

from __future__ import annotations

import math

# Disjoint sessions, one motor family, frozen detector scores (dimensionless).
SOURCE_CAL = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]  # src-cal
TARGET_CAL = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9]  # tgt-cal
TARGET_TEST_N = [2.05, 2.15, 2.25, 2.35, 2.45, 2.55, 2.65, 2.75, 2.85, 2.95]
TARGET_TEST_A = [3.2, 3.5, 3.8, 4.1, 4.4, 4.7, 5.0, 5.3]
TARGET_TEST_A_FAIL = [2.1, 2.3, 2.5, 2.7, 3.5, 4.0, 4.5, 5.0]
TOLERANCE = 0.10
Q = 0.90


def empirical_quantile(values: list[float], q: float) -> tuple[float, int]:
    """Smallest value t such that at least fraction q of values are <= t."""
    ordered = sorted(values)
    need = math.ceil(q * len(ordered) - 1e-12)
    need = max(1, min(len(ordered), need))
    return ordered[need - 1], need


def rates(threshold: float, normals: list[float], anomalies: list[float]) -> tuple[int, int, int, int]:
    far_n = sum(score > threshold for score in normals)
    rec_n = sum(score > threshold for score in anomalies)
    return far_n, len(normals), rec_n, len(anomalies)


def gamma_p90_wilson_hilferty(values: list[float]) -> dict[str, float | str]:
    count = len(values)
    mean = sum(values) / count
    var = sum((value - mean) ** 2 for value in values) / (count - 1)
    shape = mean**2 / var
    scale = var / mean
    z90 = 1.2815515655446004
    mu = 1.0 - 1.0 / (9.0 * shape)
    sd = 1.0 / math.sqrt(9.0 * shape)
    cube = mu + z90 * sd
    return {
        "k": shape,
        "theta": scale,
        "mean": mean,
        "var": var,
        "p90": (shape * scale) * (cube**3),
        "method": "wilson-hilferty",
    }


def report(
    name: str,
    threshold: float,
    normals: list[float],
    anomalies: list[float],
    baseline: tuple[float, float] | None = None,
) -> tuple[float, float]:
    far_n, n_normal, rec_n, n_anomaly = rates(threshold, normals, anomalies)
    far = far_n / n_normal
    rec = rec_n / n_anomaly
    print(f"\n== {name}")
    print(f"threshold={threshold:.6f}")
    print(f"target_normal_false_alerts={far_n}/{n_normal} rate={far:.4f}")
    print(f"target_anomaly_recall={rec_n}/{n_anomaly} rate={rec:.4f}")
    if baseline is not None:
        print(f"delta_false_alert_rate={far - baseline[0]:+.4f}")
        print(f"delta_anomaly_recall={rec - baseline[1]:+.4f}")
        drop = baseline[1] - rec
        print(f"recall_drop={drop:.4f} tolerance={TOLERANCE:.2f} proposal_allowed={drop <= TOLERANCE}")
    return far, rec


def probe_audio_anomaly() -> None:
    try:
        import audio_anomaly
        import numpy as np
    except ImportError as exc:
        print(f"\naudio-anomaly probe skipped: {exc}")
        return

    sample_rate = 16000
    time = np.arange(sample_rate) / sample_rate

    def tone(parts: list[tuple[float, float]], extra: tuple[float, float, float, float] | None = None):
        wave = np.zeros_like(time)
        for frequency, amplitude in parts:
            wave += amplitude * np.sin(2 * np.pi * frequency * time)
        if extra is not None:
            frequency, amplitude, start, stop = extra
            mask = (time >= start) & (time < stop)
            wave[mask] += amplitude * np.sin(2 * np.pi * frequency * time[mask])
        return np.clip(wave, -0.8, 0.8).astype(np.float64)

    motor = [(120.0, 0.40), (240.0, 0.20), (360.0, 0.10)]
    source = tone(motor)
    target = tone([*motor, (900.0, 0.15)])
    anomaly = tone(motor, extra=(540.0, 0.35, 0.40, 0.70))
    print("\n== audio-anomaly 0.1.0 observed (reference = one source-normal clip)")
    for name, clip in (
        ("source_normal", source),
        ("target_normal_background", target),
        ("target_anomaly_harmonic", anomaly),
    ):
        result = audio_anomaly.detect(clip, sample_rate=sample_rate, reference=source)
        kinds = sorted({event.kind for event in result.anomalies})
        print(
            f"audio-anomaly {name}: anomalies={len(result.anomalies)} "
            f"kinds={kinds} max_score={result.max_score:.3f}"
        )
    print(
        "audio-anomaly artifact is per-clip events, not a session false-alert "
        "delta or a recall-gated threshold."
    )


def main() -> None:
    print("SPLIT SIZES")
    print("source_calibration=10 session=src-cal")
    print("target_calibration=10 session=tgt-cal")
    print("target_test_normal=10 session=tgt-test")
    print("target_test_anomaly=8 session=tgt-anomaly")
    print("sessions disjoint: src-cal, tgt-cal, tgt-test, tgt-anomaly")

    source_threshold, source_rank = empirical_quantile(SOURCE_CAL, Q)
    target_threshold, target_rank = empirical_quantile(TARGET_CAL, Q)
    print(f"\nempirical q={Q} source threshold={source_threshold} (rank {source_rank}/10)")
    print(f"empirical q={Q} target threshold={target_threshold} (rank {target_rank}/10)")
    gamma = gamma_p90_wilson_hilferty(SOURCE_CAL)
    printable = {key: round(value, 6) if isinstance(value, float) else value for key, value in gamma.items()}
    print("gamma90 source-only", printable)
    ordered_target = sorted(TARGET_CAL)
    median_target = 0.5 * (ordered_target[4] + ordered_target[5])
    zhou = max(source_threshold, median_target)
    print(f"zhou max(p90_source={source_threshold}, p50_target={median_target}) = {zhou}")

    gamma_far, gamma_rec = report(
        "DCASE documented gamma-90 on source calibration only (no recalibration)",
        float(gamma["p90"]),
        TARGET_TEST_N,
        TARGET_TEST_A,
    )
    report(
        "MachineEar source empirical q=0.90 frozen",
        source_threshold,
        TARGET_TEST_N,
        TARGET_TEST_A,
        baseline=(gamma_far, gamma_rec),
    )
    report(
        "MachineEar target empirical q=0.90 reviewed recalibration",
        target_threshold,
        TARGET_TEST_N,
        TARGET_TEST_A,
        baseline=(gamma_far, gamma_rec),
    )
    print("\n-- acceptance comparison uses frozen source quantile as the before state --")
    before = report("BEFORE frozen source q=0.90", source_threshold, TARGET_TEST_N, TARGET_TEST_A)
    report("AFTER target q=0.90", target_threshold, TARGET_TEST_N, TARGET_TEST_A, baseline=before)
    report(
        "Zhou target-aware max(p90 source, p50 target) success anomalies",
        zhou,
        TARGET_TEST_N,
        TARGET_TEST_A,
        baseline=before,
    )
    print("\n-- failure anomalies: recalibration would cut recall --")
    before_fail = report(
        "BEFORE frozen source on failure anomalies",
        source_threshold,
        TARGET_TEST_N,
        TARGET_TEST_A_FAIL,
    )
    report(
        "AFTER target q=0.90 on failure anomalies",
        target_threshold,
        TARGET_TEST_N,
        TARGET_TEST_A_FAIL,
        baseline=before_fail,
    )
    report(
        "Zhou on failure anomalies (no recall gate)",
        zhou,
        TARGET_TEST_N,
        TARGET_TEST_A_FAIL,
        baseline=before_fail,
    )
    probe_audio_anomaly()


if __name__ == "__main__":
    main()
