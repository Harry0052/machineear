# Project contract

Decision carried forward from `docs/overlap.md`: proceed. MachineEar is a separate repository. It audits a frozen anomaly score when the microphone or background changes. It does not replace a factory detector, raise a live alarm, or diagnose a mechanical fault.

## User and task

Industrial reliability engineers who already have a normal-reference score for one motor family and need to know whether a new recording setup still deserves the old alert threshold.

The shipped task is offline and clip-based:

1. Load a manifest of fixed-duration mono WAV clips with session, microphone domain, split, and label.
2. Fit a standardized mean log-spectrum centroid on source-fit normals only, then freeze it.
3. Set the alert threshold at a declared empirical quantile of disjoint source-calibration normals.
4. Score target sessions with that frozen model.
5. Propose a new quantile, with its own version, only from operator-approved target normals that pass size and label checks.
6. Compare both thresholds on the same held-out target clips. Report false-alert rate, anomaly recall, the underlying counts, and the split sizes.
7. Mark the proposal unsuccessful when held-out anomaly recall falls by more than the predeclared tolerance.

Anomaly means unusual relative to the recorded source baseline. It does not mean a named failure mode.

## Out of scope

Factory controls, live alarms, fault diagnosis, autonomous threshold changes, streaming audio, and workplace recordings. Optional benchmark audio stays behind an explicit import and is not on the default path. No paid service and no credentials.

## Minimum dependencies

- Python 3.11 or newer. Development host for this contract: CPython 3.12.3.
- NumPy, SciPy, and scikit-learn for array math, the windowed spectrum, and the frozen standardizer.
- pytest for the acceptance tests.
- Standard-library `wave`, `argparse`, `csv`, `json`, and `html` for clips, the CLI, and reports.

Torch, TensorFlow, and any package that downloads DCASE or MIMII audio are not required. `audio-anomaly` was used only as an overlap probe and is not a dependency.

## Fixture provenance

Default audio is generated in this repository from documented seeds: a motor-like sum of harmonics, a changed background tone for the second microphone domain, and labeled anomalies that insert a different harmonic for a short interval. The manifest is CSV with `clip_id`, `session_id`, `microphone_domain`, WAV path, and `split`. Source fit, source validation, target approved normals, and held-out target sessions use different session ids. No remote fetch at runtime.

## Contracts the implementation must keep

- Sample rate is 16 kHz. Scores are dimensionless Euclidean distances in standardized mean log-spectrum space. Time unit is seconds. Rate quantities are counts divided by the split size.
- Empirical quantile q: the smallest calibration score `t` such that at least fraction q of the calibration scores are less than or equal to `t`. A clip alerts when its score is strictly greater than `t`. Hand check from the overlap probe: scores 1.0, 1.1, …, 1.9 at q = 0.90 give `t = 1.8` (9th of 10).
- A session id that appears in more than one split is rejected before fitting.
- All-zero, clipped, truncated, and unsupported WAV inputs return a named diagnostic and omit anomaly status.
- Unlabeled clips stay uncertain. They are not used as normals and they are not counted as confirmed anomalies.
- The proposed threshold is a different version from the source threshold.

## Measurable acceptance case

Seeded synthetic motor family, one source microphone domain and one shifted domain (extra background tone).

- Known tones produce finite features at 16 kHz, and a bin-centered tone peaks in the expected FFT bin.
- The source 0.90 threshold flags target normals more often than the reviewed target 0.90 threshold on the same held-out normal clips.
- Held-out anomaly recall on the strong-anomaly fixture does not fall by more than 0.25 absolute. The report includes both rates, both counts, and the split sizes.
- A second fixture, whose labeled anomalies sit inside the shifted-normal score range, must leave the proposal unsuccessful because recall worsens by more than 0.25.
- Exact false-alert and recall counts for the seeded showcase are pinned in the fixture oracle once the WAV generator is frozen. Those integers, not a re-fit, are the regression gate.

Predeclared showcase bounds, before the WAV scores exist:

- Target-calibration size at least 8 approved normals.
- Recall-drop tolerance 0.25 absolute.
- Source-threshold target-normal false alerts strictly greater than proposed-threshold target-normal false alerts.
- Success-fixture recall drop less than or equal to 0.25.
- Failure-fixture proposal not accepted.

## Integration

No existing tool met this acceptance case under a license this repository can carry. Build the package here.
