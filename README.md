# MachineEar

MachineEar checks whether a frozen anomaly score for one motor family still makes sense after the microphone or the background changes. It compares a source-calibration threshold with a reviewed target-calibration threshold and reports the change in false alerts and anomaly recall.

Anomaly here means unusual relative to the recorded source baseline. It is not a fault diagnosis, a live alarm, or a factory control.

The default clips are synthetic tones generated in this repository. Nothing is downloaded at install or at runtime. There is no separate hardware demo.

## Setup

Python 3.11 or newer. This repository was tested on CPython 3.12.3, Linux x86_64. Pinned libraries: NumPy 1.26.4, SciPy 1.14.1, scikit-learn 1.5.2. Tests use pytest 9.0.3 and ruff 0.8.4.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

NumPy, SciPy, and scikit-learn are BSD-3-Clause. pytest and ruff are MIT. Those licenses are compatible with the MIT license in `LICENSE`.

## Demo

```bash
machineear demo --output demo-out
```

Expected result for version 0.1.0, microphone domain `mic-b`:

- Status `ok`. Anomaly status is computed.
- Source threshold `source-empirical-q90-v1`: target-normal false alerts **10/10**, anomaly recall **8/8**.
- Proposed threshold `target-empirical-q90-v1`: false alerts **2/10**, anomaly recall **8/8**. The proposal is accepted. Recall drop is 0.00, inside the 0.25 tolerance.
- Split sizes: source_fit 8, source_validation 4, source_calibration 10, target_calibration 10, target_test 18.

The human summary is printed to stdout. The files are:

- `demo-out/summary.html` — domain selector, thresholds, session counts, score histogram, alert rate, and the clips whose classification changed
- `demo-out/result.json` — machine-readable result
- `demo-out/scores.csv` — one row per clip, including the WAV path and SHA-256
- `demo-out/summary.txt` — the same summary as stdout

Add `--json` to print `result.json` on stdout and the human summary on stderr.

Other packaged cases:

```bash
machineear demo --output demo-out --case recall-loss
machineear demo --output demo-out --case insufficient
machineear demo --output demo-out --case missing
machineear demo --output demo-out --case overlap
machineear demo --output demo-out --case invalid
```

`recall-loss` exits 0 and marks the proposal unsuccessful because recall falls from 8/8 to 4/8. `insufficient` exits 0 and does not propose a threshold. `missing`, `overlap`, and `invalid` exit 2, omit anomaly status, and print the recovery step on stderr.

## Checks

```bash
machineear validate --manifest src/machineear/fixtures/showcase/manifest.csv
machineear validate --manifest src/machineear/fixtures/invalid/manifest.csv
python -m pytest
python -m ruff check src tests scripts
```

The showcase command exits 0. The invalid command exits 2 and names `all_zero`, `clipped`, `truncated`, `unsupported_channels`, and `unsupported_sample_rate`.

Help: `machineear --help`. Version: `machineear --version`.

## What the score is

Each clip becomes a mean log-spectrum at 16 kHz (FFT length 512, hop 256, Hann window). A scikit-learn `StandardScaler` is fit on source-fit normals only and then frozen. Per-bin scales are floored at the maximum of the 10th percentile and `1e-3`. The anomaly score is the Euclidean distance from the centroid of those standardized source-fit vectors. A clip alerts when its score is strictly greater than the threshold.

The source threshold is the empirical 0.90 quantile of the disjoint source-calibration normals: the smallest score `t` such that at least 90% of those scores are less than or equal to `t`. The proposed threshold uses the same rule on operator-approved target normals and has its own version, `target-empirical-q90-v1`. It is not accepted when held-out anomaly recall falls by more than 0.25.

Hand checks and the pinned counts are in `docs/oracle.md`.

## Fixtures

`python scripts/generate_fixtures.py` rewrites the WAV files from the seeds in `src/machineear/fixtures/fixture-manifest.json`. The audio is generated here. It is covered by the MIT license. Do not add workplace recordings or license-restricted datasets. See `CONTRIBUTING.md`.

## Limits

A new microphone and a real fault can move the same score. The report will not adopt a new threshold unless the target normals were approved and the held-out recall loss stays inside tolerance. Unlabeled clips stay uncertain. Sessions that appear in more than one split are rejected before fitting.

## Security

See `SECURITY.md`. Report vulnerabilities through a private GitHub advisory.

## License

MIT. See `LICENSE`. Changes are listed in `CHANGELOG.md`.
