# Grokbot progress

## P01 — Research overlap and confirm the MVP

- **Phase and status:** P01 PASS
- **Files changed:** `docs/overlap.md`, `docs/project-contract.md`, `docs/grokbot-progress.md`, `docs/research/p01_shared_fixture_probe.py`, `docs/research/p01-probe-output.txt`
- **Commands and results:**
  - Read the repository. It contained `README.md` (`# machineear`) and `LICENSE` (MIT, copyright 2026 Harry Malhi). Remote `origin` is `https://github.com/Harry0052/machineear`, branch `main`, commit `31d7642`.
  - Searched the web and GitHub on 2026-11-23 for industrial acoustic anomaly calibration, DCASE threshold drift, MIMII DG domain shift, and session-split calibration. Inspected the DCASE 2023 Task 2 page, Zenodo record 7882613, `nttcslab/dcase2023_task2_baseline_ae` commit `f44242ec1f78f6cc34f53f43fb88be1ce5d13d47` (license Other / `LICENSEv2.1.pdf`), `dinhoitt/DCASE` commit `3cccafd65134cedce68bf83e2ed10f6eb3315079` (same license file; `LICENSE` is 60 bytes pointing at the PDF), `merlresearch/anomaly-score-normalization` release 1.1.0 commit `b9998bd73d3ab376eb373f4ce79876eb4f586986` (AGPL-3.0), PyPI `audio-anomaly` 0.1.0 (MIT), and PyPI `anomaly-watcher` 0.2.0 (AGPL-3.0-or-later).
  - `python3 --version` → Python 3.12.3. NumPy 1.26.4 was present. SciPy was not installed at probe time.
  - `python3 -m pip install --user audio-anomaly==0.1.0` succeeded. The package is not a project dependency.
  - Observed shared-fixture probe, recorded in `docs/research/p01-probe-output.txt`. Frozen source empirical 0.90 threshold 1.8: target-normal false alerts 10/10, anomaly recall 8/8. Reviewed target 0.90 threshold 2.8: false alerts 2/10 (change −0.80), recall 8/8 (change 0.00). Failure anomaly list: recall 8/8 to 4/8, drop 0.50, proposal refused. Documented DCASE gamma-90 approximation 1.848759 stayed at 10/10 and 8/8 with no second threshold. Zhou max(p90, p50) = 2.45 still emitted a threshold when recall fell to 6/8. audio-anomaly 0.1.0 on three synthetic tones flagged the background shift as `tonal` (max score 50.256) and the harmonic insert as `burst` (max score 37.681) and did not report session rates.
- **Artifact paths:** `docs/overlap.md`, `docs/project-contract.md`, `docs/research/p01-probe-output.txt`
- **Open issues:** Zenodo 7882613 displays both a CC BY-NC-SA 4.0 condition of use and a CC BY 4.0 widget. Challenge audio stays out of the repo. The gamma figure is a Wilson–Hilferty approximation, not a trained autoencoder. MERL and anomaly-watcher were not executed.
- **Decision:** proceed. No checked tool covers the acceptance case under the MIT license.
- **Next phase:** P02

## P02 — Create the repository and data contracts

- **Phase and status:** P02 PASS
- **Files changed:** package scaffold under `src/machineear/`, `scripts/generate_fixtures.py`, `src/machineear/fixtures/`, `tests/test_wav.py`, `tests/test_cli.py`, `pyproject.toml`, `requirements.txt`, `README.md`, `.gitignore`
- **Commands and results:**
  - `python3 scripts/generate_fixtures.py` wrote 128 synthetic WAV files, about 4.2 MB, under `src/machineear/fixtures/`.
  - `python -m pip install -e ".[test]"` installed NumPy 1.26.4, SciPy 1.14.1, scikit-learn 1.5.2, pytest, and ruff.
  - `machineear validate --manifest src/machineear/fixtures/showcase/manifest.csv` exited 0: `accepted 50 clips`.
  - `machineear validate --manifest src/machineear/fixtures/invalid/manifest.csv` exited 2 and named `all_zero`, `clipped`, `truncated`, `unsupported_channels`, and `unsupported_sample_rate`. Anomaly status was omitted.
  - A manifest with `wav_path` `../tone.wav` exited 2 with `path_escape`.
- **Artifact paths:** `src/machineear/fixtures/fixture-manifest.json`, `src/machineear/fixtures/showcase/manifest.csv`, `src/machineear/fixtures/invalid/manifest.csv`
- **Open issues:** none for this gate.
- **Next phase:** P03

## P03 — Build the working core

- **Phase and status:** P03 PASS
- **Files changed:** `src/machineear/features.py`, `model.py`, `calibrate.py`, `pipeline.py`, `src/machineear/fixtures/oracle.json`, `docs/oracle.md`, `tests/test_features.py`, `tests/test_calibration.py`, `tests/test_acceptance.py`
- **Commands and results:**
  - `python -m pytest` — 20 passed after the later safeguard tests were added; the acceptance file alone pins the oracle.
  - Showcase, domain `mic-b`: source threshold 21.472203, false alerts 10/10, anomaly recall 8/8. Proposed threshold 436.177441, false alerts 2/10, recall 8/8, recall drop 0.00, status `ok`.
  - Recall-loss fixture: false alerts 10/10 to 1/10, recall 8/8 to 4/8, recall drop 0.50, proposal `accepted=false`, status `calibration_unsuccessful`.
  - Hand quantile 1.0…1.9 at 0.90 is 1.8. A 125 Hz tone peaks in FFT bin 4.
  - Reference clip ids are the source-fit clips only.
- **Artifact paths:** `src/machineear/fixtures/oracle.json`, `docs/oracle.md`
- **Open issues:** none for this gate. Counts are for the synthetic fixture.
- **Next phase:** P04

## P04 — Ship the user workflow and reports

- **Phase and status:** P04 PASS
- **Files changed:** `src/machineear/cli.py`, `src/machineear/report.py`, README demo section
- **Commands and results:**
  - `machineear demo --output /tmp/machineear-demo` exited 0 in 0.038 s and wrote `summary.html`, `result.json`, `scores.csv`, and `summary.txt`.
  - `machineear demo --json` put JSON on stdout and the human summary on stderr.
  - `machineear demo --case invalid` exited 2. The report status is `rejected_input` and names the silent file.
  - `machineear --help`, `validate --help`, and `demo --help` exit 0.
  - Browser check of `http://127.0.0.1:8765/summary.html`: the "Microphone domain" combobox defaults to mic-b and the live status reads "Source threshold false alerts 10/10. Proposed false alerts 2/10." Selecting mic-a announces "No labeled target-test clips in this domain." Focus outline is solid `rgb(11, 58, 130)`. At 390 px width, `scrollWidth` equals `clientWidth`.
  - Clip `showcase-target_test-normal-00` is in the changed-classification list with score 434.939006, source alert true, proposed alert false, and its WAV path is on the CSV row.
- **Artifact paths:** `demo-out/summary.html` (local run under `/tmp/machineear-demo`), `src/machineear/report.py`
- **Open issues:** favicon request 404 when the file is served by a static server. The report has no favicon. Not a product failure.
- **Next phase:** P05

## P05 — Verify correctness, security, and performance

- **Phase and status:** P05 PASS
- **Files changed:** `tests/test_safeguards.py`, `docs/validation.md`, pytest pin moved to 9.0.3
- **Commands and results:**
  - `python -m compileall -q src tests scripts` passed.
  - `ruff check src tests scripts` passed.
  - `pytest`: 20 passed.
  - `pip-audit` 2.9.0 on numpy==1.26.4, scipy==1.14.1, scikit-learn==1.5.2, pytest==9.0.3, ruff==0.8.4: no known vulnerabilities. pytest 8.3.4 had PYSEC-2026-1845 and was not kept.
  - Oversized WAV (1 MiB + 1 byte) returns `oversized` and is not scored.
  - `label=unknown` is uncertain, has no alert bit, and is outside both rate denominators.
  - Demo files written are only `result.json`, `scores.csv`, `summary.html`, `summary.txt`.
  - Performance budget is 10 seconds. Observed demo time 0.038 s. The safeguard test fails a run that reaches 10 s.
  - Full command log, host, and residual limits are in `docs/validation.md`.
- **Artifact paths:** `docs/validation.md`
- **Open issues:** synthetic fixture only. No external dataset was measured.
- **Next phase:** P06

## P06 — Prepare a reproducible open-source release

- **Phase and status:** P06 PASS
- **Files changed:** `README.md`, `SECURITY.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `requirements.txt`, `.github/workflows/ci.yml`
- **Commands and results:**
  - Clean copy at `/tmp/machineear-clean` (no `.venv`, no `.git`): `pip install -e ".[test]"`, `compileall`, `ruff check`, `pytest` (20 passed), `machineear demo`. `result.json` metrics, thresholds, and recall drop matched `oracle.json` showcase.
  - `python -m build` produced `machineear-0.1.0-py3-none-any.whl` and the sdist. A fresh venv `pip install` of the wheel ran `machineear demo` and matched the same oracle metrics, so the fixtures are inside the package.
  - A search of the clean tree for `AKIA`, private-key blocks, `x-access-token`, and `api_key` found no matches.
  - Workflow uses `contents: read` plus `actions: write` for the demo artifact. Actions are pinned to commit SHAs: checkout `11d5960a326750d5838078e36cf38b85af677262`, setup-python `a26af69be951a213d495a4c3e4e4022e16d87065`, upload-artifact `ea165f8d65b6e75b540449e92b4886f43607fa02`. No secrets.
- **Artifact paths:** `.github/workflows/ci.yml`, `README.md`, `SECURITY.md`
- **Open issues:** hosted CI is not green until P07. The local clean-room run is the P06 evidence.
- **Next phase:** P07

