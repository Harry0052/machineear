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

