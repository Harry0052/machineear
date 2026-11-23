# Overlap review

Access date for every source below: 2026-11-23.

Decision: **proceed** with a separate MachineEar repository.

MachineEar is a frozen-score threshold audit for one motor family. It measures how a microphone or background change moves alert counts, and it accepts a new quantile only when held-out anomaly recall stays inside a predeclared tolerance. The tools checked here train or replace a detector, rank clips by AUC, or flag events inside one recording. None of them records the acceptance metric on disjoint source-fit, source-calibration, target-calibration, and target-test sessions, and none rejects the new threshold when recall worsens past that tolerance.

The shared fixture, the expected artifact, and the metric were held constant for the threshold rules:

- Input: one dimensionless frozen-score table. Source calibration `src-cal` has 10 scores. Target calibration `tgt-cal` has 10 scores. Target test normals `tgt-test` has 10 scores. Target anomalies have 8 scores. A second anomaly list puts four of those eight inside the shifted normal range. Sessions are disjoint.
- Artifact: target-normal false-alert count and rate, target anomaly-recall count and rate, the change in both rates, and the split sizes.
- Alert rule: score strictly greater than the threshold.
- Tolerance used in the probe: absolute recall drop of 0.10.

Observed probe: `docs/research/p01_shared_fixture_probe.py`. Transcript: `docs/research/p01-probe-output.txt`. Command: `python3 docs/research/p01_shared_fixture_probe.py` (audio-anomaly 0.1.0 was installed for that run only).

## Capability matrix

| Capability | DCASE 2023 Task 2 | NTT / dinhoitt autoencoder baseline | MERL score normalization | audio-anomaly 0.1.0 | anomaly-watcher 0.2.0 | MachineEar MVP |
| --- | --- | --- | --- | --- | --- | --- |
| Industrial machine audio, normal vs anomaly | Yes, task definition | Yes, trains an autoencoder | Yes, post-processes ASD scores | Yes, in-clip events | No, tabular or MQTT sensors | Yes, synthetic motor tones |
| Source and target domain shift | Yes | Yes, inside the trained detector | Yes, for AUC | A reference recording, not sessions | No | Yes, microphone domain on the manifest |
| Frozen scorer, then a separate threshold audit | Threshold is part of the submission; ranking uses AUC | Gamma 90th percentile of training scores, no before/after recall gate | Normalizes scores; evaluates AUC | Sensitivity is a fixed score cut | Quantile mode on sensor windows | Yes |
| Disjoint fit, calibration, and test sessions, overlap rejected | Protocol splits exist; this check is not a product | Not a user-facing rejection | Not found | Not found | Not found | Yes |
| Operator-reviewed target normals required before a new threshold | Ten target-normal training clips exist in the dataset | Uses training normals | No | No | Calibration CSV, not audio review | Yes |
| Report change in target-normal false alerts and anomaly recall, with counts | Precision, recall, and F1 are published from submitted decisions; the ranking metric is AUC and pAUC | Summary CSV of AUC, pAUC, precision, recall, F1 | Official or simple AUC | Per-clip event list | Window scores | Yes, this is the acceptance metric |
| Refuse the new threshold when held-out recall drop exceeds a tolerance | Not the task rule | Not found | Not found | Not found | Not found | Yes |
| Runs on synthetic WAV without a non-commercial dataset | Dataset is separate | Requires the challenge downloads | Expects DCASE-style embeddings | Yes | Bundled sensor CSV | Yes |
| License compatible with this MIT repository | Task text is public; development audio is stated CC BY-NC-SA 4.0 | `LICENSEv2.1.pdf`, SPDX `NOASSERTION` | AGPL-3.0 | MIT | AGPL-3.0-or-later | MIT |

## Probes

### DCASE 2023 Task 2 first-shot anomalous sound detection

- Primary URL: https://dcase.community/challenge2023/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring
- Version: DCASE 2023 Challenge Task 2. The page states the challenge has ended. Baseline repository linked from the task is https://github.com/nttcslab/dcase2023_task2_baseline_ae (see the next row for the commit).
- Code license: the task page is public documentation. It is not a code release.
- Dataset terms: development audio is described by the organizers as created by Hitachi and NTT. Zenodo record 7882613 states CC BY-NC-SA 4.0 in the condition-of-use text. The Zenodo license widget also displays Creative Commons Attribution 4.0. Those two statements disagree. MachineEar will not import the audio. https://zenodo.org/records/7882613
- Feature evidence, from the task page: each development section has 990 source-normal training clips, 10 target-normal training clips, and 100 normal plus 100 anomalous test clips. Labels include machine type, section, normal/anomaly, and domain attributes. A clip is anomalous when its score exceeds a threshold. Official ranking uses AUC and pAUC and does not require a chosen threshold. Participants must still submit decisions. The baseline sets that decision threshold at the 90th percentile of a gamma distribution fitted to training anomaly scores.
- What already matches: normal/anomaly labels, source versus target domain, and a frozen operating threshold on normal scores.
- Evidence class: documentation for the task rules. The same acceptance metric was then measured by reimplementing only the published gamma-90 decision on the shared score table (observed run of the rule, not of the dataset or the trainer).
- Observed result: gamma 90th percentile of the 10 source-calibration scores was approximately 1.848759 (shape 22.936364, scale 0.063218, Wilson–Hilferty). Target-normal false alerts were 10/10 and anomaly recall was 8/8. There is no second threshold, so the change in both rates is zero. The rule does not consult target calibration, session overlap, or a recall tolerance.

### DCASE Task 2 autoencoder baseline

- Primary URL: https://github.com/dinhoitt/DCASE
- The playbook names this repository. It is a public copy of the NTT baseline scripts (`train.py`, `baseline.yaml`, `01_train_2026t2.sh`, `LICENSEv2.1.pdf`) plus extra experiment files (`SincNet-master`, `envelope.py`, `meldataset.py`, `chat.md`). The maintained upstream is https://github.com/nttcslab/dcase2023_task2_baseline_ae
- Version / commit: `dinhoitt/DCASE` at `3cccafd65134cedce68bf83e2ed10f6eb3315079` (2026-05-14, message `fixed medasd`). Upstream release v5.3.0 is commit `f44242ec1f78f6cc34f53f43fb88be1ce5d13d47` (2026-07-17).
- Code license: both repositories are GitHub license `Other` / SPDX `NOASSERTION`. The 60-byte `LICENSE` file says the license is in `LICENSEv2.1.pdf`. That is not MIT. MachineEar will not vendor this code.
- Dataset terms: training scripts download DCASE challenge archives. Same non-commercial concern as above. Not imported.
- Feature evidence: README describes training an autoencoder and writing anomaly scores plus AUC, pAUC, precision, recall, and F1. Simple-autoencoder mode scores reconstruction error. Threshold text on the task page is the gamma 90th percentile. Nothing in the repository listing is a session-overlap rejection or a recall-gated recalibration report for a scorer the user already froze.
- What already matches: log-spectral features, a normal-only reference, and a percentile threshold.
- Evidence class: documentation and repository listing. The trainer was not executed. It needs the challenge dataset, PyTorch, and a non-MIT license. The decision rule was included in the observed score-table probe above.
- Effort to decide recalibration from per-session evidence: the baseline does not emit that decision. A reviewer would have to post-process score CSVs by hand. That gap is the MachineEar workflow.

Zhou and colleagues describe a related rule in a DCASE 2026 technical report, not as a reusable tool: threshold = max(90th percentile of source-train scores, 50th percentile of target-train normal scores), using only train-normal data. Primary URL: https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Zhou_15_t2.pdf

Observed on the same score table: threshold = max(1.8, 2.45) = 2.45. On the separated anomaly list, false alerts moved from 10/10 to 5/10 and recall stayed 8/8. On the overlapping anomaly list, recall moved from 8/8 to 6/8 (drop 0.25) and the rule still produced a threshold. It has no recall gate and no session-overlap check. Evidence class: documentation for the formula, observed run of that formula on the shared table. The authors' detector was not executed.

### MERL anomaly-score normalization

- Primary URL: https://github.com/merlresearch/anomaly-score-normalization
- Version / commit: release 1.1.0, commit `b9998bd73d3ab376eb373f4ce79876eb4f586986` (2025-11-17).
- Code license: AGPL-3.0. Incompatible with embedding into this MIT repository.
- Dataset terms: README options describe DCASE official versus simple AUC on source and target embeddings. No dataset was downloaded.
- Feature evidence: local-density normalization (`ratio`, `difference`, `LOF`) and optional source k-means. Metrics are AUC, not a change in false-alert rate against a recall tolerance.
- What already matches: explicit source versus target domains for anomalous sound scores.
- Evidence class: documentation. Unresolved as an executed run. AGPL plus the expectation of external embeddings made a local run a poor fit for a license-compatible comparison, and the published metric is AUC rather than the acceptance artifact.

### audio-anomaly 0.1.0 (closest PyPI package)

- Primary URL: https://pypi.org/project/audio-anomaly/0.1.0/
- Version: 0.1.0, uploaded 2026-10-15. Requires Python >= 3.9 and NumPy >= 1.23.
- Code license: MIT.
- Dataset terms: none. The probe used three one-second synthetic tones at 16 kHz generated in the probe process: motor harmonics 120/240/360 Hz, the same motor plus a 900 Hz background, and the motor plus a 540 Hz segment from 0.40 s to 0.70 s.
- Feature evidence, observed: with the source tone as `reference`, the source clip had 0 anomalies (max score 0.000), the background-shifted clip had 1 `tonal` event (max score 50.256), and the harmonic-fault clip had 1 `burst` event (max score 37.681).
- What already matches: a reference recording, log-frequency energy, and a deterministic WAV report. A background change and a mechanical-sounding change can both raise a flag.
- What does not match: there is no manifest of sessions, no frozen centroid left untouched while only the threshold moves, no split-size report, and no comparison of false-alert rate against anomaly recall. One event on the shifted normal clip would be a false alert if treated as a clip decision, but the tool does not say that, and it does not refuse a new cut when recall falls.
- Evidence class: observed run.

### anomaly-watcher 0.2.0 (registry neighbor, not a fit)

- Primary URL: https://pypi.org/project/anomaly-watcher/0.2.0/
- Version: 0.2.0. Requires Python >= 3.13. License AGPL-3.0-or-later.
- Feature evidence, from the package description: Isolation Forest on timestamped sensor CSV or MQTT, with `--threshold-mode max|quantile|std`. Bundled fixtures are room climate, not WAV. No machine-audio domain shift.
- Evidence class: documentation. Not run. It does not accept the shared audio score table or the WAV fixture, so it cannot produce the acceptance artifact.

### MIMII DG

- Paper: Kota Dohi and colleagues, DCASE 2022 Workshop, "MIMII DG: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection for Domain Generalization Task". https://dcase.community/documents/workshop2022/proceedings/DCASE2022Workshop_Dohi_62.pdf
- Data URL named by the paper: https://zenodo.org/record/6529888
- This is a dataset and a benchmark statement, not a calibration-audit tool. Domain shifts include speed, load, background noise, and microphone location. MachineEar does not import it. An optional later import would need a recorded license check first. The paper's baselines are detectors, and the reported outcome is detection performance under shift, not the price of a reviewed threshold change.

## Observed gap on the acceptance metric

Before state, frozen empirical 0.90 quantile of source calibration (threshold 1.8):

- target-normal false alerts 10/10
- target anomaly recall 8/8

After state, empirical 0.90 quantile of reviewed target calibration (threshold 2.8):

- target-normal false alerts 2/10 (change −0.80)
- target anomaly recall 8/8 (change 0.00)
- recall drop 0.00, inside tolerance 0.10

Failure anomaly list, same thresholds:

- recall 8/8 becomes 4/8 (drop 0.50), so a recall-gated proposal is refused
- Zhou's rule still emits threshold 2.45 and recall falls to 6/8 with no refusal

That is the concrete hypothesis worth testing on real waveforms: a frozen centroid score plus a reviewed quantile can show the false-alert change and can refuse a recalibration whose recall cost is too high. Existing workflows do not do both.

## Recommendation

Do not integrate MachineEar into `nttcslab/dcase2023_task2_baseline_ae` or `dinhoitt/DCASE`, and do not relicense AGPL normalization or sensor code into this repository. Those projects already cover detector training, AUC, and in-clip event finding. They do not cover the acceptance case, and the detector stacks pull in a non-MIT license plus a dataset whose stated terms are non-commercial.

Proceed with a small MIT Python package. Generate the default WAV fixtures in-repo. Keep any DCASE or MIMII import behind an explicit, license-reviewed path that is off by default.

## Uncertainties

- The Zenodo widget for record 7882613 does not match the organizers' CC BY-NC-SA 4.0 sentence. Treated as non-commercial until a human resolves it.
- The NTT `LICENSEv2.1.pdf` text was not copied into this repository. SPDX remains `NOASSERTION`.
- The gamma 90th percentile in the probe is a Wilson–Hilferty approximation of the published rule, not `scipy.stats.gamma` and not the trained autoencoder. The numerical value (1.848759) sat close to the empirical 0.90 quantile (1.8) and produced the same 10/10 and 8/8 counts.
- The audio-anomaly run used one clip per condition, not the 10/10/8 split sizes. It still shows the missing artifact: events, not a session recall gate.
- MERL and anomaly-watcher were not executed.
