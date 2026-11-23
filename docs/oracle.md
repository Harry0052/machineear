# Oracle

The seeded numbers below are the acceptance pin. `src/machineear/fixtures/oracle.json` is the machine-readable copy. Counts come from the generated WAV fixtures, not from refitting a threshold by hand on the test labels.

## Hand checks

Empirical quantile. Ordered scores `1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9`. For q = 0.90 and n = 10, the rule keeps the smallest score that covers at least 9 observations. That score is `1.8`. A clip alerts only when its score is strictly greater than the threshold.

FFT bin. For a 125 Hz tone, `n_fft = 512`, and sample rate 16000 Hz:

`k = 125 * 512 / 16000 = 4`

The mean log-spectrum of a 1.0 s tone at that frequency peaks in bin 4.

## Showcase, domain mic-b

Split sizes: source_fit 8, source_validation 4, source_calibration 10, target_calibration 10, target_test 18 (10 normal, 8 anomaly).

| Threshold | False alerts on target normals | Anomaly recall |
| --- | --- | --- |
| Source `source-empirical-q90-v1` | 10/10 | 8/8 |
| Proposed `target-empirical-q90-v1`, accepted | 2/10 | 8/8 |

Change in false-alert rate: −0.80. Change in anomaly recall: 0.00. Recall drop 0.00, inside the 0.25 tolerance. Status `ok`.

## Recall-loss fixture

Same split sizes. The labeled anomalies are a 40 ms, amplitude 0.004 harmonic inside the shifted background, so their scores stay near the target normals.

| Threshold | False alerts on target normals | Anomaly recall |
| --- | --- | --- |
| Source | 10/10 | 8/8 |
| Proposed, not accepted | 1/10 | 4/8 |

Recall drop 0.50, which is above 0.25. Status `calibration_unsuccessful`. Anomaly status is still reported so the cost is visible.

## Other states

- `insufficient`: three approved target normals. No proposed threshold. Status `insufficient_normals`.
- `invalid`: `all_zero`, `clipped`, `truncated`, `unsupported_channels`, `unsupported_sample_rate`. Anomaly status omitted.
- `overlap`: one session in two splits, rejected before fitting.
- `missing_metadata`: empty `microphone_domain`. Anomaly status omitted.
