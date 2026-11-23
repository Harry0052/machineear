# Security

MachineEar reads local WAV files and writes a report to a directory you choose. It does not phone home, and it does not take credentials.

## Reporting a vulnerability

Open a private GitHub security advisory on [Harry0052/machineear](https://github.com/Harry0052/machineear). Please do not file a public issue for an unfixed vulnerability.

Include the version or commit, the command you ran, and whether the problem is a crash, a path that writes outside the output directory, or a manifest that is treated as trusted code. Do not include workplace recordings or other private audio.

## What we consider in scope

- Reading a manifest or WAV that escapes the manifest directory.
- Writing a report outside the requested output directory.
- A crafted WAV that bypasses the size, duration, sample-rate, or channel limit and still receives an anomaly score.

Anomaly scores on synthetic or unusual-but-valid audio are a product result, not a vulnerability.
