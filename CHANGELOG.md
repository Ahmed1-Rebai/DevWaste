# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Version numbers follow
[Semantic Versioning](https://semver.org/). No version has been tagged yet;
everything below is unreleased work on `0.1.0`.

## [Unreleased]

### Added

- Real demo recording, screenshots and asciinema cast of `scan`, `train` and `monitor` on the FlakeRadar Docker stack (`docs/media/`).
- `subtle-oscillation` temporal anomaly scenario in the benchmark suite.
- Online adaptive threshold recalibration in `devwaste monitor` (`--adaptive-rate`).
- Extreme Value Theory (peaks-over-threshold) threshold calibration (`train --threshold-method evt`).
- `devwaste report`: self-contained offline HTML diagnostic report.
- `devwaste monitor`: real-time scoring of running containers with per-feature attribution and persistence filtering (`--consecutive`).
- Post-incident recovery margin and persistence filtering in evaluation.
- Controlled-incident experiment in a real container (`scripts/real_incident_experiment.py`) and its results.
- Per-feature error attribution; calibrated rolling baseline; multiple test runs per scenario.
- MIT license, CI workflow (Python 3.11–3.13, CPU-only PyTorch) and Docker development environment.
- Synthetic experiment script and documented results (`docs/results.md`).
- Unit test suite that needs no Docker, GPU or network.
- Core package: Docker collection, SQLite storage, leakage-safe data pipeline, LSTM autoencoder, baselines, evaluation and CLI.

### Fixed

- Per-feature scale floor so near-constant training features no longer exaggerate small shifts.
- Rolling baseline scale floor for zero-variance metrics.
