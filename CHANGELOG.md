# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The package version is
still `0.1.0`; no release has been tagged yet.

## [Unreleased]

### Added
- Demo recording and screenshots of `devwaste scan`, `train`, and `monitor`
  on a real local Docker stack (`docs/media/`).
- `subtle-oscillation` synthetic scenario.
- Online adaptive threshold recalibration (`monitor --adaptive-rate`).
- Extreme Value Theory (Peaks-Over-Threshold) threshold calibration
  (`train --threshold-method evt`).
- Self-contained offline HTML reports (`devwaste report`).
- Real-time monitoring of running containers (`devwaste monitor`).
- Recovery margin and persistence filtering in evaluation
  (`evaluate --recovery-margin`, `--consecutive`).

### Fixed
- Rolling baseline no longer produces NaN scores on zero-variance metrics.

## Initial MVP (2026-10-17 to 2026-10-18, untagged)

Package version `0.1.0`. Not tagged and not published to PyPI.

### Added
- Typer CLI: `scan`, `collect`, `simulate`, `train`, `evaluate`, `cleanup --dry-run`.
- Read-only Docker metrics collector with SQLite storage and CSV export.
- Reproducible synthetic scenarios and leakage-safe preprocessing.
- PyTorch LSTM Autoencoder with validation-quantile threshold.
- Baselines (z-score, rolling statistics, fixed threshold) and per-feature
  error attribution.
- Synthetic and real-container incident experiments (`docs/results.md`).
- GitHub Actions CI (Ruff and pytest on CPU), MIT license.

### Fixed
- Per-feature scale floor so near-constant training metrics do not
  exaggerate small shifts.
