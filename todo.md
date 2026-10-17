# DevWaste AI - Task List

## Done (MVP)

- [x] Test suite passes: `pytest` (110 unit tests, ~2 s, no Docker/GPU/network)
- [x] Ruff lint + format clean
- [x] End-to-end experiment script: `scripts/run_experiment.py`
- [x] README rewritten against the actual CLI (commands verified)
- [x] docs/architecture.md with Mermaid diagrams (system, model, data flow)
- [x] Synthetic training experiment run; results in docs/results.md
- [x] CONTRIBUTING.md
- [x] Tests for CLI, SQLite storage, training pipeline, evaluation, baselines, cleanup safety, leakage
- [x] Real Docker collection smoke-tested (scan, collect, cleanup --dry-run)
- [x] Artifact serialization verified (round-trip test, weights_only load)
- [x] Non-editable install + CLI smoke test in CI
- [x] Committed (local); MIT LICENSE added
- [x] Per-feature error attribution in reports
- [x] Rolling baseline calibrated like the LSTM
- [x] Synthetic experiment over 3 seeds x 3 independent test runs
- [x] Controlled incident experiment in a real container (scripts/real_incident_experiment.py)
- [x] Scale floor for near-constant features (found by the real-container run)
- [x] No secrets in code; no TODO/debug leftovers

## Next

- [ ] Push to GitHub: needs a token with the `workflow` scope (current PAT is rejected for .github/workflows/ci.yml)
- [ ] Reduce LSTM false positives during post-incident recovery (train on data containing recoveries, or add a recovery margin)
- [ ] Real incident experiment: more runs, subtle shape-only incidents (where the LSTM should help), network enabled
- [ ] Rolling baseline is unstable on short real data (run 2 flagged nothing)
- [ ] Online mode: collect + score continuously against a trained model
- [ ] Verify GitHub Actions run on the remote once pushed
