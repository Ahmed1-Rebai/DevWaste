# DevWaste AI - Task List

## Done (MVP)

- [x] Test suite passes: `pytest` (105 unit tests, ~2 s, no Docker/GPU/network)
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
- [x] No secrets in code; no TODO/debug leftovers

## Next

- [ ] Commit the project (repository has no commits yet)
- [ ] Add a LICENSE file (pyproject declares MIT)
- [ ] Evaluate on real labelled incidents (e.g. run stress workloads in unprivileged containers and label their time ranges)
- [ ] Per-feature reconstruction error in reports (which metric drove the flag)
- [ ] Tune/replace the rolling baseline (untuned, ~75% window FPR)
- [ ] More independent synthetic runs per scenario for tighter estimates
- [ ] Online mode: collect + score continuously against a trained model
- [ ] Verify GitHub Actions run on the remote once pushed
