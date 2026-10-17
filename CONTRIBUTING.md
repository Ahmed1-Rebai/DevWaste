# Contributing to DevWaste AI

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional, smaller
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff check .
ruff format --check .
pytest
```

CI runs the same commands on Python 3.11–3.13 using CPU-only PyTorch.

## Ground rules

- **No LLMs or external AI APIs.** The detector is a local PyTorch model.
- **Diagnostic only.** Code must never stop, remove, or prune Docker resources. Cleanup output is advice for a human.
- **Unit tests need no Docker, GPU, or network.** Mock the Docker client: `DockerCollector(client=...)` and `CleanupAdvisor(client)` accept test doubles. Tests that need a real daemon get `@pytest.mark.integration` and are deselected by default.
- **No data leakage.** Split chronologically per series before windowing, and fit preprocessing on training data only.
- **Honest results.** Report numbers you actually obtained with `scripts/run_experiment.py` or the CLI, include baselines, and state clearly when data is synthetic.
- Use type hints on public functions and keep modules small and focused.
