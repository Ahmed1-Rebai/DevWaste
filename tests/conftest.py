"""Shared fixtures. Nothing here needs Docker, a GPU, or network access."""

from pathlib import Path

import pytest

from devwaste.data.simulator import SyntheticDataGenerator
from devwaste.models.train import train_model


def write_scenario(path: Path, scenario: str, seed: int, n_samples: int = 400) -> Path:
    df = SyntheticDataGenerator(seed=seed).generate_scenario(
        scenario, n_samples=n_samples
    )
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def normal_csv(data_dir) -> Path:
    return write_scenario(data_dir / "normal.csv", "normal", seed=1)


@pytest.fixture(scope="session")
def cpu_spike_csv(data_dir) -> Path:
    return write_scenario(data_dir / "cpu_spike.csv", "cpu-spike", seed=2)


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory, normal_csv) -> Path:
    """A tiny model trained for a few epochs, shared across tests."""
    output_dir = tmp_path_factory.mktemp("artifacts")
    train_model(
        data_path=normal_csv,
        output_dir=output_dir,
        window_size=8,
        hidden_dim=8,
        epochs=3,
        batch_size=64,
        device="cpu",
        log=None,
    )
    return output_dir
