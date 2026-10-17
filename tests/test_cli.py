"""Tests for the Typer CLI (Docker mocked; training kept tiny)."""

import json
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from devwaste.cli import app
from tests.test_collectors import STATS, make_container

runner = CliRunner()


@pytest.fixture
def mock_collector():
    """Patch DockerCollector so CLI commands never touch a real daemon."""
    from devwaste.collectors.docker_collector import DockerCollector

    container = make_container()
    container.stats.return_value = STATS
    client = Mock()
    client.containers.list.return_value = [container]
    client.images.list.return_value = []
    client.volumes.list.return_value = []
    client.networks.list.return_value = []

    with patch(
        "devwaste.collectors.docker_collector.DockerCollector",
        side_effect=lambda: DockerCollector(client=client),
    ):
        yield client


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("scan", "collect", "simulate", "train", "evaluate", "cleanup"):
        assert command in result.output


def test_version():
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_simulate(tmp_path):
    output = tmp_path / "sub" / "data.csv"
    result = runner.invoke(
        app,
        [
            "simulate",
            "--scenario",
            "cpu-spike",
            "--output",
            str(output),
            "--n-samples",
            "50",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(pd.read_csv(output)) == 50


def test_simulate_invalid_scenario(tmp_path):
    result = runner.invoke(
        app, ["simulate", "--scenario", "nope", "--output", str(tmp_path / "x.csv")]
    )

    assert result.exit_code == 1


def test_train_and_evaluate(tmp_path, normal_csv, cpu_spike_csv):
    model_dir = tmp_path / "model"
    result = runner.invoke(
        app,
        [
            "train",
            "--data",
            str(normal_csv),
            "--output-dir",
            str(model_dir),
            "--epochs",
            "2",
            "--window-size",
            "8",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Training complete" in result.output

    report = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--data",
            str(cpu_spike_csv),
            "--model",
            str(model_dir),
            "--output",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "LSTM Autoencoder" in result.output
    assert "baseline: zscore" in result.output
    assert json.loads(report.read_text())["labelled"] is True

    result = runner.invoke(
        app,
        ["evaluate", "--data", str(cpu_spike_csv), "--model", str(model_dir), "--json"],
    )
    assert result.exit_code == 0
    assert "model_metrics" in json.loads(result.output)


def test_train_missing_file(tmp_path):
    result = runner.invoke(app, ["train", "--data", str(tmp_path / "missing.csv")])

    assert result.exit_code == 1


def test_evaluate_missing_model(tmp_path, normal_csv):
    result = runner.invoke(
        app, ["evaluate", "--data", str(normal_csv), "--model", str(tmp_path / "nope")]
    )

    assert result.exit_code == 1


def test_scan_json(mock_collector):
    result = runner.invoke(app, ["scan", "--json"])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["containers"][0]["name"] == "web"
    assert data["containers"][0]["metrics"]["cpu_percent"] == pytest.approx(100.0)


def test_scan_table(mock_collector):
    result = runner.invoke(app, ["scan", "--no-stats"])

    assert result.exit_code == 0, result.output
    assert "web" in result.output


def test_scan_without_docker():
    with patch(
        "devwaste.collectors.docker_collector.DockerCollector",
        side_effect=RuntimeError("Cannot connect to Docker daemon"),
    ):
        result = runner.invoke(app, ["scan"])

    assert result.exit_code == 1


def test_collect_exports_deltas(tmp_path, mock_collector):
    output = tmp_path / "collected.csv"
    with patch("devwaste.cli.time.sleep"):
        result = runner.invoke(
            app,
            [
                "collect",
                "--interval",
                "1",
                "--duration",
                "3",
                "--db",
                str(tmp_path / "m.db"),
                "--output",
                str(output),
            ],
        )

    assert result.exit_code == 0, result.output
    df = pd.read_csv(output)
    assert len(df) == 2  # 3 rounds, first dropped by delta conversion
    assert (df["network_rx_bytes"] == 0).all()  # constant counters -> zero deltas


def test_cleanup_dry_run(mock_collector):
    result = runner.invoke(app, ["cleanup", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "No actions were performed" in result.output


def test_cleanup_refuses_execution(mock_collector):
    result = runner.invoke(app, ["cleanup", "--no-dry-run"])

    assert result.exit_code == 2
    mock_collector.containers.prune.assert_not_called()
