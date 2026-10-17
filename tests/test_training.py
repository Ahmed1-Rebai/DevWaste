"""Tests for the training pipeline, artifacts, and evaluation."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from devwaste.data.dataset import chronological_split, load_dataset
from devwaste.models.artifacts import ModelArtifacts
from devwaste.models.score import evaluate_model
from devwaste.models.train import train_model


class TestTraining:
    def test_artifacts_saved(self, trained_model):
        assert (trained_model / "model.pt").exists()
        metadata = json.loads((trained_model / "metadata.json").read_text())

        assert metadata["window_size"] == 8
        assert metadata["threshold"] > 0
        assert metadata["baseline_state"]["zscore_threshold"] > 0
        assert len(metadata["training_config"]["train_loss_history"]) == 3

    def test_scaler_fitted_on_training_split_only(self, trained_model, normal_csv):
        """Leakage check: stored scaler mean equals the training split's mean."""
        artifacts = ModelArtifacts.load(trained_model)
        df = load_dataset(normal_csv)
        train_df, val_df = chronological_split(df, 0.2)

        expected = train_df["cpu_percent"].mean()
        stored = artifacts.feature_engineer_state["scaler_mean"][0]

        assert stored == pytest.approx(expected)
        assert stored != pytest.approx(df["cpu_percent"].mean())

    def test_anomalous_windows_excluded(self, tmp_path, cpu_spike_csv):
        results = train_model(
            cpu_spike_csv, tmp_path, window_size=8, epochs=1, device="cpu", log=None
        )

        assert results["n_excluded_anomalous_windows"] > 0

    def test_deterministic_with_seed(self, tmp_path, normal_csv):
        kwargs = dict(
            window_size=8, hidden_dim=8, epochs=2, device="cpu", log=None, seed=3
        )
        r1 = train_model(normal_csv, tmp_path / "a", **kwargs)
        r2 = train_model(normal_csv, tmp_path / "b", **kwargs)

        assert r1["threshold"] == pytest.approx(r2["threshold"])
        w1 = torch.load(tmp_path / "a" / "model.pt", weights_only=True)
        w2 = torch.load(tmp_path / "b" / "model.pt", weights_only=True)
        assert all(torch.equal(w1[k], w2[k]) for k in w1)

    def test_unlabelled_data_treated_as_normal(self, tmp_path, normal_csv):
        path = tmp_path / "unlabelled.csv"
        pd.read_csv(normal_csv).drop(columns=["label"]).to_csv(path, index=False)

        results = train_model(
            path, tmp_path / "m", window_size=8, epochs=1, device="cpu", log=None
        )

        assert results["n_excluded_anomalous_windows"] == 0

    def test_too_little_data(self, tmp_path, normal_csv):
        path = tmp_path / "tiny.csv"
        pd.read_csv(normal_csv).head(12).to_csv(path, index=False)

        with pytest.raises(ValueError, match="Validation split too short"):
            train_model(path, tmp_path / "m", window_size=8, epochs=1, log=None)


class TestArtifacts:
    def test_roundtrip_reproduces_scores(self, trained_model):
        artifacts = ModelArtifacts.load(trained_model)
        model = artifacts.build_model()
        x = torch.randn(4, artifacts.window_size, artifacts.input_dim)

        reloaded = ModelArtifacts.load(trained_model).build_model()

        assert torch.allclose(
            model.calculate_reconstruction_error(x),
            reloaded.calculate_reconstruction_error(x),
        )

    def test_missing_files(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ModelArtifacts.load(tmp_path)


class TestEvaluation:
    def test_labelled_evaluation(self, trained_model, cpu_spike_csv):
        results = evaluate_model(cpu_spike_csv, trained_model)

        assert results["labelled"]
        assert results["n_windows"] == 400 - 8 + 1
        assert 0 < results["anomaly_prevalence"] < 1
        assert set(results["baseline_metrics"]) == {"zscore", "threshold", "rolling"}
        for metrics in [
            results["model_metrics"],
            *results["baseline_metrics"].values(),
        ]:
            assert 0 <= metrics["precision"] <= 1
            assert 0 <= metrics["recall"] <= 1
        assert results["model_metrics"]["auc"] is not None
        json.dumps(results)  # report must be JSON-serialisable

    def test_unlabelled_evaluation(self, tmp_path, trained_model, cpu_spike_csv):
        path = tmp_path / "unlabelled.csv"
        pd.read_csv(cpu_spike_csv).drop(columns=["label"]).to_csv(path, index=False)

        results = evaluate_model(path, trained_model, top_k=3)

        assert not results["labelled"]
        assert "model_metrics" not in results
        assert len(results["top_windows"]) == 3
        scores = [w["score"] for w in results["top_windows"]]
        assert scores == sorted(scores, reverse=True)

    def test_inference_latency_is_small(self, trained_model, cpu_spike_csv):
        results = evaluate_model(cpu_spike_csv, trained_model)

        assert results["inference_ms_per_window"] < 1000

    def test_scores_spikes_higher_than_normal(self, trained_model, cpu_spike_csv):
        """Even a briefly trained model ranks anomalous windows above normal ones."""
        results = evaluate_model(cpu_spike_csv, trained_model)

        assert results["model_metrics"]["auc"] > 0.5
        assert np.isfinite(results["threshold"])
