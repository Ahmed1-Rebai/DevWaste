"""Tests for feature engineering and dataset utilities."""

import numpy as np
import pandas as pd
import pytest

from devwaste.data.dataset import chronological_split, load_dataset
from devwaste.data.features import FeatureEngineer, cumulative_to_deltas


def make_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2024-01-01", periods=n, freq="1s"),
            "cpu_percent": rng.uniform(20, 80, n),
            "memory_percent": rng.uniform(30, 70, n),
            "network_rx_bytes": rng.uniform(1000, 10000, n),
            "network_tx_bytes": rng.uniform(500, 5000, n),
            "block_read_bytes": rng.uniform(100, 1000, n),
            "block_write_bytes": rng.uniform(50, 500, n),
            "label": np.zeros(n, dtype=int),
        }
    )


class TestFeatureEngineer:
    """Test FeatureEngineer class."""

    @pytest.fixture
    def sample_data(self):
        return make_df()

    def test_fit_transform(self, sample_data):
        engineer = FeatureEngineer()
        X = engineer.fit_transform(sample_data)

        assert X.shape == (100, 6)
        assert engineer.is_fitted
        np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-9)

    def test_transform_unfitted_raises_error(self, sample_data):
        with pytest.raises(ValueError, match="Scaler not fitted"):
            FeatureEngineer().transform(sample_data)

    def test_transform_missing_column_raises(self, sample_data):
        engineer = FeatureEngineer()
        engineer.fit(sample_data)

        with pytest.raises(ValueError, match="Missing required feature columns"):
            engineer.transform(sample_data.drop(columns=["cpu_percent"]))

    def test_inverse_transform(self, sample_data):
        """Round trip through log1p + scaling recovers original values."""
        engineer = FeatureEngineer()
        X_original = engineer.inverse_transform(engineer.fit_transform(sample_data))

        original = sample_data[engineer.get_feature_columns()].to_numpy()
        np.testing.assert_allclose(X_original, original, rtol=1e-6)

    def test_clean_data_handles_missing(self):
        df = make_df(10)
        df.loc[[1, 3, 5], "cpu_percent"] = np.nan
        df.loc[0, "memory_percent"] = np.nan

        cleaned = FeatureEngineer().clean_data(df)

        assert not cleaned[FeatureEngineer().get_feature_columns()].isna().any().any()
        # forward fill uses the previous (past) value
        assert cleaned.loc[1, "cpu_percent"] == df.loc[0, "cpu_percent"]

    def test_clean_data_handles_infinite_and_negative(self):
        df = make_df(10)
        df.loc[1, "cpu_percent"] = np.inf
        df.loc[2, "cpu_percent"] = -np.inf
        df.loc[3, "network_rx_bytes"] = -5

        cleaned = FeatureEngineer().clean_data(df)

        assert np.isfinite(cleaned["cpu_percent"]).all()
        assert (cleaned["network_rx_bytes"] >= 0).all()

    def test_clean_data_fills_within_series_only(self):
        """Missing values are never filled from another container's data."""
        df = pd.concat(
            [make_df(5).assign(container_id="a"), make_df(5).assign(container_id="b")]
        )
        df["timestamp"] = list(pd.date_range("2024-01-01", periods=5, freq="1s")) * 2
        df.loc[df["container_id"] == "b", "cpu_percent"] = np.nan

        cleaned = FeatureEngineer().clean_data(df.reset_index(drop=True))

        assert (cleaned.loc[cleaned["container_id"] == "b", "cpu_percent"] == 0).all()

    def test_validate_schema(self, sample_data):
        is_valid, missing = FeatureEngineer().validate_schema(sample_data)

        assert is_valid
        assert missing == []

    def test_validate_schema_missing_columns(self):
        is_valid, missing = FeatureEngineer().validate_schema(
            pd.DataFrame({"cpu_percent": [1.0]})
        )

        assert not is_valid
        assert "memory_percent" in missing

    def test_get_state_and_set_state(self, sample_data):
        engineer = FeatureEngineer()
        X = engineer.fit_transform(sample_data)

        restored = FeatureEngineer()
        restored.set_state(engineer.get_state())

        assert restored.is_fitted
        assert restored.feature_names == engineer.feature_names
        np.testing.assert_allclose(restored.transform(sample_data), X)


class TestCumulativeToDeltas:
    def test_per_container_deltas(self):
        df = pd.DataFrame(
            {
                "timestamp": ["t1", "t1", "t2", "t2", "t3"],
                "container_id": ["a", "b", "a", "b", "a"],
                "network_rx_bytes": [100, 1000, 150, 1100, 40],  # a restarts at t3
                "network_tx_bytes": [0, 0, 10, 20, 30],
                "block_read_bytes": [0, 0, 0, 0, 0],
                "block_write_bytes": [0, 0, 0, 0, 0],
            }
        )

        out = cumulative_to_deltas(df)

        assert len(out) == 3  # first observation of each container dropped
        a = out[out["container_id"] == "a"]["network_rx_bytes"].tolist()
        b = out[out["container_id"] == "b"]["network_rx_bytes"].tolist()
        assert a == [50, 0]  # counter reset clipped to 0
        assert b == [100]


class TestDataset:
    def test_load_dataset_rejects_bad_schema(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"cpu_percent": [1, 2]}).to_csv(path, index=False)

        with pytest.raises(ValueError, match="missing required columns"):
            load_dataset(path)

    def test_chronological_split_per_series(self):
        df = pd.concat(
            [make_df(10).assign(run_id="r1"), make_df(20).assign(run_id="r2")],
            ignore_index=True,
        )

        train, val = chronological_split(df, 0.2)

        assert train.groupby("run_id").size().to_dict() == {"r1": 8, "r2": 16}
        assert val.groupby("run_id").size().to_dict() == {"r1": 2, "r2": 4}
        for run in ("r1", "r2"):
            assert (
                train[train["run_id"] == run]["timestamp"].max()
                < val[val["run_id"] == run]["timestamp"].min()
            )


def test_scale_floor_for_near_constant_feature():
    """A tiny level shift in a near-constant feature stays a small z-score."""
    df = make_df(100)
    df["memory_percent"] = 2.9 + np.random.default_rng(0).normal(0, 0.01, 100)
    engineer = FeatureEngineer()
    engineer.fit(df)

    shifted = df.assign(memory_percent=df["memory_percent"] + 0.8)
    z = engineer.transform(shifted)[:, engineer.feature_names.index("memory_percent")]

    assert engineer.get_state()["scaler_scale"][1] == 1.0
    assert np.abs(z).max() < 1.5
