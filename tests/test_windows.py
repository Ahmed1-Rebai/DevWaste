"""Tests for window generation."""

import numpy as np
import pandas as pd
import pytest

from devwaste.data.windows import WindowGenerator


class TestWindowGenerator:
    """Test WindowGenerator class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        n_samples = 100
        X = np.random.randn(n_samples, 6)
        y = np.zeros(n_samples, dtype=int)
        y[50:60] = 1

        metadata = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2024-01-01", periods=n_samples, freq="1s"
                ),
                "container_id": ["container1"] * n_samples,
            }
        )

        return X, y, metadata

    def test_create_windows_basic(self, sample_data):
        """Test basic window creation."""
        X, y, metadata = sample_data
        generator = WindowGenerator(window_size=10, stride=1)

        X_windows, y_windows, window_metadata = generator.create_windows(X, y, metadata)

        assert X_windows.shape == (91, 10, 6)
        assert len(y_windows) == 91
        assert len(window_metadata) == 91

    def test_window_stride(self, sample_data):
        """Test window creation with stride."""
        X, y, metadata = sample_data
        generator = WindowGenerator(window_size=10, stride=5)

        X_windows, y_windows, window_metadata = generator.create_windows(X, y, metadata)

        assert X_windows.shape[0] == 19

    def test_window_metadata_timestamps(self, sample_data):
        """Test that window metadata includes timestamps."""
        X, y, metadata = sample_data
        generator = WindowGenerator(window_size=10, stride=1)

        X_windows, y_windows, window_metadata = generator.create_windows(X, y, metadata)

        assert "start_timestamp" in window_metadata.columns
        assert "end_timestamp" in window_metadata.columns

    def test_window_label_propagation(self, sample_data):
        """Test that anomaly labels propagate to windows."""
        X, y, metadata = sample_data
        generator = WindowGenerator(window_size=10, stride=1)

        X_windows, y_windows, _ = generator.create_windows(X, y, metadata)

        assert y_windows.max() == 1
        assert y_windows.sum() > 0

    def test_insufficient_samples(self):
        """Test error when not enough samples."""
        X = np.random.randn(5, 6)
        generator = WindowGenerator(window_size=10, stride=1)

        with pytest.raises(ValueError, match="Not enough samples"):
            generator.create_windows(X)

    def test_create_windows_from_df(self):
        """Test creating windows directly from DataFrame."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(start="2024-01-01", periods=50, freq="1s"),
                "cpu_percent": np.random.randn(50) * 10 + 50,
                "memory_percent": np.random.randn(50) * 5 + 40,
                "network_rx_bytes": np.random.randn(50) * 1000 + 5000,
                "network_tx_bytes": np.random.randn(50) * 500 + 2500,
                "block_read_bytes": np.random.randn(50) * 100 + 500,
                "block_write_bytes": np.random.randn(50) * 50 + 250,
                "label": np.zeros(50, dtype=int),
            }
        )
        df.loc[20:25, "label"] = 1

        generator = WindowGenerator(window_size=5, stride=1)
        feature_cols = [
            "cpu_percent",
            "memory_percent",
            "network_rx_bytes",
            "network_tx_bytes",
            "block_read_bytes",
            "block_write_bytes",
        ]

        X_windows, y_windows, window_metadata = generator.create_windows_from_df(
            df, feature_cols, "label"
        )

        assert X_windows.shape[0] == 46
        assert y_windows.sum() > 0


class TestWindowsBySeries:
    """Windows never cross container/run boundaries."""

    @pytest.fixture
    def two_series(self):
        df = (
            pd.DataFrame(
                {
                    "timestamp": list(
                        pd.date_range("2024-01-01", periods=12, freq="1s")
                    )
                    * 2,
                    "run_id": ["r1"] * 12 + ["r2"] * 12,
                    "label": [0] * 12 + [0] * 11 + [1],
                }
            )
            .sort_values("timestamp", kind="stable")
            .reset_index(drop=True)
        )
        # feature encodes the run so mixing would be visible
        X = np.where(df["run_id"] == "r1", 1.0, 2.0).reshape(-1, 1)
        return X, df

    def test_no_window_mixes_series(self, two_series):
        X, df = two_series
        generator = WindowGenerator(window_size=5)

        Xw, yw, meta = generator.create_windows_by_series(X, df, df["label"].to_numpy())

        assert Xw.shape == (16, 5, 1)  # 8 windows per series
        assert all(len(np.unique(w)) == 1 for w in Xw)
        assert set(meta["run_id"]) == {"r1", "r2"}
        assert yw.sum() == 1  # only the last r2 window covers its last sample

    def test_metadata_points_to_original_rows(self, two_series):
        X, df = two_series
        _, _, meta = WindowGenerator(window_size=5).create_windows_by_series(X, df)

        for _, row in meta.iterrows():
            assert df.iloc[row["start_sample"]]["run_id"] == row["run_id"]
            assert df.iloc[row["end_sample"] - 1]["run_id"] == row["run_id"]
            assert df.iloc[row["start_sample"]]["timestamp"] == row["start_timestamp"]

    def test_short_series_skipped(self, two_series):
        X, df = two_series
        df = df.copy()
        df.loc[df.index[:3], "run_id"] = "tiny"

        Xw, _, meta = WindowGenerator(window_size=5).create_windows_by_series(X, df)

        assert "tiny" not in set(meta["run_id"])

    def test_windows_from_points_aligns(self, two_series):
        X, df = two_series
        generator = WindowGenerator(window_size=5)
        points = df["label"].to_numpy()

        _, yw, _ = generator.create_windows_by_series(X, df, points)

        np.testing.assert_array_equal(generator.windows_from_points(points, df), yw)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            WindowGenerator(window_size=0)
