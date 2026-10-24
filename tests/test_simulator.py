"""Tests for synthetic data generator."""

import pandas as pd
import pytest

from devwaste.data.simulator import SyntheticDataGenerator


class TestSyntheticDataGenerator:
    """Test SyntheticDataGenerator class."""

    def test_generate_normal_scenario(self):
        """Test normal scenario generation."""
        generator = SyntheticDataGenerator(seed=42)
        df = generator.generate_scenario("normal", n_samples=100)

        assert len(df) == 100
        assert "label" in df.columns
        assert df["label"].sum() == 0
        assert df["cpu_percent"].between(0, 100).all()

    def test_generate_cpu_spike_scenario(self):
        """Test CPU spike scenario generation."""
        generator = SyntheticDataGenerator(seed=42)
        df = generator.generate_scenario("cpu-spike", n_samples=100)

        assert len(df) == 100
        assert df["label"].sum() > 0
        assert df["cpu_percent"].between(0, 100).all()

    def test_generate_memory_growth_scenario(self):
        """Test memory growth scenario generation."""
        generator = SyntheticDataGenerator(seed=42)
        df = generator.generate_scenario("memory-growth", n_samples=100)

        assert len(df) == 100
        assert df["memory_percent"].between(0, 100).all()
        assert df["memory_percent"].iloc[-1] > df["memory_percent"].iloc[0]

    def test_generate_network_spike_scenario(self):
        """Test network spike scenario generation."""
        generator = SyntheticDataGenerator(seed=42)
        df = generator.generate_scenario("network-spike", n_samples=100)

        assert len(df) == 100
        assert df["label"].sum() > 0

    def test_generate_oscillating_scenario(self):
        """Test oscillating scenario generation."""
        generator = SyntheticDataGenerator(seed=42)
        df = generator.generate_scenario("oscillating", n_samples=100)

        assert len(df) == 100
        assert df["cpu_percent"].between(0, 100).all()

    def test_generate_subtle_oscillation_scenario(self):
        """Test subtle oscillation scenario generation."""
        generator = SyntheticDataGenerator(seed=42)
        df = generator.generate_scenario("subtle-oscillation", n_samples=100)

        assert len(df) == 100
        assert df["cpu_percent"].between(0, 100).all()
        assert df["label"].sum() > 0
        # Peak values should not saturate or hit extreme threshold 80%
        assert (df["cpu_percent"] < 75).all()

    def test_invalid_scenario_raises_error(self):
        """Test that invalid scenario raises ValueError."""
        generator = SyntheticDataGenerator(seed=42)

        with pytest.raises(ValueError, match="Unknown scenario"):
            generator.generate_scenario("invalid-scenario")

    def test_deterministic_generation(self):
        """Test that generation is deterministic with same seed."""
        generator1 = SyntheticDataGenerator(seed=42)
        generator2 = SyntheticDataGenerator(seed=42)

        df1 = generator1.generate_scenario("normal", n_samples=100)
        df2 = generator2.generate_scenario("normal", n_samples=100)

        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        generator1 = SyntheticDataGenerator(seed=42)
        generator2 = SyntheticDataGenerator(seed=123)

        df1 = generator1.generate_scenario("normal", n_samples=100)
        df2 = generator2.generate_scenario("normal", n_samples=100)

        # Data should be different
        assert not df1.equals(df2)

    def test_custom_container_metadata(self):
        """Test custom container metadata."""
        generator = SyntheticDataGenerator(seed=42)
        df = generator.generate_scenario(
            "normal",
            n_samples=50,
            container_id="custom-id",
            container_name="custom-name",
        )

        assert df["container_id"].iloc[0] == "custom-id"
        assert df["container_name"].iloc[0] == "custom-name"

    def test_all_required_columns_present(self):
        """Test that all required columns are present."""
        generator = SyntheticDataGenerator(seed=42)
        df = generator.generate_scenario("normal", n_samples=50)

        required_columns = [
            "timestamp",
            "cpu_percent",
            "memory_usage",
            "memory_limit",
            "memory_percent",
            "network_rx_bytes",
            "network_tx_bytes",
            "block_read_bytes",
            "block_write_bytes",
            "label",
        ]

        for col in required_columns:
            assert col in df.columns


class TestScenarioDesign:
    """Labels and metadata of the synthetic scenarios."""

    @pytest.mark.parametrize("scenario", SyntheticDataGenerator.SCENARIOS)
    def test_metadata_and_bounds(self, scenario):
        df = SyntheticDataGenerator(seed=7).generate_scenario(scenario, n_samples=300)

        assert (df["scenario"] == scenario).all()
        assert (df["run_id"] == f"{scenario}-seed7").all()
        assert df["timestamp"].is_monotonic_increasing
        assert df[["cpu_percent", "memory_percent"]].stack().between(0, 100).all()
        assert (df.filter(like="_bytes") >= 0).all().all()

    def test_spikes_are_labelled(self):
        df = SyntheticDataGenerator(seed=7).generate_scenario(
            "cpu-spike", n_samples=500
        )

        assert df.loc[df["label"] == 1, "cpu_percent"].min() >= 75
        assert 0.03 < df["label"].mean() < 0.1

    def test_memory_growth_label_after_onset(self):
        df = SyntheticDataGenerator(seed=7).generate_scenario(
            "memory-growth", n_samples=500
        )

        assert df["label"].iloc[:200].sum() == 0
        assert df["label"].iloc[-1] == 1

    def test_sub_second_interval(self):
        df = SyntheticDataGenerator().generate_scenario(
            "normal", n_samples=3, interval=0.5
        )

        assert (df["timestamp"].diff().dropna() == pd.Timedelta("500ms")).all()

    def test_does_not_touch_global_rng(self):
        import numpy as np

        np.random.seed(0)
        expected = np.random.rand()
        np.random.seed(0)
        SyntheticDataGenerator(seed=1).generate_scenario("normal", n_samples=10)
        assert np.random.rand() == expected
