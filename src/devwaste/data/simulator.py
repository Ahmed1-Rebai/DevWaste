"""Synthetic data generator for testing and experimentation.

Every scenario starts from the same "normal" baseline process and injects a
specific anomaly pattern with known labels. Generation is fully determined by
the seed: an isolated ``numpy.random.Generator`` is created per call and
timestamps start at a fixed epoch.

Synthetic data is for controlled experiments only. It does not represent
real production workloads.
"""

import numpy as np
import pandas as pd

MEMORY_LIMIT_BYTES = 8 * 1024**3
DEFAULT_START = "2024-01-01T00:00:00"


class SyntheticDataGenerator:
    """Generate reproducible synthetic time-series data."""

    SCENARIOS = ["normal", "cpu-spike", "memory-growth", "network-spike", "oscillating"]

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed."""
        self.seed = seed

    def generate_scenario(
        self,
        scenario: str,
        n_samples: int = 1000,
        interval: float = 1.0,
        container_id: str = "synthetic",
        container_name: str = "test-container",
        run_id: str | None = None,
        start: str = DEFAULT_START,
    ) -> pd.DataFrame:
        """
        Generate synthetic data for a specific scenario.

        Args:
            scenario: One of :attr:`SCENARIOS`
            n_samples: Number of observations
            interval: Seconds between observations
            container_id: Container identifier written to every row
            container_name: Container name written to every row
            run_id: Simulation run identifier (defaults to ``<scenario>-seed<seed>``)
            start: Timestamp of the first observation

        Returns:
            DataFrame with timestamp, metric columns, ``label`` (1 = anomalous
            observation), and ``scenario``/``run_id``/container metadata.
        """
        if scenario not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}. Valid: {self.SCENARIOS}")
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if interval <= 0:
            raise ValueError("interval must be > 0")

        rng = np.random.default_rng(self.seed)
        timestamps = pd.Timestamp(start) + pd.to_timedelta(
            np.arange(n_samples) * interval, unit="s"
        )

        df = self._generate_normal(rng, n_samples)
        label = np.zeros(n_samples, dtype=int)

        if scenario == "cpu-spike":
            label = self._inject_cpu_spikes(rng, df)
        elif scenario == "memory-growth":
            label = self._inject_memory_growth(rng, df)
        elif scenario == "network-spike":
            label = self._inject_network_spikes(rng, df)
        elif scenario == "oscillating":
            label = self._inject_oscillation(rng, df)

        df["memory_usage"] = df["memory_percent"] / 100 * MEMORY_LIMIT_BYTES
        df["memory_limit"] = float(MEMORY_LIMIT_BYTES)
        df.insert(0, "timestamp", timestamps)
        df["label"] = label
        df["scenario"] = scenario
        df["run_id"] = run_id or f"{scenario}-seed{self.seed}"
        df["container_id"] = container_id
        df["container_name"] = container_name

        columns = [
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
            "scenario",
            "run_id",
            "container_id",
            "container_name",
        ]
        return df[columns]

    @staticmethod
    def _generate_normal(rng: np.random.Generator, n: int) -> pd.DataFrame:
        """Baseline process: noisy CPU with a slow cycle, drifting memory, steady I/O."""
        t = np.arange(n)
        cpu = 35 + 5 * np.sin(2 * np.pi * t / 120) + rng.normal(0, 6, n)
        memory = 45 + np.cumsum(rng.normal(0, 0.3, n))
        return pd.DataFrame(
            {
                "cpu_percent": np.clip(cpu, 5, 70),
                "memory_percent": np.clip(memory, 30, 60),
                "network_rx_bytes": np.abs(rng.normal(10_000, 2_500, n)),
                "network_tx_bytes": np.abs(rng.normal(5_000, 1_500, n)),
                "block_read_bytes": np.abs(rng.normal(500, 120, n)),
                "block_write_bytes": np.abs(rng.normal(300, 90, n)),
            }
        )

    @staticmethod
    def _burst_mask(
        rng: np.random.Generator, n: int, coverage: float, min_len: int, max_len: int
    ) -> np.ndarray:
        """Random non-overlapping bursts covering roughly ``coverage`` of ``n``."""
        mask = np.zeros(n, dtype=bool)
        target = max(1, int(n * coverage))
        attempts = 0
        while mask.sum() < target and attempts < 10 * n:
            attempts += 1
            length = int(rng.integers(min_len, max_len + 1))
            start = int(rng.integers(0, max(1, n - length + 1)))
            # keep a gap of normal data around each burst
            lo, hi = max(0, start - 1), min(n, start + length + 1)
            if mask[lo:hi].any():
                continue
            mask[start : start + length] = True
        return mask

    def _inject_cpu_spikes(
        self, rng: np.random.Generator, df: pd.DataFrame
    ) -> np.ndarray:
        """Short bursts of near-saturated CPU (~5% of samples)."""
        mask = self._burst_mask(rng, len(df), 0.05, 2, 6)
        df.loc[mask, "cpu_percent"] = np.clip(rng.normal(90, 5, mask.sum()), 75, 100)
        return mask.astype(int)

    def _inject_network_spikes(
        self, rng: np.random.Generator, df: pd.DataFrame
    ) -> np.ndarray:
        """Short bursts of 5-10x network traffic (~5% of samples)."""
        mask = self._burst_mask(rng, len(df), 0.05, 2, 6)
        factor = rng.uniform(5, 10, mask.sum())
        df.loc[mask, "network_rx_bytes"] *= factor
        df.loc[mask, "network_tx_bytes"] *= factor
        return mask.astype(int)

    @staticmethod
    def _inject_memory_growth(rng: np.random.Generator, df: pd.DataFrame) -> np.ndarray:
        """
        Memory leak: linear growth starting at 40% of the series.

        Labelled anomalous once the leak adds more than 10 percentage points
        above the baseline.
        """
        n = len(df)
        onset = int(n * 0.4)
        leak = np.zeros(n)
        leak[onset:] = np.linspace(0, 45, n - onset)
        df["memory_percent"] = np.clip(df["memory_percent"] + leak, 0, 100)
        return (leak > 10).astype(int)

    @staticmethod
    def _inject_oscillation(rng: np.random.Generator, df: pd.DataFrame) -> np.ndarray:
        """
        Resource thrashing: large CPU/memory oscillation from 50% of the series.

        Every observation after the onset is labelled anomalous.
        """
        n = len(df)
        onset = n // 2
        t = np.arange(n - onset)
        df.loc[onset:, "cpu_percent"] = np.clip(
            50 + 40 * np.sin(2 * np.pi * t / 20) + rng.normal(0, 4, n - onset), 0, 100
        )
        df.loc[onset:, "memory_percent"] = np.clip(
            df["memory_percent"].to_numpy()[onset:]
            + 15 * np.sin(2 * np.pi * t / 20 + np.pi / 4),
            0,
            100,
        )
        label = np.zeros(n, dtype=int)
        label[onset:] = 1
        return label
