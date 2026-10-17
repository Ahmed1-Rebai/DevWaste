"""Configuration settings for DevWaste AI."""

from pathlib import Path
from typing import Final

# Default paths are relative to the current working directory so that an
# installed package never writes into site-packages.
DEFAULT_DATA_DIR: Final = Path("data")
DEFAULT_ARTIFACTS_DIR: Final = Path("artifacts")
DEFAULT_DB_PATH: Final = DEFAULT_DATA_DIR / "metrics.db"

# Default metrics collection
DEFAULT_COLLECTION_INTERVAL: Final = 5  # seconds
DEFAULT_COLLECTION_DURATION: Final = 120  # seconds

# Model defaults
DEFAULT_WINDOW_SIZE: Final = 10
DEFAULT_HIDDEN_DIM: Final = 16
DEFAULT_EPOCHS: Final = 50
DEFAULT_BATCH_SIZE: Final = 32
DEFAULT_LEARNING_RATE: Final = 0.001
DEFAULT_THRESHOLD_QUANTILE: Final = 0.95  # for threshold calibration
DEFAULT_SEED: Final = 42

# Model input features. Byte counters are per-interval deltas, not cumulative
# totals (see devwaste.data.features.cumulative_to_deltas).
FEATURE_COLUMNS: Final = [
    "cpu_percent",
    "memory_percent",
    "network_rx_bytes",
    "network_tx_bytes",
    "block_read_bytes",
    "block_write_bytes",
]

# Heavy-tailed byte features are log1p-transformed before scaling.
LOG_FEATURES: Final = [
    "network_rx_bytes",
    "network_tx_bytes",
    "block_read_bytes",
    "block_write_bytes",
]

# Minimum per-feature scale after preprocessing. A feature that is nearly
# constant during training would otherwise turn a harmless small level shift
# into an extreme z-score. Percent features: 1 percentage point. Byte
# features (log1p space): 0.1, i.e. roughly a 10% change.
MIN_FEATURE_SCALE: Final = {
    "cpu_percent": 1.0,
    "memory_percent": 1.0,
    "network_rx_bytes": 0.1,
    "network_tx_bytes": 0.1,
    "block_read_bytes": 0.1,
    "block_write_bytes": 0.1,
}

# Columns identifying one independent time series. Windows never cross
# series boundaries.
SERIES_KEYS: Final = ["container_id", "run_id"]

# Safety
DEFAULT_DRY_RUN: Final = True
