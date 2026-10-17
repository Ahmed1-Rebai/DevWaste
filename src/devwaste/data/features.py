"""Feature engineering for time-series data."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from devwaste.config import (
    FEATURE_COLUMNS,
    LOG_FEATURES,
    MIN_FEATURE_SCALE,
    SERIES_KEYS,
)

COUNTER_COLUMNS = [
    "network_rx_bytes",
    "network_tx_bytes",
    "block_read_bytes",
    "block_write_bytes",
]


def series_keys(df: pd.DataFrame) -> list[str]:
    """Return the series-identifying columns present in ``df``."""
    return [k for k in SERIES_KEYS if k in df.columns]


def cumulative_to_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Docker's cumulative byte counters to per-interval deltas.

    Docker reports network and block I/O as totals since container start.
    The model expects per-interval values, matching the synthetic generator.
    The first observation of each series has no predecessor and is dropped.
    Negative deltas (counter reset after a restart) are clipped to 0.
    """
    df = df.copy()
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", kind="stable")

    keys = series_keys(df)
    counters = [c for c in COUNTER_COLUMNS if c in df.columns]
    if keys:
        diffs = df.groupby(keys, sort=False)[counters].diff()
    else:
        diffs = df[counters].diff()

    df[counters] = diffs.clip(lower=0)
    return df.dropna(subset=counters).reset_index(drop=True)


class FeatureEngineer:
    """Validate, clean, and scale model input features."""

    def __init__(self):
        """Initialize feature engineer."""
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.log_features: list[str] = []
        self.is_fitted = False

    def get_feature_columns(self) -> list[str]:
        """Get list of feature column names."""
        return list(FEATURE_COLUMNS)

    def validate_schema(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """Validate that DataFrame has required columns."""
        required = self.get_feature_columns()
        missing = [col for col in required if col not in df.columns]
        return len(missing) == 0, missing

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data: handle missing, infinite, and invalid values.

        Rows are sorted by timestamp (stable, so series order is kept).
        Missing values are forward-filled within each series (past values
        only); leading gaps are back-filled from the first valid value and
        anything still missing becomes 0. Negative values are clipped to 0
        since no supported metric can be negative.
        """
        df = df.copy()

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

        features = [f for f in self.get_feature_columns() if f in df.columns]
        for col in features:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[features] = df[features].replace([np.inf, -np.inf], np.nan)

        keys = series_keys(df)
        if keys:
            grouped = df.groupby(keys, sort=False)[features]
            df[features] = grouped.ffill()
            df[features] = df.groupby(keys, sort=False)[features].bfill()
        else:
            df[features] = df[features].ffill().bfill()

        df[features] = df[features].fillna(0).clip(lower=0)
        return df

    def _prepare(self, df: pd.DataFrame, columns: list[str]) -> np.ndarray:
        values = df[columns].to_numpy(dtype=float).copy()
        for i, col in enumerate(columns):
            if col in self.log_features:
                values[:, i] = np.log1p(values[:, i])
        return values

    def fit(self, df: pd.DataFrame):
        """
        Fit scaler on data. Call this on the training split only.

        Each feature's scale is floored at ``MIN_FEATURE_SCALE`` so that
        near-constant training features do not exaggerate small shifts.
        """
        is_valid, missing = self.validate_schema(df)
        if not is_valid:
            raise ValueError(f"Missing required feature columns: {missing}")

        self.feature_names = self.get_feature_columns()
        self.log_features = [f for f in self.feature_names if f in LOG_FEATURES]
        self.scaler.fit(self._prepare(df, self.feature_names))
        floor = np.array([MIN_FEATURE_SCALE.get(f, 0.0) for f in self.feature_names])
        self.scaler.scale_ = np.maximum(self.scaler.scale_, floor)
        self.scaler.var_ = self.scaler.scale_**2
        self.is_fitted = True

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        return self.scaler.transform(self._prepare(df, self.feature_names))

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform data."""
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data back to the original units."""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        values = self.scaler.inverse_transform(X)
        for i, col in enumerate(self.feature_names):
            if col in self.log_features:
                values[:, i] = np.expm1(values[:, i])
        return values

    def get_state(self) -> dict:
        """Get scaler state for serialization."""
        return {
            "feature_names": self.feature_names,
            "log_features": self.log_features,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "is_fitted": self.is_fitted,
        }

    def set_state(self, state: dict):
        """Restore scaler state from serialization."""
        self.feature_names = list(state["feature_names"])
        self.log_features = list(state.get("log_features", []))
        self.scaler.mean_ = np.array(state["scaler_mean"])
        self.scaler.scale_ = np.array(state["scaler_scale"])
        self.scaler.var_ = self.scaler.scale_**2
        self.scaler.n_features_in_ = len(self.feature_names)
        self.is_fitted = state["is_fitted"]
