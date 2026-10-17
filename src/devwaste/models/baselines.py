"""Baseline anomaly detectors for comparison with the LSTM Autoencoder."""

import numpy as np
import pandas as pd

from devwaste.config import FEATURE_COLUMNS
from devwaste.data.features import series_keys

BASELINE_METHODS = ["zscore", "threshold", "rolling"]


class ZScoreBaseline:
    """
    Window-level z-score detector.

    Uses the same training-fitted scaler as the LSTM, so each scaled value is
    a z-score relative to training data. A window's score is the largest
    absolute z-score it contains. The threshold is calibrated with the same
    rule as the LSTM (a quantile of validation scores), which makes the
    comparison like-for-like.
    """

    def __init__(self, threshold: float | None = None):
        """Initialize with an optional pre-calibrated threshold."""
        self.threshold = threshold

    @staticmethod
    def score_windows(X_windows: np.ndarray) -> np.ndarray:
        """Max absolute z-score per window, shape (n_windows,)."""
        return np.abs(X_windows).max(axis=(1, 2))

    def calibrate(self, X_val_windows: np.ndarray, quantile: float) -> float:
        """Set the threshold to a quantile of validation window scores."""
        self.threshold = float(np.quantile(self.score_windows(X_val_windows), quantile))
        return self.threshold

    def predict_windows(self, X_windows: np.ndarray) -> np.ndarray:
        """Binary window predictions."""
        if self.threshold is None:
            raise ValueError("ZScoreBaseline threshold not calibrated")
        return (self.score_windows(X_windows) > self.threshold).astype(int)


class ThresholdBaseline:
    """
    Fixed-threshold rule detector.

    Flags observations where any metric exceeds a fixed threshold. Note that
    rules like ``cpu_percent > 80`` resemble how some synthetic scenarios are
    generated, so this baseline can look unrealistically strong on them.
    """

    DEFAULT_THRESHOLDS = {"cpu_percent": 80.0, "memory_percent": 80.0}

    def __init__(self, thresholds: dict[str, float] | None = None):
        """
        Initialize threshold baseline.

        Args:
            thresholds: Dictionary mapping feature names to thresholds
        """
        self.thresholds = thresholds or dict(self.DEFAULT_THRESHOLDS)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict point-level anomalies using fixed thresholds.

        Args:
            df: DataFrame with metrics

        Returns:
            Binary predictions array
        """
        predictions = np.zeros(len(df), dtype=int)

        for feature, threshold in self.thresholds.items():
            if feature in df.columns:
                feature_preds = (df[feature].to_numpy() > threshold).astype(int)
                predictions = np.maximum(predictions, feature_preds)

        return predictions


class RollingBaseline:
    """
    Rolling statistics-based anomaly detector.

    Flags observations that deviate from the rolling mean of the preceding
    observations (past only) of the same series by more than
    ``std_threshold`` rolling standard deviations, on any feature.
    """

    def __init__(
        self,
        window_size: int = 10,
        std_threshold: float = 3.0,
        features: list[str] | None = None,
    ):
        """
        Initialize rolling baseline.

        Args:
            window_size: Rolling window size for statistics
            std_threshold: Number of standard deviations for anomaly detection
            features: Features to check (defaults to all model features)
        """
        self.window_size = window_size
        self.std_threshold = std_threshold
        self.features = features or list(FEATURE_COLUMNS)

    def _series_predict(self, df: pd.DataFrame) -> np.ndarray:
        predictions = np.zeros(len(df), dtype=int)
        for feature in self.features:
            if feature not in df.columns:
                continue
            series = df[feature].astype(float)
            past = series.shift(1).rolling(window=self.window_size, min_periods=2)
            mean, std = past.mean(), past.std().replace(0, np.nan)
            z = ((series - mean) / std).abs().fillna(0).to_numpy()
            predictions = np.maximum(predictions, (z > self.std_threshold).astype(int))
        return predictions

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict point-level anomalies using rolling statistics.

        Args:
            df: DataFrame with metrics, sorted in time order

        Returns:
            Binary predictions array aligned with ``df``
        """
        keys = series_keys(df)
        if not keys:
            return self._series_predict(df)

        predictions = np.zeros(len(df), dtype=int)
        for positions in df.groupby(keys, sort=False, dropna=False).indices.values():
            predictions[positions] = self._series_predict(df.iloc[positions])
        return predictions
