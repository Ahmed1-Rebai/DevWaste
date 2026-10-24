"""Baseline anomaly detectors for comparison with the LSTM Autoencoder."""

import numpy as np
import pandas as pd

from devwaste.config import FEATURE_COLUMNS, LOG_FEATURES, MIN_FEATURE_SCALE
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

    A point's score is its largest absolute deviation, over all features,
    from the rolling mean of the preceding ``window_size`` observations of
    the same series (past only), in rolling standard deviations.

    Used at window level with a threshold calibrated on validation windows
    (like the LSTM), or at point level with a fixed ``std_threshold``.
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
            std_threshold: Point-level threshold used by :meth:`predict`
            features: Features to check (defaults to all model features)
        """
        self.window_size = window_size
        self.std_threshold = std_threshold
        self.features = features or list(FEATURE_COLUMNS)

    def _series_scores(self, df: pd.DataFrame) -> np.ndarray:
        scores = np.zeros(len(df))
        for feature in self.features:
            if feature not in df.columns:
                continue
            series = df[feature].astype(float)
            if feature in LOG_FEATURES:
                series = np.log1p(series)
            past = series.shift(1).rolling(window=self.window_size, min_periods=2)
            mean = past.mean()
            scale_floor = MIN_FEATURE_SCALE.get(feature, 1.0)
            std = np.maximum(past.std().fillna(0.0), scale_floor)
            valid_mask = past.count() >= 2
            z = ((series - mean) / std).abs()
            z = z.where(valid_mask, 0.0).fillna(0.0).to_numpy()
            scores = np.maximum(scores, z)
        return scores

    def score_points(self, df: pd.DataFrame) -> np.ndarray:
        """
        Point-level rolling z-scores, computed independently per series.

        Args:
            df: DataFrame with metrics, sorted in time order

        Returns:
            Score array aligned with ``df``
        """
        keys = series_keys(df)
        if not keys:
            return self._series_scores(df)

        scores = np.zeros(len(df))
        for positions in df.groupby(keys, sort=False, dropna=False).indices.values():
            scores[positions] = self._series_scores(df.iloc[positions])
        return scores

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Point-level predictions using the fixed ``std_threshold``."""
        return (self.score_points(df) > self.std_threshold).astype(int)
