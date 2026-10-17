"""Tests for evaluation metrics and baseline detectors."""

import numpy as np
import pandas as pd
import pytest

from devwaste.evaluation.metrics import calculate_metrics
from devwaste.models.baselines import RollingBaseline, ThresholdBaseline, ZScoreBaseline


class TestMetrics:
    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1])
        m = calculate_metrics(y, y, scores=np.array([0.1, 0.2, 0.8, 0.9]))

        assert m["precision"] == m["recall"] == m["f1_score"] == 1.0
        assert m["false_positive_rate"] == 0.0
        assert m["confusion_matrix"] == [[2, 0], [0, 2]]
        assert m["auc"] == 1.0

    def test_known_values(self):
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred = np.array([1, 0, 0, 0, 1, 0])
        m = calculate_metrics(y_true, y_pred)

        assert m["precision"] == 0.5
        assert m["recall"] == 0.5
        assert m["false_positive_rate"] == 0.25
        assert m["confusion_matrix"] == [[3, 1], [1, 1]]

    def test_single_class_keeps_2x2_matrix(self):
        m = calculate_metrics(np.zeros(5), np.zeros(5), scores=np.zeros(5))

        assert m["confusion_matrix"] == [[5, 0], [0, 0]]
        assert m["auc"] is None
        assert m["precision"] == 0.0

    def test_empty(self):
        m = calculate_metrics(np.array([]), np.array([]))

        assert m["n_samples"] == 0


class TestBaselines:
    def test_zscore_calibration(self):
        rng = np.random.default_rng(0)
        X_val = rng.normal(size=(200, 5, 2))
        baseline = ZScoreBaseline()
        threshold = baseline.calibrate(X_val, 0.95)

        assert baseline.predict_windows(X_val).mean() == pytest.approx(0.05, abs=0.01)
        outlier = np.full((1, 5, 2), threshold + 1)
        assert baseline.predict_windows(outlier)[0] == 1

    def test_zscore_requires_calibration(self):
        with pytest.raises(ValueError):
            ZScoreBaseline().predict_windows(np.zeros((1, 2, 2)))

    def test_threshold_baseline(self):
        df = pd.DataFrame({"cpu_percent": [50, 90, 10], "memory_percent": [10, 10, 85]})

        np.testing.assert_array_equal(ThresholdBaseline().predict(df), [0, 1, 1])

    def test_rolling_baseline_flags_jump(self):
        values = np.r_[np.random.default_rng(0).normal(50, 1, 30), 90]
        df = pd.DataFrame({"cpu_percent": values})

        preds = RollingBaseline(window_size=10, features=["cpu_percent"]).predict(df)

        assert preds[-1] == 1
        assert preds[:-1].sum() <= 2

    def test_rolling_baseline_per_series(self):
        """A level difference between two containers is not an anomaly."""
        df = pd.DataFrame(
            {
                "container_id": ["a"] * 20 + ["b"] * 20,
                "cpu_percent": np.r_[np.full(20, 10.0), np.full(20, 90.0)]
                + np.tile([0.0, 1.0], 20),
            }
        )

        preds = RollingBaseline(window_size=5, features=["cpu_percent"]).predict(df)

        assert preds.sum() == 0


def test_rolling_scores_are_past_only():
    """Changing a future value never changes an earlier score."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"cpu_percent": rng.normal(50, 1, 40)})
    baseline = RollingBaseline(window_size=10, features=["cpu_percent"])
    before = baseline.score_points(df)

    df.loc[30, "cpu_percent"] = 500
    after = baseline.score_points(df)

    np.testing.assert_array_equal(before[:30], after[:30])
    assert after[30] > before[30]
