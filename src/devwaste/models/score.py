"""Model scoring and evaluation."""

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from devwaste.data.dataset import get_labels, load_dataset
from devwaste.data.features import FeatureEngineer
from devwaste.data.windows import WindowGenerator
from devwaste.evaluation.metrics import calculate_metrics
from devwaste.models.artifacts import ModelArtifacts
from devwaste.models.baselines import RollingBaseline, ThresholdBaseline, ZScoreBaseline
from devwaste.models.train import feature_errors


def evaluate_model(
    data_path: Path,
    model_dir: Path,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Score a dataset with a trained model and, if labelled, evaluate it.

    The dataset is cleaned and scaled with the stored training scaler, then
    windowed per series. Windows whose reconstruction error exceeds the
    calibrated threshold are flagged. When the data has a ``label`` column,
    window-level metrics are computed for the model and every baseline on
    exactly the same windows.

    Args:
        data_path: Path to data CSV
        model_dir: Directory containing trained model artifacts
        top_k: Number of highest-scoring windows to include in the result

    Returns:
        Dictionary with detection summary, metrics, and baseline metrics
    """
    artifacts = ModelArtifacts.load(model_dir)

    feature_engineer = FeatureEngineer()
    feature_engineer.set_state(artifacts.feature_engineer_state)

    df = load_dataset(data_path, feature_engineer)
    y = get_labels(df)
    X = feature_engineer.transform(df)

    window_generator = WindowGenerator(window_size=artifacts.window_size, stride=1)
    X_windows, y_windows, window_meta = window_generator.create_windows_by_series(
        X, df, y
    )

    model = artifacts.build_model()
    start = time.perf_counter()
    per_feature = feature_errors(model, X_windows)
    elapsed = time.perf_counter() - start
    errors = per_feature.mean(axis=1)

    threshold = float(artifacts.threshold)
    predictions = (errors > threshold).astype(int)
    feature_names = feature_engineer.feature_names
    top_feature = np.asarray(feature_names)[per_feature.argmax(axis=1)]
    flagged_drivers = pd.Series(top_feature[predictions == 1]).value_counts()

    results: dict[str, Any] = {
        "data_path": str(data_path),
        "model_dir": str(model_dir),
        "threshold": threshold,
        "threshold_method": artifacts.threshold_method,
        "n_windows": int(len(X_windows)),
        "n_flagged": int(predictions.sum()),
        "flagged_fraction": float(predictions.mean()),
        "inference_ms_per_window": 1000 * elapsed / max(len(X_windows), 1),
        "labelled": y is not None,
        "flagged_by_feature": {k: int(v) for k, v in flagged_drivers.items()},
        "top_windows": _top_windows(
            window_meta, errors, per_feature, feature_names, predictions, top_k
        ),
    }

    if y is None:
        return results

    results["anomaly_prevalence"] = float(np.mean(y_windows))
    results["model_metrics"] = calculate_metrics(y_windows, predictions, errors)
    results["baseline_metrics"] = evaluate_baselines(
        df, X_windows, y_windows, window_generator, artifacts
    )
    return results


def evaluate_baselines(
    df: pd.DataFrame,
    X_windows: np.ndarray,
    y_windows: np.ndarray,
    window_generator: WindowGenerator,
    artifacts: ModelArtifacts,
) -> dict[str, dict[str, Any]]:
    """
    Evaluate all baseline detectors on the same windows as the model.

    Args:
        df: Cleaned (unscaled) DataFrame the windows were built from
        X_windows: Scaled windows
        y_windows: Window labels
        window_generator: Window generator used for the model windows
        artifacts: Model artifacts (provide calibrated baseline thresholds)

    Returns:
        Mapping of baseline name to metrics
    """
    results: dict[str, dict[str, Any]] = {}

    zscore_threshold = artifacts.baseline_state.get("zscore_threshold")
    if zscore_threshold is not None:
        zscore = ZScoreBaseline(threshold=zscore_threshold)
        results["zscore"] = calculate_metrics(
            y_windows,
            zscore.predict_windows(X_windows),
            ZScoreBaseline.score_windows(X_windows),
        )

    point_preds = ThresholdBaseline().predict(df)
    window_preds = window_generator.windows_from_points(point_preds, df).astype(int)
    results["threshold"] = calculate_metrics(y_windows, window_preds)

    rolling_threshold = artifacts.baseline_state.get("rolling_threshold")
    if rolling_threshold is not None:
        point_scores = RollingBaseline(window_size=artifacts.window_size).score_points(
            df
        )
        window_scores = window_generator.windows_from_points(point_scores, df)
        results["rolling"] = calculate_metrics(
            y_windows, (window_scores > rolling_threshold).astype(int), window_scores
        )

    return results


def _top_windows(
    window_meta: pd.DataFrame,
    errors: np.ndarray,
    per_feature: np.ndarray,
    feature_names: list[str],
    predictions: np.ndarray,
    k: int,
) -> list[dict[str, Any]]:
    """
    Highest-scoring windows with metadata and per-feature attribution.

    ``feature_share`` is each feature's fraction of the window's total
    reconstruction error; ``top_feature`` is the largest contributor.
    """
    order = np.argsort(errors)[::-1][:k]
    top = window_meta.iloc[order].copy()
    top["score"] = errors[order]
    top["flagged"] = predictions[order].astype(bool)
    shares = per_feature[order] / np.maximum(
        per_feature[order].sum(axis=1, keepdims=True), 1e-12
    )
    top["top_feature"] = [feature_names[i] for i in shares.argmax(axis=1)]
    top["feature_share"] = [
        {name: round(float(v), 4) for name, v in zip(feature_names, row, strict=True)}
        for row in shares
    ]
    for col in ("start_timestamp", "end_timestamp"):
        if col in top.columns:
            top[col] = pd.to_datetime(top[col]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return [
        {k: (v.item() if isinstance(v, np.generic) else v) for k, v in row.items()}
        for row in top.to_dict(orient="records")
    ]
