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


def apply_persistence_filter(
    predictions: np.ndarray,
    window_meta: pd.DataFrame,
    consecutive: int = 1,
) -> np.ndarray:
    """
    Apply persistence filter to window predictions per series.

    A window is flagged only if at least `consecutive` consecutive windows
    exceeded the threshold.
    """
    if consecutive <= 1 or len(predictions) == 0:
        return predictions

    filtered = np.zeros_like(predictions)
    keys = [k for k in ("container_id", "run_id") if k in window_meta.columns]

    def _filter_series(indices: np.ndarray):
        s_preds = pd.Series(predictions[indices])
        rolling_count = s_preds.rolling(
            window=consecutive, min_periods=consecutive
        ).sum()
        confirmed = (rolling_count == consecutive).to_numpy()
        filtered[indices] = confirmed.astype(int)

    if keys:
        for idxs in window_meta.groupby(keys, sort=False).indices.values():
            _filter_series(np.asarray(idxs))
    else:
        _filter_series(np.arange(len(predictions)))

    return filtered


def find_recovery_windows(
    y_windows: np.ndarray,
    window_meta: pd.DataFrame,
    recovery_margin: float,
) -> np.ndarray:
    """
    Identify normal windows that fall within `recovery_margin` seconds after an incident.
    """
    if recovery_margin <= 0 or len(y_windows) == 0:
        return np.zeros(len(y_windows), dtype=bool)

    recovery_mask = np.zeros(len(y_windows), dtype=bool)
    has_ts = (
        "start_timestamp" in window_meta.columns
        and "end_timestamp" in window_meta.columns
    )
    keys = [k for k in ("container_id", "run_id") if k in window_meta.columns]

    def _mark_series(indices: np.ndarray):
        s_y = y_windows[indices]
        s_meta = window_meta.iloc[indices]
        if has_ts:
            starts = pd.to_datetime(s_meta["start_timestamp"], utc=True)
            ends = pd.to_datetime(s_meta["end_timestamp"], utc=True)
        else:
            starts = s_meta["start_sample"].to_numpy()
            ends = s_meta["end_sample"].to_numpy()

        last_incident_end = None
        for i, idx in enumerate(indices):
            if s_y[i] == 1:
                last_incident_end = ends.iloc[i] if has_ts else ends[i]
            elif last_incident_end is not None:
                start_i = starts.iloc[i] if has_ts else starts[i]
                if has_ts:
                    diff_sec = (start_i - last_incident_end).total_seconds()
                else:
                    diff_sec = float(start_i - last_incident_end)
                if 0 <= diff_sec <= recovery_margin:
                    recovery_mask[idx] = True
                else:
                    last_incident_end = None

    if keys:
        for idxs in window_meta.groupby(keys, sort=False).indices.values():
            _mark_series(np.asarray(idxs))
    else:
        _mark_series(np.arange(len(y_windows)))

    return recovery_mask


def evaluate_model(
    data_path: Path,
    model_dir: Path,
    top_k: int = 10,
    consecutive: int = 1,
    recovery_margin: float = 0.0,
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
        consecutive: Minimum consecutive anomalous windows required for an alert
        recovery_margin: Seconds after an incident to treat as recovery transition

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
    raw_predictions = (errors > threshold).astype(int)
    predictions = apply_persistence_filter(
        raw_predictions, window_meta, consecutive=consecutive
    )
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
        "consecutive": consecutive,
        "recovery_margin": recovery_margin,
        "flagged_by_feature": {k: int(v) for k, v in flagged_drivers.items()},
        "top_windows": _top_windows(
            window_meta, errors, per_feature, feature_names, predictions, top_k
        ),
    }

    if y is None:
        return results

    recovery_mask = find_recovery_windows(
        y_windows, window_meta, recovery_margin=recovery_margin
    )
    eval_mask = ~recovery_mask
    results["n_recovery_windows_excluded"] = int(recovery_mask.sum())
    results["anomaly_prevalence"] = (
        float(np.mean(y_windows[eval_mask])) if eval_mask.any() else 0.0
    )
    results["model_metrics"] = calculate_metrics(
        y_windows[eval_mask], predictions[eval_mask], errors[eval_mask]
    )
    results["baseline_metrics"] = evaluate_baselines(
        df,
        X_windows,
        y_windows,
        window_generator,
        artifacts,
        eval_mask=eval_mask,
        consecutive=consecutive,
        window_meta=window_meta,
    )
    return results


def evaluate_baselines(
    df: pd.DataFrame,
    X_windows: np.ndarray,
    y_windows: np.ndarray,
    window_generator: WindowGenerator,
    artifacts: ModelArtifacts,
    eval_mask: np.ndarray | None = None,
    consecutive: int = 1,
    window_meta: pd.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Evaluate all baseline detectors on the same windows as the model.

    Args:
        df: Cleaned (unscaled) DataFrame the windows were built from
        X_windows: Scaled windows
        y_windows: Window labels
        window_generator: Window generator used for the model windows
        artifacts: Model artifacts (provide calibrated baseline thresholds)
        eval_mask: Optional boolean mask of windows to evaluate on
        consecutive: Minimum consecutive anomalous windows required for an alert
        window_meta: Optional window metadata DataFrame (required if consecutive > 1)

    Returns:
        Mapping of baseline name to metrics
    """
    results: dict[str, dict[str, Any]] = {}
    if eval_mask is None:
        eval_mask = np.ones(len(y_windows), dtype=bool)

    zscore_threshold = artifacts.baseline_state.get("zscore_threshold")
    if zscore_threshold is not None:
        zscore = ZScoreBaseline(threshold=zscore_threshold)
        z_preds = zscore.predict_windows(X_windows)
        if consecutive > 1 and window_meta is not None:
            z_preds = apply_persistence_filter(
                z_preds, window_meta, consecutive=consecutive
            )
        results["zscore"] = calculate_metrics(
            y_windows[eval_mask],
            z_preds[eval_mask],
            ZScoreBaseline.score_windows(X_windows)[eval_mask],
        )

    point_preds = ThresholdBaseline().predict(df)
    window_preds = window_generator.windows_from_points(point_preds, df).astype(int)
    if consecutive > 1 and window_meta is not None:
        window_preds = apply_persistence_filter(
            window_preds, window_meta, consecutive=consecutive
        )
    results["threshold"] = calculate_metrics(
        y_windows[eval_mask], window_preds[eval_mask]
    )

    rolling_threshold = artifacts.baseline_state.get("rolling_threshold")
    if rolling_threshold is not None:
        point_scores = RollingBaseline(window_size=artifacts.window_size).score_points(
            df
        )
        window_scores = window_generator.windows_from_points(point_scores, df)
        rolling_preds = (window_scores > rolling_threshold).astype(int)
        if consecutive > 1 and window_meta is not None:
            rolling_preds = apply_persistence_filter(
                rolling_preds, window_meta, consecutive=consecutive
            )
        results["rolling"] = calculate_metrics(
            y_windows[eval_mask],
            rolling_preds[eval_mask],
            window_scores[eval_mask],
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
