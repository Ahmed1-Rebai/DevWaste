"""Real-time online container anomaly monitoring."""

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from devwaste.collectors.docker_collector import ContainerMetric
from devwaste.data.features import FeatureEngineer
from devwaste.models.artifacts import ModelArtifacts
from devwaste.models.train import feature_errors


class OnlineMonitor:
    """
    Online time-series anomaly monitor for Docker containers.

    Maintains per-container sliding window buffers, converts streaming
    cumulative byte counters to deltas, and scores each completed window
    against a trained LSTM Autoencoder with per-feature attribution.
    """

    def __init__(
        self,
        model_dir: Path,
        threshold_multiplier: float = 1.0,
        consecutive: int = 1,
    ):
        """
        Initialize the monitor with a trained model.

        Args:
            model_dir: Directory containing trained model artifacts
            threshold_multiplier: Multiplier on the calibrated threshold
            consecutive: Number of consecutive anomalous windows required for alert
        """
        self.artifacts = ModelArtifacts.load(model_dir)
        self.feature_engineer = FeatureEngineer()
        self.feature_engineer.set_state(self.artifacts.feature_engineer_state)
        self.model = self.artifacts.build_model()

        self.threshold = float(self.artifacts.threshold) * threshold_multiplier
        self.window_size = int(self.artifacts.window_size)
        self.consecutive = max(1, consecutive)

        self.prev_metrics: dict[str, ContainerMetric] = {}
        self.window_buffers: dict[str, deque[dict[str, float]]] = {}
        self.streak: dict[str, int] = {}

    def process_metric(self, metric: ContainerMetric) -> dict[str, Any]:
        """
        Process a single container metric observation.

        Args:
            metric: Observation from DockerCollector

        Returns:
            Dictionary containing window evaluation results or warming status
        """
        cid = metric.container_id
        prev = self.prev_metrics.get(cid)
        self.prev_metrics[cid] = metric

        if prev is None:
            # First observation has no predecessor to calculate byte deltas
            return {
                "status": "warming",
                "container_id": cid,
                "container_name": metric.container_name,
                "timestamp": metric.timestamp,
                "samples": 0,
                "needed": self.window_size,
            }

        # Calculate per-interval byte deltas, clipping counter resets at 0
        delta_rx = max(0.0, metric.network_rx_bytes - prev.network_rx_bytes)
        delta_tx = max(0.0, metric.network_tx_bytes - prev.network_tx_bytes)
        delta_block_read = max(0.0, metric.block_read_bytes - prev.block_read_bytes)
        delta_block_write = max(0.0, metric.block_write_bytes - prev.block_write_bytes)

        point = {
            "cpu_percent": float(metric.cpu_percent),
            "memory_percent": float(metric.memory_percent),
            "network_rx_bytes": float(delta_rx),
            "network_tx_bytes": float(delta_tx),
            "block_read_bytes": float(delta_block_read),
            "block_write_bytes": float(delta_block_write),
        }

        if cid not in self.window_buffers:
            self.window_buffers[cid] = deque(maxlen=self.window_size)
            self.streak[cid] = 0

        buf = self.window_buffers[cid]
        buf.append(point)

        if len(buf) < self.window_size:
            return {
                "status": "warming",
                "container_id": cid,
                "container_name": metric.container_name,
                "timestamp": metric.timestamp,
                "samples": len(buf),
                "needed": self.window_size,
            }

        # Form window DataFrame and scale
        df_win = pd.DataFrame(list(buf), columns=self.feature_engineer.feature_names)
        X_scaled = self.feature_engineer.transform(df_win)
        X_window = X_scaled[np.newaxis, :, :]  # shape (1, window_size, n_features)

        per_feature = feature_errors(self.model, X_window)
        error = float(per_feature.mean())
        raw_anomalous = bool(error > self.threshold)

        if raw_anomalous:
            self.streak[cid] += 1
        else:
            self.streak[cid] = 0

        is_alert = self.streak[cid] >= self.consecutive

        shares = per_feature[0] / max(float(per_feature[0].sum()), 1e-12)
        top_idx = int(per_feature[0].argmax())
        feature_names = self.feature_engineer.feature_names
        top_feature = feature_names[top_idx]
        top_share = float(shares[top_idx])

        return {
            "status": "ready",
            "timestamp": metric.timestamp,
            "container_id": cid,
            "container_name": metric.container_name,
            "window_size": self.window_size,
            "score": round(error, 4),
            "threshold": round(self.threshold, 4),
            "is_anomalous": is_alert,
            "raw_anomalous": raw_anomalous,
            "streak": self.streak[cid],
            "top_feature": top_feature,
            "top_feature_share": round(top_share, 4),
            "metrics": {
                "cpu_percent": round(float(metric.cpu_percent), 2),
                "memory_percent": round(float(metric.memory_percent), 2),
                "network_rx_bytes": delta_rx,
                "network_tx_bytes": delta_tx,
                "block_read_bytes": delta_block_read,
                "block_write_bytes": delta_block_write,
            },
        }

    def process_round(self, metrics: list[ContainerMetric]) -> list[dict[str, Any]]:
        """
        Process a batch of metrics from a single collection round.

        Args:
            metrics: List of observations from all running containers

        Returns:
            List of results for each container
        """
        return [self.process_metric(m) for m in metrics]
