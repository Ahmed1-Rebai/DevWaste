"""Sliding window utilities for time-series data."""

import numpy as np
import pandas as pd

from devwaste.data.features import series_keys


class WindowGenerator:
    """Generate sliding windows from time-series data."""

    def __init__(self, window_size: int = 10, stride: int = 1):
        """Initialize window generator."""
        if window_size < 1 or stride < 1:
            raise ValueError("window_size and stride must be >= 1")
        self.window_size = window_size
        self.stride = stride

    def n_windows(self, n_samples: int) -> int:
        """Number of windows that fit in ``n_samples`` observations."""
        if n_samples < self.window_size:
            return 0
        return (n_samples - self.window_size) // self.stride + 1

    def create_windows(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        metadata: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Create sliding windows from a single contiguous series.

        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Labels array of shape (n_samples,)
            metadata: DataFrame with additional metadata (timestamps, container_id, etc.)

        Returns:
            X_windows: Array of shape (n_windows, window_size, n_features)
            y_windows: Array of shape (n_windows,) - max label in window
            window_metadata: DataFrame with window-level metadata
        """
        n_samples = X.shape[0]

        if n_samples < self.window_size:
            raise ValueError(
                f"Not enough samples ({n_samples}) for window size {self.window_size}"
            )

        n_windows = self.n_windows(n_samples)
        starts = np.arange(n_windows) * self.stride
        ends = starts + self.window_size

        idx = starts[:, None] + np.arange(self.window_size)[None, :]
        X_windows = np.asarray(X, dtype=float)[idx]

        if y is not None:
            # If any sample in the window is anomalous, the window is anomalous
            y_windows = np.asarray(y, dtype=int)[idx].max(axis=1)
        else:
            y_windows = np.zeros(n_windows, dtype=int)

        window_meta_dict = {
            "window_index": np.arange(n_windows),
            "start_sample": starts,
            "end_sample": ends,
        }

        if metadata is not None:
            if "timestamp" in metadata.columns:
                window_meta_dict["start_timestamp"] = metadata["timestamp"].to_numpy()[starts]
                window_meta_dict["end_timestamp"] = metadata["timestamp"].to_numpy()[ends - 1]
            for col in ("container_id", "container_name", "run_id", "scenario"):
                if col in metadata.columns:
                    window_meta_dict[col] = metadata[col].to_numpy()[starts]

        window_metadata = pd.DataFrame(window_meta_dict)

        return X_windows, y_windows, window_metadata

    def create_windows_by_series(
        self,
        X: np.ndarray,
        df: pd.DataFrame,
        y: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Create windows independently for each series in ``df``.

        Series are identified by ``container_id``/``run_id`` columns when
        present, so no window mixes observations from different containers
        or simulation runs. Series shorter than the window are skipped.

        Args:
            X: Feature array aligned row-for-row with ``df``
            df: DataFrame providing series keys and metadata
            y: Optional labels aligned with ``df``

        Returns:
            Same as :meth:`create_windows`, concatenated over series.
        """
        keys = series_keys(df)
        if not keys:
            return self.create_windows(X, y, df)

        X_parts, y_parts, meta_parts = [], [], []
        for group_pos in df.groupby(keys, sort=False, dropna=False).indices.values():
            if len(group_pos) < self.window_size:
                continue
            Xw, yw, meta = self.create_windows(
                X[group_pos],
                None if y is None else np.asarray(y)[group_pos],
                df.iloc[group_pos],
            )
            meta["start_sample"] = group_pos[meta["start_sample"].to_numpy()]
            meta["end_sample"] = group_pos[meta["end_sample"].to_numpy() - 1] + 1
            X_parts.append(Xw)
            y_parts.append(yw)
            meta_parts.append(meta)

        if not X_parts:
            raise ValueError(
                f"No series (container/run) has at least {self.window_size} samples "
                "for windowing; collect for longer or use a smaller --window-size"
            )

        window_metadata = pd.concat(meta_parts, ignore_index=True)
        window_metadata["window_index"] = np.arange(len(window_metadata))
        return np.concatenate(X_parts), np.concatenate(y_parts), window_metadata

    def create_windows_from_df(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        label_column: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Create windows directly from DataFrame.

        Args:
            df: DataFrame with features and optional labels
            feature_columns: List of column names to use as features
            label_column: Optional column name for labels

        Returns:
            X_windows: Array of shape (n_windows, window_size, n_features)
            y_windows: Array of shape (n_windows,)
            window_metadata: DataFrame with window metadata
        """
        X = df[feature_columns].to_numpy(dtype=float)

        if label_column and label_column in df.columns:
            y = df[label_column].to_numpy()
        else:
            y = None

        return self.create_windows_by_series(X, df, y)

    def windows_from_points(self, point_values: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """
        Aggregate point-level values to windows (max over each window).

        Uses the same series grouping as :meth:`create_windows_by_series`,
        so the output aligns with windows built from the same ``df``.
        """
        values = np.asarray(point_values, dtype=float).reshape(-1, 1)
        Xw, _, _ = self.create_windows_by_series(values, df)
        return Xw[:, :, 0].max(axis=1)
