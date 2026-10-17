"""Loading and splitting metric datasets."""

from pathlib import Path

import numpy as np
import pandas as pd

from devwaste.data.features import FeatureEngineer, series_keys


def load_dataset(path: Path, feature_engineer: FeatureEngineer | None = None) -> pd.DataFrame:
    """
    Load a metrics CSV, validate its schema, and clean it.

    Args:
        path: CSV produced by ``devwaste simulate`` or ``devwaste collect``
        feature_engineer: Engineer used for validation/cleaning

    Returns:
        Cleaned DataFrame sorted by timestamp

    Raises:
        ValueError: If the file is empty or required feature columns are missing
    """
    feature_engineer = feature_engineer or FeatureEngineer()
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")

    is_valid, missing = feature_engineer.validate_schema(df)
    if not is_valid:
        raise ValueError(f"Dataset {path} is missing required columns: {missing}")

    return feature_engineer.clean_data(df)


def get_labels(df: pd.DataFrame) -> np.ndarray | None:
    """Return integer labels if the dataset has a ``label`` column, else None."""
    if "label" not in df.columns:
        return None
    return df["label"].fillna(0).astype(int).to_numpy()


def chronological_split(
    df: pd.DataFrame, validation_split: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split each series in time order: the first part trains, the rest validates.

    Splitting happens on raw observations before any windowing, so no window
    can contain both training and validation samples.

    Args:
        df: Cleaned DataFrame sorted by timestamp
        validation_split: Fraction of each series used for validation

    Returns:
        (train_df, val_df)
    """
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1")

    keys = series_keys(df)
    if keys:
        position = df.groupby(keys, sort=False, dropna=False).cumcount()
        size = df.groupby(keys, sort=False, dropna=False)[df.columns[0]].transform("size")
    else:
        position = pd.Series(np.arange(len(df)), index=df.index)
        size = pd.Series(len(df), index=df.index)

    is_train = position < (size * (1 - validation_split)).astype(int)
    return df[is_train].reset_index(drop=True), df[~is_train].reset_index(drop=True)
