"""Model training pipeline."""

import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from devwaste.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_THRESHOLD_QUANTILE,
    DEFAULT_WINDOW_SIZE,
)
from devwaste.data.dataset import chronological_split, get_labels, load_dataset
from devwaste.data.features import FeatureEngineer, series_keys
from devwaste.data.windows import WindowGenerator
from devwaste.models.artifacts import ModelArtifacts
from devwaste.models.baselines import RollingBaseline, ZScoreBaseline
from devwaste.models.lstm_autoencoder import LSTMAutoencoder


def set_seed(seed: int):
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def feature_errors(
    model: LSTMAutoencoder, X_windows: np.ndarray, batch_size: int = 1024
) -> np.ndarray:
    """
    Per-window, per-feature reconstruction errors, shape (n_windows, n_features).

    Each entry is the squared error of one feature averaged over the window's
    timesteps; the mean over features equals the window's overall MSE.
    """
    device = next(model.parameters()).device
    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(X_windows), batch_size):
            batch = torch.as_tensor(
                X_windows[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            squared = (batch - model(batch)) ** 2
            errors.append(squared.mean(dim=1).cpu().numpy())
    if not errors:
        return np.zeros((0, X_windows.shape[-1] if X_windows.ndim == 3 else 0))
    return np.concatenate(errors)


def reconstruction_errors(
    model: LSTMAutoencoder, X_windows: np.ndarray, batch_size: int = 1024
) -> np.ndarray:
    """Per-window reconstruction errors (MSE), computed in batches."""
    return feature_errors(model, X_windows, batch_size).mean(axis=1)


def calibrate_evt_threshold(
    errors: np.ndarray,
    initial_quantile: float = 0.90,
    target_quantile: float = 0.95,
) -> float:
    """
    Calibrate threshold using Extreme Value Theory (Peaks-Over-Threshold / GPD).

    Fits a Generalized Pareto Distribution to the tail excesses above an
    initial quantile u, then extrapolates the extreme threshold at
    target_quantile.

    Args:
        errors: 1D array of validation reconstruction errors
        initial_quantile: Quantile used as the POT base threshold u
        target_quantile: Desired tail quantile for anomaly threshold

    Returns:
        Estimated threshold float
    """
    errors = np.asarray(errors, dtype=float)
    if len(errors) < 10:
        return float(np.quantile(errors, target_quantile))

    u = float(np.quantile(errors, initial_quantile))
    excesses = errors[errors > u] - u
    n = len(errors)
    n_u = len(excesses)

    if n_u < 5:
        return float(np.quantile(errors, target_quantile))

    mean_y = float(np.mean(excesses))
    var_y = float(np.var(excesses, ddof=1))

    if var_y <= 1e-12:
        return float(np.quantile(errors, target_quantile))

    gamma = float(np.clip(0.5 * (mean_y**2 / var_y - 1.0), -0.5, 0.5))
    sigma = float(0.5 * mean_y * (mean_y**2 / var_y + 1.0))

    if sigma <= 0:
        return float(np.quantile(errors, target_quantile))

    tail_prob = max(1.0 - target_quantile, 1e-6)
    ratio = (n / n_u) * tail_prob

    if abs(gamma) < 1e-4:
        z_q = u - sigma * np.log(ratio)
    else:
        z_q = u + (sigma / gamma) * (ratio ** (-gamma) - 1.0)

    return float(max(z_q, u))


def train_model(
    data_path: Path,
    output_dir: Path,
    window_size: int = DEFAULT_WINDOW_SIZE,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    num_layers: int = 1,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    threshold_quantile: float = DEFAULT_THRESHOLD_QUANTILE,
    threshold_method: str = "quantile",
    validation_split: float = 0.2,
    early_stopping_patience: int = 10,
    seed: int = DEFAULT_SEED,
    device: str | None = None,
    log: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """
    Train the LSTM Autoencoder on normal behaviour.

    Pipeline: load and clean -> chronological split per series -> fit scaler
    on normal training rows -> window each split per series -> drop windows
    containing labelled anomalies -> train with early stopping on validation
    loss -> calibrate thresholds on normal validation windows -> save.

    Args:
        data_path: Path to training data CSV. A ``label`` column is optional;
            when present, anomalous observations are excluded from training
            and calibration. Without it, all data is assumed normal.
        output_dir: Directory to save model artifacts
        window_size: Sequence length
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        epochs: Maximum number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        threshold_quantile: Quantile of validation errors used as threshold
        validation_split: Fraction of each series used for validation
        early_stopping_patience: Epochs without validation improvement before stopping
        seed: Random seed for reproducibility
        device: ``"cpu"``, ``"cuda"``, or None to auto-detect
        log: Progress callback (None to silence)

    Returns:
        Dictionary with training results
    """
    log = log or (lambda _msg: None)
    set_seed(seed)
    torch_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    feature_engineer = FeatureEngineer()
    df = load_dataset(data_path, feature_engineer)
    if get_labels(df) is None:
        df["label"] = 0

    train_df, val_df = chronological_split(df, validation_split)

    keys = series_keys(val_df)
    longest_val = val_df.groupby(keys).size().max() if keys else len(val_df)
    if longest_val < window_size:
        min_rows = math.ceil(window_size / validation_split)
        raise ValueError(
            f"Validation split too short: no series has {window_size} validation "
            f"samples. Each container/run needs at least ~{min_rows} rows; collect "
            "for longer, reduce --window-size, or raise --validation-split"
        )

    # Fit the scaler on normal training observations only
    normal_train = train_df[train_df["label"] == 0]
    if normal_train.empty:
        raise ValueError("Training split contains no normal observations")
    feature_engineer.fit(normal_train)

    window_generator = WindowGenerator(window_size=window_size, stride=1)
    X_train, y_train, _ = window_generator.create_windows_by_series(
        feature_engineer.transform(train_df), train_df, train_df["label"].to_numpy()
    )
    X_val, y_val, _ = window_generator.create_windows_by_series(
        feature_engineer.transform(val_df), val_df, val_df["label"].to_numpy()
    )

    n_excluded = int(y_train.sum() + y_val.sum())
    X_train, X_val = X_train[y_train == 0], X_val[y_val == 0]
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError(
            "Not enough normal windows for training/validation; "
            "provide more normal data or reduce --window-size"
        )

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.as_tensor(X_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    X_val_tensor = torch.as_tensor(X_val, dtype=torch.float32, device=torch_device)

    input_dim = X_train.shape[2]
    model = LSTMAutoencoder(
        input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    log(f"Training on {torch_device}")
    log(
        f"Normal windows - train: {len(X_train)}, validation: {len(X_val)} "
        f"(excluded {n_excluded} windows containing labelled anomalies)"
    )

    best_val_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    patience_counter = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (x,) in train_loader:
            x = x.to(torch_device)
            optimizer.zero_grad()
            loss = model.get_loss(x, model(x))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(x)
        train_loss = epoch_loss / len(X_train)

        model.eval()
        with torch.no_grad():
            val_loss = model.get_loss(X_val_tensor, model(X_val_tensor)).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch == 0 or (epoch + 1) % 10 == 0:
            log(
                f"Epoch {epoch + 1}/{epochs} - train loss {train_loss:.4f}, val loss {val_loss:.4f}"
            )

        if patience_counter >= early_stopping_patience:
            log(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)

    val_errors = reconstruction_errors(model, X_val)
    if threshold_method == "evt":
        threshold = calibrate_evt_threshold(
            val_errors,
            initial_quantile=min(0.90, threshold_quantile),
            target_quantile=threshold_quantile,
        )
        threshold_method_name = f"evt_{threshold_quantile}"
    else:
        threshold = float(np.quantile(val_errors, threshold_quantile))
        threshold_method_name = f"quantile_{threshold_quantile}"

    zscore_threshold = ZScoreBaseline().calibrate(X_val, threshold_quantile)
    rolling_scores = window_generator.windows_from_points(
        RollingBaseline(window_size=window_size).score_points(val_df), val_df
    )
    rolling_threshold = float(
        np.quantile(rolling_scores[y_val == 0], threshold_quantile)
    )
    log(f"Threshold ({threshold_method_name}): {threshold:.4f}")

    results = {
        "final_train_loss": train_losses[-1],
        "best_val_loss": best_val_loss,
        "threshold": threshold,
        "threshold_method": threshold_method_name,
        "n_train_windows": int(len(X_train)),
        "n_val_windows": int(len(X_val)),
        "n_excluded_anomalous_windows": n_excluded,
        "epochs_trained": len(train_losses),
    }

    artifacts = ModelArtifacts(
        model_state_dict={k: v.cpu() for k, v in best_state.items()},
        feature_engineer_state=feature_engineer.get_state(),
        window_size=window_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        input_dim=input_dim,
        threshold=threshold,
        threshold_method=f"quantile_{threshold_quantile}",
        baseline_state={
            "zscore_threshold": zscore_threshold,
            "rolling_threshold": rolling_threshold,
        },
        training_config={
            "data_path": str(data_path),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "validation_split": validation_split,
            "early_stopping_patience": early_stopping_patience,
            "seed": seed,
            **results,
            "train_loss_history": train_losses,
            "val_loss_history": val_losses,
        },
    )
    artifacts.save(output_dir)

    return results
