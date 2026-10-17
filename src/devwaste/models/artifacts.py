"""Model artifacts serialization and loading."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from devwaste import __version__


@dataclass
class ModelArtifacts:
    """
    Container for all model-related artifacts.

    Saved as two files: ``model.pt`` (PyTorch state dict, loaded with
    ``weights_only=True``) and ``metadata.json`` (everything else).
    """

    model_state_dict: dict[str, Any]
    feature_engineer_state: dict[str, Any]
    window_size: int
    hidden_dim: int
    input_dim: int
    threshold: float
    threshold_method: str
    training_config: dict[str, Any]
    num_layers: int = 1
    baseline_state: dict[str, Any] = field(default_factory=dict)
    model_version: str = __version__

    def save(self, output_dir: Path):
        """
        Save all artifacts to directory.

        Args:
            output_dir: Directory to save artifacts
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model_state_dict, output_dir / "model.pt")

        metadata = {
            "model_version": self.model_version,
            "window_size": self.window_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "input_dim": self.input_dim,
            "threshold": float(self.threshold),
            "threshold_method": self.threshold_method,
            "training_config": self.training_config,
            "feature_engineer_state": self.feature_engineer_state,
            "baseline_state": self.baseline_state,
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, input_dir: Path) -> "ModelArtifacts":
        """
        Load artifacts from directory.

        Args:
            input_dir: Directory containing artifacts

        Returns:
            ModelArtifacts instance
        """
        metadata_path = input_dir / "metadata.json"
        model_path = input_dir / "model.pt"
        for path in (metadata_path, model_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing model artifact: {path}")

        with open(metadata_path) as f:
            metadata = json.load(f)

        model_state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

        return cls(
            model_state_dict=model_state_dict,
            feature_engineer_state=metadata["feature_engineer_state"],
            window_size=metadata["window_size"],
            hidden_dim=metadata["hidden_dim"],
            input_dim=metadata["input_dim"],
            threshold=metadata["threshold"],
            threshold_method=metadata["threshold_method"],
            training_config=metadata["training_config"],
            num_layers=metadata.get("num_layers", 1),
            baseline_state=metadata.get("baseline_state", {}),
            model_version=metadata.get("model_version", "0.1.0"),
        )

    def build_model(self):
        """Instantiate the LSTM Autoencoder with the stored weights (eval mode)."""
        from devwaste.models.lstm_autoencoder import LSTMAutoencoder

        model = LSTMAutoencoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
        model.load_state_dict(self.model_state_dict)
        model.eval()
        return model
