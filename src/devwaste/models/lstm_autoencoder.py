"""LSTM Autoencoder model for anomaly detection."""

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """
    LSTM-based Autoencoder for time-series anomaly detection.

    The encoder compresses an input window into its final hidden state (the
    latent vector). The decoder receives that latent vector repeated at every
    timestep and reconstructs the window. Anomalies are detected based on
    high reconstruction error.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        """
        Initialize LSTM Autoencoder.

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden (latent) dimension size
            num_layers: Number of LSTM layers in encoder and decoder
            dropout: Dropout between stacked LSTM layers (only used when
                num_layers > 1, as in ``torch.nn.LSTM``)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence to latent representation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            latent: Last layer's final hidden state, shape (batch_size, hidden_dim)
        """
        _, (hidden, _) = self.encoder(x)
        return hidden[-1]

    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Decode latent representation to an output sequence.

        Args:
            latent: Latent tensor of shape (batch_size, hidden_dim)
            seq_len: Desired output sequence length

        Returns:
            reconstructed: Tensor of shape (batch_size, seq_len, input_dim)
        """
        decoder_input = latent.unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(decoder_input)
        return self.output_proj(decoded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            reconstructed: Reconstructed tensor of same shape as input
        """
        return self.decode(self.encode(x), x.shape[1])

    def calculate_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate reconstruction error for input sequences.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            errors: Per-window mean squared error of shape (batch_size,)
        """
        reconstructed = self.forward(x)
        return torch.mean((x - reconstructed) ** 2, dim=(1, 2))

    def get_loss(self, x: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Calculate reconstruction loss.

        Args:
            x: Original input
            reconstructed: Reconstructed output

        Returns:
            loss: Mean squared error loss
        """
        return nn.functional.mse_loss(reconstructed, x)
