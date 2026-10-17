"""Tests for LSTM Autoencoder model."""

import pytest
import torch

from devwaste.models.lstm_autoencoder import LSTMAutoencoder


class TestLSTMAutoencoder:
    """Test LSTMAutoencoder class."""

    @pytest.fixture
    def sample_input(self):
        torch.manual_seed(0)
        return torch.randn(32, 10, 6)

    def test_model_initialization(self):
        model = LSTMAutoencoder(input_dim=6, hidden_dim=16)

        assert model.input_dim == 6
        assert model.hidden_dim == 16
        assert model.num_layers == 1

    def test_forward_pass(self, sample_input):
        model = LSTMAutoencoder(input_dim=6, hidden_dim=16)

        assert model(sample_input).shape == sample_input.shape

    def test_encode_decode(self, sample_input):
        model = LSTMAutoencoder(input_dim=6, hidden_dim=16)

        latent = model.encode(sample_input)
        reconstructed = model.decode(latent, sample_input.shape[1])

        assert latent.shape == (32, 16)
        assert reconstructed.shape == sample_input.shape

    def test_reconstruction_error(self, sample_input):
        model = LSTMAutoencoder(input_dim=6, hidden_dim=16)

        errors = model.calculate_reconstruction_error(sample_input)

        assert errors.shape == (sample_input.shape[0],)
        assert (errors >= 0).all()

    @pytest.mark.parametrize("num_layers", [2, 3])
    def test_multiple_layers(self, num_layers):
        model = LSTMAutoencoder(input_dim=6, hidden_dim=16, num_layers=num_layers)
        x = torch.randn(8, 10, 6)

        assert model(x).shape == x.shape

    def test_dropout_only_between_layers(self):
        """Dropout applies between stacked layers and is disabled in eval mode."""
        single = LSTMAutoencoder(input_dim=6, hidden_dim=16, dropout=0.5)
        assert single.encoder.dropout == 0.0

        torch.manual_seed(0)
        model = LSTMAutoencoder(input_dim=6, hidden_dim=16, num_layers=2, dropout=0.5)
        x = torch.randn(8, 10, 6)

        model.train()
        assert not torch.allclose(model(x), model(x))

        model.eval()
        assert torch.allclose(model(x), model(x))

    def test_can_overfit_small_batch(self):
        """Sanity check: training reduces reconstruction loss."""
        torch.manual_seed(0)
        model = LSTMAutoencoder(input_dim=3, hidden_dim=16)
        x = torch.sin(torch.linspace(0, 6, 10)).repeat(16, 3, 1).transpose(1, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        initial = model.get_loss(x, model(x)).item()
        for _ in range(100):
            optimizer.zero_grad()
            loss = model.get_loss(x, model(x))
            loss.backward()
            optimizer.step()

        assert loss.item() < initial * 0.5
