"""
High-Level LOB Generator

Provides a unified interface for generating synthetic LOB snapshots
using any of the three generative model families (VAE, GAN, Diffusion).
"""

import numpy as np
from typing import List, Optional

from .lob_snapshot import LobSnapshot, generate_test_snapshots
from .vae import LobVae, LobVaeTrainer
from .gan import LobGan
from .diffusion import LobDiffusion


class LobGenerator:
    """
    High-level generator for synthetic LOB states.

    Wraps VAE, GAN, or Diffusion models with a simple train/generate interface
    that operates on LobSnapshot objects directly.

    Args:
        num_levels: Number of LOB price levels per side.
        hidden_dim: Hidden layer size for all models.
        latent_dim: Latent/noise dimensionality.
        model_type: One of 'vae', 'gan', 'diffusion'. Default 'vae'.
    """

    def __init__(
        self,
        num_levels: int = 10,
        hidden_dim: int = 64,
        latent_dim: int = 8,
        model_type: str = "vae",
    ):
        self.num_levels = num_levels
        self.input_dim = num_levels * 4
        self.model_type = model_type
        self.reference_mid_price = 50000.0

        if model_type == "vae":
            self.model = LobVae(
                input_dim=self.input_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
            )
            self._trainer = None
        elif model_type == "gan":
            self.model = LobGan(
                input_dim=self.input_dim,
                noise_dim=latent_dim,
                hidden_dim=hidden_dim,
            )
        elif model_type == "diffusion":
            self.model = LobDiffusion(
                input_dim=self.input_dim,
                hidden_dim=hidden_dim,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def _snapshots_to_features(self, snapshots: List[LobSnapshot]) -> np.ndarray:
        """Convert list of snapshots to feature matrix."""
        return np.array(
            [s.to_feature_vector(self.num_levels) for s in snapshots],
            dtype=np.float32,
        )

    def train(
        self,
        snapshots: List[LobSnapshot],
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ) -> List:
        """
        Train the generative model on real LOB snapshots.

        Args:
            snapshots: List of real LobSnapshot objects.
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            learning_rate: Learning rate.

        Returns:
            Training history (list of loss dicts or floats).
        """
        if snapshots:
            self.reference_mid_price = np.mean(
                [s.mid_price() for s in snapshots]
            )

        data = self._snapshots_to_features(snapshots)

        if self.model_type == "vae":
            self._trainer = LobVaeTrainer(
                self.model, learning_rate=learning_rate
            )
            return self._trainer.train(data, epochs=epochs, batch_size=batch_size)
        elif self.model_type == "gan":
            return self.model.train(data, epochs=epochs, batch_size=batch_size)
        elif self.model_type == "diffusion":
            return self.model.train(data, epochs=epochs, batch_size=batch_size)

    def generate_features(self, count: int = 1) -> np.ndarray:
        """Generate synthetic LOB feature vectors."""
        if self.model_type == "vae":
            return self.model.generate(count).numpy()
        elif self.model_type == "gan":
            return self.model.generate(count)
        elif self.model_type == "diffusion":
            return self.model.generate(count)

    def generate_snapshot(self) -> LobSnapshot:
        """Generate a single synthetic LOB snapshot."""
        features = self.generate_features(1)[0]
        return LobSnapshot.from_feature_vector(
            features, self.reference_mid_price, self.num_levels
        )

    def generate_snapshots(self, count: int) -> List[LobSnapshot]:
        """Generate multiple synthetic LOB snapshots."""
        features = self.generate_features(count)
        return [
            LobSnapshot.from_feature_vector(
                features[i], self.reference_mid_price, self.num_levels
            )
            for i in range(count)
        ]


if __name__ == "__main__":
    # Quick test with each model type
    test_data = generate_test_snapshots(50, 10, 50000.0)

    for model_type in ["vae", "gan", "diffusion"]:
        print(f"\n--- Testing {model_type.upper()} generator ---")
        gen = LobGenerator(num_levels=10, hidden_dim=32, latent_dim=8, model_type=model_type)
        history = gen.train(test_data, epochs=3, batch_size=16)
        print(f"  Training complete ({len(history)} epochs)")

        snapshots = gen.generate_snapshots(5)
        print(f"  Generated {len(snapshots)} snapshots")
        print(f"  Sample mid-price: {snapshots[0].mid_price():.2f}")
        print(f"  Sample spread: {snapshots[0].spread():.4f}")

    print("\nGenerator test passed!")
