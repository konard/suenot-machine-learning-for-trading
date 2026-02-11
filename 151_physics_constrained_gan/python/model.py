"""
Physics-Constrained GAN Model Module

This module implements the core components of a Physics-Constrained WGAN-GP
for generating realistic financial time series.

Components:
    - Generator: Temporal convolutional generator producing synthetic returns
    - Critic: 1D convolutional critic for Wasserstein distance estimation
    - PhysicsLoss: Financial physics constraint penalties
    - PhysicsConstrainedGAN: Combined model with training utilities

Key Physics Constraints:
    - Martingale property (no-arbitrage)
    - Volatility clustering (GARCH-like ACF of |r|)
    - Fat tails (excess kurtosis matching)
    - Leverage effect (negative return-volatility correlation)
    - Autocorrelation structure (zero ACF for r, positive ACF for r^2)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")


@dataclass
class GANConfig:
    """Configuration for Physics-Constrained GAN."""

    # Architecture
    latent_dim: int = 128
    seq_len: int = 256
    gen_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    n_features: int = 1  # Number of output features (1 = univariate returns)

    # Conditional generation
    n_regimes: int = 4   # bull, bear, sideways, crisis
    n_vol_levels: int = 4  # low, medium, high, extreme
    cond_embed_dim: int = 16
    use_conditional: bool = True

    # WGAN-GP parameters
    n_critic: int = 5         # Critic updates per generator update
    gp_lambda: float = 10.0   # Gradient penalty coefficient
    lr_gen: float = 1e-4
    lr_critic: float = 1e-4
    beta1: float = 0.0        # Adam beta1 (0 for WGAN-GP)
    beta2: float = 0.9        # Adam beta2

    # Physics constraint weights
    lambda_martingale: float = 1.0
    lambda_volatility: float = 0.5
    lambda_kurtosis: float = 0.3
    lambda_leverage: float = 0.2
    lambda_autocorr: float = 0.3
    lambda_moment: float = 0.2

    # Physics targets (computed from real data, defaults are typical crypto)
    target_kurtosis: float = 8.0
    target_leverage_corr: float = -0.25
    target_vol_acf_lags: int = 20
    target_return_acf_lags: int = 10

    # Training
    batch_size: int = 64
    epochs: int = 500
    device: str = "cpu"


def _check_torch():
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for model operations. "
            "Install with: pip install torch"
        )


class Generator(nn.Module):
    """
    Temporal Convolutional Generator for financial time series.

    Takes a latent noise vector z (and optional condition embedding)
    and produces a sequence of synthetic log-returns.

    Architecture:
        Linear -> Reshape -> [ConvTranspose1d -> BN -> LeakyReLU] x N -> Tanh
    """

    def __init__(self, config: GANConfig):
        _check_torch()
        super().__init__()
        self.config = config

        # Condition embedding
        cond_dim = 0
        if config.use_conditional:
            self.regime_embed = nn.Embedding(config.n_regimes, config.cond_embed_dim)
            self.vol_embed = nn.Embedding(config.n_vol_levels, config.cond_embed_dim)
            cond_dim = config.cond_embed_dim * 2

        input_dim = config.latent_dim + cond_dim
        init_length = config.seq_len // (2 ** len(config.gen_hidden_dims))
        self.init_channels = config.gen_hidden_dims[0]
        self.init_length = init_length

        # Project and reshape
        self.project = nn.Sequential(
            nn.Linear(input_dim, self.init_channels * init_length),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Transposed convolution layers
        layers = []
        in_channels = self.init_channels
        for i, out_channels in enumerate(config.gen_hidden_dims[1:]):
            layers.extend([
                nn.ConvTranspose1d(
                    in_channels, out_channels,
                    kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_channels = out_channels

        # Final layer to output dimension
        layers.extend([
            nn.ConvTranspose1d(
                in_channels, config.n_features,
                kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),  # Bound returns to [-1, 1]
        ])

        self.conv_layers = nn.Sequential(*layers)

        # Scale factor for returns (Tanh outputs [-1,1], real returns are smaller)
        self.return_scale = nn.Parameter(torch.tensor(0.05), requires_grad=True)

    def forward(
        self,
        z: "torch.Tensor",
        regime: Optional["torch.Tensor"] = None,
        vol_level: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Generate synthetic returns.

        Args:
            z: Latent noise [batch_size, latent_dim]
            regime: Regime labels [batch_size] (optional)
            vol_level: Volatility level labels [batch_size] (optional)

        Returns:
            Synthetic log-returns [batch_size, seq_len]
        """
        # Build conditioned input
        if self.config.use_conditional and regime is not None and vol_level is not None:
            r_emb = self.regime_embed(regime)
            v_emb = self.vol_embed(vol_level)
            z = torch.cat([z, r_emb, v_emb], dim=-1)

        # Project and reshape
        x = self.project(z)
        x = x.view(-1, self.init_channels, self.init_length)

        # Upsampling convolutions
        x = self.conv_layers(x)

        # Ensure correct sequence length
        x = x[:, :, :self.config.seq_len]

        # Squeeze feature dim and scale
        returns = x.squeeze(1) * self.return_scale

        return returns

    def generate(
        self,
        n_paths: int,
        regime: Optional[int] = None,
        vol_level: Optional[int] = None,
        device: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate synthetic return paths (convenience method).

        Args:
            n_paths: Number of paths to generate
            regime: Regime index (0=bull, 1=bear, 2=sideways, 3=crisis)
            vol_level: Volatility level (0=low, 1=medium, 2=high, 3=extreme)
            device: Device override

        Returns:
            Numpy array of synthetic returns [n_paths, seq_len]
        """
        dev = device or self.config.device
        self.eval()
        with torch.no_grad():
            z = torch.randn(n_paths, self.config.latent_dim, device=dev)

            r_tensor = None
            v_tensor = None
            if self.config.use_conditional:
                r_idx = regime if regime is not None else 0
                v_idx = vol_level if vol_level is not None else 1
                r_tensor = torch.full((n_paths,), r_idx, dtype=torch.long, device=dev)
                v_tensor = torch.full((n_paths,), v_idx, dtype=torch.long, device=dev)

            returns = self.forward(z, r_tensor, v_tensor)
        return returns.cpu().numpy()


class Critic(nn.Module):
    """
    1D Convolutional Critic (Discriminator) for WGAN-GP.

    Takes a sequence of returns and outputs a scalar Wasserstein score.
    No sigmoid activation -- outputs unbounded real values.

    Uses LayerNorm instead of BatchNorm (required for WGAN-GP gradient penalty).
    """

    def __init__(self, config: GANConfig):
        _check_torch()
        super().__init__()
        self.config = config

        layers = []
        in_channels = config.n_features

        for i, out_channels in enumerate(config.critic_hidden_dims):
            layers.extend([
                nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=4, stride=2, padding=1
                ),
                nn.LayerNorm([out_channels, config.seq_len // (2 ** (i + 1))]),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Compute flattened size
        final_length = config.seq_len // (2 ** len(config.critic_hidden_dims))
        flat_size = config.critic_hidden_dims[-1] * final_length

        # Condition embedding for conditional critic
        cond_dim = 0
        if config.use_conditional:
            self.regime_embed = nn.Embedding(config.n_regimes, config.cond_embed_dim)
            self.vol_embed = nn.Embedding(config.n_vol_levels, config.cond_embed_dim)
            cond_dim = config.cond_embed_dim * 2

        self.output = nn.Sequential(
            nn.Linear(flat_size + cond_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        returns: "torch.Tensor",
        regime: Optional["torch.Tensor"] = None,
        vol_level: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        Compute Wasserstein critic score.

        Args:
            returns: Return sequences [batch_size, seq_len]
            regime: Regime labels [batch_size] (optional)
            vol_level: Volatility level labels [batch_size] (optional)

        Returns:
            Critic scores [batch_size, 1]
        """
        # Add channel dimension
        x = returns.unsqueeze(1)  # [B, 1, T]

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Concatenate condition
        if self.config.use_conditional and regime is not None and vol_level is not None:
            r_emb = self.regime_embed(regime)
            v_emb = self.vol_embed(vol_level)
            x = torch.cat([x, r_emb, v_emb], dim=-1)

        return self.output(x)


class PhysicsLoss(nn.Module):
    """
    Physics constraint penalties for financial time series generation.

    Implements differentiable penalties for:
        - Martingale property (no-arbitrage)
        - Volatility clustering (ACF of absolute returns)
        - Fat tails (excess kurtosis)
        - Leverage effect (return-volatility correlation)
        - Autocorrelation structure
        - Moment matching
    """

    def __init__(self, config: GANConfig):
        _check_torch()
        super().__init__()
        self.config = config

        # Register target statistics as buffers (not parameters)
        self.register_buffer(
            "target_vol_acf",
            torch.zeros(config.target_vol_acf_lags),
        )
        self.register_buffer(
            "target_sq_return_acf",
            torch.zeros(config.target_return_acf_lags),
        )
        self.register_buffer(
            "target_mean", torch.tensor(0.0),
        )
        self.register_buffer(
            "target_std", torch.tensor(0.03),
        )

    def set_targets_from_data(self, real_returns: "torch.Tensor"):
        """
        Compute target statistics from real return data.

        Args:
            real_returns: Real return sequences [n_samples, seq_len]
        """
        flat = real_returns.reshape(-1)

        # Target moments
        self.target_mean.fill_(flat.mean().item())
        self.target_std.fill_(flat.std().item())

        # Target ACF of absolute returns
        abs_returns = real_returns.abs()
        for k in range(self.config.target_vol_acf_lags):
            acf_k = self._compute_acf(abs_returns, k + 1)
            self.target_vol_acf[k] = acf_k

        # Target ACF of squared returns
        sq_returns = real_returns ** 2
        for k in range(self.config.target_return_acf_lags):
            acf_k = self._compute_acf(sq_returns, k + 1)
            self.target_sq_return_acf[k] = acf_k

    def _compute_acf(self, x: "torch.Tensor", lag: int) -> "torch.Tensor":
        """Compute autocorrelation at a given lag across batch."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        T = x.size(-1)
        if lag >= T:
            return torch.tensor(0.0, device=x.device)

        x_centered = x - x.mean(dim=-1, keepdim=True)
        var = (x_centered ** 2).mean(dim=-1)

        cov = (x_centered[:, :-lag] * x_centered[:, lag:]).mean(dim=-1)
        acf = cov / (var + 1e-8)
        return acf.mean()

    def martingale_loss(self, gen_returns: "torch.Tensor") -> "torch.Tensor":
        """
        Martingale constraint: E[r_t] should be approximately zero.

        For risk-neutral pricing, discounted prices are martingales,
        so expected returns should equal the risk-free rate (approx 0 for short horizons).
        """
        mean_returns = gen_returns.mean(dim=-1)  # [batch_size]
        return (mean_returns ** 2).mean()

    def volatility_clustering_loss(self, gen_returns: "torch.Tensor") -> "torch.Tensor":
        """
        Volatility clustering: ACF of |r_t| should match real data.

        Real financial returns exhibit slowly decaying autocorrelation
        of absolute values, reflecting GARCH-like clustering.
        """
        abs_returns = gen_returns.abs()
        loss = torch.tensor(0.0, device=gen_returns.device)

        for k in range(self.config.target_vol_acf_lags):
            gen_acf_k = self._compute_acf(abs_returns, k + 1)
            target_acf_k = self.target_vol_acf[k]
            loss = loss + (gen_acf_k - target_acf_k) ** 2

        return loss / self.config.target_vol_acf_lags

    def kurtosis_loss(self, gen_returns: "torch.Tensor") -> "torch.Tensor":
        """
        Fat tails: excess kurtosis should match target.

        Financial returns are leptokurtic (heavier tails than Gaussian).
        Typical excess kurtosis for daily returns: 3-30.
        """
        mean = gen_returns.mean(dim=-1, keepdim=True)
        var = gen_returns.var(dim=-1, keepdim=True).clamp(min=1e-8)
        centered = gen_returns - mean

        fourth_moment = (centered ** 4).mean(dim=-1)
        kurtosis = fourth_moment / (var.squeeze(-1) ** 2) - 3.0  # Excess kurtosis

        target = self.config.target_kurtosis
        return ((kurtosis - target) ** 2).mean()

    def leverage_effect_loss(self, gen_returns: "torch.Tensor") -> "torch.Tensor":
        """
        Leverage effect: Corr(r_t, |r_{t+1}|) should be negative.

        Negative returns tend to increase future volatility.
        Typical correlation: -0.1 to -0.4.
        """
        r_t = gen_returns[:, :-1]
        abs_r_next = gen_returns[:, 1:].abs()

        # Compute correlation
        r_mean = r_t.mean(dim=-1, keepdim=True)
        a_mean = abs_r_next.mean(dim=-1, keepdim=True)

        r_centered = r_t - r_mean
        a_centered = abs_r_next - a_mean

        cov = (r_centered * a_centered).mean(dim=-1)
        r_std = r_centered.std(dim=-1).clamp(min=1e-8)
        a_std = a_centered.std(dim=-1).clamp(min=1e-8)

        corr = cov / (r_std * a_std)
        target = self.config.target_leverage_corr

        return ((corr - target) ** 2).mean()

    def autocorrelation_loss(self, gen_returns: "torch.Tensor") -> "torch.Tensor":
        """
        Autocorrelation structure:
        - Returns r_t should have near-zero ACF
        - Squared returns r_t^2 should have positive, slowly decaying ACF
        """
        loss = torch.tensor(0.0, device=gen_returns.device)

        # Penalize non-zero ACF of raw returns (should be ~0)
        for k in range(1, min(6, gen_returns.size(-1))):
            acf_k = self._compute_acf(gen_returns, k)
            loss = loss + acf_k ** 2

        # Match ACF of squared returns
        sq_returns = gen_returns ** 2
        for k in range(self.config.target_return_acf_lags):
            gen_acf_k = self._compute_acf(sq_returns, k + 1)
            target_acf_k = self.target_sq_return_acf[k]
            loss = loss + (gen_acf_k - target_acf_k) ** 2

        n_terms = 5 + self.config.target_return_acf_lags
        return loss / n_terms

    def moment_matching_loss(self, gen_returns: "torch.Tensor") -> "torch.Tensor":
        """
        Moment matching: mean and std of generated returns should match real data.
        """
        gen_mean = gen_returns.mean()
        gen_std = gen_returns.std()

        mean_loss = (gen_mean - self.target_mean) ** 2
        std_loss = (gen_std - self.target_std) ** 2

        return mean_loss + std_loss

    def forward(self, gen_returns: "torch.Tensor") -> Tuple["torch.Tensor", Dict[str, float]]:
        """
        Compute total physics loss with all constraints.

        Args:
            gen_returns: Generated return sequences [batch_size, seq_len]

        Returns:
            Tuple of (total_physics_loss, dict of individual losses)
        """
        cfg = self.config

        l_martingale = self.martingale_loss(gen_returns)
        l_volatility = self.volatility_clustering_loss(gen_returns)
        l_kurtosis = self.kurtosis_loss(gen_returns)
        l_leverage = self.leverage_effect_loss(gen_returns)
        l_autocorr = self.autocorrelation_loss(gen_returns)
        l_moment = self.moment_matching_loss(gen_returns)

        total = (
            cfg.lambda_martingale * l_martingale
            + cfg.lambda_volatility * l_volatility
            + cfg.lambda_kurtosis * l_kurtosis
            + cfg.lambda_leverage * l_leverage
            + cfg.lambda_autocorr * l_autocorr
            + cfg.lambda_moment * l_moment
        )

        details = {
            "martingale": l_martingale.item(),
            "volatility_clustering": l_volatility.item(),
            "kurtosis": l_kurtosis.item(),
            "leverage_effect": l_leverage.item(),
            "autocorrelation": l_autocorr.item(),
            "moment_matching": l_moment.item(),
            "total_physics": total.item(),
        }

        return total, details


class PhysicsConstrainedGAN:
    """
    Complete Physics-Constrained WGAN-GP system.

    Combines Generator, Critic, and PhysicsLoss into a unified interface
    for training, generation, and evaluation.
    """

    def __init__(self, config: Optional[GANConfig] = None):
        _check_torch()
        self.config = config or GANConfig()
        self.device = torch.device(self.config.device)

        # Initialize components
        self.generator = Generator(self.config).to(self.device)
        self.critic = Critic(self.config).to(self.device)
        self.physics_loss = PhysicsLoss(self.config).to(self.device)

        # Optimizers
        self.opt_gen = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.lr_gen,
            betas=(self.config.beta1, self.config.beta2),
        )
        self.opt_critic = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.lr_critic,
            betas=(self.config.beta1, self.config.beta2),
        )

    def compute_gradient_penalty(
        self,
        real: "torch.Tensor",
        fake: "torch.Tensor",
        regime: Optional["torch.Tensor"] = None,
        vol_level: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        """
        WGAN-GP gradient penalty.

        Interpolates between real and fake samples and penalizes
        the gradient norm deviating from 1.
        """
        batch_size = real.size(0)
        epsilon = torch.rand(batch_size, 1, device=self.device)
        epsilon = epsilon.expand_as(real)

        interpolated = epsilon * real + (1 - epsilon) * fake
        interpolated.requires_grad_(True)

        critic_score = self.critic(interpolated, regime, vol_level)

        gradients = torch.autograd.grad(
            outputs=critic_score,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_score),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1.0) ** 2).mean()

        return penalty

    def train_critic_step(
        self,
        real_returns: "torch.Tensor",
        regime: Optional["torch.Tensor"] = None,
        vol_level: Optional["torch.Tensor"] = None,
    ) -> Dict[str, float]:
        """
        Single critic training step.

        Returns dict with critic loss components.
        """
        batch_size = real_returns.size(0)
        self.critic.train()
        self.opt_critic.zero_grad()

        # Generate fake data
        z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        with torch.no_grad():
            fake_returns = self.generator(z, regime, vol_level)

        # Critic scores
        real_score = self.critic(real_returns, regime, vol_level)
        fake_score = self.critic(fake_returns, regime, vol_level)

        # Wasserstein loss
        w_loss = fake_score.mean() - real_score.mean()

        # Gradient penalty
        gp = self.compute_gradient_penalty(real_returns, fake_returns, regime, vol_level)

        # Total critic loss
        critic_loss = w_loss + self.config.gp_lambda * gp

        critic_loss.backward()
        self.opt_critic.step()

        return {
            "critic_loss": critic_loss.item(),
            "wasserstein_distance": -w_loss.item(),
            "gradient_penalty": gp.item(),
        }

    def train_generator_step(
        self,
        batch_size: int,
        regime: Optional["torch.Tensor"] = None,
        vol_level: Optional["torch.Tensor"] = None,
    ) -> Dict[str, float]:
        """
        Single generator training step with physics constraints.

        Returns dict with generator loss components.
        """
        self.generator.train()
        self.opt_gen.zero_grad()

        # Generate fake data
        z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        fake_returns = self.generator(z, regime, vol_level)

        # Adversarial loss (want critic to score fakes highly)
        fake_score = self.critic(fake_returns, regime, vol_level)
        adv_loss = -fake_score.mean()

        # Physics penalty
        physics_total, physics_details = self.physics_loss(fake_returns)

        # Total generator loss
        gen_loss = adv_loss + physics_total

        gen_loss.backward()
        self.opt_gen.step()

        result = {
            "gen_loss": gen_loss.item(),
            "adversarial_loss": adv_loss.item(),
        }
        result.update(physics_details)
        return result

    def generate(
        self,
        n_paths: int = 100,
        condition: Optional[Dict[str, str]] = None,
        as_prices: bool = False,
        initial_price: float = 100.0,
    ) -> np.ndarray:
        """
        Generate synthetic return or price paths.

        Args:
            n_paths: Number of paths to generate
            condition: Dict with 'regime' and 'volatility' keys
            as_prices: If True, convert returns to price paths
            initial_price: Starting price for price paths

        Returns:
            Numpy array [n_paths, seq_len]
        """
        regime_map = {"bull": 0, "bear": 1, "sideways": 2, "crisis": 3}
        vol_map = {"low": 0, "medium": 1, "high": 2, "extreme": 3}

        regime_idx = None
        vol_idx = None
        if condition:
            regime_idx = regime_map.get(condition.get("regime", "bull"), 0)
            vol_idx = vol_map.get(condition.get("volatility", "medium"), 1)

        returns = self.generator.generate(
            n_paths, regime=regime_idx, vol_level=vol_idx, device=str(self.device)
        )

        if as_prices:
            prices = initial_price * np.cumprod(1.0 + returns, axis=-1)
            return prices

        return returns

    def save(self, path: str):
        """Save model state dicts."""
        torch.save({
            "generator": self.generator.state_dict(),
            "critic": self.critic.state_dict(),
            "physics_loss": self.physics_loss.state_dict(),
            "opt_gen": self.opt_gen.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "config": self.config,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model state dicts."""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.physics_loss.load_state_dict(checkpoint["physics_loss"])
        self.opt_gen.load_state_dict(checkpoint["opt_gen"])
        self.opt_critic.load_state_dict(checkpoint["opt_critic"])
        print(f"Model loaded from {path}")

    def evaluate_statistics(self, n_samples: int = 1000) -> Dict[str, float]:
        """
        Evaluate statistical properties of generated data.

        Returns dict with computed statistics.
        """
        returns = self.generate(n_paths=n_samples)

        flat = returns.flatten()
        stats = {
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "skewness": float(_skewness(flat)),
            "kurtosis": float(_kurtosis(flat)),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
        }

        # ACF of absolute returns (average across paths)
        abs_returns = np.abs(returns)
        acf_1 = float(np.mean([
            _acf(abs_returns[i], 1) for i in range(min(100, n_samples))
        ]))
        acf_5 = float(np.mean([
            _acf(abs_returns[i], 5) for i in range(min(100, n_samples))
        ]))
        stats["acf_abs_lag1"] = acf_1
        stats["acf_abs_lag5"] = acf_5

        return stats


def _skewness(x: np.ndarray) -> float:
    """Compute skewness."""
    n = len(x)
    if n < 3:
        return 0.0
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if std < 1e-10:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis."""
    n = len(x)
    if n < 4:
        return 0.0
    mean = np.mean(x)
    std = np.std(x, ddof=1)
    if std < 1e-10:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 4) - 3.0)


def _acf(x: np.ndarray, lag: int) -> float:
    """Compute autocorrelation at given lag."""
    n = len(x)
    if lag >= n:
        return 0.0
    mean = np.mean(x)
    var = np.var(x)
    if var < 1e-10:
        return 0.0
    cov = np.mean((x[:-lag] - mean) * (x[lag:] - mean))
    return float(cov / var)
