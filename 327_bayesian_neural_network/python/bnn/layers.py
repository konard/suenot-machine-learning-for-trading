"""
Bayesian Neural Network Layers.

Implements weight uncertainty using variational inference.
Based on "Weight Uncertainty in Neural Networks" (Blundell et al., 2015).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer with weight uncertainty.

    Each weight and bias is represented as a distribution:
    w ~ N(mu, sigma^2) where sigma = softplus(rho)

    Uses the reparameterization trick for backpropagation:
    w = mu + sigma * epsilon, where epsilon ~ N(0, 1)

    Args:
        in_features: Size of input features
        out_features: Size of output features
        prior_sigma_1: First prior standard deviation (for mixture)
        prior_sigma_2: Second prior standard deviation (for mixture)
        prior_pi: Mixture coefficient for scale mixture prior
        bias: Whether to include bias term
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma_1: float = 1.0,
        prior_sigma_2: float = 0.0025,
        prior_pi: float = 0.5,
        bias: bool = True
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Prior parameters (scale mixture of two Gaussians)
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi

        # Weight parameters (variational posterior)
        # mu: mean of the weight distribution
        # rho: parameterizes sigma via softplus(rho)
        self.weight_mu = nn.Parameter(
            torch.zeros(out_features, in_features)
        )
        self.weight_rho = nn.Parameter(
            torch.zeros(out_features, in_features)
        )

        # Bias parameters
        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.zeros(out_features))
            self.bias_rho = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        # Initialize parameters
        self.reset_parameters()

        # Store sampled weights for KL computation
        self.weight_sample: Optional[torch.Tensor] = None
        self.bias_sample: Optional[torch.Tensor] = None

    def reset_parameters(self):
        """Initialize parameters using Kaiming initialization for mu."""
        # Initialize mu with Kaiming uniform
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))

        # Initialize rho to give small initial variance
        # softplus(-5) ≈ 0.007, so initial sigma ≈ 0.007
        nn.init.constant_(self.weight_rho, -5.0)

        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_rho, -5.0)

    def _reparameterize(
        self,
        mu: torch.Tensor,
        rho: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick: sample from N(mu, sigma^2).

        sigma = softplus(rho) = log(1 + exp(rho))
        w = mu + sigma * epsilon, epsilon ~ N(0, 1)

        Args:
            mu: Mean of the distribution
            rho: Parameter for standard deviation (sigma = softplus(rho))

        Returns:
            Sampled values from the distribution
        """
        sigma = F.softplus(rho)
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon

    def forward(
        self,
        x: torch.Tensor,
        sample: bool = True
    ) -> torch.Tensor:
        """
        Forward pass with optional weight sampling.

        Args:
            x: Input tensor of shape (batch_size, in_features)
            sample: If True, sample weights; if False, use mean weights

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        if sample:
            # Sample weights using reparameterization trick
            weight = self._reparameterize(self.weight_mu, self.weight_rho)
            self.weight_sample = weight

            if self.use_bias:
                bias = self._reparameterize(self.bias_mu, self.bias_rho)
                self.bias_sample = bias
            else:
                bias = None
                self.bias_sample = None
        else:
            # Use mean weights (no sampling)
            weight = self.weight_mu
            bias = self.bias_mu if self.use_bias else None
            self.weight_sample = weight
            self.bias_sample = bias

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence from variational posterior to prior.

        KL[q(w|theta) || p(w)] where:
        - q(w|theta) = N(mu, sigma^2) is the variational posterior
        - p(w) is the scale mixture prior

        Returns:
            KL divergence (scalar tensor)
        """
        kl = self._kl_divergence_weights(
            self.weight_mu,
            self.weight_rho,
            self.weight_sample
        )

        if self.use_bias:
            kl += self._kl_divergence_weights(
                self.bias_mu,
                self.bias_rho,
                self.bias_sample
            )

        return kl

    def _kl_divergence_weights(
        self,
        mu: torch.Tensor,
        rho: torch.Tensor,
        weight_sample: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute KL divergence for a set of weights.

        Uses the log-uniform prior approximation:
        KL ≈ log q(w) - log p(w)

        Args:
            mu: Mean of variational posterior
            rho: Rho parameter of variational posterior
            weight_sample: Sampled weights

        Returns:
            KL divergence contribution
        """
        sigma = F.softplus(rho)

        if weight_sample is None:
            weight_sample = mu

        # Log probability under variational posterior (Gaussian)
        log_posterior = self._log_gaussian(weight_sample, mu, sigma)

        # Log probability under scale mixture prior
        log_prior = self._log_scale_mixture_prior(weight_sample)

        return (log_posterior - log_prior).sum()

    def _log_gaussian(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Log probability of x under Gaussian(mu, sigma^2).

        log N(x|mu, sigma^2) = -0.5 * log(2*pi) - log(sigma) - 0.5 * ((x-mu)/sigma)^2
        """
        return (
            -0.5 * math.log(2 * math.pi)
            - torch.log(sigma)
            - 0.5 * ((x - mu) / sigma) ** 2
        )

    def _log_scale_mixture_prior(self, x: torch.Tensor) -> torch.Tensor:
        """
        Log probability under scale mixture of Gaussians prior.

        p(w) = pi * N(0, sigma_1^2) + (1-pi) * N(0, sigma_2^2)
        """
        sigma_1 = torch.tensor(self.prior_sigma_1, device=x.device)
        sigma_2 = torch.tensor(self.prior_sigma_2, device=x.device)
        pi = self.prior_pi

        # Log prob under first Gaussian
        log_p1 = (
            -0.5 * math.log(2 * math.pi)
            - torch.log(sigma_1)
            - 0.5 * (x / sigma_1) ** 2
        )

        # Log prob under second Gaussian
        log_p2 = (
            -0.5 * math.log(2 * math.pi)
            - torch.log(sigma_2)
            - 0.5 * (x / sigma_2) ** 2
        )

        # Log-sum-exp for numerical stability
        # log(pi * exp(log_p1) + (1-pi) * exp(log_p2))
        log_pi = math.log(pi)
        log_one_minus_pi = math.log(1 - pi)

        return torch.logsumexp(
            torch.stack([log_pi + log_p1, log_one_minus_pi + log_p2], dim=0),
            dim=0
        )

    def get_weight_statistics(self) -> dict:
        """
        Get statistics about the weight distributions.

        Returns:
            Dictionary with mean and std statistics
        """
        weight_sigma = F.softplus(self.weight_rho)

        stats = {
            'weight_mu_mean': self.weight_mu.mean().item(),
            'weight_mu_std': self.weight_mu.std().item(),
            'weight_sigma_mean': weight_sigma.mean().item(),
            'weight_sigma_std': weight_sigma.std().item(),
        }

        if self.use_bias:
            bias_sigma = F.softplus(self.bias_rho)
            stats.update({
                'bias_mu_mean': self.bias_mu.mean().item(),
                'bias_mu_std': self.bias_mu.std().item(),
                'bias_sigma_mean': bias_sigma.mean().item(),
                'bias_sigma_std': bias_sigma.std().item(),
            })

        return stats

    def extra_repr(self) -> str:
        """String representation."""
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'bias={self.use_bias}, '
            f'prior_sigma_1={self.prior_sigma_1}, '
            f'prior_sigma_2={self.prior_sigma_2}, '
            f'prior_pi={self.prior_pi}'
        )


class BayesianConv1d(nn.Module):
    """
    Bayesian 1D Convolutional Layer (for time series).

    Similar to BayesianLinear but for convolutional operations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        prior_sigma_1: float = 1.0,
        prior_sigma_2: float = 0.0025,
        prior_pi: float = 0.5
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Prior parameters
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi = prior_pi

        # Weight parameters
        self.weight_mu = nn.Parameter(
            torch.zeros(out_channels, in_channels, kernel_size)
        )
        self.weight_rho = nn.Parameter(
            torch.zeros(out_channels, in_channels, kernel_size)
        )

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_channels))
        self.bias_rho = nn.Parameter(torch.zeros(out_channels))

        self.reset_parameters()

        self.weight_sample: Optional[torch.Tensor] = None
        self.bias_sample: Optional[torch.Tensor] = None

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, -5.0)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -5.0)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Forward pass with optional sampling."""
        if sample:
            weight_sigma = F.softplus(self.weight_rho)
            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)

            bias_sigma = F.softplus(self.bias_rho)
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)

            self.weight_sample = weight
            self.bias_sample = bias
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            self.weight_sample = weight
            self.bias_sample = bias

        return F.conv1d(x, weight, bias, self.stride, self.padding)

    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence (simplified version)."""
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        # Simplified KL for Gaussian posterior vs Gaussian prior
        kl_weight = 0.5 * torch.sum(
            (self.weight_mu ** 2 + weight_sigma ** 2) / self.prior_sigma_1 ** 2
            - 1
            + 2 * math.log(self.prior_sigma_1)
            - 2 * torch.log(weight_sigma)
        )

        kl_bias = 0.5 * torch.sum(
            (self.bias_mu ** 2 + bias_sigma ** 2) / self.prior_sigma_1 ** 2
            - 1
            + 2 * math.log(self.prior_sigma_1)
            - 2 * torch.log(bias_sigma)
        )

        return kl_weight + kl_bias
