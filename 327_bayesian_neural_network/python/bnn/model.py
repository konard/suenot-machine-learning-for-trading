"""
Bayesian Neural Network Model for Trading.

Implements a full BNN for price direction prediction with uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Optional

from .layers import BayesianLinear


@dataclass
class BayesianNNConfig:
    """Configuration for Bayesian Neural Network."""

    # Architecture
    input_dim: int = 20
    hidden_dims: List[int] = None
    num_classes: int = 3  # up, neutral, down

    # Prior parameters
    prior_sigma_1: float = 1.0
    prior_sigma_2: float = 0.0025
    prior_pi: float = 0.5

    # Regularization
    dropout_rate: float = 0.1

    # Output type
    heteroscedastic: bool = True  # Predict aleatoric uncertainty

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]


class BayesianNN(nn.Module):
    """
    Bayesian Neural Network for trading signal prediction.

    Features:
    - Weight uncertainty via variational inference
    - Epistemic uncertainty from weight distributions
    - Optional heteroscedastic output for aleatoric uncertainty
    - Monte Carlo inference for uncertainty estimation

    Args:
        config: BayesianNNConfig with model parameters
    """

    def __init__(self, config: BayesianNNConfig):
        super().__init__()
        self.config = config

        # Build Bayesian layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Input layer
        prev_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            self.layers.append(
                BayesianLinear(
                    prev_dim,
                    hidden_dim,
                    prior_sigma_1=config.prior_sigma_1,
                    prior_sigma_2=config.prior_sigma_2,
                    prior_pi=config.prior_pi
                )
            )
            self.dropouts.append(nn.Dropout(config.dropout_rate))
            prev_dim = hidden_dim

        # Output head for classification
        self.classifier = BayesianLinear(
            prev_dim,
            config.num_classes,
            prior_sigma_1=config.prior_sigma_1,
            prior_sigma_2=config.prior_sigma_2,
            prior_pi=config.prior_pi
        )

        # Optional heteroscedastic head for return prediction
        if config.heteroscedastic:
            # Predicts mean and log-variance of returns
            self.return_head = BayesianLinear(
                prev_dim,
                2,  # mu and log_sigma
                prior_sigma_1=config.prior_sigma_1,
                prior_sigma_2=config.prior_sigma_2,
                prior_pi=config.prior_pi
            )

    def forward(
        self,
        x: torch.Tensor,
        sample: bool = True
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through the network.

        Args:
            x: Input features of shape (batch_size, input_dim)
            sample: If True, sample weights; if False, use mean weights

        Returns:
            logits: Classification logits of shape (batch_size, num_classes)
            return_dist: If heteroscedastic, tuple of (mean, log_var) for returns
        """
        # Pass through hidden layers
        h = x
        for layer, dropout in zip(self.layers, self.dropouts):
            h = layer(h, sample=sample)
            h = F.leaky_relu(h, negative_slope=0.1)
            h = dropout(h)

        # Classification output
        logits = self.classifier(h, sample=sample)

        # Return prediction (heteroscedastic)
        if self.config.heteroscedastic:
            return_out = self.return_head(h, sample=sample)
            return_mu = return_out[:, 0]
            return_log_var = return_out[:, 1]
            return logits, (return_mu, return_log_var)

        return logits, None

    def kl_divergence(self) -> torch.Tensor:
        """
        Compute total KL divergence for all Bayesian layers.

        Returns:
            Total KL divergence (scalar tensor)
        """
        kl = torch.tensor(0.0, device=next(self.parameters()).device)

        for layer in self.layers:
            kl += layer.kl_divergence()

        kl += self.classifier.kl_divergence()

        if self.config.heteroscedastic:
            kl += self.return_head.kl_divergence()

        return kl

    def predict_proba(
        self,
        x: torch.Tensor,
        n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class probabilities with uncertainty estimation.

        Performs Monte Carlo inference by sampling weights multiple times.

        Args:
            x: Input features of shape (batch_size, input_dim)
            n_samples: Number of MC samples

        Returns:
            mean_probs: Mean predicted probabilities (batch_size, num_classes)
            epistemic_uncertainty: Variance of predictions (batch_size,)
        """
        # Set to evaluation mode for inference
        training_mode = self.training
        self.train(False)
        probs_list = []

        with torch.no_grad():
            for _ in range(n_samples):
                logits, _ = self.forward(x, sample=True)
                probs = F.softmax(logits, dim=-1)
                probs_list.append(probs)

        # Restore training mode
        self.train(training_mode)

        # Stack all predictions
        probs_stack = torch.stack(probs_list, dim=0)  # (n_samples, batch, classes)

        # Mean prediction
        mean_probs = probs_stack.mean(dim=0)

        # Epistemic uncertainty: variance of predictions
        # Total variance across classes
        epistemic_uncertainty = probs_stack.var(dim=0).sum(dim=-1)

        return mean_probs, epistemic_uncertainty

    def predict_returns(
        self,
        x: torch.Tensor,
        n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict returns with uncertainty decomposition.

        Args:
            x: Input features
            n_samples: Number of MC samples

        Returns:
            mean_return: Mean predicted return
            epistemic_uncertainty: Uncertainty from model (reducible)
            aleatoric_uncertainty: Uncertainty from data (irreducible)
        """
        if not self.config.heteroscedastic:
            raise ValueError("Model not configured for return prediction")

        # Set to evaluation mode
        training_mode = self.training
        self.train(False)
        return_means = []
        return_vars = []

        with torch.no_grad():
            for _ in range(n_samples):
                _, (mu, log_var) = self.forward(x, sample=True)
                return_means.append(mu)
                return_vars.append(torch.exp(log_var))

        # Restore training mode
        self.train(training_mode)

        # Stack predictions
        means_stack = torch.stack(return_means, dim=0)
        vars_stack = torch.stack(return_vars, dim=0)

        # Mean prediction
        mean_return = means_stack.mean(dim=0)

        # Epistemic uncertainty: variance of means
        epistemic_uncertainty = means_stack.var(dim=0)

        # Aleatoric uncertainty: mean of variances
        aleatoric_uncertainty = vars_stack.mean(dim=0)

        return mean_return, epistemic_uncertainty, aleatoric_uncertainty

    def get_weight_statistics(self) -> dict:
        """Get statistics about all weight distributions."""
        stats = {}
        for i, layer in enumerate(self.layers):
            layer_stats = layer.get_weight_statistics()
            for key, value in layer_stats.items():
                stats[f'layer_{i}_{key}'] = value

        classifier_stats = self.classifier.get_weight_statistics()
        for key, value in classifier_stats.items():
            stats[f'classifier_{key}'] = value

        return stats


class BayesianLSTM(nn.Module):
    """
    Bayesian LSTM for sequence modeling.

    Applies Bayesian treatment to the output layer while using
    standard LSTM for temporal processing.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
        prior_sigma_1: float = 1.0,
        prior_sigma_2: float = 0.0025,
        prior_pi: float = 0.5
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Standard LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Bayesian output layers
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.fc1 = BayesianLinear(
            hidden_dim, hidden_dim // 2,
            prior_sigma_1=prior_sigma_1,
            prior_sigma_2=prior_sigma_2,
            prior_pi=prior_pi
        )

        self.fc2 = BayesianLinear(
            hidden_dim // 2, num_classes,
            prior_sigma_1=prior_sigma_1,
            prior_sigma_2=prior_sigma_2,
            prior_pi=prior_pi
        )

    def forward(
        self,
        x: torch.Tensor,
        sample: bool = True
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, seq_len, input_dim)
            sample: Whether to sample Bayesian weights

        Returns:
            logits: Classification logits
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)

        # Take last time step
        last_hidden = lstm_out[:, -1, :]

        # Bayesian layers
        h = self.bn(last_hidden)
        h = self.dropout(h)
        h = F.relu(self.fc1(h, sample=sample))
        h = self.dropout(h)
        logits = self.fc2(h, sample=sample)

        return logits

    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence for Bayesian layers."""
        return self.fc1.kl_divergence() + self.fc2.kl_divergence()

    def predict_proba(
        self,
        x: torch.Tensor,
        n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Monte Carlo prediction with uncertainty."""
        training_mode = self.training
        self.train(False)
        probs_list = []

        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x, sample=True)
                probs = F.softmax(logits, dim=-1)
                probs_list.append(probs)

        self.train(training_mode)

        probs_stack = torch.stack(probs_list, dim=0)
        mean_probs = probs_stack.mean(dim=0)
        epistemic_uncertainty = probs_stack.var(dim=0).sum(dim=-1)

        return mean_probs, epistemic_uncertainty
