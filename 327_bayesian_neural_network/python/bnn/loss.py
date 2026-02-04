"""
Loss functions for Bayesian Neural Networks.

Implements ELBO (Evidence Lower Bound) and related loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def get_kl_weight(
    epoch: int,
    warmup_epochs: int = 10,
    annealing_type: str = 'linear'
) -> float:
    """
    Compute KL weight for KL annealing.

    KL annealing helps prevent posterior collapse early in training
    by gradually increasing the KL term weight.

    Args:
        epoch: Current training epoch (0-indexed)
        warmup_epochs: Number of epochs for warmup
        annealing_type: Type of annealing ('linear', 'sigmoid', 'cyclical')

    Returns:
        KL weight between 0 and 1
    """
    if epoch >= warmup_epochs:
        return 1.0

    if annealing_type == 'linear':
        return (epoch + 1) / warmup_epochs

    elif annealing_type == 'sigmoid':
        # Sigmoid annealing: slower start and end
        import math
        x = 10 * (epoch / warmup_epochs - 0.5)  # Scale to [-5, 5]
        return 1 / (1 + math.exp(-x))

    elif annealing_type == 'cyclical':
        # Cyclical annealing (useful for VAEs)
        cycle_length = warmup_epochs
        position = epoch % cycle_length
        return position / cycle_length

    else:
        return (epoch + 1) / warmup_epochs


def elbo_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    n_samples: int = 1,
    kl_weight: float = 1.0,
    reduction: str = 'mean'
) -> Tuple[torch.Tensor, dict]:
    """
    Compute Evidence Lower Bound (ELBO) loss.

    ELBO = E_q[log p(y|x,w)] - beta * KL[q(w|theta) || p(w)]

    The negative ELBO is minimized during training.

    Args:
        model: Bayesian neural network model
        x: Input features of shape (batch_size, input_dim)
        y: Target labels of shape (batch_size,)
        n_samples: Number of Monte Carlo samples for likelihood estimation
        kl_weight: Weight for KL divergence term (beta for KL annealing)
        reduction: How to reduce the loss ('mean', 'sum', 'none')

    Returns:
        loss: Negative ELBO loss
        metrics: Dictionary with loss components
    """
    batch_size = x.size(0)

    # Compute negative log-likelihood (expected over MC samples)
    nll = torch.tensor(0.0, device=x.device)

    for _ in range(n_samples):
        logits, return_dist = model(x, sample=True)

        # Classification loss
        nll += F.cross_entropy(logits, y, reduction='sum')

    nll = nll / n_samples

    # KL divergence
    kl_div = model.kl_divergence()

    # Scale KL by batch size for proper ELBO
    # KL is summed over all weights, but NLL is averaged
    # We want: loss = E[NLL] + KL/N where N is dataset size
    # Approximation: KL_scaled = KL * batch_size / dataset_size
    # For simplicity, we use: KL_scaled = KL / batch_size
    kl_scaled = kl_div / batch_size

    # Total loss
    if reduction == 'mean':
        nll = nll / batch_size
        loss = nll + kl_weight * kl_scaled
    elif reduction == 'sum':
        loss = nll + kl_weight * kl_div
    else:
        loss = nll + kl_weight * kl_scaled

    metrics = {
        'loss': loss.item(),
        'nll': nll.item() if reduction == 'mean' else (nll / batch_size).item(),
        'kl': kl_div.item(),
        'kl_scaled': kl_scaled.item(),
        'kl_weight': kl_weight
    }

    return loss, metrics


def heteroscedastic_loss(
    model: nn.Module,
    x: torch.Tensor,
    y_class: torch.Tensor,
    y_return: torch.Tensor,
    n_samples: int = 1,
    kl_weight: float = 1.0,
    return_weight: float = 1.0
) -> Tuple[torch.Tensor, dict]:
    """
    ELBO loss with heteroscedastic return prediction.

    Combines:
    1. Classification loss for direction
    2. Gaussian NLL for return prediction (captures aleatoric uncertainty)
    3. KL divergence for weight distributions

    Args:
        model: Bayesian neural network with heteroscedastic output
        x: Input features
        y_class: Target class labels
        y_return: Target returns
        n_samples: MC samples
        kl_weight: Weight for KL term
        return_weight: Weight for return prediction loss

    Returns:
        loss: Combined loss
        metrics: Dictionary with loss components
    """
    batch_size = x.size(0)

    class_nll = torch.tensor(0.0, device=x.device)
    return_nll = torch.tensor(0.0, device=x.device)

    for _ in range(n_samples):
        logits, (return_mu, return_log_var) = model(x, sample=True)

        # Classification NLL
        class_nll += F.cross_entropy(logits, y_class, reduction='sum')

        # Gaussian NLL for returns (heteroscedastic)
        # NLL = 0.5 * log(var) + 0.5 * (y - mu)^2 / var
        return_var = torch.exp(return_log_var)
        return_nll += 0.5 * torch.sum(
            return_log_var + (y_return - return_mu) ** 2 / return_var
        )

    class_nll = class_nll / (n_samples * batch_size)
    return_nll = return_nll / (n_samples * batch_size)

    # KL divergence
    kl_div = model.kl_divergence()
    kl_scaled = kl_div / batch_size

    # Total loss
    loss = class_nll + return_weight * return_nll + kl_weight * kl_scaled

    metrics = {
        'loss': loss.item(),
        'class_nll': class_nll.item(),
        'return_nll': return_nll.item(),
        'kl': kl_div.item(),
        'kl_scaled': kl_scaled.item()
    }

    return loss, metrics


class ELBOLoss(nn.Module):
    """
    ELBO Loss module for easy integration with training loops.

    Example:
        criterion = ELBOLoss(kl_weight=1.0, n_samples=3)
        loss, metrics = criterion(model, x, y)
    """

    def __init__(
        self,
        n_samples: int = 1,
        kl_weight: float = 1.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.n_samples = n_samples
        self.kl_weight = kl_weight
        self.reduction = reduction

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute ELBO loss."""
        return elbo_loss(
            model, x, y,
            n_samples=self.n_samples,
            kl_weight=self.kl_weight,
            reduction=self.reduction
        )

    def set_kl_weight(self, kl_weight: float):
        """Update KL weight (for annealing)."""
        self.kl_weight = kl_weight


class FocalLossWithUncertainty(nn.Module):
    """
    Focal Loss adapted for Bayesian NNs.

    Focal loss helps with class imbalance by down-weighting easy examples.
    Combined with uncertainty, it can focus on hard but confident examples.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        uncertainty_weight: float = 0.1
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.uncertainty_weight = uncertainty_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model predictions (batch, num_classes)
            targets: Target labels (batch,)
            uncertainty: Optional epistemic uncertainty (batch,)

        Returns:
            loss: Focal loss value
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze()

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weighting for class imbalance
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight

        # Uncertainty weighting (optional)
        if uncertainty is not None:
            # Down-weight uncertain examples
            uncertainty_weight = 1.0 / (1.0 + self.uncertainty_weight * uncertainty)
            focal_weight = focal_weight * uncertainty_weight

        loss = focal_weight * ce_loss
        return loss.mean()
