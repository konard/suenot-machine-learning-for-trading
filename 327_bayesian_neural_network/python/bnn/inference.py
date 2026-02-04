"""
Monte Carlo Inference for Bayesian Neural Networks.

Provides utilities for uncertainty estimation and prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm


@dataclass
class UncertaintyDecomposition:
    """
    Container for decomposed uncertainty estimates.

    Attributes:
        mean_prediction: Mean prediction across MC samples
        epistemic: Epistemic (model) uncertainty - reducible with more data
        aleatoric: Aleatoric (data) uncertainty - irreducible noise
        total: Total uncertainty (epistemic + aleatoric)
        samples: Optional raw MC samples for analysis
    """
    mean_prediction: torch.Tensor
    epistemic: torch.Tensor
    aleatoric: Optional[torch.Tensor] = None
    total: Optional[torch.Tensor] = None
    samples: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.aleatoric is not None and self.total is None:
            self.total = self.epistemic + self.aleatoric
        elif self.total is None:
            self.total = self.epistemic


class MCInference:
    """
    Monte Carlo Inference engine for Bayesian Neural Networks.

    Performs multiple forward passes with weight sampling to estimate
    predictive distributions and uncertainties.

    Args:
        model: Bayesian neural network model
        n_samples: Number of MC samples for inference
        device: Device to run inference on
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 100,
        device: str = 'cpu'
    ):
        self.model = model
        self.n_samples = n_samples
        self.device = device
        self.model.to(device)

    def predict_classification(
        self,
        x: torch.Tensor,
        return_samples: bool = False
    ) -> UncertaintyDecomposition:
        """
        Predict class probabilities with uncertainty.

        Args:
            x: Input features (batch_size, input_dim)
            return_samples: Whether to return raw MC samples

        Returns:
            UncertaintyDecomposition with probabilities and uncertainties
        """
        x = x.to(self.device)
        self.model.train(False)
        samples = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                logits, _ = self.model(x, sample=True)
                probs = F.softmax(logits, dim=-1)
                samples.append(probs)

        # Stack: (n_samples, batch_size, num_classes)
        samples_tensor = torch.stack(samples, dim=0)

        # Mean prediction
        mean_probs = samples_tensor.mean(dim=0)

        # Epistemic uncertainty: variance of predictions
        # Sum variance across classes to get scalar per sample
        epistemic = samples_tensor.var(dim=0).sum(dim=-1)

        return UncertaintyDecomposition(
            mean_prediction=mean_probs,
            epistemic=epistemic,
            samples=samples_tensor if return_samples else None
        )

    def predict_regression(
        self,
        x: torch.Tensor,
        return_samples: bool = False
    ) -> UncertaintyDecomposition:
        """
        Predict returns with decomposed uncertainty.

        For heteroscedastic models, separates epistemic and aleatoric uncertainty.

        Args:
            x: Input features
            return_samples: Whether to return raw MC samples

        Returns:
            UncertaintyDecomposition with mean, epistemic, and aleatoric uncertainties
        """
        x = x.to(self.device)
        self.model.train(False)

        mean_samples = []
        var_samples = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                _, (mu, log_var) = self.model(x, sample=True)
                mean_samples.append(mu)
                var_samples.append(torch.exp(log_var))

        means_tensor = torch.stack(mean_samples, dim=0)
        vars_tensor = torch.stack(var_samples, dim=0)

        # Mean prediction
        mean_return = means_tensor.mean(dim=0)

        # Epistemic: variance of means (model uncertainty)
        epistemic = means_tensor.var(dim=0)

        # Aleatoric: mean of variances (data uncertainty)
        aleatoric = vars_tensor.mean(dim=0)

        return UncertaintyDecomposition(
            mean_prediction=mean_return,
            epistemic=epistemic,
            aleatoric=aleatoric,
            samples=means_tensor if return_samples else None
        )

    def predict_with_confidence_interval(
        self,
        x: torch.Tensor,
        confidence: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with confidence intervals.

        Args:
            x: Input features
            confidence: Confidence level (e.g., 0.95 for 95% CI)

        Returns:
            mean: Mean prediction
            lower: Lower bound of confidence interval
            upper: Upper bound of confidence interval
        """
        result = self.predict_classification(x, return_samples=True)
        samples = result.samples  # (n_samples, batch, classes)

        mean = samples.mean(dim=0)

        # Compute percentiles
        alpha = (1 - confidence) / 2
        lower = torch.quantile(samples, alpha, dim=0)
        upper = torch.quantile(samples, 1 - alpha, dim=0)

        return mean, lower, upper

    def calibration_analysis(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Analyze calibration of uncertainty estimates.

        A well-calibrated model should have predicted confidence
        match observed accuracy across different confidence levels.

        Args:
            x: Input features
            y: True labels
            n_bins: Number of bins for calibration analysis

        Returns:
            Dictionary with calibration metrics
        """
        result = self.predict_classification(x)
        probs = result.mean_prediction
        uncertainties = result.epistemic

        # Get predictions and confidences
        confidences, predictions = probs.max(dim=-1)
        accuracies = (predictions == y).float()

        # Bin by confidence
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=self.device)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i in range(n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            mask = (confidences >= lower) & (confidences < upper)

            if mask.sum() > 0:
                bin_acc = accuracies[mask].mean().item()
                bin_conf = confidences[mask].mean().item()
                bin_count = mask.sum().item()
            else:
                bin_acc = 0
                bin_conf = (lower + upper).item() / 2
                bin_count = 0

            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            bin_counts.append(bin_count)

        # Expected Calibration Error (ECE)
        total_samples = sum(bin_counts)
        ece = sum(
            (count / total_samples) * abs(acc - conf)
            for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts)
            if count > 0
        )

        # Maximum Calibration Error (MCE)
        mce = max(
            abs(acc - conf)
            for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts)
            if count > 0
        ) if any(c > 0 for c in bin_counts) else 0

        return {
            'ece': ece,
            'mce': mce,
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts,
            'mean_accuracy': accuracies.mean().item(),
            'mean_confidence': confidences.mean().item()
        }

    def uncertainty_stratified_accuracy(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_quantiles: int = 5
    ) -> Dict[str, List[float]]:
        """
        Compute accuracy stratified by uncertainty levels.

        A good model should have higher accuracy on low-uncertainty predictions.

        Args:
            x: Input features
            y: True labels
            n_quantiles: Number of uncertainty quantiles

        Returns:
            Dictionary with accuracy per uncertainty quantile
        """
        result = self.predict_classification(x)
        probs = result.mean_prediction
        uncertainties = result.epistemic

        predictions = probs.argmax(dim=-1)
        correct = (predictions == y).float()

        # Compute quantile boundaries
        quantile_boundaries = torch.quantile(
            uncertainties,
            torch.linspace(0, 1, n_quantiles + 1, device=self.device)
        )

        quantile_accuracies = []
        quantile_counts = []
        quantile_means = []

        for i in range(n_quantiles):
            lower = quantile_boundaries[i]
            upper = quantile_boundaries[i + 1]

            if i == n_quantiles - 1:
                mask = (uncertainties >= lower) & (uncertainties <= upper)
            else:
                mask = (uncertainties >= lower) & (uncertainties < upper)

            if mask.sum() > 0:
                quantile_acc = correct[mask].mean().item()
                quantile_count = mask.sum().item()
                quantile_mean_unc = uncertainties[mask].mean().item()
            else:
                quantile_acc = 0
                quantile_count = 0
                quantile_mean_unc = (lower + upper).item() / 2

            quantile_accuracies.append(quantile_acc)
            quantile_counts.append(quantile_count)
            quantile_means.append(quantile_mean_unc)

        return {
            'quantile_accuracies': quantile_accuracies,
            'quantile_counts': quantile_counts,
            'quantile_mean_uncertainties': quantile_means
        }


def predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute predictive entropy from probability distribution.

    H[p] = -sum(p * log(p))

    Args:
        probs: Probability tensor (batch, num_classes)

    Returns:
        entropy: Entropy per sample (batch,)
    """
    # Avoid log(0)
    eps = 1e-10
    return -torch.sum(probs * torch.log(probs + eps), dim=-1)


def mutual_information(mc_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute mutual information from MC samples.

    MI = H[E[p]] - E[H[p]]

    This measures the information gain about the prediction
    from knowing the weights.

    Args:
        mc_probs: MC probability samples (n_samples, batch, num_classes)

    Returns:
        mi: Mutual information per sample (batch,)
    """
    # Mean prediction entropy
    mean_probs = mc_probs.mean(dim=0)
    mean_entropy = predictive_entropy(mean_probs)

    # Mean of entropies
    individual_entropies = predictive_entropy(mc_probs)  # (n_samples, batch)
    expected_entropy = individual_entropies.mean(dim=0)

    # MI = H[E[p]] - E[H[p]]
    return mean_entropy - expected_entropy


def brier_score(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute Brier score for probability predictions.

    Brier = mean((probs - one_hot_targets)^2)

    Lower is better. Perfect = 0.

    Args:
        probs: Predicted probabilities (batch, num_classes)
        targets: True labels (batch,)

    Returns:
        brier: Brier score
    """
    num_classes = probs.size(-1)
    one_hot = F.one_hot(targets, num_classes).float()
    return torch.mean((probs - one_hot) ** 2)
