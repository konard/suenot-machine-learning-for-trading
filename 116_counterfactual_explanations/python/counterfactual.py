"""
Counterfactual explanation generation for trading models.

This module provides methods for generating counterfactual explanations
that identify minimal changes to input features that would flip a model's prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class CounterfactualResult:
    """Result of counterfactual generation."""
    original: torch.Tensor
    counterfactual: torch.Tensor
    original_class: int
    target_class: int
    original_prob: float
    counterfactual_prob: float
    is_valid: bool
    num_features_changed: int
    l1_distance: float
    l2_distance: float
    changed_features: List[Tuple[int, float, float]]  # (index, original, counterfactual)


class CounterfactualGenerator(nn.Module):
    """
    Neural network-based counterfactual generator.

    Uses an encoder-decoder architecture to generate counterfactual
    explanations conditioned on a target class.

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, hidden_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        target_class: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate counterfactual for given input and target class.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            target_class: Target class indices of shape (batch_size,)

        Returns:
            Counterfactual tensor of shape (batch_size, input_dim)
        """
        # Encode input
        z = self.encoder(x)

        # Get target class embedding
        class_emb = self.class_embedding(target_class)

        # Concatenate and decode
        z_combined = torch.cat([z, class_emb], dim=-1)
        delta = self.decoder(z_combined)

        # Generate counterfactual as perturbation
        x_cf = x + delta

        return x_cf


class CounterfactualOptimizer:
    """
    Gradient-based counterfactual optimizer.

    Generates counterfactual explanations by optimizing an input perturbation
    to achieve a target prediction while minimizing distance from the original.

    Args:
        model: Trading classifier model
        lambda_proximity: Weight for proximity (distance) loss
        lambda_validity: Weight for validity (classification) loss
        lambda_sparsity: Weight for sparsity loss
        actionable_mask: Optional mask for actionable features (1=can change, 0=fixed)
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_proximity: float = 1.0,
        lambda_validity: float = 1.0,
        lambda_sparsity: float = 0.1,
        actionable_mask: Optional[torch.Tensor] = None
    ):
        self.model = model
        self.lambda_proximity = lambda_proximity
        self.lambda_validity = lambda_validity
        self.lambda_sparsity = lambda_sparsity
        self.actionable_mask = actionable_mask

    def generate(
        self,
        x: torch.Tensor,
        target_class: Union[int, torch.Tensor],
        num_steps: int = 100,
        lr: float = 0.01,
        feature_names: Optional[List[str]] = None
    ) -> CounterfactualResult:
        """
        Generate counterfactual via gradient descent.

        Args:
            x: Original input tensor (1, num_features)
            target_class: Desired prediction class
            num_steps: Number of optimization steps
            lr: Learning rate
            feature_names: Optional list of feature names for reporting

        Returns:
            CounterfactualResult with counterfactual and metadata
        """
        if isinstance(target_class, int):
            target_class = torch.tensor([target_class])

        # Get original prediction
        with torch.no_grad():
            orig_probs = F.softmax(self.model(x), dim=-1)
            orig_class = orig_probs.argmax(dim=-1).item()
            orig_prob = orig_probs[0, orig_class].item()

        # Initialize counterfactual
        x = x.clone().detach()
        x_cf = x.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x_cf], lr=lr)

        best_cf = None
        best_loss = float('inf')

        for step in range(num_steps):
            optimizer.zero_grad()

            # Compute losses
            loss, _ = self._compute_loss(x, x_cf, target_class)

            # Backpropagate
            loss.backward()

            # Apply actionability mask to gradients
            if self.actionable_mask is not None:
                x_cf.grad.data *= self.actionable_mask

            optimizer.step()

            # Track best valid counterfactual
            if self._is_valid(x_cf, target_class) and loss.item() < best_loss:
                best_loss = loss.item()
                best_cf = x_cf.clone().detach()

        # Use best found or final
        final_cf = best_cf if best_cf is not None else x_cf.detach()

        # Get final prediction
        with torch.no_grad():
            cf_probs = F.softmax(self.model(final_cf), dim=-1)
            cf_class = cf_probs.argmax(dim=-1).item()
            cf_prob = cf_probs[0, target_class[0]].item()

        # Compute metrics
        is_valid = cf_class == target_class[0].item()
        l1_dist = torch.norm(x - final_cf, p=1).item()
        l2_dist = torch.norm(x - final_cf, p=2).item()

        # Find changed features
        changed = []
        threshold = 0.01
        for i in range(x.shape[1]):
            orig_val = x[0, i].item()
            cf_val = final_cf[0, i].item()
            if abs(orig_val - cf_val) > threshold:
                changed.append((i, orig_val, cf_val))

        return CounterfactualResult(
            original=x,
            counterfactual=final_cf,
            original_class=orig_class,
            target_class=target_class[0].item(),
            original_prob=orig_prob,
            counterfactual_prob=cf_prob,
            is_valid=is_valid,
            num_features_changed=len(changed),
            l1_distance=l1_dist,
            l2_distance=l2_dist,
            changed_features=changed
        )

    def _compute_loss(
        self,
        x: torch.Tensor,
        x_cf: torch.Tensor,
        target_class: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss for counterfactual optimization."""
        # Proximity loss (L1)
        proximity = torch.norm(x - x_cf, p=1)

        # Validity loss (cross-entropy to target)
        logits = self.model(x_cf)
        validity = F.cross_entropy(logits, target_class)

        # Sparsity loss (approximate L0)
        sparsity = torch.sum(1 - torch.exp(-torch.abs(x - x_cf) / 0.1))

        total_loss = (
            self.lambda_proximity * proximity +
            self.lambda_validity * validity +
            self.lambda_sparsity * sparsity
        )

        return total_loss, {
            'proximity': proximity.item(),
            'validity': validity.item(),
            'sparsity': sparsity.item()
        }

    def _is_valid(
        self,
        x_cf: torch.Tensor,
        target_class: torch.Tensor,
        threshold: float = 0.5
    ) -> bool:
        """Check if counterfactual achieves target class."""
        with torch.no_grad():
            probs = F.softmax(self.model(x_cf), dim=-1)
            return probs[0, target_class[0]].item() > threshold


class DiCE:
    """
    Diverse Counterfactual Explanations (DiCE) implementation.

    Generates multiple diverse counterfactual explanations for
    comprehensive understanding of model behavior.

    Args:
        model: Trading classifier model
        num_counterfactuals: Number of counterfactuals to generate
        diversity_weight: Weight for diversity loss
    """

    def __init__(
        self,
        model: nn.Module,
        num_counterfactuals: int = 4,
        diversity_weight: float = 0.5
    ):
        self.model = model
        self.num_counterfactuals = num_counterfactuals
        self.diversity_weight = diversity_weight

    def generate(
        self,
        x: torch.Tensor,
        target_class: int,
        num_steps: int = 200,
        lr: float = 0.01
    ) -> List[CounterfactualResult]:
        """
        Generate diverse counterfactuals.

        Args:
            x: Original input tensor (1, num_features)
            target_class: Desired prediction class
            num_steps: Number of optimization steps
            lr: Learning rate

        Returns:
            List of CounterfactualResult objects
        """
        target = torch.tensor([target_class])
        x = x.clone().detach()

        # Initialize multiple counterfactuals
        cfs = [x.clone().requires_grad_(True) for _ in range(self.num_counterfactuals)]
        optimizer = torch.optim.Adam(cfs, lr=lr)

        for step in range(num_steps):
            optimizer.zero_grad()

            total_loss = 0.0

            for cf in cfs:
                # Validity loss
                logits = self.model(cf)
                validity = F.cross_entropy(logits, target)

                # Proximity loss
                proximity = torch.norm(x - cf, p=1)

                total_loss += validity + proximity

            # Diversity loss (pairwise distance between counterfactuals)
            diversity_loss = 0.0
            for i in range(len(cfs)):
                for j in range(i + 1, len(cfs)):
                    diversity_loss -= torch.norm(cfs[i] - cfs[j], p=1)

            total_loss += self.diversity_weight * diversity_loss

            total_loss.backward()
            optimizer.step()

        # Create results
        results = []
        with torch.no_grad():
            orig_probs = F.softmax(self.model(x), dim=-1)
            orig_class = orig_probs.argmax(dim=-1).item()

            for cf in cfs:
                cf_probs = F.softmax(self.model(cf), dim=-1)
                cf_class = cf_probs.argmax(dim=-1).item()

                changed = []
                for i in range(x.shape[1]):
                    orig_val = x[0, i].item()
                    cf_val = cf[0, i].item()
                    if abs(orig_val - cf_val) > 0.01:
                        changed.append((i, orig_val, cf_val))

                results.append(CounterfactualResult(
                    original=x,
                    counterfactual=cf.detach(),
                    original_class=orig_class,
                    target_class=target_class,
                    original_prob=orig_probs[0, orig_class].item(),
                    counterfactual_prob=cf_probs[0, target_class].item(),
                    is_valid=(cf_class == target_class),
                    num_features_changed=len(changed),
                    l1_distance=torch.norm(x - cf, p=1).item(),
                    l2_distance=torch.norm(x - cf, p=2).item(),
                    changed_features=changed
                ))

        return results


def format_explanation(
    result: CounterfactualResult,
    feature_names: Optional[List[str]] = None
) -> str:
    """
    Format counterfactual result as human-readable explanation.

    Args:
        result: CounterfactualResult object
        feature_names: Optional list of feature names

    Returns:
        Formatted explanation string
    """
    class_names = ['SELL', 'HOLD', 'BUY']

    lines = [
        "Counterfactual Explanation",
        "=" * 40,
        f"Original prediction: {class_names[result.original_class]} "
        f"({result.original_prob:.1%} confidence)",
        f"Target prediction: {class_names[result.target_class]}",
        f"Counterfactual valid: {result.is_valid}",
        "",
        f"To change from {class_names[result.original_class]} to "
        f"{class_names[result.target_class]}:",
    ]

    if result.changed_features:
        for idx, orig, cf in result.changed_features:
            name = feature_names[idx] if feature_names else f"Feature {idx}"
            direction = "increase" if cf > orig else "decrease"
            diff = abs(cf - orig)
            lines.append(f"  - {name}: {orig:.3f} -> {cf:.3f} "
                        f"({direction} by {diff:.3f})")
    else:
        lines.append("  No valid counterfactual found")

    lines.extend([
        "",
        f"Number of features changed: {result.num_features_changed}",
        f"L1 distance: {result.l1_distance:.4f}",
        f"L2 distance: {result.l2_distance:.4f}",
    ])

    return "\n".join(lines)
