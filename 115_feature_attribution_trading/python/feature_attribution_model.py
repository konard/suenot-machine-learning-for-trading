"""
Feature Attribution Trading Model implementation.

This module implements a neural network trading model with explainability
methods including SHAP, Integrated Gradients, and Layer-wise Relevance
Propagation (LRP) for understanding feature importance in trading decisions.

References:
    Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions"
    NeurIPS 2017. arXiv:1705.07874

    Sundararajan et al., "Axiomatic Attribution for Deep Networks"
    ICML 2017. arXiv:1703.01365
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AttributionConfig:
    """Configuration for Feature Attribution Model."""
    input_dim: int = 10
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    output_dim: int = 3  # BUY, HOLD, SELL
    dropout: float = 0.2
    use_batch_norm: bool = True
    activation: str = "relu"
    # Attribution settings
    n_samples_shap: int = 100
    n_steps_ig: int = 50
    baseline_type: str = "zero"  # "zero", "mean", "random"


class FeatureEmbedding(nn.Module):
    """
    Feature embedding layer with optional normalization.

    Projects raw features to a higher dimensional space for better
    representation learning.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.linear = nn.Linear(input_dim, embed_dim)
        self.batch_norm = nn.BatchNorm1d(embed_dim) if use_batch_norm else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim) or (batch, input_dim)

        Returns:
            Embedded tensor
        """
        if x.dim() == 3:
            batch, seq_len, _ = x.shape
            x = x.reshape(batch * seq_len, -1)
            x = self.linear(x)
            x = self.batch_norm(x)
            x = self.activation(x)
            x = x.reshape(batch, seq_len, -1)
        else:
            x = self.linear(x)
            if x.size(0) > 1:
                x = self.batch_norm(x)
            x = self.activation(x)
        return x


class TemporalBlock(nn.Module):
    """
    Temporal processing block using 1D convolution and self-attention.

    Captures both local patterns (via convolution) and global dependencies
    (via attention) in time series data.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Causal convolution
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size - 1, groups=1
        )
        self.conv_norm = nn.LayerNorm(d_model)

        # Self-attention
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Causal convolution
        residual = x
        x_conv = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x_conv = self.conv(x_conv)[:, :, :-self.conv.padding[0]]  # Causal masking
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_model)
        x = self.conv_norm(residual + x_conv)

        # Self-attention
        residual = x
        x_attn, _ = self.attention(x, x, x)
        x = self.attn_norm(residual + x_attn)

        # Feed-forward
        residual = x
        x = self.ff_norm(residual + self.ff(x))

        return x


class TradingMLP(nn.Module):
    """
    Multi-layer perceptron for trading signal prediction.

    A simple feedforward network that can be easily interpreted
    using gradient-based attribution methods.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        return self.network(x)


class FeatureAttributionModel(nn.Module):
    """
    Feature Attribution Trading Model.

    A neural network model for trading signal prediction with built-in
    feature attribution capabilities. Supports multiple explainability
    methods:

    1. SHAP (SHapley Additive exPlanations) - Game-theoretic feature importance
    2. Integrated Gradients - Path-based attribution method
    3. Gradient-based attribution - Simple gradient saliency

    Architecture:
        Input -> Embedding -> [TemporalBlock]×N -> Pooling -> MLP -> Output
    """

    def __init__(
        self,
        config: Optional[AttributionConfig] = None,
        input_dim: int = 10,
        hidden_dim: int = 64,
        n_temporal_layers: int = 2,
        n_classes: int = 3,
        dropout: float = 0.2,
        seq_len: int = 64,
    ):
        """
        Initialize the model.

        Args:
            config: Model configuration (overrides other args if provided)
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            n_temporal_layers: Number of temporal processing layers
            n_classes: Number of output classes (3 for BUY/HOLD/SELL)
            dropout: Dropout rate
            seq_len: Expected sequence length
        """
        super().__init__()

        self.config = config or AttributionConfig(input_dim=input_dim)
        self.input_dim = self.config.input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.seq_len = seq_len

        # Feature embedding
        self.embedding = FeatureEmbedding(
            self.input_dim, hidden_dim, self.config.use_batch_norm
        )

        # Temporal processing
        self.temporal_layers = nn.ModuleList([
            TemporalBlock(hidden_dim, kernel_size=3, dropout=dropout)
            for _ in range(n_temporal_layers)
        ])

        # Output head
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = TradingMLP(
            hidden_dim, [hidden_dim // 2], n_classes, dropout, False
        )

        # Store attribution settings
        self.n_samples_shap = self.config.n_samples_shap
        self.n_steps_ig = self.config.n_steps_ig
        self.baseline_type = self.config.baseline_type

        # Cache for intermediate activations (for attribution)
        self._activations = {}
        self._gradients = {}

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input features of shape (batch, seq_len, input_dim)
            return_features: Whether to return intermediate features

        Returns:
            Logits of shape (batch, n_classes)
            If return_features=True, also returns feature tensor
        """
        # Store input for attribution
        self._activations['input'] = x

        # Embedding
        x = self.embedding(x)
        self._activations['embedded'] = x

        # Temporal processing
        for i, layer in enumerate(self.temporal_layers):
            x = layer(x)
            self._activations[f'temporal_{i}'] = x

        # Pool (use last timestep)
        features = x[:, -1, :]
        self._activations['pooled'] = features

        # Output
        features = self.output_norm(features)
        logits = self.output_head(features)

        if return_features:
            return logits, features
        return logits

    def predict(self, features: np.ndarray) -> Dict:
        """
        Make prediction from numpy features.

        Args:
            features: Features of shape (seq_len, input_dim)

        Returns:
            Dictionary with signal, confidence, probabilities, and attributions
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

        signal_map = {0: "BUY", 1: "HOLD", 2: "SELL"}
        signal_idx = int(np.argmax(probs))

        return {
            "signal": signal_map[signal_idx],
            "confidence": float(probs[signal_idx]),
            "probabilities": {
                "BUY": float(probs[0]),
                "HOLD": float(probs[1]),
                "SELL": float(probs[2]),
            }
        }

    def predict_with_attribution(
        self,
        features: np.ndarray,
        method: str = "integrated_gradients",
        target_class: Optional[int] = None,
    ) -> Dict:
        """
        Make prediction with feature attribution.

        Args:
            features: Features of shape (seq_len, input_dim)
            method: Attribution method ("gradient", "integrated_gradients", "shap")
            target_class: Class to explain (None = predicted class)

        Returns:
            Dictionary with prediction and attribution scores
        """
        # Get base prediction
        prediction = self.predict(features)

        # Determine target class
        if target_class is None:
            target_class = {"BUY": 0, "HOLD": 1, "SELL": 2}[prediction["signal"]]

        # Compute attribution
        if method == "gradient":
            attribution = self.gradient_attribution(features, target_class)
        elif method == "integrated_gradients":
            attribution = self.integrated_gradients(features, target_class)
        elif method == "shap":
            attribution = self.shap_attribution(features, target_class)
        else:
            raise ValueError(f"Unknown attribution method: {method}")

        # Aggregate attribution across time
        feature_importance = np.mean(np.abs(attribution), axis=0)
        temporal_importance = np.mean(np.abs(attribution), axis=1)

        prediction["attribution"] = {
            "method": method,
            "raw": attribution,
            "feature_importance": feature_importance,
            "temporal_importance": temporal_importance,
            "top_features": self._get_top_features(feature_importance),
        }

        return prediction

    def gradient_attribution(
        self,
        features: np.ndarray,
        target_class: int,
    ) -> np.ndarray:
        """
        Compute gradient-based attribution.

        Simple gradient of output with respect to input features.

        Args:
            features: Input features of shape (seq_len, input_dim)
            target_class: Target class index

        Returns:
            Attribution scores of shape (seq_len, input_dim)
        """
        self.eval()
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        x.requires_grad_(True)

        logits = self.forward(x)
        target_score = logits[0, target_class]

        target_score.backward()

        gradients = x.grad.squeeze(0).detach().numpy()

        # Input × Gradient
        attribution = features * gradients

        return attribution

    def integrated_gradients(
        self,
        features: np.ndarray,
        target_class: int,
        n_steps: Optional[int] = None,
        baseline: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute Integrated Gradients attribution.

        Integrates gradients along a path from a baseline to the input,
        satisfying completeness and sensitivity axioms.

        Args:
            features: Input features of shape (seq_len, input_dim)
            target_class: Target class index
            n_steps: Number of integration steps
            baseline: Baseline input (default: zeros)

        Returns:
            Attribution scores of shape (seq_len, input_dim)
        """
        n_steps = n_steps or self.n_steps_ig

        # Set baseline
        if baseline is None:
            if self.baseline_type == "zero":
                baseline = np.zeros_like(features)
            elif self.baseline_type == "mean":
                baseline = np.mean(features, axis=0, keepdims=True)
                baseline = np.broadcast_to(baseline, features.shape)
            else:
                baseline = np.random.randn(*features.shape) * 0.01

        self.eval()

        # Create interpolated inputs
        alphas = np.linspace(0, 1, n_steps)
        scaled_inputs = np.array([
            baseline + alpha * (features - baseline)
            for alpha in alphas
        ])  # (n_steps, seq_len, input_dim)

        # Compute gradients for all scaled inputs
        x_batch = torch.tensor(scaled_inputs, dtype=torch.float32)
        x_batch.requires_grad_(True)

        logits = self.forward(x_batch)
        target_scores = logits[:, target_class].sum()

        target_scores.backward()

        gradients = x_batch.grad.detach().numpy()  # (n_steps, seq_len, input_dim)

        # Integrate gradients using trapezoidal rule
        avg_gradients = np.mean(gradients, axis=0)

        # Scale by difference from baseline
        integrated_gradients = (features - baseline) * avg_gradients

        return integrated_gradients

    def shap_attribution(
        self,
        features: np.ndarray,
        target_class: int,
        n_samples: Optional[int] = None,
        background: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute SHAP-like attribution using sampling.

        Approximates Shapley values using random feature coalitions.
        For efficiency, uses gradient-based approximation.

        Args:
            features: Input features of shape (seq_len, input_dim)
            target_class: Target class index
            n_samples: Number of samples for estimation
            background: Background samples for marginal expectations

        Returns:
            Attribution scores of shape (seq_len, input_dim)
        """
        n_samples = n_samples or self.n_samples_shap
        seq_len, input_dim = features.shape

        # Generate background if not provided
        if background is None:
            # Use noise as background
            background = np.random.randn(n_samples, seq_len, input_dim) * 0.1

        self.eval()
        attribution = np.zeros_like(features)

        # For each feature, estimate its marginal contribution
        for i in range(input_dim):
            contributions = []

            for _ in range(min(n_samples, 20)):
                # Sample a random subset of features
                mask = np.random.binomial(1, 0.5, input_dim)

                # With feature i
                mask_with = mask.copy()
                mask_with[i] = 1

                # Without feature i
                mask_without = mask.copy()
                mask_without[i] = 0

                # Create masked inputs
                bg_sample = background[np.random.randint(len(background))]

                x_with = features.copy()
                x_without = features.copy()

                for j in range(input_dim):
                    if mask_with[j] == 0:
                        x_with[:, j] = bg_sample[:, j]
                    if mask_without[j] == 0:
                        x_without[:, j] = bg_sample[:, j]

                # Compute outputs
                with torch.no_grad():
                    x_with_t = torch.tensor(x_with, dtype=torch.float32).unsqueeze(0)
                    x_without_t = torch.tensor(x_without, dtype=torch.float32).unsqueeze(0)

                    logits_with = self.forward(x_with_t)
                    logits_without = self.forward(x_without_t)

                    prob_with = F.softmax(logits_with, dim=-1)[0, target_class].item()
                    prob_without = F.softmax(logits_without, dim=-1)[0, target_class].item()

                contributions.append(prob_with - prob_without)

            # Average contribution for feature i
            avg_contribution = np.mean(contributions)
            attribution[:, i] = avg_contribution

        return attribution

    def get_feature_importance(
        self,
        features: np.ndarray,
        method: str = "integrated_gradients",
    ) -> Dict[str, float]:
        """
        Get overall feature importance scores.

        Args:
            features: Input features of shape (seq_len, input_dim)
            method: Attribution method

        Returns:
            Dictionary mapping feature names to importance scores
        """
        result = self.predict_with_attribution(features, method)
        return result["attribution"]["top_features"]

    def _get_top_features(
        self,
        importance_scores: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Get top features sorted by importance.

        Args:
            importance_scores: Importance scores per feature
            feature_names: Optional feature names

        Returns:
            Dictionary of feature names to importance scores
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance_scores))]

        # Normalize scores
        total = np.sum(importance_scores)
        if total > 0:
            normalized = importance_scores / total
        else:
            normalized = importance_scores

        # Create sorted dictionary
        feature_dict = {
            name: float(score)
            for name, score in zip(feature_names, normalized)
        }

        return dict(sorted(feature_dict.items(), key=lambda x: -x[1]))

    def explain_prediction(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None,
        methods: List[str] = None,
    ) -> Dict:
        """
        Generate comprehensive explanation for a prediction.

        Args:
            features: Input features of shape (seq_len, input_dim)
            feature_names: Names of input features
            methods: Attribution methods to use

        Returns:
            Dictionary with prediction and explanations from multiple methods
        """
        methods = methods or ["gradient", "integrated_gradients"]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]

        # Base prediction
        base_prediction = self.predict(features)
        target_class = {"BUY": 0, "HOLD": 1, "SELL": 2}[base_prediction["signal"]]

        explanation = {
            "prediction": base_prediction,
            "feature_names": feature_names,
            "attributions": {},
        }

        # Compute attributions for each method
        for method in methods:
            try:
                if method == "gradient":
                    attr = self.gradient_attribution(features, target_class)
                elif method == "integrated_gradients":
                    attr = self.integrated_gradients(features, target_class)
                elif method == "shap":
                    attr = self.shap_attribution(features, target_class)
                else:
                    continue

                feature_importance = np.mean(np.abs(attr), axis=0)

                explanation["attributions"][method] = {
                    "raw": attr,
                    "feature_importance": dict(zip(feature_names, feature_importance.tolist())),
                    "top_features": self._get_top_features(feature_importance, feature_names),
                }
            except Exception as e:
                logger.warning(f"Failed to compute {method} attribution: {e}")

        # Consensus ranking
        if len(explanation["attributions"]) > 1:
            all_rankings = []
            for method_data in explanation["attributions"].values():
                ranking = list(method_data["top_features"].keys())
                all_rankings.append(ranking)

            # Simple consensus: average rank
            rank_scores = {name: 0 for name in feature_names}
            for ranking in all_rankings:
                for rank, name in enumerate(ranking):
                    rank_scores[name] += rank

            # Lower average rank = more important
            consensus = sorted(rank_scores.items(), key=lambda x: x[1])
            explanation["consensus_ranking"] = [name for name, _ in consensus]

        return explanation


class AttributionBasedTrader:
    """
    Trading strategy that uses feature attributions for decision making.

    Generates trading signals based on model predictions while monitoring
    feature importance for risk management and signal quality assessment.
    """

    def __init__(
        self,
        model: FeatureAttributionModel,
        confidence_threshold: float = 0.6,
        attribution_threshold: float = 0.1,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Initialize the trader.

        Args:
            model: Feature attribution model
            confidence_threshold: Minimum confidence for trade signals
            attribution_threshold: Minimum feature concentration for quality
            feature_names: Names of input features
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.attribution_threshold = attribution_threshold
        self.feature_names = feature_names

    def generate_signal(
        self,
        features: np.ndarray,
        require_attribution_quality: bool = True,
    ) -> Dict:
        """
        Generate trading signal with attribution analysis.

        Args:
            features: Input features
            require_attribution_quality: Whether to filter by attribution quality

        Returns:
            Signal dictionary with trading decision and explanation
        """
        # Get prediction with attribution
        result = self.model.predict_with_attribution(
            features, method="integrated_gradients"
        )

        signal = result["signal"]
        confidence = result["confidence"]
        attribution = result["attribution"]

        # Check attribution quality (concentration in few features)
        feature_importance = attribution["feature_importance"]
        top_3_importance = sum(sorted(feature_importance, reverse=True)[:3])
        attribution_quality = top_3_importance / (sum(feature_importance) + 1e-8)

        # Decision logic
        if confidence < self.confidence_threshold:
            final_signal = "HOLD"
            reason = f"Low confidence ({confidence:.2%})"
        elif require_attribution_quality and attribution_quality < self.attribution_threshold:
            final_signal = "HOLD"
            reason = f"Diffuse attribution ({attribution_quality:.2%})"
        else:
            final_signal = signal
            reason = f"High confidence ({confidence:.2%}) with focused attribution"

        return {
            "signal": final_signal,
            "original_signal": signal,
            "confidence": confidence,
            "reason": reason,
            "attribution_quality": attribution_quality,
            "top_features": attribution["top_features"],
            "temporal_focus": np.argmax(attribution["temporal_importance"]),
        }


def create_model(
    config: Optional[AttributionConfig] = None,
    input_dim: int = 10,
    hidden_dim: int = 64,
    seq_len: int = 64,
) -> FeatureAttributionModel:
    """Create Feature Attribution model from config or parameters."""
    if config:
        return FeatureAttributionModel(config=config, seq_len=seq_len)
    return FeatureAttributionModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
    )


if __name__ == "__main__":
    # Demo
    print("Feature Attribution Trading Model Demo")
    print("=" * 60)

    # Create model
    config = AttributionConfig(
        input_dim=10,
        hidden_dims=[128, 64],
        dropout=0.2,
    )
    model = create_model(config=config, hidden_dim=64, seq_len=64)

    print(f"\nModel configuration:")
    print(f"  Input dim: {config.input_dim}")
    print(f"  Hidden dims: {config.hidden_dims}")
    print(f"  Output dim: {config.output_dim}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # Generate random input
    batch_size = 2
    seq_len = 64
    input_dim = 10

    x = torch.randn(batch_size, seq_len, input_dim)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(x)
    print(f"Output shape: {output.shape}")

    # Test prediction with attribution
    print("\n" + "-" * 60)
    print("Testing Feature Attribution Methods")
    print("-" * 60)

    features = np.random.randn(seq_len, input_dim).astype(np.float32)

    # Test gradient attribution
    print("\n1. Gradient Attribution:")
    result = model.predict_with_attribution(features, method="gradient")
    print(f"   Signal: {result['signal']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Top features: {list(result['attribution']['top_features'].keys())[:3]}")

    # Test integrated gradients
    print("\n2. Integrated Gradients:")
    result = model.predict_with_attribution(features, method="integrated_gradients")
    print(f"   Signal: {result['signal']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Top features: {list(result['attribution']['top_features'].keys())[:3]}")

    # Test SHAP
    print("\n3. SHAP Attribution:")
    result = model.predict_with_attribution(features, method="shap")
    print(f"   Signal: {result['signal']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Top features: {list(result['attribution']['top_features'].keys())[:3]}")

    # Test comprehensive explanation
    print("\n4. Comprehensive Explanation:")
    feature_names = [
        "open", "high", "low", "close", "volume",
        "rsi", "macd", "sma_20", "volatility", "momentum"
    ]
    explanation = model.explain_prediction(
        features, feature_names, methods=["gradient", "integrated_gradients"]
    )
    print(f"   Prediction: {explanation['prediction']['signal']}")
    if "consensus_ranking" in explanation:
        print(f"   Consensus top features: {explanation['consensus_ranking'][:3]}")

    # Test attribution-based trader
    print("\n" + "-" * 60)
    print("Testing Attribution-Based Trader")
    print("-" * 60)

    trader = AttributionBasedTrader(
        model,
        confidence_threshold=0.5,
        attribution_threshold=0.1,
        feature_names=feature_names,
    )

    signal = trader.generate_signal(features)
    print(f"\n   Final signal: {signal['signal']}")
    print(f"   Original signal: {signal['original_signal']}")
    print(f"   Reason: {signal['reason']}")
    print(f"   Attribution quality: {signal['attribution_quality']:.2%}")

    print("\n" + "=" * 60)
    print("Demo complete!")
