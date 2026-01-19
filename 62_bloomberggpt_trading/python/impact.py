"""
News Impact Prediction Module

This module provides tools for predicting the market impact of financial
news events using LLM embeddings combined with market features.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime


@dataclass
class ImpactPrediction:
    """Prediction of news impact on market."""
    symbol: str
    direction: float      # -1 (negative) to 1 (positive)
    magnitude_pct: float  # Expected price change in percent
    confidence: float     # 0 to 1
    interpretation: str   # Human-readable interpretation
    features_used: Dict   # Features that contributed to prediction


class NewsImpactPredictor(nn.Module):
    """
    Predict market impact of financial news.

    This model combines LLM embeddings with market data to predict
    the magnitude and direction of price moves following news events.

    Architecture:
    - Text encoder: Processes LLM embeddings of news text
    - Market encoder: Processes current market state features
    - Combined predictor: Outputs direction, magnitude, and confidence
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        market_features: int = 10,
        hidden_dim: int = 256,
        dropout: float = 0.2
    ):
        """
        Initialize the impact predictor.

        Args:
            embedding_dim: Dimension of LLM text embeddings
            market_features: Number of market state features
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.market_features = market_features
        self.hidden_dim = hidden_dim

        # Text encoding branch
        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Market features branch
        self.market_encoder = nn.Sequential(
            nn.Linear(market_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )

        # Combined prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # [direction, log_magnitude, confidence_logit]
        )

    def forward(
        self,
        text_embedding: torch.Tensor,
        market_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict news impact.

        Args:
            text_embedding: LLM embedding of news text [batch, embedding_dim]
            market_features: Market state features [batch, market_features]

        Returns:
            Dict with direction (-1 to 1), magnitude (0 to inf), confidence (0 to 1)
        """
        text_encoded = self.text_encoder(text_embedding)
        market_encoded = self.market_encoder(market_features)

        combined = torch.cat([text_encoded, market_encoded], dim=-1)
        output = self.predictor(combined)

        return {
            "direction": torch.tanh(output[:, 0]),           # -1 to 1
            "magnitude": torch.exp(output[:, 1]),            # Positive, log-scale
            "confidence": torch.sigmoid(output[:, 2])        # 0 to 1
        }


class MarketFeatureExtractor:
    """
    Extract market features for impact prediction.

    Features include volatility, volume, momentum, and market conditions.
    """

    # Feature indices
    FEATURE_NAMES = [
        "volatility_1h",       # Recent 1h volatility
        "volume_ratio",        # Volume vs 20-day average
        "bid_ask_spread",      # Current spread (%)
        "hours_since_open",    # Time of day
        "day_of_week",         # Day of week (0-4)
        "vix_level",           # VIX or volatility index
        "momentum_1h",         # 1h momentum
        "momentum_1d",         # 1d momentum
        "price",               # Current price
        "avg_daily_volume",    # Average daily volume
    ]

    def __init__(self, data_source=None):
        """
        Initialize feature extractor.

        Args:
            data_source: Optional data source for real-time features
        """
        self.data_source = data_source

    def extract_features(
        self,
        symbol: str,
        timestamp: Optional[datetime] = None
    ) -> np.ndarray:
        """
        Extract market features for a symbol.

        Args:
            symbol: Stock/crypto symbol
            timestamp: Time to extract features for (default: now)

        Returns:
            Feature array of shape (10,)
        """
        timestamp = timestamp or datetime.now()

        if self.data_source:
            # In production, fetch from real data source
            return self._fetch_real_features(symbol, timestamp)
        else:
            # Return mock features for demo
            return self._mock_features(symbol, timestamp)

    def _mock_features(
        self,
        symbol: str,
        timestamp: datetime
    ) -> np.ndarray:
        """Generate mock features for demonstration."""
        np.random.seed(hash(symbol + str(timestamp.date())) % 2**32)

        return np.array([
            np.random.uniform(0.01, 0.05),    # volatility_1h
            np.random.uniform(0.5, 2.0),      # volume_ratio
            np.random.uniform(0.0005, 0.002), # bid_ask_spread
            timestamp.hour + timestamp.minute / 60,  # hours_since_open
            timestamp.weekday() if timestamp.weekday() < 5 else 4,  # day_of_week
            np.random.uniform(12, 30),        # vix_level
            np.random.uniform(-0.02, 0.02),   # momentum_1h
            np.random.uniform(-0.05, 0.05),   # momentum_1d
            np.random.uniform(50, 500),       # price
            np.random.uniform(1e6, 1e8),      # avg_daily_volume
        ], dtype=np.float32)

    def _fetch_real_features(
        self,
        symbol: str,
        timestamp: datetime
    ) -> np.ndarray:
        """Fetch real features from data source."""
        # Implementation depends on data source
        # For now, fall back to mock
        return self._mock_features(symbol, timestamp)


class ImpactPredictionPipeline:
    """
    End-to-end pipeline for news impact prediction.

    Combines LLM embeddings with market features to predict impact.
    """

    def __init__(
        self,
        impact_model: Optional[NewsImpactPredictor] = None,
        embedding_model=None,
        feature_extractor: Optional[MarketFeatureExtractor] = None
    ):
        """
        Initialize the pipeline.

        Args:
            impact_model: Trained impact prediction model
            embedding_model: Model to generate text embeddings
            feature_extractor: Market feature extractor
        """
        self.impact_model = impact_model or NewsImpactPredictor()
        self.embedding_model = embedding_model
        self.feature_extractor = feature_extractor or MarketFeatureExtractor()

    def predict(
        self,
        news_text: str,
        symbol: str,
        timestamp: Optional[datetime] = None
    ) -> ImpactPrediction:
        """
        Predict impact of news on a symbol.

        Args:
            news_text: Financial news text
            symbol: Stock/crypto symbol
            timestamp: Time of the news

        Returns:
            ImpactPrediction with predicted impact
        """
        timestamp = timestamp or datetime.now()

        # Get text embedding
        if self.embedding_model:
            text_embedding = self._get_embedding(news_text)
        else:
            # Mock embedding for demo
            text_embedding = torch.randn(1, self.impact_model.embedding_dim)

        # Get market features
        market_features = self.feature_extractor.extract_features(symbol, timestamp)
        market_features_tensor = torch.tensor(
            market_features, dtype=torch.float32
        ).unsqueeze(0)

        # Predict
        self.impact_model.eval()
        with torch.no_grad():
            prediction = self.impact_model(text_embedding, market_features_tensor)

        direction = prediction["direction"].item()
        magnitude = prediction["magnitude"].item()
        confidence = prediction["confidence"].item()

        return ImpactPrediction(
            symbol=symbol,
            direction=direction,
            magnitude_pct=magnitude * 100,  # Convert to percentage
            confidence=confidence,
            interpretation=self._interpret(direction, magnitude, confidence),
            features_used={
                name: market_features[i]
                for i, name in enumerate(MarketFeatureExtractor.FEATURE_NAMES)
            }
        )

    def _get_embedding(self, text: str) -> torch.Tensor:
        """Get text embedding from embedding model."""
        # Implementation depends on embedding model
        # Example with sentence-transformers:
        # embedding = self.embedding_model.encode(text, convert_to_tensor=True)
        # return embedding.unsqueeze(0)
        raise NotImplementedError("Embedding model not configured")

    def _interpret(
        self,
        direction: float,
        magnitude: float,
        confidence: float
    ) -> str:
        """Generate human-readable interpretation."""
        if confidence < 0.4:
            return "Low confidence prediction - uncertain impact"

        dir_str = "positive" if direction > 0.1 else "negative" if direction < -0.1 else "neutral"
        mag_pct = magnitude * 100

        if mag_pct > 5:
            mag_str = "major"
        elif mag_pct > 2:
            mag_str = "significant"
        elif mag_pct > 0.5:
            mag_str = "moderate"
        else:
            mag_str = "minor"

        return (
            f"Expected {mag_str} {dir_str} impact "
            f"({mag_pct:.1f}% move, {confidence:.0%} confidence)"
        )


def train_impact_model(
    model: NewsImpactPredictor,
    train_data: List[Dict],
    epochs: int = 100,
    learning_rate: float = 0.001
) -> Dict[str, List[float]]:
    """
    Train the impact prediction model.

    Args:
        model: NewsImpactPredictor model
        train_data: List of training examples with text_embedding,
                    market_features, and actual_returns
        epochs: Number of training epochs
        learning_rate: Learning rate

    Returns:
        Dict with training history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"loss": [], "direction_acc": [], "magnitude_mse": []}

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        correct_direction = 0
        total = 0

        for batch in train_data:
            optimizer.zero_grad()

            text_emb = batch["text_embedding"]
            market_feat = batch["market_features"]
            actual_returns = batch["actual_returns"]

            # Forward pass
            pred = model(text_emb, market_feat)

            # Calculate loss
            # Direction: MSE between predicted and actual sign
            actual_direction = torch.sign(actual_returns)
            direction_loss = nn.MSELoss()(pred["direction"], actual_direction)

            # Magnitude: MSE in log space
            actual_magnitude = torch.abs(actual_returns)
            magnitude_loss = nn.MSELoss()(
                torch.log(pred["magnitude"] + 1e-6),
                torch.log(actual_magnitude + 1e-6)
            )

            # Combined loss
            loss = direction_loss + 0.5 * magnitude_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct_direction += (
                (pred["direction"] * actual_direction > 0).float().sum().item()
            )
            total += len(actual_returns)

        history["loss"].append(epoch_loss / len(train_data))
        history["direction_acc"].append(correct_direction / total)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Loss={history['loss'][-1]:.4f}, "
                  f"Direction Acc={history['direction_acc'][-1]:.2%}")

    return history


if __name__ == "__main__":
    # Demo usage
    print("News Impact Prediction Demo\n" + "=" * 50)

    # Create pipeline with mock models
    pipeline = ImpactPredictionPipeline()

    # Test predictions
    test_cases = [
        ("Apple reports record quarterly revenue", "AAPL"),
        ("Tesla stock plunges on delivery miss", "TSLA"),
        ("Fed signals potential rate cuts", "SPY"),
    ]

    for news, symbol in test_cases:
        prediction = pipeline.predict(news, symbol)
        print(f"\nNews: {news}")
        print(f"Symbol: {symbol}")
        print(f"Direction: {prediction.direction:+.2f}")
        print(f"Magnitude: {prediction.magnitude_pct:.2f}%")
        print(f"Confidence: {prediction.confidence:.2%}")
        print(f"Interpretation: {prediction.interpretation}")
