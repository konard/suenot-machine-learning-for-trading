#!/usr/bin/env python3
"""
Task-Agnostic Representation Learning for Financial Trading

This module implements task-agnostic representation learning for financial markets,
enabling a single model to solve multiple trading tasks simultaneously through
shared representations.

Supported Tasks:
- Direction Prediction (up/down/sideways)
- Volatility Estimation
- Market Regime Classification
- Return Prediction

Usage:
    from task_agnostic_trading import MultiTaskModel, BybitClient

    # Create model
    model = MultiTaskModel(input_dim=20, embedding_dim=64)

    # Train on multiple tasks
    model.fit(features, task_labels)

    # Make predictions
    predictions = model.predict(query_features)

Requirements:
    pip install torch numpy pandas requests aiohttp
"""

import math
import json
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np

# Optional: PyTorch for neural network implementation
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using numpy-only implementation.")

# Optional: aiohttp for async HTTP
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    import urllib.request
    AIOHTTP_AVAILABLE = False


# ============================================================================
# Enums and Data Classes
# ============================================================================

class Direction(Enum):
    """Market direction."""
    UP = 0
    DOWN = 1
    SIDEWAYS = 2

class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING = 0
    RANGING = 1
    VOLATILE = 2
    CRASH = 3
    RECOVERY = 4

class VolatilityLevel(Enum):
    """Volatility level."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    EXTREME = 3

class TaskType(Enum):
    """Task types for multi-task learning."""
    DIRECTION = "direction"
    VOLATILITY = "volatility"
    REGIME = "regime"
    RETURNS = "returns"

class EncoderType(Enum):
    """Encoder architecture types."""
    TRANSFORMER = "transformer"
    CNN = "cnn"
    MOE = "moe"

class FusionMethod(Enum):
    """Decision fusion methods."""
    VOTING = "voting"
    WEIGHTED_CONFIDENCE = "weighted_confidence"
    RULE_BASED = "rule_based"


@dataclass
class DirectionPrediction:
    """Direction prediction result."""
    direction: Direction
    confidence: float
    probabilities: List[float]

@dataclass
class VolatilityPrediction:
    """Volatility prediction result."""
    volatility_pct: float
    level: VolatilityLevel
    confidence: float
    lower_bound: float
    upper_bound: float

@dataclass
class RegimePrediction:
    """Regime prediction result."""
    regime: MarketRegime
    confidence: float
    probabilities: List[float]
    risk_level: int
    recommendation: str

@dataclass
class ReturnsPrediction:
    """Returns prediction result."""
    return_pct: float
    confidence: float
    lower_bound: float
    upper_bound: float
    risk_adjusted: float

@dataclass
class MultiTaskPrediction:
    """Combined predictions from all tasks."""
    direction: Optional[DirectionPrediction] = None
    volatility: Optional[VolatilityPrediction] = None
    regime: Optional[RegimePrediction] = None
    returns: Optional[ReturnsPrediction] = None

    def average_confidence(self) -> float:
        """Get average confidence across all predictions."""
        confidences = []
        if self.direction:
            confidences.append(self.direction.confidence)
        if self.volatility:
            confidences.append(self.volatility.confidence)
        if self.regime:
            confidences.append(self.regime.confidence)
        if self.returns:
            confidences.append(self.returns.confidence)
        return sum(confidences) / len(confidences) if confidences else 0.0

@dataclass
class TradingDecision:
    """Trading decision result."""
    action: str  # "LONG", "SHORT", "FLAT", "HOLD"
    position_size: float
    confidence: float
    task_agreement: float
    reasoning: List[str]


# ============================================================================
# Market Data Types
# ============================================================================

@dataclass
class Kline:
    """OHLCV candlestick data."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float

    def return_pct(self) -> float:
        """Calculate candle return percentage."""
        return (self.close - self.open) / self.open if self.open > 0 else 0.0

    def range(self) -> float:
        """Calculate high-low range."""
        return self.high - self.low

    def body_size(self) -> float:
        """Calculate candle body size."""
        return abs(self.close - self.open)

    def is_bullish(self) -> bool:
        """Check if bullish candle."""
        return self.close > self.open

@dataclass
class OrderBook:
    """Order book snapshot."""
    timestamp: int
    bids: List[Tuple[float, float]]  # (price, quantity)
    asks: List[Tuple[float, float]]

    def imbalance(self, depth: int = 10) -> float:
        """Calculate order book imbalance."""
        bid_vol = sum(q for _, q in self.bids[:depth])
        ask_vol = sum(q for _, q in self.asks[:depth])
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0

    def spread_pct(self) -> Optional[float]:
        """Calculate spread percentage."""
        if self.bids and self.asks:
            bid = self.bids[0][0]
            ask = self.asks[0][0]
            return (ask - bid) / bid if bid > 0 else None
        return None


# ============================================================================
# PyTorch Neural Network Components (if available)
# ============================================================================

if TORCH_AVAILABLE:
    class TransformerEncoder(nn.Module):
        """Transformer encoder for market data."""

        def __init__(self, input_dim: int, embedding_dim: int, num_heads: int = 4,
                     num_layers: int = 2, ff_dim: int = 128, dropout: float = 0.1):
            super().__init__()
            self.input_projection = nn.Linear(input_dim, embedding_dim)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_projection = nn.Linear(embedding_dim, embedding_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, features)
            x = self.input_projection(x)
            x = x.unsqueeze(1)  # (batch, 1, embedding)
            x = self.transformer(x)
            x = x.squeeze(1)  # (batch, embedding)
            return self.output_projection(x)


    class CNNEncoder(nn.Module):
        """CNN encoder for local pattern extraction."""

        def __init__(self, input_dim: int, embedding_dim: int,
                     kernel_sizes: List[int] = [3, 5, 7], num_filters: int = 32):
            super().__init__()
            self.convs = nn.ModuleList([
                nn.Conv1d(1, num_filters, k, padding=k//2)
                for k in kernel_sizes
            ])
            conv_out_dim = num_filters * len(kernel_sizes)
            self.fc = nn.Sequential(
                nn.Linear(conv_out_dim, embedding_dim * 2),
                nn.ReLU(),
                nn.Linear(embedding_dim * 2, embedding_dim)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, features)
            x = x.unsqueeze(1)  # (batch, 1, features)
            conv_outputs = [F.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
            x = torch.cat(conv_outputs, dim=1)
            return self.fc(x)


    class MoEEncoder(nn.Module):
        """Mixture of Experts encoder."""

        def __init__(self, input_dim: int, embedding_dim: int,
                     num_experts: int = 4, top_k: int = 2):
            super().__init__()
            self.num_experts = num_experts
            self.top_k = top_k

            self.input_proj = nn.Linear(input_dim, embedding_dim)
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim * 2),
                    nn.ReLU(),
                    nn.Linear(embedding_dim * 2, embedding_dim)
                )
                for _ in range(num_experts)
            ])
            self.gate = nn.Linear(embedding_dim, num_experts)
            self.output_proj = nn.Linear(embedding_dim, embedding_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.input_proj(x)

            # Gating
            gate_logits = self.gate(x)
            gate_weights = F.softmax(gate_logits, dim=-1)

            # Top-k selection
            top_weights, top_indices = torch.topk(gate_weights, self.top_k, dim=-1)
            top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

            # Compute expert outputs
            output = torch.zeros_like(x)
            for i in range(self.top_k):
                for j in range(self.num_experts):
                    mask = (top_indices[:, i] == j).float().unsqueeze(-1)
                    expert_out = self.experts[j](x)
                    output = output + mask * top_weights[:, i:i+1] * expert_out

            return self.output_proj(output)


    class DirectionHead(nn.Module):
        """Direction prediction head."""

        def __init__(self, embedding_dim: int, hidden_dim: int = 32, num_classes: int = 3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

        def predict(self, x: torch.Tensor) -> DirectionPrediction:
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            if len(probs.shape) > 1:
                probs = probs[0]
            probs_list = probs.detach().cpu().numpy().tolist()
            max_idx = int(probs.argmax())
            return DirectionPrediction(
                direction=Direction(max_idx),
                confidence=float(probs[max_idx]),
                probabilities=probs_list
            )


    class VolatilityHead(nn.Module):
        """Volatility estimation head."""

        def __init__(self, embedding_dim: int, hidden_dim: int = 32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)  # mean, log_var
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

        def predict(self, x: torch.Tensor) -> VolatilityPrediction:
            out = self.forward(x)
            if len(out.shape) > 1:
                out = out[0]
            mean = float(out[0])
            log_var = float(out[1])
            std = math.sqrt(math.exp(log_var))
            vol_pct = abs(mean) * 100

            level = VolatilityLevel.LOW
            if vol_pct >= 5:
                level = VolatilityLevel.EXTREME
            elif vol_pct >= 3:
                level = VolatilityLevel.HIGH
            elif vol_pct >= 1:
                level = VolatilityLevel.MEDIUM

            return VolatilityPrediction(
                volatility_pct=vol_pct,
                level=level,
                confidence=1.0 / (1.0 + std),
                lower_bound=abs(mean - 2*std) * 100,
                upper_bound=abs(mean + 2*std) * 100
            )


    class RegimeHead(nn.Module):
        """Regime classification head."""

        def __init__(self, embedding_dim: int, hidden_dim: int = 32, num_regimes: int = 5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_regimes)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

        def predict(self, x: torch.Tensor) -> RegimePrediction:
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            if len(probs.shape) > 1:
                probs = probs[0]
            probs_list = probs.detach().cpu().numpy().tolist()
            max_idx = int(probs.argmax())
            regime = MarketRegime(max_idx)

            risk_levels = {
                MarketRegime.TRENDING: 2,
                MarketRegime.RANGING: 1,
                MarketRegime.VOLATILE: 4,
                MarketRegime.CRASH: 5,
                MarketRegime.RECOVERY: 3
            }

            recommendations = {
                MarketRegime.TRENDING: "Follow the trend with tight stops",
                MarketRegime.RANGING: "Trade range bounds",
                MarketRegime.VOLATILE: "Reduce position size",
                MarketRegime.CRASH: "Stay out or hedge",
                MarketRegime.RECOVERY: "Scale in carefully"
            }

            return RegimePrediction(
                regime=regime,
                confidence=float(probs[max_idx]),
                probabilities=probs_list,
                risk_level=risk_levels[regime],
                recommendation=recommendations[regime]
            )


    class ReturnsHead(nn.Module):
        """Returns prediction head."""

        def __init__(self, embedding_dim: int, hidden_dim: int = 32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)  # mean, log_var
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

        def predict(self, x: torch.Tensor) -> ReturnsPrediction:
            out = self.forward(x)
            if len(out.shape) > 1:
                out = out[0]
            mean = float(out[0])
            log_var = float(out[1])
            std = math.sqrt(math.exp(log_var))

            return ReturnsPrediction(
                return_pct=mean * 100,
                confidence=1.0 / (1.0 + std),
                lower_bound=(mean - 1.96*std) * 100,
                upper_bound=(mean + 1.96*std) * 100,
                risk_adjusted=mean / std if std > 1e-6 else 0.0
            )


    class MultiTaskModel(nn.Module):
        """Multi-task model with shared encoder and task-specific heads."""

        def __init__(self, input_dim: int, embedding_dim: int = 64,
                     encoder_type: EncoderType = EncoderType.TRANSFORMER,
                     tasks: Optional[List[TaskType]] = None):
            super().__init__()
            self.input_dim = input_dim
            self.embedding_dim = embedding_dim

            # Create encoder
            if encoder_type == EncoderType.TRANSFORMER:
                self.encoder = TransformerEncoder(input_dim, embedding_dim)
            elif encoder_type == EncoderType.CNN:
                self.encoder = CNNEncoder(input_dim, embedding_dim)
            else:
                self.encoder = MoEEncoder(input_dim, embedding_dim)

            # Create task heads
            self.tasks = tasks or list(TaskType)
            self.direction_head = DirectionHead(embedding_dim) if TaskType.DIRECTION in self.tasks else None
            self.volatility_head = VolatilityHead(embedding_dim) if TaskType.VOLATILITY in self.tasks else None
            self.regime_head = RegimeHead(embedding_dim) if TaskType.REGIME in self.tasks else None
            self.returns_head = ReturnsHead(embedding_dim) if TaskType.RETURNS in self.tasks else None

        def forward(self, x: torch.Tensor) -> Dict[TaskType, torch.Tensor]:
            """Forward pass through encoder and all task heads."""
            embedding = self.encoder(x)
            outputs = {}

            if self.direction_head:
                outputs[TaskType.DIRECTION] = self.direction_head(embedding)
            if self.volatility_head:
                outputs[TaskType.VOLATILITY] = self.volatility_head(embedding)
            if self.regime_head:
                outputs[TaskType.REGIME] = self.regime_head(embedding)
            if self.returns_head:
                outputs[TaskType.RETURNS] = self.returns_head(embedding)

            return outputs

        def predict(self, x: np.ndarray) -> MultiTaskPrediction:
            """Make predictions for all tasks."""
            self.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(x, dtype=torch.float32)
                if len(x_tensor.shape) == 1:
                    x_tensor = x_tensor.unsqueeze(0)

                embedding = self.encoder(x_tensor)

                prediction = MultiTaskPrediction()
                if self.direction_head:
                    prediction.direction = self.direction_head.predict(embedding)
                if self.volatility_head:
                    prediction.volatility = self.volatility_head.predict(embedding)
                if self.regime_head:
                    prediction.regime = self.regime_head.predict(embedding)
                if self.returns_head:
                    prediction.returns = self.returns_head.predict(embedding)

                return prediction


# ============================================================================
# NumPy-only Implementation (fallback when PyTorch not available)
# ============================================================================

class NumpyMultiTaskModel:
    """NumPy-only multi-task model for when PyTorch is not available."""

    def __init__(self, input_dim: int, embedding_dim: int = 64):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Initialize weights with Xavier initialization
        scale = np.sqrt(2.0 / (input_dim + embedding_dim))
        self.encoder_w1 = np.random.randn(input_dim, embedding_dim) * scale
        self.encoder_w2 = np.random.randn(embedding_dim, embedding_dim) * scale

        # Task heads
        scale_head = np.sqrt(2.0 / (embedding_dim + 32))
        self.direction_w1 = np.random.randn(embedding_dim, 32) * scale_head
        self.direction_w2 = np.random.randn(32, 3) * np.sqrt(2.0 / 35)

        self.volatility_w1 = np.random.randn(embedding_dim, 32) * scale_head
        self.volatility_w2 = np.random.randn(32, 2) * np.sqrt(2.0 / 34)

        self.regime_w1 = np.random.randn(embedding_dim, 32) * scale_head
        self.regime_w2 = np.random.randn(32, 5) * np.sqrt(2.0 / 37)

        self.returns_w1 = np.random.randn(embedding_dim, 32) * scale_head
        self.returns_w2 = np.random.randn(32, 2) * np.sqrt(2.0 / 34)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to embedding."""
        h = self._relu(x @ self.encoder_w1)
        return h @ self.encoder_w2

    def predict(self, x: np.ndarray) -> MultiTaskPrediction:
        """Make predictions for all tasks."""
        embedding = self.encode(x)

        # Direction
        dir_h = self._relu(embedding @ self.direction_w1)
        dir_logits = dir_h @ self.direction_w2
        dir_probs = self._softmax(dir_logits)
        max_idx = int(np.argmax(dir_probs))
        direction = DirectionPrediction(
            direction=Direction(max_idx),
            confidence=float(dir_probs[max_idx]),
            probabilities=dir_probs.tolist()
        )

        # Volatility
        vol_h = self._relu(embedding @ self.volatility_w1)
        vol_out = vol_h @ self.volatility_w2
        vol_mean = float(vol_out[0])
        vol_log_var = float(vol_out[1])
        vol_std = np.sqrt(np.exp(vol_log_var))
        vol_pct = abs(vol_mean) * 100

        level = VolatilityLevel.LOW
        if vol_pct >= 5:
            level = VolatilityLevel.EXTREME
        elif vol_pct >= 3:
            level = VolatilityLevel.HIGH
        elif vol_pct >= 1:
            level = VolatilityLevel.MEDIUM

        volatility = VolatilityPrediction(
            volatility_pct=vol_pct,
            level=level,
            confidence=1.0 / (1.0 + vol_std),
            lower_bound=abs(vol_mean - 2*vol_std) * 100,
            upper_bound=abs(vol_mean + 2*vol_std) * 100
        )

        # Regime
        reg_h = self._relu(embedding @ self.regime_w1)
        reg_logits = reg_h @ self.regime_w2
        reg_probs = self._softmax(reg_logits)
        reg_idx = int(np.argmax(reg_probs))
        regime_type = MarketRegime(reg_idx)

        risk_levels = {
            MarketRegime.TRENDING: 2, MarketRegime.RANGING: 1,
            MarketRegime.VOLATILE: 4, MarketRegime.CRASH: 5, MarketRegime.RECOVERY: 3
        }
        recommendations = {
            MarketRegime.TRENDING: "Follow the trend",
            MarketRegime.RANGING: "Trade range bounds",
            MarketRegime.VOLATILE: "Reduce position size",
            MarketRegime.CRASH: "Stay out or hedge",
            MarketRegime.RECOVERY: "Scale in carefully"
        }

        regime = RegimePrediction(
            regime=regime_type,
            confidence=float(reg_probs[reg_idx]),
            probabilities=reg_probs.tolist(),
            risk_level=risk_levels[regime_type],
            recommendation=recommendations[regime_type]
        )

        # Returns
        ret_h = self._relu(embedding @ self.returns_w1)
        ret_out = ret_h @ self.returns_w2
        ret_mean = float(ret_out[0])
        ret_log_var = float(ret_out[1])
        ret_std = np.sqrt(np.exp(ret_log_var))

        returns = ReturnsPrediction(
            return_pct=ret_mean * 100,
            confidence=1.0 / (1.0 + ret_std),
            lower_bound=(ret_mean - 1.96*ret_std) * 100,
            upper_bound=(ret_mean + 1.96*ret_std) * 100,
            risk_adjusted=ret_mean / ret_std if ret_std > 1e-6 else 0.0
        )

        return MultiTaskPrediction(
            direction=direction,
            volatility=volatility,
            regime=regime,
            returns=returns
        )


# ============================================================================
# Feature Extraction
# ============================================================================

class FeatureExtractor:
    """Extract features from market data."""

    def __init__(self, ma_windows: List[int] = [5, 10, 20],
                 rsi_period: int = 14, normalize: bool = True):
        self.ma_windows = ma_windows
        self.rsi_period = rsi_period
        self.normalize = normalize

    def extract_from_klines(self, klines: List[Kline]) -> np.ndarray:
        """Extract features from kline data."""
        if not klines:
            return np.array([])

        closes = np.array([k.close for k in klines])
        highs = np.array([k.high for k in klines])
        lows = np.array([k.low for k in klines])
        volumes = np.array([k.volume for k in klines])

        features = []

        # Return
        last = klines[-1]
        features.append(last.return_pct())

        # Moving average ratios
        for window in self.ma_windows:
            if len(closes) >= window:
                ma = np.mean(closes[-window:])
                features.append(last.close / ma - 1)

        # RSI
        if len(closes) >= self.rsi_period + 1:
            features.append((self._compute_rsi(closes) - 50) / 50)

        # ATR ratio
        if len(klines) >= 14:
            atr = self._compute_atr(klines)
            features.append(atr / last.close)

        # Volume ratio
        if len(volumes) >= 20:
            vol_ma = np.mean(volumes[-20:])
            features.append(last.volume / vol_ma - 1)

        # Volatility
        if len(closes) >= 20:
            returns = np.diff(closes) / closes[:-1]
            features.append(np.std(returns[-20:]))

        # Momentum
        for period in [5, 10, 20]:
            if len(closes) > period:
                roc = (last.close - closes[-1-period]) / closes[-1-period]
                features.append(roc)

        # Range ratio
        if len(klines) > 0:
            avg_range = np.mean([k.range() for k in klines])
            if avg_range > 0:
                features.append(last.range() / avg_range - 1)

        # Body ratio
        if last.range() > 0:
            features.append(last.body_size() / last.range())
        else:
            features.append(0.5)

        # Direction
        features.append(1.0 if last.is_bullish() else -1.0)

        features = np.array(features)

        if self.normalize:
            features = np.tanh(np.clip(features, -5, 5))

        return features

    def _compute_rsi(self, closes: np.ndarray) -> float:
        """Compute RSI indicator."""
        changes = np.diff(closes)[-self.rsi_period:]
        gains = np.sum(changes[changes > 0])
        losses = np.abs(np.sum(changes[changes < 0]))

        if losses == 0:
            return 100.0
        if gains == 0:
            return 0.0

        rs = gains / losses
        return 100 - 100 / (1 + rs)

    def _compute_atr(self, klines: List[Kline], period: int = 14) -> float:
        """Compute Average True Range."""
        tr_values = []
        for i in range(1, len(klines)):
            prev_close = klines[i-1].close
            high = klines[i].high
            low = klines[i].low
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)

        return np.mean(tr_values[-period:])

    def extract_from_ohlcv(self, ohlcv: np.ndarray) -> np.ndarray:
        """Extract features from OHLCV numpy array.

        Args:
            ohlcv: Array of shape (n, 5) with columns [open, high, low, close, volume]

        Returns:
            Feature vector as numpy array
        """
        import time
        klines = []
        for i, row in enumerate(ohlcv):
            klines.append(Kline(
                timestamp=int(time.time() * 1000) + i * 60000,  # fake timestamps
                open=row[0],
                high=row[1],
                low=row[2],
                close=row[3],
                volume=row[4] if len(row) > 4 else 1000.0,
                turnover=row[3] * row[4] if len(row) > 4 else row[3] * 1000.0
            ))
        return self.extract_from_klines(klines)


# ============================================================================
# Decision Fusion
# ============================================================================

class DecisionFusion:
    """Fuse multi-task predictions into trading decisions."""

    def __init__(self, method: FusionMethod = FusionMethod.WEIGHTED_CONFIDENCE,
                 min_confidence: float = 0.6):
        self.method = method
        self.min_confidence = min_confidence

    def fuse(self, prediction: MultiTaskPrediction) -> TradingDecision:
        """Fuse predictions into a trading decision."""
        if self.method == FusionMethod.VOTING:
            return self._voting_fusion(prediction)
        elif self.method == FusionMethod.RULE_BASED:
            return self._rule_based_fusion(prediction)
        else:
            return self._weighted_fusion(prediction)

    def _weighted_fusion(self, prediction: MultiTaskPrediction) -> TradingDecision:
        """Weighted confidence fusion."""
        bullish_score = 0.0
        bearish_score = 0.0
        reasoning = []

        if prediction.direction:
            d = prediction.direction
            weight = 0.3 * d.confidence
            if d.direction == Direction.UP:
                bullish_score += weight
            elif d.direction == Direction.DOWN:
                bearish_score += weight
            reasoning.append(f"Direction: {d.direction.name} ({d.confidence*100:.1f}%)")

        if prediction.returns:
            r = prediction.returns
            weight = 0.3 * r.confidence
            if r.return_pct > 0:
                bullish_score += weight * min(r.return_pct / 5, 1)
            else:
                bearish_score += weight * min(abs(r.return_pct) / 5, 1)
            reasoning.append(f"Expected return: {r.return_pct:.2f}%")

        if prediction.volatility:
            v = prediction.volatility
            vol_factor = 1.0 / (1 + v.volatility_pct / 5)
            bullish_score *= vol_factor
            bearish_score *= vol_factor
            reasoning.append(f"Volatility: {v.volatility_pct:.2f}% ({v.level.name})")

        if prediction.regime:
            reg = prediction.regime
            risk_factor = 1.0 / (reg.risk_level / 2)
            bullish_score *= risk_factor
            bearish_score *= risk_factor
            if reg.regime == MarketRegime.CRASH:
                bearish_score *= 2
                reasoning.append("CRASH regime - bearish bias")
            reasoning.append(f"Regime: {reg.regime.name} (risk {reg.risk_level})")

        net_score = bullish_score - bearish_score
        confidence = prediction.average_confidence()

        if net_score > 0.2:
            action = "LONG"
        elif net_score < -0.2:
            action = "SHORT"
        else:
            action = "FLAT"

        position_size = abs(net_score) if confidence >= self.min_confidence else 0.0

        return TradingDecision(
            action=action if position_size > 0 else "HOLD",
            position_size=min(position_size, 1.0),
            confidence=confidence,
            task_agreement=1.0 - abs(bullish_score - bearish_score) / (bullish_score + bearish_score + 0.01),
            reasoning=reasoning
        )

    def _voting_fusion(self, prediction: MultiTaskPrediction) -> TradingDecision:
        """Simple voting fusion."""
        long_votes = 0
        short_votes = 0
        reasoning = []

        if prediction.direction:
            if prediction.direction.direction == Direction.UP:
                long_votes += 1
            elif prediction.direction.direction == Direction.DOWN:
                short_votes += 1
            reasoning.append(f"Direction: {prediction.direction.direction.name}")

        if prediction.returns:
            if prediction.returns.return_pct > 0.5:
                long_votes += 1
            elif prediction.returns.return_pct < -0.5:
                short_votes += 1
            reasoning.append(f"Returns: {prediction.returns.return_pct:.2f}%")

        total = long_votes + short_votes
        if long_votes > short_votes:
            action = "LONG"
            agreement = long_votes / max(total, 1)
        elif short_votes > long_votes:
            action = "SHORT"
            agreement = short_votes / max(total, 1)
        else:
            action = "FLAT"
            agreement = 0.5

        confidence = prediction.average_confidence()

        return TradingDecision(
            action=action if confidence >= self.min_confidence else "HOLD",
            position_size=agreement * confidence if confidence >= self.min_confidence else 0.0,
            confidence=confidence,
            task_agreement=agreement,
            reasoning=reasoning
        )

    def _rule_based_fusion(self, prediction: MultiTaskPrediction) -> TradingDecision:
        """Rule-based fusion."""
        reasoning = []

        # Rule 1: Never trade in crash regime
        if prediction.regime and prediction.regime.regime == MarketRegime.CRASH:
            reasoning.append("RULE: Crash regime - stay flat")
            return TradingDecision(
                action="FLAT",
                position_size=0.0,
                confidence=prediction.average_confidence(),
                task_agreement=0.0,
                reasoning=reasoning
            )

        # Rule 2: Direction and returns must agree
        dir_bullish = (prediction.direction and
                       prediction.direction.direction == Direction.UP and
                       prediction.direction.confidence > 0.6)
        dir_bearish = (prediction.direction and
                       prediction.direction.direction == Direction.DOWN and
                       prediction.direction.confidence > 0.6)
        ret_bullish = (prediction.returns and
                       prediction.returns.return_pct > 0.5 and
                       prediction.returns.confidence > 0.5)
        ret_bearish = (prediction.returns and
                       prediction.returns.return_pct < -0.5 and
                       prediction.returns.confidence > 0.5)

        if dir_bullish and ret_bullish:
            action = "LONG"
            reasoning.append("RULE: Direction and returns both bullish")
        elif dir_bearish and ret_bearish:
            action = "SHORT"
            reasoning.append("RULE: Direction and returns both bearish")
        else:
            action = "FLAT"
            reasoning.append("RULE: No agreement - stay flat")

        # Volatility adjustment
        vol_adj = 1.0
        if prediction.volatility and prediction.volatility.volatility_pct > 3:
            vol_adj = 0.5
            reasoning.append("RULE: High volatility - reduced position")

        position_size = 0.8 * vol_adj if action != "FLAT" else 0.0

        return TradingDecision(
            action=action,
            position_size=position_size,
            confidence=prediction.average_confidence(),
            task_agreement=1.0 if action != "FLAT" else 0.0,
            reasoning=reasoning
        )


# ============================================================================
# Bybit API Client
# ============================================================================

class BybitClient:
    """Bybit API client for fetching market data."""

    def __init__(self, testnet: bool = False):
        if testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"
        self.testnet = testnet

    def get_klines(self, symbol: str, interval: str = "5",
                   limit: int = 200) -> List[Kline]:
        """Fetch klines synchronously."""
        url = f"{self.base_url}/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit={limit}"

        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode())
        except Exception as e:
            print(f"Error fetching klines: {e}")
            return []

        if data.get("retCode") != 0:
            print(f"API error: {data.get('retMsg')}")
            return []

        klines = []
        for item in data.get("result", {}).get("list", []):
            if len(item) >= 7:
                klines.append(Kline(
                    timestamp=int(item[0]),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    turnover=float(item[6])
                ))

        klines.reverse()  # Bybit returns newest first
        return klines

    async def get_klines_async(self, symbol: str, interval: str = "5",
                               limit: int = 200) -> List[Kline]:
        """Fetch klines asynchronously."""
        if not AIOHTTP_AVAILABLE:
            return self.get_klines(symbol, interval, limit)

        url = f"{self.base_url}/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit={limit}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                data = await response.json()

        if data.get("retCode") != 0:
            print(f"API error: {data.get('retMsg')}")
            return []

        klines = []
        for item in data.get("result", {}).get("list", []):
            if len(item) >= 7:
                klines.append(Kline(
                    timestamp=int(item[0]),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                    turnover=float(item[6])
                ))

        klines.reverse()
        return klines


# ============================================================================
# Example Usage
# ============================================================================

def main():
    """Example usage of the task-agnostic trading system."""
    print("=== Task-Agnostic Trading: Python Example ===\n")

    # Initialize components
    extractor = FeatureExtractor()
    client = BybitClient()

    # Create model
    if TORCH_AVAILABLE:
        print("Using PyTorch model")
        model = MultiTaskModel(input_dim=15, embedding_dim=64)
    else:
        print("Using NumPy model (PyTorch not available)")
        model = NumpyMultiTaskModel(input_dim=15, embedding_dim=64)

    fusion = DecisionFusion(method=FusionMethod.WEIGHTED_CONFIDENCE)

    # Fetch data
    print("\nFetching BTCUSDT data from Bybit...")
    klines = client.get_klines("BTCUSDT", "5", 100)

    if klines:
        print(f"Fetched {len(klines)} klines")
        print(f"Latest price: ${klines[-1].close:,.2f}")

        # Extract features
        features = extractor.extract_from_klines(klines)
        print(f"Extracted {len(features)} features")

        # Make prediction
        prediction = model.predict(features)

        print("\n--- Multi-Task Predictions ---")
        if prediction.direction:
            print(f"Direction: {prediction.direction.direction.name} "
                  f"({prediction.direction.confidence*100:.1f}%)")

        if prediction.volatility:
            print(f"Volatility: {prediction.volatility.volatility_pct:.2f}% "
                  f"- {prediction.volatility.level.name}")

        if prediction.regime:
            print(f"Regime: {prediction.regime.regime.name} "
                  f"(risk: {prediction.regime.risk_level})")
            print(f"  → {prediction.regime.recommendation}")

        if prediction.returns:
            print(f"Expected return: {prediction.returns.return_pct:.2f}% "
                  f"[{prediction.returns.lower_bound:.2f}%, {prediction.returns.upper_bound:.2f}%]")

        # Fuse into decision
        decision = fusion.fuse(prediction)

        print(f"\n--- Trading Decision ---")
        print(f"Action: {decision.action}")
        print(f"Position size: {decision.position_size*100:.1f}%")
        print(f"Confidence: {decision.confidence*100:.1f}%")
        print("Reasoning:")
        for reason in decision.reasoning:
            print(f"  - {reason}")
    else:
        print("Could not fetch market data")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
