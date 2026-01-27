"""
Task-Agnostic Representation Learning for Trading

This module implements a task-agnostic multi-task learning system for financial
markets. A single universal encoder produces shared representations that serve
multiple downstream trading tasks simultaneously:

1. Trend Prediction (up / sideways / down)
2. Volatility Forecasting (continuous)
3. Regime Detection (trending / mean-reverting / volatile / calm)
4. Risk Assessment (low / medium / high)

Key Ideas:
- Universal Encoder: shared backbone maps raw features to a representation space
- Task-Specific Heads: lightweight layers for each trading objective
- Gradient Harmonization: balances gradients so no task dominates training
- Decision Fusion: combines multi-task outputs into unified trading signals

Supports both stock market (Yahoo Finance) and cryptocurrency (Bybit) data.

Reference:
    Lan et al., "Task-Agnostic Representation Learning", 2019
    https://arxiv.org/abs/1907.12157
"""

import math
import json
import hashlib
import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional, Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


# ---------------------------------------------------------------------------
# Configuration data classes
# ---------------------------------------------------------------------------

class TaskType(Enum):
    """Types of trading tasks."""
    TREND_PREDICTION = auto()
    VOLATILITY_FORECAST = auto()
    REGIME_DETECTION = auto()
    RISK_ASSESSMENT = auto()


@dataclass
class EncoderConfig:
    """Configuration for the universal encoder."""
    input_dim: int = 20
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    repr_dim: int = 16
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    use_residual: bool = True


@dataclass
class TaskHeadConfig:
    """Configuration for a single task head."""
    task_type: TaskType = TaskType.TREND_PREDICTION
    input_dim: int = 16
    hidden_dim: int = 32
    output_dim: int = 3
    task_weight: float = 1.0


@dataclass
class FusionConfig:
    """Configuration for decision fusion."""
    method: str = "weighted_average"  # "weighted_average", "attention", "majority_vote"
    task_weights: List[float] = field(default_factory=lambda: [0.35, 0.20, 0.25, 0.20])
    confidence_threshold: float = 0.6
    risk_tolerance: float = 0.5


@dataclass
class TrainerConfig:
    """Configuration for multi-task training."""
    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32
    weight_decay: float = 1e-4
    val_split: float = 0.2
    patience: int = 10
    harmonization: str = "grad_norm"  # "none", "grad_norm", "uncertainty"
    grad_norm_alpha: float = 1.5


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100_000.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    risk_free_rate: float = 0.05
    periods_per_year: float = 365.0


@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    buy_threshold: float = 0.15
    strong_buy_threshold: float = 0.40
    sell_threshold: float = -0.15
    strong_sell_threshold: float = -0.40
    max_position_size: float = 1.0
    min_confidence: float = 0.5


@dataclass
class BybitConfig:
    """Configuration for Bybit API client."""
    base_url: str = "https://api.bybit.com"
    timeout_secs: int = 30
    testnet: bool = False


# ---------------------------------------------------------------------------
# PyTorch Models
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class UniversalEncoder(nn.Module):
        """Shared encoder producing task-agnostic representations.

        Maps raw market features -> shared representation space using:
        - Stacked dense layers with batch normalization
        - ReLU activations
        - Optional residual connections
        - L2 normalization of output representations
        """

        def __init__(self, config: EncoderConfig):
            super().__init__()
            self.config = config

            layers = []
            prev_dim = config.input_dim
            for hidden_dim in config.hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if config.use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config.dropout_rate))
                prev_dim = hidden_dim

            self.backbone = nn.Sequential(*layers)
            self.projection = nn.Linear(prev_dim, config.repr_dim)

            # Residual projection if dimensions change
            if config.use_residual and config.input_dim != config.hidden_dims[0]:
                self.residual_proj = nn.Linear(config.input_dim, config.hidden_dims[0])
            else:
                self.residual_proj = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.backbone(x)
            repr_out = self.projection(h)
            # L2 normalize
            repr_out = F.normalize(repr_out, p=2, dim=-1)
            return repr_out


    class TaskHead(nn.Module):
        """Task-specific head mapping representations to task outputs.

        Supports classification (softmax) and regression (sigmoid) tasks.
        """

        def __init__(self, config: TaskHeadConfig):
            super().__init__()
            self.config = config
            self.hidden = nn.Linear(config.input_dim, config.hidden_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
            self.output = nn.Linear(config.hidden_dim, config.output_dim)

        def forward(self, representations: torch.Tensor) -> torch.Tensor:
            h = self.dropout(self.relu(self.hidden(representations)))
            out = self.output(h)
            if self.config.task_type in (
                TaskType.TREND_PREDICTION,
                TaskType.REGIME_DETECTION,
                TaskType.RISK_ASSESSMENT,
            ):
                return F.softmax(out, dim=-1)
            else:
                return torch.sigmoid(out)


    class TaskAgnosticModel(nn.Module):
        """Complete task-agnostic multi-task model.

        Combines a universal encoder with multiple task-specific heads
        for simultaneous prediction across trading objectives.
        """

        def __init__(
            self,
            encoder_config: EncoderConfig,
            task_configs: Optional[List[TaskHeadConfig]] = None,
        ):
            super().__init__()
            self.encoder = UniversalEncoder(encoder_config)

            if task_configs is None:
                repr_dim = encoder_config.repr_dim
                task_configs = [
                    TaskHeadConfig(TaskType.TREND_PREDICTION, repr_dim, 32, 3),
                    TaskHeadConfig(TaskType.VOLATILITY_FORECAST, repr_dim, 32, 1),
                    TaskHeadConfig(TaskType.REGIME_DETECTION, repr_dim, 32, 4),
                    TaskHeadConfig(TaskType.RISK_ASSESSMENT, repr_dim, 32, 3),
                ]

            self.heads = nn.ModuleDict()
            self.task_configs = {}
            for tc in task_configs:
                key = tc.task_type.name.lower()
                self.heads[key] = TaskHead(tc)
                self.task_configs[key] = tc

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            """Encode input features into task-agnostic representations."""
            return self.encoder(x)

        def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
            """Forward pass through encoder and all task heads."""
            representations = self.encode(x)
            outputs = {}
            for key, head in self.heads.items():
                outputs[key] = head(representations)
            return outputs

        def predict_single(self, features: np.ndarray) -> Dict[str, np.ndarray]:
            """Predict on numpy array input."""
            self.eval()
            with torch.no_grad():
                x = torch.FloatTensor(features)
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                outputs = self.forward(x)
                return {k: v.numpy() for k, v in outputs.items()}


# ---------------------------------------------------------------------------
# Gradient Harmonization
# ---------------------------------------------------------------------------

class GradientHarmonizer:
    """Balances gradients from multiple tasks during training.

    Methods:
    - GradNorm: adjusts weights so tasks train at similar rates
    - Uncertainty: weights tasks by inverse uncertainty
    """

    def __init__(self, n_tasks: int, method: str = "grad_norm", alpha: float = 1.5):
        self.n_tasks = n_tasks
        self.method = method
        self.alpha = alpha
        self.task_weights = np.ones(n_tasks)
        self.initial_losses: Optional[np.ndarray] = None

    def update(self, task_losses: np.ndarray) -> np.ndarray:
        """Update task weights based on current losses."""
        if self.method == "none":
            return self.task_weights

        if self.method == "grad_norm":
            return self._grad_norm_update(task_losses)
        elif self.method == "uncertainty":
            return self._uncertainty_update(task_losses)
        return self.task_weights

    def weighted_loss(self, task_losses: np.ndarray) -> float:
        """Compute weighted total loss."""
        return float(np.sum(task_losses * self.task_weights))

    def _grad_norm_update(self, task_losses: np.ndarray) -> np.ndarray:
        if self.initial_losses is None:
            self.initial_losses = task_losses.copy()
            return self.task_weights

        ratios = task_losses / (self.initial_losses + 1e-10)
        avg_ratio = np.mean(ratios)
        if avg_ratio < 1e-10:
            return self.task_weights

        relative_rates = ratios / avg_ratio
        target_weights = relative_rates ** (-self.alpha)

        # Smooth update
        self.task_weights += 0.01 * (target_weights - self.task_weights)
        self.task_weights = np.clip(self.task_weights, 0.1, 5.0)

        # Normalize
        self.task_weights *= self.n_tasks / np.sum(self.task_weights)
        return self.task_weights

    def _uncertainty_update(self, task_losses: np.ndarray) -> np.ndarray:
        log_sigmas = np.log(task_losses + 1e-10)
        self.task_weights = 1.0 / (2.0 * np.exp(2.0 * log_sigmas) + 1e-10)
        self.task_weights = np.clip(self.task_weights, 0.1, 5.0)
        self.task_weights *= self.n_tasks / np.sum(self.task_weights)
        return self.task_weights


# ---------------------------------------------------------------------------
# Multi-Task Trainer
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class MultiTaskTrainer:
        """Trains the task-agnostic model on multiple tasks simultaneously."""

        def __init__(self, config: TrainerConfig):
            self.config = config

        def train(
            self,
            model: TaskAgnosticModel,
            features: np.ndarray,
            labels: Dict[str, np.ndarray],
        ) -> Dict[str, Any]:
            """Train the model on multi-task data.

            Args:
                model: TaskAgnosticModel to train
                features: Input features (n_samples, n_features)
                labels: Dict mapping task key -> labels array

            Returns:
                Training results dict
            """
            n_samples = features.shape[0]
            val_size = int(n_samples * self.config.val_split)
            train_size = n_samples - val_size

            # Split data
            x_train = torch.FloatTensor(features[:train_size])
            x_val = torch.FloatTensor(features[train_size:])
            y_train = {k: torch.FloatTensor(v[:train_size]) for k, v in labels.items()}
            y_val = {k: torch.FloatTensor(v[train_size:]) for k, v in labels.items()}

            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

            n_tasks = len(model.heads)
            harmonizer = GradientHarmonizer(
                n_tasks,
                method=self.config.harmonization,
                alpha=self.config.grad_norm_alpha,
            )

            history = []
            best_val_loss = float("inf")
            best_epoch = 0
            patience_counter = 0

            for epoch in range(self.config.epochs):
                # Training
                model.train()
                train_losses = self._train_epoch(model, x_train, y_train, optimizer)

                # Validation
                model.eval()
                with torch.no_grad():
                    val_losses = self._compute_losses(model, x_val, y_val)

                # Update harmonizer
                task_loss_array = np.array(list(train_losses.values()))
                harmonizer.update(task_loss_array)
                total_train = harmonizer.weighted_loss(task_loss_array)
                total_val = sum(val_losses.values())

                history.append({
                    "epoch": epoch,
                    "train_loss": total_train,
                    "val_loss": total_val,
                    "task_losses": dict(train_losses),
                    "task_weights": harmonizer.task_weights.tolist(),
                })

                # Early stopping
                if total_val < best_val_loss:
                    best_val_loss = total_val
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        break

            return {
                "history": history,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "final_weights": harmonizer.task_weights.tolist(),
                "early_stopped": patience_counter >= self.config.patience,
            }

        def _train_epoch(
            self,
            model: TaskAgnosticModel,
            x_train: torch.Tensor,
            y_train: Dict[str, torch.Tensor],
            optimizer: optim.Optimizer,
        ) -> Dict[str, float]:
            n = x_train.shape[0]
            bs = self.config.batch_size
            epoch_losses: Dict[str, float] = {k: 0.0 for k in model.heads}
            n_batches = (n + bs - 1) // bs

            for i in range(0, n, bs):
                batch_x = x_train[i : i + bs]
                outputs = model(batch_x)

                total_loss = torch.tensor(0.0, requires_grad=True)
                for key, pred in outputs.items():
                    if key in y_train:
                        target = y_train[key][i : i + bs]
                        tc = model.task_configs.get(key)
                        if tc and tc.task_type in (
                            TaskType.TREND_PREDICTION,
                            TaskType.REGIME_DETECTION,
                            TaskType.RISK_ASSESSMENT,
                        ):
                            loss = -torch.sum(target * torch.log(pred + 1e-10)) / pred.shape[0]
                        else:
                            loss = F.mse_loss(pred, target)

                        total_loss = total_loss + loss
                        epoch_losses[key] += loss.item() / n_batches

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            return epoch_losses

        def _compute_losses(
            self,
            model: TaskAgnosticModel,
            x: torch.Tensor,
            y: Dict[str, torch.Tensor],
        ) -> Dict[str, float]:
            outputs = model(x)
            losses = {}
            for key, pred in outputs.items():
                if key in y:
                    tc = model.task_configs.get(key)
                    if tc and tc.task_type in (
                        TaskType.TREND_PREDICTION,
                        TaskType.REGIME_DETECTION,
                        TaskType.RISK_ASSESSMENT,
                    ):
                        loss = -torch.sum(y[key] * torch.log(pred + 1e-10)) / pred.shape[0]
                    else:
                        loss = F.mse_loss(pred, y[key])
                    losses[key] = loss.item()
            return losses


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    returns_window: int = 5
    volatility_window: int = 20
    rsi_period: int = 14
    short_ma: int = 10
    long_ma: int = 30
    bb_std: float = 2.0
    normalize: bool = True


class FeatureExtractor:
    """Extract task-agnostic features from OHLCV data.

    Features include:
    - Returns (1-day, N-day)
    - Momentum (RSI, MACD, MA crossover)
    - Volatility (realized, Bollinger position, ATR ratio)
    - Volume (ratio, trend)
    - Candle patterns (body ratio, shadows)
    - Range (percentage, close position)
    - Trend (strength, consistency)
    - Distribution (skewness, kurtosis)
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    def extract(self, klines: List[Dict], idx: int) -> Optional[np.ndarray]:
        """Extract features for a single time step.

        Args:
            klines: List of OHLCV dicts with keys: open, high, low, close, volume
            idx: Index of the time step to extract features for

        Returns:
            Feature vector as numpy array, or None if insufficient data
        """
        min_lookback = max(self.config.long_ma, self.config.volatility_window)
        if idx < min_lookback or idx >= len(klines):
            return None

        features = []

        # Returns
        features.append(self._return_1d(klines, idx))
        features.append(self._return_nd(klines, idx))

        # Momentum
        features.append(self._rsi(klines, idx))
        features.append(self._macd_signal(klines, idx))
        features.append(self._ma_crossover(klines, idx))

        # Volatility
        features.append(self._realized_vol(klines, idx))
        features.append(self._bb_position(klines, idx))
        features.append(self._atr_ratio(klines, idx))

        # Volume
        features.append(self._volume_ratio(klines, idx))
        features.append(self._volume_trend(klines, idx))

        # Candle patterns
        body, upper, lower = self._candle_features(klines, idx)
        features.extend([body, upper, lower])

        # Range
        features.append(self._range_pct(klines, idx))
        features.append(self._close_position(klines, idx))

        # Trend
        features.append(self._trend_strength(klines, idx))
        features.append(self._trend_consistency(klines, idx))

        # Distribution
        features.append(self._skewness(klines, idx))
        features.append(self._kurtosis(klines, idx))

        arr = np.array(features, dtype=np.float64)
        if self.config.normalize:
            std = arr.std()
            if std > 1e-10:
                arr = (arr - arr.mean()) / std
        return arr

    def extract_all(self, klines: List[Dict]) -> np.ndarray:
        """Extract features for all valid time steps."""
        min_lookback = max(self.config.long_ma, self.config.volatility_window)
        results = []
        for idx in range(min_lookback, len(klines)):
            feat = self.extract(klines, idx)
            if feat is not None:
                results.append(feat)
        return np.array(results) if results else np.empty((0, 19))

    @staticmethod
    def feature_dim() -> int:
        return 19

    # --- Private helpers ---

    def _return_1d(self, k, idx):
        prev = k[idx - 1]["close"]
        return (k[idx]["close"] - prev) / prev if prev > 0 else 0.0

    def _return_nd(self, k, idx):
        w = min(self.config.returns_window, idx)
        prev = k[idx - w]["close"]
        return (k[idx]["close"] - prev) / prev if prev > 0 else 0.0

    def _rsi(self, k, idx):
        period = min(self.config.rsi_period, idx)
        gains = losses = 0.0
        for i in range(idx - period + 1, idx + 1):
            change = k[i]["close"] - k[i - 1]["close"]
            if change > 0:
                gains += change
            else:
                losses += abs(change)
        avg_gain = gains / period
        avg_loss = losses / period
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            return (100.0 - 100.0 / (1.0 + rs)) / 100.0
        return 1.0

    def _macd_signal(self, k, idx):
        short = self._ma(k, idx, self.config.short_ma)
        long = self._ma(k, idx, self.config.long_ma)
        return (short - long) / long if long > 0 else 0.0

    def _ma_crossover(self, k, idx):
        s1 = self._ma(k, idx, self.config.short_ma)
        l1 = self._ma(k, idx, self.config.long_ma)
        if idx > 0:
            s0 = self._ma(k, idx - 1, self.config.short_ma)
            l0 = self._ma(k, idx - 1, self.config.long_ma)
            if s1 > l1 and s0 <= l0:
                return 1.0
            if s1 < l1 and s0 >= l0:
                return -1.0
        return 0.0

    def _realized_vol(self, k, idx):
        w = min(self.config.volatility_window, idx)
        rets = []
        for i in range(idx - w + 1, idx + 1):
            prev = k[i - 1]["close"]
            if prev > 0:
                rets.append(math.log(k[i]["close"] / prev))
            else:
                rets.append(0.0)
        return float(np.std(rets)) if rets else 0.0

    def _bb_position(self, k, idx):
        w = min(self.config.volatility_window, idx)
        ma = self._ma(k, idx, w)
        vol = self._realized_vol(k, idx) * k[idx]["close"]
        upper = ma + self.config.bb_std * vol
        lower = ma - self.config.bb_std * vol
        width = upper - lower
        return (k[idx]["close"] - lower) / width if width > 0 else 0.5

    def _atr_ratio(self, k, idx):
        w = min(14, min(self.config.volatility_window, idx))
        atr = 0.0
        for i in range(idx - w + 1, idx + 1):
            tr = max(
                k[i]["high"] - k[i]["low"],
                abs(k[i]["high"] - k[i - 1]["close"]),
                abs(k[i]["low"] - k[i - 1]["close"]),
            )
            atr += tr
        atr /= w
        return atr / k[idx]["close"] if k[idx]["close"] > 0 else 0.0

    def _volume_ratio(self, k, idx):
        w = min(20, idx)
        avg = sum(k[i]["volume"] for i in range(idx - w, idx)) / w
        return k[idx]["volume"] / avg if avg > 0 else 1.0

    def _volume_trend(self, k, idx):
        w = min(20, idx)
        sw = min(5, idx)
        avg_long = sum(k[i]["volume"] for i in range(idx - w, idx)) / w
        avg_short = sum(k[i]["volume"] for i in range(idx - sw, idx)) / max(sw, 1)
        return (avg_short - avg_long) / avg_long if avg_long > 0 else 0.0

    def _candle_features(self, k, idx):
        c = k[idx]
        rng = c["high"] - c["low"]
        if rng <= 0:
            return 0.0, 0.0, 0.0
        body = abs(c["close"] - c["open"]) / rng
        if c["close"] > c["open"]:
            upper = (c["high"] - c["close"]) / rng
            lower = (c["open"] - c["low"]) / rng
        else:
            upper = (c["high"] - c["open"]) / rng
            lower = (c["close"] - c["low"]) / rng
        return body, upper, lower

    def _range_pct(self, k, idx):
        c = k[idx]
        return (c["high"] - c["low"]) / c["low"] if c["low"] > 0 else 0.0

    def _close_position(self, k, idx):
        c = k[idx]
        rng = c["high"] - c["low"]
        return (c["close"] - c["low"]) / rng if rng > 0 else 0.5

    def _trend_strength(self, k, idx):
        w = min(10, idx)
        prices = [k[i]["close"] for i in range(idx - w, idx + 1)]
        n = len(prices)
        x_mean = (n - 1) / 2.0
        y_mean = sum(prices) / n
        num = sum((i - x_mean) * (p - y_mean) for i, p in enumerate(prices))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0.0
        return slope / y_mean if y_mean > 0 else 0.0

    def _trend_consistency(self, k, idx):
        w = min(10, idx)
        ups = sum(
            1 for i in range(idx - w + 1, idx + 1) if k[i]["close"] > k[i - 1]["close"]
        )
        return ups / w

    def _skewness(self, k, idx):
        w = min(20, idx)
        rets = [
            (k[i]["close"] - k[i - 1]["close"]) / k[i - 1]["close"]
            if k[i - 1]["close"] > 0
            else 0.0
            for i in range(idx - w + 1, idx + 1)
        ]
        n = len(rets)
        mean = sum(rets) / n
        std = (sum((r - mean) ** 2 for r in rets) / n) ** 0.5
        if std < 1e-10:
            return 0.0
        return sum(((r - mean) / std) ** 3 for r in rets) / n

    def _kurtosis(self, k, idx):
        w = min(20, idx)
        rets = [
            (k[i]["close"] - k[i - 1]["close"]) / k[i - 1]["close"]
            if k[i - 1]["close"] > 0
            else 0.0
            for i in range(idx - w + 1, idx + 1)
        ]
        n = len(rets)
        mean = sum(rets) / n
        std = (sum((r - mean) ** 2 for r in rets) / n) ** 0.5
        if std < 1e-10:
            return 0.0
        return sum(((r - mean) / std) ** 4 for r in rets) / n - 3.0

    def _ma(self, k, idx, window):
        start = max(0, idx - window + 1)
        prices = [k[i]["close"] for i in range(start, idx + 1)]
        return sum(prices) / len(prices) if prices else 0.0


# ---------------------------------------------------------------------------
# Bybit API Client
# ---------------------------------------------------------------------------

class BybitClient:
    """Async client for Bybit exchange API.

    Fetches kline (candlestick) data for cryptocurrency pairs.
    """

    def __init__(self, config: Optional[BybitConfig] = None):
        self.config = config or BybitConfig()

    async def fetch_klines(
        self, symbol: str, interval: str = "60", limit: int = 200
    ) -> List[Dict]:
        """Fetch OHLCV kline data from Bybit.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candle interval in minutes (e.g., "60" for 1h)
            limit: Number of candles to fetch

        Returns:
            List of dicts with keys: timestamp, open, high, low, close, volume, turnover
        """
        if not HAS_AIOHTTP:
            print("aiohttp not installed. Generating synthetic data.")
            return self._generate_synthetic_klines(symbol, limit)

        url = f"{self.config.base_url}/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": str(limit),
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=self.config.timeout_secs)) as resp:
                    data = await resp.json()

            if data.get("retCode") != 0:
                print(f"Bybit API error: {data.get('retMsg')}")
                return self._generate_synthetic_klines(symbol, limit)

            klines = []
            for item in reversed(data["result"]["list"]):
                klines.append({
                    "timestamp": int(item[0]),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                    "turnover": float(item[6]),
                })
            return klines

        except Exception as e:
            print(f"Bybit fetch failed: {e}. Using synthetic data.")
            return self._generate_synthetic_klines(symbol, limit)

    async def fetch_multi_klines(
        self, symbols: List[str], interval: str = "60", limit: int = 200
    ) -> Dict[str, List[Dict]]:
        """Fetch klines for multiple symbols concurrently."""
        tasks = [self.fetch_klines(s, interval, limit) for s in symbols]
        results = await asyncio.gather(*tasks)
        return dict(zip(symbols, results))

    @staticmethod
    def _generate_synthetic_klines(symbol: str, n: int) -> List[Dict]:
        """Generate synthetic OHLCV data for demonstration."""
        seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16) % 10000
        rng = np.random.RandomState(seed)

        base_price = 100.0 if "BTC" not in symbol else 50000.0
        prices = [base_price]
        for _ in range(n - 1):
            ret = rng.normal(0.0002, 0.02)
            prices.append(prices[-1] * (1 + ret))

        klines = []
        for i, close in enumerate(prices):
            noise = rng.uniform(0.005, 0.02)
            klines.append({
                "timestamp": 1700000000000 + i * 3600000,
                "open": close * (1 - rng.uniform(-noise, noise)),
                "high": close * (1 + noise),
                "low": close * (1 - noise),
                "close": close,
                "volume": rng.uniform(100, 10000),
                "turnover": close * rng.uniform(100, 10000),
            })
        return klines


# ---------------------------------------------------------------------------
# Decision Fusion
# ---------------------------------------------------------------------------

class DecisionFusion:
    """Fuses outputs from multiple task heads into unified trading signals."""

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()

    def fuse(self, task_outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Fuse task outputs into trading decisions.

        Args:
            task_outputs: Dict mapping task key -> prediction array

        Returns:
            Dict with keys: signal, confidence, regime, risk_level
        """
        n = next(iter(task_outputs.values())).shape[0] if task_outputs else 0
        signals = np.zeros(n)
        confidences = np.zeros(n)
        regimes = ["Unknown"] * n
        risk_levels = np.full(n, 0.5)

        weights = self.config.task_weights
        regime_labels = ["Trending", "Mean-Reverting", "Volatile", "Calm"]

        # Trend contribution
        trend_key = "trend_prediction"
        if trend_key in task_outputs:
            preds = task_outputs[trend_key]
            for j in range(n):
                up = preds[j, 0] if preds.shape[1] > 0 else 0.33
                down = preds[j, 2] if preds.shape[1] > 2 else 0.33
                signals[j] += weights[0] * (up - down)
                confidences[j] += weights[0] * max(up, down)

        # Volatility contribution
        vol_key = "volatility_forecast"
        if vol_key in task_outputs:
            preds = task_outputs[vol_key]
            for j in range(n):
                vol = preds[j, 0]
                confidences[j] *= (1.0 - 0.3 * vol)
                risk_levels[j] = vol

        # Regime contribution
        regime_key = "regime_detection"
        if regime_key in task_outputs:
            preds = task_outputs[regime_key]
            for j in range(n):
                best = int(np.argmax(preds[j]))
                regimes[j] = regime_labels[best] if best < len(regime_labels) else "Unknown"
                multiplier = [1.2, 0.8, 0.6, 1.0][best] if best < 4 else 1.0
                signals[j] *= multiplier

        # Risk contribution
        risk_key = "risk_assessment"
        if risk_key in task_outputs:
            preds = task_outputs[risk_key]
            for j in range(n):
                high_risk = preds[j, 2] if preds.shape[1] > 2 else 0.33
                adj = 1.0 - high_risk * (1.0 - self.config.risk_tolerance)
                signals[j] *= adj
                risk_levels[j] = max(risk_levels[j], high_risk)

        signals = np.clip(signals, -1.0, 1.0)
        confidences = np.clip(confidences, 0.0, 1.0)

        # Apply confidence threshold
        for j in range(n):
            if confidences[j] < self.config.confidence_threshold:
                signals[j] = 0.0

        return {
            "signal": signals,
            "confidence": confidences,
            "regime": regimes,
            "risk_level": risk_levels,
        }


# ---------------------------------------------------------------------------
# Signal Generator
# ---------------------------------------------------------------------------

class SignalGenerator:
    """Converts model outputs into actionable trading signals."""

    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()

    def generate(self, fused: Dict[str, Any]) -> List[Dict]:
        """Generate trading signals from fused predictions.

        Returns list of signal dicts with: type, raw_signal, confidence,
        regime, risk_level, position_size.
        """
        signals = fused["signal"]
        confidences = fused["confidence"]
        regimes = fused["regime"]
        risk_levels = fused["risk_level"]

        results = []
        for i in range(len(signals)):
            raw = signals[i]
            conf = confidences[i]
            risk = risk_levels[i]

            if conf < self.config.min_confidence:
                sig_type = "Hold"
            elif raw >= self.config.strong_buy_threshold:
                sig_type = "Strong Buy"
            elif raw >= self.config.buy_threshold:
                sig_type = "Buy"
            elif raw <= self.config.strong_sell_threshold:
                sig_type = "Strong Sell"
            elif raw <= self.config.sell_threshold:
                sig_type = "Sell"
            else:
                sig_type = "Hold"

            base_size = {"Strong Buy": 1.0, "Buy": 0.5, "Hold": 0.0, "Sell": 0.5, "Strong Sell": 1.0}[sig_type]
            risk_adj = 1.0 - risk * 0.5
            pos_size = min(base_size * risk_adj * conf, self.config.max_position_size)

            results.append({
                "type": sig_type,
                "raw_signal": float(raw),
                "confidence": float(conf),
                "regime": regimes[i] if isinstance(regimes, list) else str(regimes[i]),
                "risk_level": float(risk),
                "position_size": float(pos_size),
            })

        return results


# ---------------------------------------------------------------------------
# Backtesting Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Backtests trading signals against historical price data."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(self, signals: List[Dict], prices: np.ndarray) -> Dict[str, Any]:
        """Run backtest.

        Args:
            signals: List of signal dicts from SignalGenerator
            prices: Array of close prices

        Returns:
            Dict with metrics, equity_curve, trades
        """
        n = len(prices)
        equity = self.config.initial_capital
        equity_curve = [equity]
        trades = []
        returns = []

        current_pos = 0.0
        entry_price = 0.0
        entry_idx = 0

        direction_map = {
            "Strong Buy": 1.0, "Buy": 1.0, "Hold": 0.0,
            "Sell": -1.0, "Strong Sell": -1.0,
        }

        for i in range(n):
            sig = signals[i]
            target = direction_map.get(sig["type"], 0.0) * sig["position_size"]

            if abs(target - current_pos) > 0.01:
                # Close existing
                if abs(current_pos) > 0.01 and i > 0:
                    exit_p = prices[i] * (1.0 - self.config.slippage * np.sign(current_pos))
                    ret = current_pos * (exit_p - entry_price) / entry_price
                    cost = self.config.transaction_cost * abs(current_pos)
                    net = ret - cost
                    equity *= (1.0 + net)
                    trades.append({
                        "entry_idx": entry_idx, "exit_idx": i,
                        "entry_price": entry_price, "exit_price": exit_p,
                        "direction": np.sign(current_pos), "pnl": equity * net,
                        "return_pct": net * 100.0,
                    })

                # Open new
                if abs(target) > 0.01:
                    entry_price = prices[i] * (1.0 + self.config.slippage * np.sign(target))
                    entry_idx = i
                    equity *= (1.0 - self.config.transaction_cost * abs(target))

                current_pos = target

            # Track equity
            if abs(current_pos) > 0.01 and i > 0:
                unrealized = current_pos * (prices[i] - entry_price) / entry_price
                equity_curve.append(equity * (1.0 + unrealized))
            else:
                equity_curve.append(equity)

            if len(equity_curve) >= 2:
                prev = equity_curve[-2]
                curr = equity_curve[-1]
                returns.append((curr - prev) / prev if prev > 0 else 0.0)

        # Close final position
        if abs(current_pos) > 0.01:
            ret = current_pos * (prices[-1] - entry_price) / entry_price
            cost = self.config.transaction_cost * abs(current_pos)
            equity *= (1.0 + ret - cost)
            equity_curve[-1] = equity

        # Metrics
        returns_arr = np.array(returns) if returns else np.zeros(1)
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (np.array(peak) - np.array(equity_curve)) / np.array(peak)

        winning = [t for t in trades if t["pnl"] > 0]
        losing = [t for t in trades if t["pnl"] <= 0]

        total_return = (equity - self.config.initial_capital) / self.config.initial_capital * 100
        years = len(returns_arr) / self.config.periods_per_year
        ann_return = ((equity / self.config.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

        daily_rf = self.config.risk_free_rate / self.config.periods_per_year
        std_ret = float(np.std(returns_arr))
        sharpe = (float(np.mean(returns_arr)) - daily_rf) / std_ret * np.sqrt(self.config.periods_per_year) if std_ret > 1e-10 else 0

        down_rets = returns_arr[returns_arr < daily_rf]
        down_std = float(np.sqrt(np.mean((down_rets - daily_rf) ** 2))) if len(down_rets) > 0 else 0
        sortino = (float(np.mean(returns_arr)) - daily_rf) / down_std * np.sqrt(self.config.periods_per_year) if down_std > 1e-10 else 0

        max_dd = float(np.max(drawdowns)) * 100
        win_rate = len(winning) / len(trades) * 100 if trades else 0
        gross_profit = sum(t["pnl"] for t in winning)
        gross_loss = sum(abs(t["pnl"]) for t in losing)
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

        return {
            "metrics": {
                "total_return_pct": total_return,
                "annualized_return": ann_return,
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "max_drawdown_pct": max_dd,
                "win_rate": win_rate,
                "profit_factor": pf,
                "total_trades": len(trades),
                "final_equity": equity,
            },
            "equity_curve": equity_curve,
            "trades": trades,
        }


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    """Demonstrate the task-agnostic trading system."""
    print("=" * 60)
    print("Task-Agnostic Representation Learning for Trading")
    print("=" * 60)

    # 1. Generate synthetic data
    print("\n--- Step 1: Generate Synthetic Market Data ---")
    client = BybitClient()
    loop = asyncio.new_event_loop()
    klines = loop.run_until_complete(client.fetch_klines("BTCUSDT", "60", 250))
    loop.close()
    print(f"Generated {len(klines)} klines for BTCUSDT")

    # 2. Extract features
    print("\n--- Step 2: Extract Task-Agnostic Features ---")
    extractor = FeatureExtractor()
    features = extractor.extract_all(klines)
    print(f"Extracted features: {features.shape}")

    if not HAS_TORCH:
        print("\nPyTorch not installed. Skipping model training demo.")
        print("Install with: pip install torch")
        _demo_without_torch(features, klines)
        return

    # 3. Create model
    print("\n--- Step 3: Create Task-Agnostic Model ---")
    n_features = features.shape[1]
    encoder_cfg = EncoderConfig(input_dim=n_features, hidden_dims=[64, 32], repr_dim=16)
    model = TaskAgnosticModel(encoder_cfg)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    for key in model.heads:
        print(f"  Head: {key}")

    # 4. Prepare labels (synthetic)
    print("\n--- Step 4: Prepare Multi-Task Labels ---")
    n = features.shape[0]
    labels = {
        "trend_prediction": np.eye(3)[np.random.randint(0, 3, n)],
        "volatility_forecast": np.random.uniform(0, 1, (n, 1)).astype(np.float32),
        "regime_detection": np.eye(4)[np.random.randint(0, 4, n)],
        "risk_assessment": np.eye(3)[np.random.randint(0, 3, n)],
    }
    print(f"Labels prepared for {len(labels)} tasks")

    # 5. Train
    print("\n--- Step 5: Train with Gradient Harmonization ---")
    trainer = MultiTaskTrainer(TrainerConfig(epochs=30, batch_size=32, patience=10))
    result = trainer.train(model, features, labels)
    print(f"Training complete: {len(result['history'])} epochs")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Best val loss: {result['best_val_loss']:.4f}")
    print(f"Final task weights: {[f'{w:.3f}' for w in result['final_weights']]}")

    # 6. Predict
    print("\n--- Step 6: Multi-Task Predictions ---")
    outputs = model.predict_single(features)
    for key, val in outputs.items():
        print(f"  {key}: shape={val.shape}")

    # 7. Fuse and generate signals
    print("\n--- Step 7: Decision Fusion & Signal Generation ---")
    fusion = DecisionFusion()
    fused = fusion.fuse(outputs)
    sig_gen = SignalGenerator()
    signals = sig_gen.generate(fused)

    counts = {}
    for s in signals:
        counts[s["type"]] = counts.get(s["type"], 0) + 1
    for sig_type, count in sorted(counts.items()):
        print(f"  {sig_type}: {count} ({count / len(signals) * 100:.0f}%)")

    # 8. Backtest
    print("\n--- Step 8: Backtest ---")
    prices = np.array([k["close"] for k in klines[-len(signals):]])
    engine = BacktestEngine()
    bt_result = engine.run(signals, prices)
    m = bt_result["metrics"]
    print(f"  Total Return:    {m['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio:    {m['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio:   {m['sortino_ratio']:.2f}")
    print(f"  Max Drawdown:    {m['max_drawdown_pct']:.2f}%")
    print(f"  Win Rate:        {m['win_rate']:.1f}%")
    print(f"  Total Trades:    {m['total_trades']}")
    print(f"  Final Equity:    ${m['final_equity']:.2f}")

    bh_return = (prices[-1] - prices[0]) / prices[0] * 100
    print(f"\n  Buy & Hold:      {bh_return:.2f}%")
    print(f"  Strategy vs B&H: {m['total_return_pct'] - bh_return:+.2f}%")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


def _demo_without_torch(features, klines):
    """Run a simplified demo without PyTorch."""
    print("\n--- Running signal generation with heuristic model ---")

    n = features.shape[0]
    # Simple heuristic: use trend and volatility features directly
    signals_raw = features[:, 0] * 10  # Scale returns as signal
    confidences = np.clip(np.abs(signals_raw) * 2, 0, 1)

    fused = {
        "signal": np.clip(signals_raw, -1, 1),
        "confidence": confidences,
        "regime": ["Unknown"] * n,
        "risk_level": np.full(n, 0.3),
    }

    sig_gen = SignalGenerator()
    signals = sig_gen.generate(fused)

    counts = {}
    for s in signals:
        counts[s["type"]] = counts.get(s["type"], 0) + 1
    for sig_type, count in sorted(counts.items()):
        print(f"  {sig_type}: {count}")

    prices = np.array([k["close"] for k in klines[-n:]])
    engine = BacktestEngine()
    result = engine.run(signals, prices)
    m = result["metrics"]
    print(f"\n  Total Return: {m['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio: {m['sharpe_ratio']:.2f}")
    print(f"  Total Trades: {m['total_trades']}")

    print("\nDONE")


if __name__ == "__main__":
    main()
