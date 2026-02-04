"""
Configuration for Variational Inference Trading
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Data fetching and preprocessing configuration"""
    symbols: List[str] = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "MATIC/USDT", "DOT/USDT"
    ])
    exchange: str = "bybit"
    timeframe: str = "1m"
    limit: int = 1000
    sequence_length: int = 60  # 1 hour of 1-min data
    prediction_horizon: int = 5  # 5 minutes ahead
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class ModelConfig:
    """VAE model configuration"""
    input_dim: int = 64
    latent_dim: int = 16
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    beta: float = 4.0  # KL weight for disentanglement
    dropout: float = 0.1
    use_batch_norm: bool = True
    activation: str = "leaky_relu"


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    max_epochs: int = 200
    early_stopping_patience: int = 20

    # Beta annealing
    beta_start: float = 0.0
    beta_end: float = 4.0
    beta_warmup_epochs: int = 50

    # Learning rate scheduler
    lr_scheduler: str = "cosine"
    lr_min: float = 1e-6

    # Gradient clipping
    gradient_clip: float = 1.0

    # Device
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"


@dataclass
class StrategyConfig:
    """Trading strategy configuration"""
    confidence_threshold: float = 0.6
    uncertainty_threshold: float = 0.05
    n_samples: int = 100  # Number of samples for uncertainty estimation

    # Position sizing
    max_position_size: float = 0.1  # 10% of portfolio
    min_position_size: float = 0.01  # 1% of portfolio

    # Risk management
    stop_loss: float = 0.02  # 2%
    take_profit: float = 0.04  # 4%

    # Regime detection
    regime_threshold: float = 0.5


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 10000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%

    # Metrics to track
    calculate_sharpe: bool = True
    calculate_sortino: bool = True
    calculate_max_drawdown: bool = True
    calculate_calmar: bool = True


@dataclass
class Config:
    """Main configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    # Paths
    data_dir: str = "./data"
    model_dir: str = "./models"
    results_dir: str = "./results"

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Random seed for reproducibility
    seed: int = 42


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()
