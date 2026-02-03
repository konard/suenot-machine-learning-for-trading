"""
SSM-Transformer Hybrid for Trading.

This package provides:
- SSMTransformerHybrid: Hybrid model combining SSM (Mamba-style) and Transformer layers
- SSMHybridTrainer: Training loop with multi-task loss weighting
- SSMHybridDataLoader: Data loading from Bybit and Yahoo Finance
- SSMHybridBacktester: Backtesting framework for strategy evaluation
"""

from python.ssm_transformer_model import SSMTransformerHybrid, SSMHybridTrainer
from python.data_loader import SSMHybridDataLoader
from python.backtest import SSMHybridBacktester

__all__ = [
    "SSMTransformerHybrid",
    "SSMHybridTrainer",
    "SSMHybridDataLoader",
    "SSMHybridBacktester",
]
