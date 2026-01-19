"""
Reformer LSH Attention Implementation

Provides:
- ReformerConfig: Model configuration
- ReformerModel: Main transformer model with LSH attention
- LSHAttention: Locality-sensitive hashing attention mechanism
- ReversibleBlock: Memory-efficient reversible layers
"""

from .model import (
    ReformerConfig,
    ReformerModel,
    LSHAttention,
    ReversibleBlock,
    ChunkedFeedForward,
)
from .data import (
    load_bybit_data,
    prepare_long_sequence_data,
    BybitDataLoader,
)
from .strategy import (
    ReformerStrategy,
    backtest_strategy,
    calculate_metrics,
)

__all__ = [
    'ReformerConfig',
    'ReformerModel',
    'LSHAttention',
    'ReversibleBlock',
    'ChunkedFeedForward',
    'load_bybit_data',
    'prepare_long_sequence_data',
    'BybitDataLoader',
    'ReformerStrategy',
    'backtest_strategy',
    'calculate_metrics',
]

__version__ = '0.1.0'
