"""
Cross-Attention Multi-Asset Trading Module

This module provides:
- CrossAttentionMultiAsset: Main transformer model for multi-asset prediction
- BybitDataLoader: Data loading from Bybit exchange
- CrossAttentionBacktest: Backtesting framework

Example:
    from cross_attention import CrossAttentionMultiAsset, BybitDataLoader

    # Load data
    loader = BybitDataLoader()
    data = loader.fetch_multi_asset(['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])

    # Create model
    model = CrossAttentionMultiAsset(
        n_assets=3,
        n_features=6,
        d_model=64
    )

    # Train and predict
    predictions = model(x)
"""

from .model import (
    CrossAttentionMultiAsset,
    CrossAttentionConfig,
    CrossAssetAttention,
    TemporalCrossAttention,
)

from .data import (
    BybitDataLoader,
    prepare_cross_attention_data,
    compute_features,
)

from .backtest import (
    CrossAttentionBacktest,
)

__all__ = [
    'CrossAttentionMultiAsset',
    'CrossAttentionConfig',
    'CrossAssetAttention',
    'TemporalCrossAttention',
    'BybitDataLoader',
    'prepare_cross_attention_data',
    'compute_features',
    'CrossAttentionBacktest',
]

__version__ = '0.1.0'
