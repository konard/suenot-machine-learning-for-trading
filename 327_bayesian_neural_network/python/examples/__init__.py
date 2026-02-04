"""
Examples for Bayesian Neural Network trading.

Available examples:
- train_bnn: Train a Bayesian Neural Network on cryptocurrency data
- backtest: Backtest the BNN trading strategy
- fetch_data: Fetch cryptocurrency data from Bybit

Usage:
    python -m examples.fetch_data --symbol BTC/USDT --days 30
    python -m examples.train_bnn --symbol BTC/USDT --days 60
    python -m examples.backtest --model bnn_model.pt --symbol BTC/USDT
"""

__all__ = ['train_bnn', 'backtest', 'fetch_data']
