"""
LLM Earnings Call Analysis Package

This package provides tools for analyzing earnings call transcripts
using Large Language Models to extract trading signals.

Modules:
    earnings_analyzer: Core analysis functionality
    bybit_client: Bybit API client for crypto data
    backtest: Backtesting framework for earnings strategies
"""

from .earnings_analyzer import (
    EarningsTranscriptParser,
    EarningsSentimentAnalyzer,
    EarningsSignalGenerator,
    TradingSignal,
    SignalDirection,
    SentimentResult,
    TranscriptSegment,
    SpeakerRole,
    analyze_earnings_call_simple
)

from .bybit_client import (
    BybitClient,
    Candle,
    Ticker,
    fetch_crypto_data_for_analysis
)

from .backtest import (
    EarningsBacktest,
    BacktestResult,
    Trade,
    run_sample_backtest
)

__version__ = "0.1.0"
__author__ = "ML Trading Team"

__all__ = [
    # Parser
    'EarningsTranscriptParser',
    'TranscriptSegment',
    'SpeakerRole',

    # Sentiment
    'EarningsSentimentAnalyzer',
    'SentimentResult',

    # Signals
    'EarningsSignalGenerator',
    'TradingSignal',
    'SignalDirection',

    # Convenience function
    'analyze_earnings_call_simple',

    # Bybit
    'BybitClient',
    'Candle',
    'Ticker',
    'fetch_crypto_data_for_analysis',

    # Backtest
    'EarningsBacktest',
    'BacktestResult',
    'Trade',
    'run_sample_backtest'
]
