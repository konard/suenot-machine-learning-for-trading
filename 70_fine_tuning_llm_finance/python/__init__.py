"""
Chapter 70: Fine-tuning LLM for Finance

This package provides tools for fine-tuning Large Language Models
for financial applications using LoRA, QLoRA, and Prefix-Tuning methods.

Modules:
    model: LoRA, QLoRA, and prefix-tuning implementations
    trainer: Training pipeline with early stopping
    data_loader: Yahoo Finance and Bybit data loaders
    signals: Trading signal generation and aggregation
    backtest: Backtesting framework for LLM signals
    evaluate: Evaluation metrics
"""

from .model import LoRALayer, FinancialSentimentLoRA, PrefixTuningLayer
from .trainer import FineTuningTrainer
from .data_loader import YahooFinanceLoader, BybitDataLoader
from .signals import TradingSignalGenerator, CryptoSignalGenerator
from .backtest import LLMBacktester
from .evaluate import evaluate_sentiment, evaluate_trading

__version__ = "1.0.0"
__author__ = "Machine Learning for Trading"
