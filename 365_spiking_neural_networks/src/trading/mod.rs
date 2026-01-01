//! Trading module for SNN-based strategies
//!
//! This module provides trading signal generation and strategy implementation
//! using Spiking Neural Networks.

mod strategy;
mod signals;

pub use strategy::{TradingStrategy, SNNTradingStrategy, TradeDecision, StrategyParams};
pub use signals::{TradingSignal, SignalStrength, MarketRegime};
