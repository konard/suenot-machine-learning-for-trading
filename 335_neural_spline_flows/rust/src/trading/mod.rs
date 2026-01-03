//! Trading module for signal generation and risk management
//!
//! This module provides trading utilities based on Neural Spline Flows:
//! - Signal generation from learned distributions
//! - Risk management with VaR/CVaR
//! - Position sizing

pub mod risk;
pub mod signals;

pub use risk::RiskManager;
pub use signals::{SignalGenerator, TradingSignal};
