//! Trading Strategies using SE Networks
//!
//! This module provides trading strategies that leverage SE blocks
//! for dynamic feature weighting.

pub mod se_momentum;
pub mod signals;

pub use se_momentum::SEMomentumStrategy;
pub use signals::{TradingSignal, Direction};
