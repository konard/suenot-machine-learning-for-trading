//! Trading strategy components
//!
//! This module provides:
//! - Market regime classification
//! - Trading signal generation from multi-task model outputs

mod regime;
mod signals;

pub use regime::{MarketRegime, RegimeClassifier, RegimeConfig, RegimeClassification};
pub use signals::{TradingSignal, SignalType, SignalGenerator, SignalConfig};
