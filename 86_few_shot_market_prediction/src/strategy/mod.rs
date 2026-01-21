//! Trading strategy components
//!
//! Provides regime classification and signal generation for trading.

mod regime;
mod signal;
mod risk;

pub use regime::{MarketRegime, RegimeClassifier};
pub use signal::{TradingSignal, SignalType, SignalGenerator};
pub use risk::{RiskManager, RiskConfig};
