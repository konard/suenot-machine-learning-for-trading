//! Trading strategy components
//!
//! Provides regime classification and signal generation for trading.

mod regime;
mod signal;
mod risk;

pub use regime::{MarketRegime, RegimeClassifier, RegimeClassification};
pub use signal::{TradingSignal, SignalType, SignalGenerator, SignalConfig};
pub use risk::{RiskManager, RiskConfig, RiskCheckResult, RiskSummary, Position};
