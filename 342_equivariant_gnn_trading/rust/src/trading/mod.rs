//! Trading Module
//!
//! Trading signals, backtesting, and risk management.

mod signal;
mod backtester;
mod position;
mod risk;
mod metrics;

pub use signal::{TradingSignal, TradeDirection};
pub use backtester::Backtester;
pub use position::Position;
pub use risk::RiskManager;
pub use metrics::TradingMetrics;
