//! Trading Module
//!
//! Trading signals, backtesting, and risk management.

mod signal;
mod position;
mod backtester;
mod risk;
mod metrics;

pub use signal::TradingSignal;
pub use position::Position;
pub use backtester::Backtester;
pub use risk::RiskManager;
pub use metrics::PerformanceMetrics;
