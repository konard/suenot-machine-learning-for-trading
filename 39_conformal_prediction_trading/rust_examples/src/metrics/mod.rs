//! Evaluation metrics for conformal prediction and trading
//!
//! - `coverage` - Coverage and interval quality metrics
//! - `trading` - Trading performance metrics

pub mod coverage;
pub mod trading;

pub use coverage::CoverageMetrics;
pub use trading::TradingMetrics;
