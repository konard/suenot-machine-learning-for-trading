//! Utility functions and helpers.

mod metrics;
mod config;

pub use metrics::{calculate_sharpe, calculate_max_drawdown, calculate_sortino};
pub use config::AppConfig;
