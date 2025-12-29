//! Utility functions and helpers

mod metrics;

pub use metrics::{
    compute_max_drawdown, compute_returns, compute_sharpe, compute_sortino, compute_win_rate,
    compute_calmar, TradingMetrics,
};
