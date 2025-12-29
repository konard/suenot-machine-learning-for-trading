//! Модуль бэктестинга
//!
//! Этот модуль содержит:
//! - Движок бэктеста
//! - Метрики производительности

pub mod engine;
pub mod metrics;

pub use engine::{BacktestConfig, BacktestEngine, BacktestResult, PortfolioSnapshot, Trade};
pub use metrics::{
    annualized_volatility, avg_loss, avg_win, cagr, calculate_all_metrics, calmar_ratio, cvar,
    information_ratio, kelly_criterion, log_return, max_drawdown, profit_factor, sharpe_ratio,
    simple_return, sortino_ratio, var, volatility, win_rate, PerformanceMetrics,
};
