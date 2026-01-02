//! Trading Performance Metrics

use serde::{Deserialize, Serialize};

/// Trading performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingMetrics {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub num_trades: usize,
    pub avg_trade_return: f64,
}

impl TradingMetrics {
    /// Calculate metrics from returns
    pub fn from_returns(returns: &[f64], risk_free_rate: f64) -> Self {
        let n = returns.len();
        if n == 0 {
            return Self::default();
        }

        let total_return = returns.iter().fold(1.0, |acc, r| acc * (1.0 + r)) - 1.0;
        let mean_return = returns.iter().sum::<f64>() / n as f64;
        let std_dev = Self::std_dev(returns);

        let sharpe_ratio = if std_dev > 1e-10 {
            (mean_return - risk_free_rate / 252.0) / std_dev * (252.0_f64).sqrt()
        } else { 0.0 };

        let downside: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_std = Self::std_dev(&downside);
        let sortino_ratio = if downside_std > 1e-10 {
            (mean_return - risk_free_rate / 252.0) / downside_std * (252.0_f64).sqrt()
        } else { 0.0 };

        let max_drawdown = Self::calculate_max_drawdown(returns);

        let wins: usize = returns.iter().filter(|&&r| r > 0.0).count();
        let win_rate = if n > 0 { wins as f64 / n as f64 } else { 0.0 };

        let gross_profit: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
        let gross_loss: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
        let profit_factor = if gross_loss > 1e-10 { gross_profit / gross_loss } else { 0.0 };

        Self {
            total_return, sharpe_ratio, sortino_ratio, max_drawdown, win_rate,
            profit_factor, num_trades: n, avg_trade_return: mean_return,
        }
    }

    fn std_dev(values: &[f64]) -> f64 {
        if values.len() < 2 { return 0.0; }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        var.sqrt()
    }

    fn calculate_max_drawdown(returns: &[f64]) -> f64 {
        let mut peak = 1.0;
        let mut max_dd = 0.0;
        let mut equity = 1.0;

        for r in returns {
            equity *= 1.0 + r;
            peak = peak.max(equity);
            let dd = (peak - equity) / peak;
            max_dd = max_dd.max(dd);
        }
        max_dd
    }
}

impl Default for TradingMetrics {
    fn default() -> Self {
        Self {
            total_return: 0.0, sharpe_ratio: 0.0, sortino_ratio: 0.0, max_drawdown: 0.0,
            win_rate: 0.0, profit_factor: 0.0, num_trades: 0, avg_trade_return: 0.0,
        }
    }
}
