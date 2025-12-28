//! Performance metrics for backtesting

use serde::{Deserialize, Serialize};

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annual_return: f64,
    /// Annualized volatility
    pub annual_volatility: f64,
    /// Sharpe ratio (assuming risk-free rate = 0)
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Calmar ratio (annual return / max drawdown)
    pub calmar_ratio: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Number of trades
    pub n_trades: usize,
    /// Average trade return
    pub avg_trade_return: f64,
    /// Best trade
    pub best_trade: f64,
    /// Worst trade
    pub worst_trade: f64,
}

impl PerformanceMetrics {
    /// Calculate metrics from returns series
    pub fn from_returns(returns: &[f64], periods_per_year: f64) -> Self {
        let n = returns.len();

        if n == 0 {
            return Self::default();
        }

        // Total return (compounded)
        let total_return = returns.iter().fold(1.0, |acc, r| acc * (1.0 + r)) - 1.0;

        // Mean return
        let mean_return = returns.iter().sum::<f64>() / n as f64;

        // Annualized return
        let annual_return = (1.0 + mean_return).powf(periods_per_year) - 1.0;

        // Volatility
        let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / n as f64;
        let volatility = variance.sqrt();
        let annual_volatility = volatility * periods_per_year.sqrt();

        // Sharpe ratio
        let sharpe_ratio = if annual_volatility > 0.0 {
            annual_return / annual_volatility
        } else {
            0.0
        };

        // Sortino ratio (downside deviation)
        let negative_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
        let downside_variance = if !negative_returns.is_empty() {
            negative_returns.iter().map(|r| r.powi(2)).sum::<f64>() / negative_returns.len() as f64
        } else {
            0.0
        };
        let downside_deviation = downside_variance.sqrt() * periods_per_year.sqrt();
        let sortino_ratio = if downside_deviation > 0.0 {
            annual_return / downside_deviation
        } else {
            0.0
        };

        // Maximum drawdown
        let mut cumulative = 1.0;
        let mut peak = 1.0;
        let mut max_drawdown = 0.0;

        for &r in returns {
            cumulative *= 1.0 + r;
            peak = peak.max(cumulative);
            let drawdown = (peak - cumulative) / peak;
            max_drawdown = max_drawdown.max(drawdown);
        }

        // Calmar ratio
        let calmar_ratio = if max_drawdown > 0.0 {
            annual_return / max_drawdown
        } else {
            0.0
        };

        // Win rate
        let wins = returns.iter().filter(|&&r| r > 0.0).count();
        let win_rate = wins as f64 / n as f64;

        // Profit factor
        let gross_profit: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
        let gross_loss: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Trade statistics
        let best_trade = returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let worst_trade = returns.iter().cloned().fold(f64::INFINITY, f64::min);

        Self {
            total_return,
            annual_return,
            annual_volatility,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            win_rate,
            profit_factor,
            n_trades: n,
            avg_trade_return: mean_return,
            best_trade,
            worst_trade,
        }
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("Performance Metrics");
        println!("===================");
        println!("Total Return:      {:>10.2}%", self.total_return * 100.0);
        println!("Annual Return:     {:>10.2}%", self.annual_return * 100.0);
        println!("Annual Volatility: {:>10.2}%", self.annual_volatility * 100.0);
        println!("Sharpe Ratio:      {:>10.2}", self.sharpe_ratio);
        println!("Sortino Ratio:     {:>10.2}", self.sortino_ratio);
        println!("Max Drawdown:      {:>10.2}%", self.max_drawdown * 100.0);
        println!("Calmar Ratio:      {:>10.2}", self.calmar_ratio);
        println!("Win Rate:          {:>10.2}%", self.win_rate * 100.0);
        println!("Profit Factor:     {:>10.2}", self.profit_factor);
        println!("Number of Trades:  {:>10}", self.n_trades);
        println!("Avg Trade Return:  {:>10.4}%", self.avg_trade_return * 100.0);
        println!("Best Trade:        {:>10.2}%", self.best_trade * 100.0);
        println!("Worst Trade:       {:>10.2}%", self.worst_trade * 100.0);
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_return: 0.0,
            annual_return: 0.0,
            annual_volatility: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            calmar_ratio: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            n_trades: 0,
            avg_trade_return: 0.0,
            best_trade: 0.0,
            worst_trade: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_calculation() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.008, 0.012];

        let metrics = PerformanceMetrics::from_returns(&returns, 252.0);

        assert!(metrics.total_return > 0.0);
        assert!(metrics.win_rate > 0.0 && metrics.win_rate < 1.0);
        assert!(metrics.max_drawdown >= 0.0);
    }
}
