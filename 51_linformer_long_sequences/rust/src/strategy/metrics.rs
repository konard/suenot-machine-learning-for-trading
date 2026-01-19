//! Performance metrics calculation for trading strategies.

/// Performance metrics for strategy evaluation.
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Total return percentage
    pub total_return: f64,
    /// Annualized return
    pub annual_return: f64,
    /// Annualized volatility
    pub annual_volatility: f64,
    /// Sharpe ratio (risk-adjusted return)
    pub sharpe_ratio: f64,
    /// Sortino ratio (downside risk-adjusted)
    pub sortino_ratio: f64,
    /// Maximum drawdown percentage
    pub max_drawdown: f64,
    /// Win rate (percentage of profitable trades)
    pub win_rate: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Number of trades
    pub num_trades: usize,
    /// Average trade return
    pub avg_trade_return: f64,
}

impl PerformanceMetrics {
    /// Calculate all metrics from returns series.
    pub fn from_returns(
        returns: &[f64],
        risk_free_rate: f64,
        periods_per_year: f64,
    ) -> Self {
        if returns.is_empty() {
            return Self::default();
        }

        let total_return = Self::calculate_total_return(returns);
        let annual_return = Self::calculate_annual_return(returns, periods_per_year);
        let annual_volatility = Self::calculate_annual_volatility(returns, periods_per_year);
        let sharpe_ratio =
            Self::calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year);
        let sortino_ratio =
            Self::calculate_sortino_ratio(returns, risk_free_rate, periods_per_year);
        let max_drawdown = Self::calculate_max_drawdown(returns);
        let (win_rate, profit_factor, num_trades, avg_trade_return) =
            Self::calculate_trade_metrics(returns);

        Self {
            total_return,
            annual_return,
            annual_volatility,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            num_trades,
            avg_trade_return,
        }
    }

    /// Calculate total cumulative return.
    fn calculate_total_return(returns: &[f64]) -> f64 {
        returns.iter().fold(1.0, |acc, &r| acc * (1.0 + r)) - 1.0
    }

    /// Calculate annualized return.
    fn calculate_annual_return(returns: &[f64], periods_per_year: f64) -> f64 {
        let total = Self::calculate_total_return(returns);
        let n_periods = returns.len() as f64;
        let years = n_periods / periods_per_year;

        if years > 0.0 {
            (1.0 + total).powf(1.0 / years) - 1.0
        } else {
            0.0
        }
    }

    /// Calculate annualized volatility (standard deviation).
    fn calculate_annual_volatility(returns: &[f64], periods_per_year: f64) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;

        variance.sqrt() * periods_per_year.sqrt()
    }

    /// Calculate Sharpe ratio.
    fn calculate_sharpe_ratio(
        returns: &[f64],
        risk_free_rate: f64,
        periods_per_year: f64,
    ) -> f64 {
        let annual_return = Self::calculate_annual_return(returns, periods_per_year);
        let annual_vol = Self::calculate_annual_volatility(returns, periods_per_year);

        if annual_vol > 0.0 {
            (annual_return - risk_free_rate) / annual_vol
        } else {
            0.0
        }
    }

    /// Calculate Sortino ratio (uses downside deviation).
    fn calculate_sortino_ratio(
        returns: &[f64],
        risk_free_rate: f64,
        periods_per_year: f64,
    ) -> f64 {
        let annual_return = Self::calculate_annual_return(returns, periods_per_year);
        let period_rf = risk_free_rate / periods_per_year;

        // Calculate downside deviation
        let downside_returns: Vec<f64> = returns
            .iter()
            .map(|&r| (r - period_rf).min(0.0))
            .collect();

        if downside_returns.is_empty() {
            return 0.0;
        }

        let downside_variance =
            downside_returns.iter().map(|&r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64;

        let downside_deviation = downside_variance.sqrt() * periods_per_year.sqrt();

        if downside_deviation > 0.0 {
            (annual_return - risk_free_rate) / downside_deviation
        } else {
            0.0
        }
    }

    /// Calculate maximum drawdown.
    fn calculate_max_drawdown(returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mut cumulative = 1.0;
        let mut peak = 1.0;
        let mut max_dd = 0.0;

        for &r in returns {
            cumulative *= 1.0 + r;
            peak = peak.max(cumulative);

            let drawdown = (peak - cumulative) / peak;
            max_dd = max_dd.max(drawdown);
        }

        max_dd
    }

    /// Calculate trade-level metrics.
    fn calculate_trade_metrics(returns: &[f64]) -> (f64, f64, usize, f64) {
        let trades: Vec<f64> = returns.iter().filter(|&&r| r != 0.0).cloned().collect();

        if trades.is_empty() {
            return (0.0, 0.0, 0, 0.0);
        }

        let num_trades = trades.len();
        let winning_trades: Vec<f64> = trades.iter().filter(|&&r| r > 0.0).cloned().collect();
        let losing_trades: Vec<f64> = trades.iter().filter(|&&r| r < 0.0).cloned().collect();

        let win_rate = winning_trades.len() as f64 / num_trades as f64;

        let gross_profit: f64 = winning_trades.iter().sum();
        let gross_loss: f64 = losing_trades.iter().map(|&r| r.abs()).sum();

        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let avg_trade_return = trades.iter().sum::<f64>() / num_trades as f64;

        (win_rate, profit_factor, num_trades, avg_trade_return)
    }

    /// Format metrics as a summary string.
    pub fn summary(&self) -> String {
        format!(
            r#"Performance Metrics Summary
============================
Total Return:      {:>8.2}%
Annual Return:     {:>8.2}%
Annual Volatility: {:>8.2}%
Sharpe Ratio:      {:>8.2}
Sortino Ratio:     {:>8.2}
Max Drawdown:      {:>8.2}%
Win Rate:          {:>8.2}%
Profit Factor:     {:>8.2}
Number of Trades:  {:>8}
Avg Trade Return:  {:>8.4}%"#,
            self.total_return * 100.0,
            self.annual_return * 100.0,
            self.annual_volatility * 100.0,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown * 100.0,
            self.win_rate * 100.0,
            self.profit_factor,
            self.num_trades,
            self.avg_trade_return * 100.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_total_return() {
        let returns = vec![0.01, 0.02, -0.01, 0.03];
        let total = PerformanceMetrics::calculate_total_return(&returns);
        assert!((total - 0.0509).abs() < 0.001);
    }

    #[test]
    fn test_max_drawdown() {
        let returns = vec![0.1, 0.05, -0.15, -0.05, 0.1];
        let dd = PerformanceMetrics::calculate_max_drawdown(&returns);
        assert!(dd > 0.15);
    }

    #[test]
    fn test_win_rate() {
        let returns = vec![0.01, -0.01, 0.02, 0.01, -0.02];
        let (win_rate, _, num_trades, _) =
            PerformanceMetrics::calculate_trade_metrics(&returns);
        assert!((win_rate - 0.6).abs() < 0.01);
        assert_eq!(num_trades, 5);
    }

    #[test]
    fn test_from_returns() {
        let returns: Vec<f64> = (0..252).map(|i| 0.001 * (i % 3 - 1) as f64).collect();
        let metrics = PerformanceMetrics::from_returns(&returns, 0.02, 252.0);

        assert!(metrics.sharpe_ratio.is_finite());
        assert!(metrics.max_drawdown >= 0.0);
    }

    #[test]
    fn test_empty_returns() {
        let metrics = PerformanceMetrics::from_returns(&[], 0.02, 252.0);
        assert_eq!(metrics.total_return, 0.0);
        assert_eq!(metrics.num_trades, 0);
    }
}
