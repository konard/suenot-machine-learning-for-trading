//! Performance metrics for backtesting

/// Performance metrics
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Annualized volatility
    pub annualized_volatility: f64,
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Number of trades
    pub num_trades: usize,
    /// Average trade return
    pub avg_trade_return: f64,
}

impl PerformanceMetrics {
    /// Calculate metrics from returns series
    pub fn from_returns(returns: &[f64], risk_free_rate: f64) -> Self {
        if returns.is_empty() {
            return Self::default();
        }

        let n = returns.len() as f64;

        // Total return (cumulative)
        let total_return: f64 = returns.iter().map(|r| 1.0 + r).product::<f64>() - 1.0;

        // Mean and std
        let mean_return = returns.iter().sum::<f64>() / n;
        let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / n;
        let std_return = variance.sqrt();

        // Annualize (assuming daily returns, 252 trading days)
        let annualized_return = mean_return * 252.0;
        let annualized_volatility = std_return * 252.0_f64.sqrt();

        // Sharpe ratio
        let sharpe_ratio = if annualized_volatility > 0.0 {
            (annualized_return - risk_free_rate) / annualized_volatility
        } else {
            0.0
        };

        // Sortino ratio (downside deviation)
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_variance = if !downside_returns.is_empty() {
            downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64
        } else {
            0.0
        };
        let downside_std = downside_variance.sqrt() * 252.0_f64.sqrt();
        let sortino_ratio = if downside_std > 0.0 {
            (annualized_return - risk_free_rate) / downside_std
        } else {
            0.0
        };

        // Maximum drawdown
        let max_drawdown = Self::calculate_max_drawdown(returns);

        // Win rate
        let wins = returns.iter().filter(|&&r| r > 0.0).count();
        let win_rate = wins as f64 / n;

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

        Self {
            total_return,
            annualized_return,
            annualized_volatility,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            num_trades: returns.len(),
            avg_trade_return: mean_return,
        }
    }

    /// Calculate maximum drawdown
    fn calculate_max_drawdown(returns: &[f64]) -> f64 {
        let mut peak = 1.0;
        let mut max_dd = 0.0;
        let mut cumulative = 1.0;

        for &ret in returns {
            cumulative *= 1.0 + ret;
            if cumulative > peak {
                peak = cumulative;
            }
            let dd = (peak - cumulative) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        max_dd
    }

    /// Print metrics summary
    pub fn summary(&self) -> String {
        format!(
            "Total Return: {:.2}%\n\
             Annualized Return: {:.2}%\n\
             Annualized Volatility: {:.2}%\n\
             Sharpe Ratio: {:.2}\n\
             Sortino Ratio: {:.2}\n\
             Max Drawdown: {:.2}%\n\
             Win Rate: {:.2}%\n\
             Profit Factor: {:.2}\n\
             Num Trades: {}",
            self.total_return * 100.0,
            self.annualized_return * 100.0,
            self.annualized_volatility * 100.0,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown * 100.0,
            self.win_rate * 100.0,
            self.profit_factor,
            self.num_trades
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_calculation() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.008, 0.012];
        let metrics = PerformanceMetrics::from_returns(&returns, 0.02);

        assert!(metrics.total_return > 0.0);
        assert!(metrics.win_rate > 0.5);
        assert!(metrics.max_drawdown >= 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        let returns = vec![0.1, -0.2, 0.05];  // 10%, -20%, 5%
        let metrics = PerformanceMetrics::from_returns(&returns, 0.0);

        // After 10%, equity = 1.1
        // After -20%, equity = 0.88, drawdown from 1.1 = (1.1 - 0.88) / 1.1 = 0.2
        assert!(metrics.max_drawdown > 0.15);
    }
}
