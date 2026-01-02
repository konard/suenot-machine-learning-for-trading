//! Performance metrics and tracking

use std::collections::VecDeque;

/// Performance metrics
#[derive(Debug, Clone, Default)]
pub struct Metrics {
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
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
    /// Total trades
    pub total_trades: u32,
    /// Winning trades
    pub winning_trades: u32,
    /// Average win
    pub avg_win: f64,
    /// Average loss
    pub avg_loss: f64,
    /// Calmar ratio (return / max drawdown)
    pub calmar_ratio: f64,
}

impl Metrics {
    /// Create new metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute metrics from returns
    pub fn from_returns(returns: &[f64], risk_free_rate: f64) -> Self {
        if returns.is_empty() {
            return Self::default();
        }

        let n = returns.len() as f64;

        // Total and annualized return
        let total_return: f64 = returns.iter().map(|r| 1.0 + r).product::<f64>() - 1.0;
        let periods_per_year = 252.0;  // Assuming daily returns
        let annualized_return = (1.0 + total_return).powf(periods_per_year / n) - 1.0;

        // Mean and std
        let mean: f64 = returns.iter().sum::<f64>() / n;
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        // Sharpe ratio
        let excess_return = mean - risk_free_rate / periods_per_year;
        let sharpe_ratio = if std > 0.0 {
            excess_return / std * periods_per_year.sqrt()
        } else {
            0.0
        };

        // Sortino ratio (downside deviation)
        let downside_returns: Vec<f64> = returns.iter()
            .filter(|&&r| r < 0.0)
            .copied()
            .collect();
        let downside_var: f64 = if !downside_returns.is_empty() {
            downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64
        } else {
            0.0
        };
        let downside_std = downside_var.sqrt();
        let sortino_ratio = if downside_std > 0.0 {
            excess_return / downside_std * periods_per_year.sqrt()
        } else {
            0.0
        };

        // Maximum drawdown
        let max_drawdown = compute_max_drawdown(returns);

        // Win rate
        let winning: Vec<_> = returns.iter().filter(|&&r| r > 0.0).collect();
        let losing: Vec<_> = returns.iter().filter(|&&r| r < 0.0).collect();
        let total_trades = returns.len() as u32;
        let winning_trades = winning.len() as u32;
        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };

        // Average win/loss
        let avg_win = if !winning.is_empty() {
            winning.iter().copied().sum::<f64>() / winning.len() as f64
        } else {
            0.0
        };
        let avg_loss = if !losing.is_empty() {
            losing.iter().copied().sum::<f64>().abs() / losing.len() as f64
        } else {
            0.0
        };

        // Profit factor
        let gross_profit: f64 = winning.iter().copied().sum();
        let gross_loss: f64 = losing.iter().copied().sum::<f64>().abs();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Calmar ratio
        let calmar_ratio = if max_drawdown > 0.0 {
            annualized_return / max_drawdown
        } else {
            0.0
        };

        Self {
            total_return,
            annualized_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            total_trades,
            winning_trades,
            avg_win,
            avg_loss,
            calmar_ratio,
        }
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("=== Performance Metrics ===");
        println!("Total Return:      {:.2}%", self.total_return * 100.0);
        println!("Annualized Return: {:.2}%", self.annualized_return * 100.0);
        println!("Sharpe Ratio:      {:.2}", self.sharpe_ratio);
        println!("Sortino Ratio:     {:.2}", self.sortino_ratio);
        println!("Max Drawdown:      {:.2}%", self.max_drawdown * 100.0);
        println!("Calmar Ratio:      {:.2}", self.calmar_ratio);
        println!("Win Rate:          {:.2}%", self.win_rate * 100.0);
        println!("Profit Factor:     {:.2}", self.profit_factor);
        println!("Total Trades:      {}", self.total_trades);
        println!("Avg Win:           {:.2}%", self.avg_win * 100.0);
        println!("Avg Loss:          {:.2}%", self.avg_loss * 100.0);
    }
}

/// Compute maximum drawdown from returns
fn compute_max_drawdown(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mut equity = 1.0;
    let mut peak = 1.0;
    let mut max_dd = 0.0;

    for &r in returns {
        equity *= 1.0 + r;
        peak = peak.max(equity);
        let dd = (peak - equity) / peak;
        max_dd = max_dd.max(dd);
    }

    max_dd
}

/// Performance tracker for live monitoring
#[derive(Debug)]
pub struct PerformanceTracker {
    /// Equity curve
    equity: Vec<f64>,
    /// Returns
    returns: VecDeque<f64>,
    /// Maximum window size
    max_window: usize,
    /// Peak equity
    peak_equity: f64,
    /// Current drawdown
    current_drawdown: f64,
    /// Risk-free rate
    risk_free_rate: f64,
}

impl PerformanceTracker {
    /// Create a new performance tracker
    pub fn new(initial_equity: f64, max_window: usize) -> Self {
        Self {
            equity: vec![initial_equity],
            returns: VecDeque::new(),
            max_window,
            peak_equity: initial_equity,
            current_drawdown: 0.0,
            risk_free_rate: 0.0,
        }
    }

    /// Set risk-free rate
    pub fn with_risk_free_rate(mut self, rate: f64) -> Self {
        self.risk_free_rate = rate;
        self
    }

    /// Update with new equity value
    pub fn update(&mut self, equity: f64) {
        let prev_equity = *self.equity.last().unwrap_or(&equity);
        let ret = if prev_equity > 0.0 {
            equity / prev_equity - 1.0
        } else {
            0.0
        };

        self.equity.push(equity);
        self.returns.push_back(ret);

        // Maintain window size
        if self.returns.len() > self.max_window {
            self.returns.pop_front();
        }

        // Update peak and drawdown
        self.peak_equity = self.peak_equity.max(equity);
        self.current_drawdown = if self.peak_equity > 0.0 {
            (self.peak_equity - equity) / self.peak_equity
        } else {
            0.0
        };
    }

    /// Get current metrics
    pub fn current_metrics(&self) -> Metrics {
        let returns: Vec<f64> = self.returns.iter().copied().collect();
        Metrics::from_returns(&returns, self.risk_free_rate)
    }

    /// Get current equity
    pub fn current_equity(&self) -> f64 {
        *self.equity.last().unwrap_or(&0.0)
    }

    /// Get current drawdown
    pub fn current_drawdown(&self) -> f64 {
        self.current_drawdown
    }

    /// Get rolling Sharpe ratio
    pub fn rolling_sharpe(&self, window: usize) -> f64 {
        if self.returns.len() < window {
            return 0.0;
        }

        let recent: Vec<f64> = self.returns.iter().rev().take(window).copied().collect();
        let metrics = Metrics::from_returns(&recent, self.risk_free_rate);
        metrics.sharpe_ratio
    }

    /// Check if drawdown exceeds threshold
    pub fn is_drawdown_exceeded(&self, threshold: f64) -> bool {
        self.current_drawdown > threshold
    }

    /// Get total return
    pub fn total_return(&self) -> f64 {
        if self.equity.len() < 2 {
            return 0.0;
        }
        let initial = self.equity[0];
        let current = *self.equity.last().unwrap();
        if initial > 0.0 {
            current / initial - 1.0
        } else {
            0.0
        }
    }
}

impl Default for PerformanceTracker {
    fn default() -> Self {
        Self::new(100000.0, 252)  // Default: $100k, 1 year of daily data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_from_returns() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015];
        let metrics = Metrics::from_returns(&returns, 0.0);

        assert!(metrics.total_return > 0.0);
        assert!(metrics.win_rate > 0.5);
        assert!(metrics.max_drawdown > 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        let returns = vec![0.1, -0.05, -0.1, 0.15, -0.02];
        let mdd = compute_max_drawdown(&returns);
        assert!(mdd > 0.0);
        assert!(mdd < 1.0);
    }

    #[test]
    fn test_performance_tracker() {
        let mut tracker = PerformanceTracker::new(100000.0, 100);

        tracker.update(102000.0);  // +2%
        tracker.update(100000.0);  // -1.96%
        tracker.update(105000.0);  // +5%

        assert!(tracker.total_return() > 0.0);
        assert!(tracker.current_drawdown() >= 0.0);
    }

    #[test]
    fn test_drawdown_check() {
        let mut tracker = PerformanceTracker::new(100000.0, 100);
        tracker.update(100000.0);
        tracker.update(90000.0);  // -10% drawdown

        assert!(tracker.is_drawdown_exceeded(0.05));
        assert!(!tracker.is_drawdown_exceeded(0.15));
    }
}
