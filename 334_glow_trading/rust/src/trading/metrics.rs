//! Performance metrics for trading strategies

use serde::{Deserialize, Serialize};

/// Comprehensive performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total return (percentage)
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Sortino ratio (annualized)
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// Average win
    pub avg_win: f64,
    /// Average loss
    pub avg_loss: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Number of trades
    pub num_trades: usize,
    /// Calmar ratio (return / max drawdown)
    pub calmar_ratio: f64,
}

impl PerformanceMetrics {
    /// Compute metrics from returns series
    pub fn from_returns(returns: &[f64], initial_capital: f64) -> Self {
        if returns.is_empty() {
            return Self::default();
        }

        // Total return
        let total_pnl: f64 = returns.iter().sum();
        let total_return = total_pnl / initial_capital;

        // Annualized return (assuming hourly data)
        let periods_per_year = 365.25 * 24.0;
        let num_periods = returns.len() as f64;
        let annualized_return = if num_periods > 0.0 {
            (1.0 + total_return).powf(periods_per_year / num_periods) - 1.0
        } else {
            0.0
        };

        // Sharpe ratio
        let sharpe_ratio = compute_sharpe_ratio(returns, periods_per_year);

        // Sortino ratio
        let sortino_ratio = compute_sortino_ratio(returns, periods_per_year);

        // Maximum drawdown
        let max_drawdown = compute_max_drawdown(returns, initial_capital);

        // Win/loss statistics
        let wins: Vec<f64> = returns.iter().filter(|&&r| r > 0.0).copied().collect();
        let losses: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

        let num_trades = wins.len() + losses.len();
        let win_rate = if num_trades > 0 {
            wins.len() as f64 / num_trades as f64
        } else {
            0.0
        };

        let avg_win = if !wins.is_empty() {
            wins.iter().sum::<f64>() / wins.len() as f64
        } else {
            0.0
        };

        let avg_loss = if !losses.is_empty() {
            losses.iter().sum::<f64>().abs() / losses.len() as f64
        } else {
            0.0
        };

        let total_wins: f64 = wins.iter().sum();
        let total_losses: f64 = losses.iter().sum::<f64>().abs();
        let profit_factor = if total_losses > 0.0 {
            total_wins / total_losses
        } else if total_wins > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Calmar ratio
        let calmar_ratio = if max_drawdown.abs() > 0.0 {
            annualized_return / max_drawdown.abs()
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
            avg_win,
            avg_loss,
            profit_factor,
            num_trades,
            calmar_ratio,
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_return: 0.0,
            annualized_return: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            max_drawdown: 0.0,
            win_rate: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            profit_factor: 0.0,
            num_trades: 0,
            calmar_ratio: 0.0,
        }
    }
}

/// Compute Sharpe ratio from returns
pub fn compute_sharpe_ratio(returns: &[f64], periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
    let std = variance.sqrt();

    if std > 0.0 {
        (mean / std) * periods_per_year.sqrt()
    } else {
        0.0
    }
}

/// Compute Sortino ratio (using downside deviation)
pub fn compute_sortino_ratio(returns: &[f64], periods_per_year: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;

    // Downside deviation (only negative returns)
    let downside: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

    if downside.is_empty() {
        return if mean > 0.0 { f64::INFINITY } else { 0.0 };
    }

    let downside_variance = downside.iter().map(|r| r.powi(2)).sum::<f64>() / downside.len() as f64;
    let downside_std = downside_variance.sqrt();

    if downside_std > 0.0 {
        (mean / downside_std) * periods_per_year.sqrt()
    } else {
        0.0
    }
}

/// Compute maximum drawdown
pub fn compute_max_drawdown(returns: &[f64], initial_capital: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mut equity = initial_capital;
    let mut peak = initial_capital;
    let mut max_dd = 0.0;

    for &ret in returns {
        equity += ret;
        if equity > peak {
            peak = equity;
        }
        let drawdown = (peak - equity) / peak;
        if drawdown > max_dd {
            max_dd = drawdown;
        }
    }

    max_dd
}

/// Compute rolling Sharpe ratio
pub fn rolling_sharpe_ratio(returns: &[f64], window: usize, periods_per_year: f64) -> Vec<f64> {
    if returns.len() < window {
        return vec![];
    }

    (0..=returns.len() - window)
        .map(|i| {
            let window_returns = &returns[i..i + window];
            compute_sharpe_ratio(window_returns, periods_per_year)
        })
        .collect()
}

/// Compute Value at Risk
pub fn compute_var(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let idx = ((1.0 - confidence) * returns.len() as f64) as usize;
    sorted.get(idx).copied().unwrap_or(0.0)
}

/// Compute Conditional Value at Risk (Expected Shortfall)
pub fn compute_cvar(returns: &[f64], confidence: f64) -> f64 {
    let var = compute_var(returns, confidence);

    let tail: Vec<f64> = returns.iter().filter(|&&r| r <= var).copied().collect();
    if tail.is_empty() {
        var
    } else {
        tail.iter().sum::<f64>() / tail.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpe_ratio() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.002];
        let sharpe = compute_sharpe_ratio(&returns, 252.0);
        assert!(sharpe.is_finite());
    }

    #[test]
    fn test_max_drawdown() {
        let returns = vec![100.0, -50.0, 20.0, -30.0, 40.0];
        let mdd = compute_max_drawdown(&returns, 10000.0);
        assert!(mdd > 0.0 && mdd < 1.0);
    }

    #[test]
    fn test_performance_metrics() {
        let returns = vec![10.0, -5.0, 20.0, -10.0, 15.0, 5.0, -2.0, 8.0];
        let metrics = PerformanceMetrics::from_returns(&returns, 10000.0);

        assert!(metrics.total_return > 0.0);
        assert!(metrics.win_rate > 0.0 && metrics.win_rate <= 1.0);
        assert!(metrics.num_trades > 0);
    }

    #[test]
    fn test_var_cvar() {
        let returns = vec![-0.05, -0.02, 0.01, 0.03, -0.01, 0.02, -0.03, 0.01, -0.04, 0.02];
        let var = compute_var(&returns, 0.95);
        let cvar = compute_cvar(&returns, 0.95);

        assert!(var <= 0.0);
        assert!(cvar <= var);
    }
}
