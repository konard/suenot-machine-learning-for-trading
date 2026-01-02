//! Performance metrics for backtesting.

use serde::{Deserialize, Serialize};

/// Performance metrics for strategy evaluation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Annualized return
    pub annual_return: f64,
    /// Annualized volatility
    pub annual_volatility: f64,
    /// Sharpe ratio (assuming 0% risk-free rate)
    pub sharpe_ratio: f64,
    /// Sortino ratio
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor
    pub profit_factor: f64,
    /// Average win
    pub avg_win: f64,
    /// Average loss
    pub avg_loss: f64,
    /// Total trades
    pub total_trades: usize,
    /// Average holding period
    pub avg_holding_period: f64,
    /// Skewness of returns
    pub skewness: f64,
    /// Kurtosis of returns
    pub kurtosis: f64,
    /// Value at Risk (95%)
    pub var_95: f64,
    /// Expected Shortfall (95%)
    pub cvar_95: f64,
}

impl PerformanceMetrics {
    /// Calculate metrics from a series of returns.
    pub fn from_returns(returns: &[f64], periods_per_year: usize) -> Self {
        if returns.is_empty() {
            return Self::default();
        }

        let n = returns.len() as f64;
        let ppf = periods_per_year as f64;

        // Mean return
        let mean: f64 = returns.iter().sum::<f64>() / n;

        // Volatility
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let volatility = variance.sqrt();

        // Downside deviation
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_variance: f64 = if !downside_returns.is_empty() {
            downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64
        } else {
            1e-10
        };
        let downside_deviation = downside_variance.sqrt();

        // Annualized metrics
        let annual_return = (1.0 + mean).powf(ppf) - 1.0;
        let annual_volatility = volatility * ppf.sqrt();

        // Sharpe ratio
        let sharpe_ratio = if volatility > 0.0 {
            (mean / volatility) * ppf.sqrt()
        } else {
            0.0
        };

        // Sortino ratio
        let sortino_ratio = if downside_deviation > 0.0 {
            (mean / downside_deviation) * ppf.sqrt()
        } else {
            0.0
        };

        // Drawdown
        let mut cumulative = Vec::with_capacity(returns.len());
        let mut cum = 1.0;
        for r in returns {
            cum *= 1.0 + r;
            cumulative.push(cum);
        }

        let mut peak = cumulative[0];
        let mut max_dd = 0.0;
        for &c in &cumulative {
            if c > peak {
                peak = c;
            }
            let dd = (peak - c) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        // Calmar ratio
        let calmar_ratio = if max_dd > 0.0 {
            annual_return / max_dd
        } else {
            0.0
        };

        // Win/loss metrics
        let wins: Vec<f64> = returns.iter().filter(|&&r| r > 0.0).cloned().collect();
        let losses: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();

        let win_rate = wins.len() as f64 / n;
        let avg_win = if !wins.is_empty() {
            wins.iter().sum::<f64>() / wins.len() as f64
        } else {
            0.0
        };
        let avg_loss = if !losses.is_empty() {
            losses.iter().sum::<f64>() / losses.len() as f64
        } else {
            0.0
        };

        let total_wins: f64 = wins.iter().sum();
        let total_losses: f64 = losses.iter().map(|l| l.abs()).sum();
        let profit_factor = if total_losses > 0.0 {
            total_wins / total_losses
        } else if total_wins > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Skewness
        let skewness = if volatility > 0.0 {
            returns.iter().map(|r| ((r - mean) / volatility).powi(3)).sum::<f64>() / n
        } else {
            0.0
        };

        // Kurtosis (excess)
        let kurtosis = if volatility > 0.0 {
            returns.iter().map(|r| ((r - mean) / volatility).powi(4)).sum::<f64>() / n - 3.0
        } else {
            0.0
        };

        // VaR and CVaR
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let var_idx = ((n * 0.05) as usize).max(1) - 1;
        let var_95 = -sorted_returns.get(var_idx).copied().unwrap_or(0.0);

        let cvar_95 = if var_idx > 0 {
            -sorted_returns[..=var_idx].iter().sum::<f64>() / (var_idx + 1) as f64
        } else {
            var_95
        };

        Self {
            annual_return,
            annual_volatility,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown: max_dd,
            calmar_ratio,
            win_rate,
            profit_factor,
            avg_win,
            avg_loss,
            total_trades: 0,
            avg_holding_period: 0.0,
            skewness,
            kurtosis,
            var_95,
            cvar_95,
        }
    }

    /// Calculate metrics from trade data.
    pub fn from_trades(
        trades: &[super::Trade],
        initial_capital: f64,
        periods_per_year: usize,
    ) -> Self {
        if trades.is_empty() {
            return Self::default();
        }

        // Calculate trade-by-trade returns
        let mut capital = initial_capital;
        let mut returns = Vec::new();

        for trade in trades {
            if trade.pnl.abs() > 0.0 {
                let ret = trade.pnl / capital;
                returns.push(ret);
                capital += trade.pnl - trade.commission;
            }
        }

        let mut metrics = Self::from_returns(&returns, periods_per_year);
        metrics.total_trades = trades.len();

        // Calculate average holding period
        if trades.len() > 1 {
            let mut holding_periods = Vec::new();
            let mut open_time: std::collections::HashMap<String, u64> = std::collections::HashMap::new();

            for trade in trades {
                if trade.quantity > 0.0 {
                    open_time.insert(trade.symbol.clone(), trade.timestamp);
                } else if let Some(&open) = open_time.get(&trade.symbol) {
                    holding_periods.push((trade.timestamp - open) as f64 / 3600.0); // Convert to hours
                }
            }

            if !holding_periods.is_empty() {
                metrics.avg_holding_period =
                    holding_periods.iter().sum::<f64>() / holding_periods.len() as f64;
            }
        }

        metrics
    }

    /// Display metrics in a formatted way.
    pub fn display(&self) -> String {
        format!(
            r#"Performance Metrics
==================
Annual Return:     {:.2}%
Annual Volatility: {:.2}%
Sharpe Ratio:      {:.2}
Sortino Ratio:     {:.2}
Max Drawdown:      {:.2}%
Calmar Ratio:      {:.2}
Win Rate:          {:.2}%
Profit Factor:     {:.2}
Avg Win:           {:.4}%
Avg Loss:          {:.4}%
Total Trades:      {}
Skewness:          {:.2}
Kurtosis:          {:.2}
VaR (95%):         {:.2}%
CVaR (95%):        {:.2}%
"#,
            self.annual_return * 100.0,
            self.annual_volatility * 100.0,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown * 100.0,
            self.calmar_ratio,
            self.win_rate * 100.0,
            self.profit_factor,
            self.avg_win * 100.0,
            self.avg_loss * 100.0,
            self.total_trades,
            self.skewness,
            self.kurtosis,
            self.var_95 * 100.0,
            self.cvar_95 * 100.0,
        )
    }
}

/// Compare two strategies.
pub struct StrategyComparison {
    pub strategy_a: PerformanceMetrics,
    pub strategy_b: PerformanceMetrics,
}

impl StrategyComparison {
    /// Create a comparison between two strategies.
    pub fn new(a: PerformanceMetrics, b: PerformanceMetrics) -> Self {
        Self {
            strategy_a: a,
            strategy_b: b,
        }
    }

    /// Get the better strategy based on Sharpe ratio.
    pub fn better_by_sharpe(&self) -> &PerformanceMetrics {
        if self.strategy_a.sharpe_ratio > self.strategy_b.sharpe_ratio {
            &self.strategy_a
        } else {
            &self.strategy_b
        }
    }

    /// Calculate information ratio (A relative to B).
    pub fn information_ratio(&self, returns_a: &[f64], returns_b: &[f64]) -> f64 {
        if returns_a.len() != returns_b.len() || returns_a.is_empty() {
            return 0.0;
        }

        let active_returns: Vec<f64> = returns_a
            .iter()
            .zip(returns_b.iter())
            .map(|(a, b)| a - b)
            .collect();

        let mean: f64 = active_returns.iter().sum::<f64>() / active_returns.len() as f64;
        let variance: f64 = active_returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / active_returns.len() as f64;
        let tracking_error = variance.sqrt();

        if tracking_error > 0.0 {
            mean / tracking_error * (252.0_f64).sqrt()
        } else {
            0.0
        }
    }
}

/// Rolling performance metrics.
pub struct RollingMetrics {
    /// Window size
    window: usize,
    /// Rolling Sharpe ratios
    pub sharpe_ratios: Vec<f64>,
    /// Rolling volatilities
    pub volatilities: Vec<f64>,
    /// Rolling returns
    pub returns: Vec<f64>,
}

impl RollingMetrics {
    /// Calculate rolling metrics.
    pub fn new(returns: &[f64], window: usize, periods_per_year: usize) -> Self {
        let ppf = periods_per_year as f64;
        let mut sharpe_ratios = Vec::new();
        let mut volatilities = Vec::new();
        let mut rolling_returns = Vec::new();

        for i in window..=returns.len() {
            let window_returns = &returns[i - window..i];

            let mean: f64 = window_returns.iter().sum::<f64>() / window as f64;
            let variance: f64 = window_returns
                .iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>()
                / window as f64;
            let vol = variance.sqrt();

            let sharpe = if vol > 0.0 {
                (mean / vol) * ppf.sqrt()
            } else {
                0.0
            };

            sharpe_ratios.push(sharpe);
            volatilities.push(vol * ppf.sqrt());
            rolling_returns.push(mean * ppf);
        }

        Self {
            window,
            sharpe_ratios,
            volatilities,
            returns: rolling_returns,
        }
    }

    /// Get average rolling Sharpe ratio.
    pub fn avg_sharpe(&self) -> f64 {
        if self.sharpe_ratios.is_empty() {
            return 0.0;
        }
        self.sharpe_ratios.iter().sum::<f64>() / self.sharpe_ratios.len() as f64
    }

    /// Get Sharpe ratio stability (lower is better).
    pub fn sharpe_stability(&self) -> f64 {
        if self.sharpe_ratios.len() < 2 {
            return 0.0;
        }
        let mean = self.avg_sharpe();
        let variance: f64 = self
            .sharpe_ratios
            .iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f64>()
            / self.sharpe_ratios.len() as f64;
        variance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_from_returns() {
        // Positive returns
        let returns = vec![0.01, 0.02, -0.01, 0.03, 0.01, -0.02, 0.02];
        let metrics = PerformanceMetrics::from_returns(&returns, 252);

        assert!(metrics.annual_return > 0.0);
        assert!(metrics.sharpe_ratio > 0.0);
        assert!(metrics.win_rate > 0.5);
    }

    #[test]
    fn test_max_drawdown() {
        // Returns that should produce a drawdown
        let returns = vec![0.1, -0.15, -0.1, 0.05, 0.1];
        let metrics = PerformanceMetrics::from_returns(&returns, 252);

        assert!(metrics.max_drawdown > 0.0);
        assert!(metrics.max_drawdown < 1.0);
    }

    #[test]
    fn test_var_cvar() {
        let returns: Vec<f64> = (-10..10).map(|i| i as f64 / 100.0).collect();
        let metrics = PerformanceMetrics::from_returns(&returns, 252);

        // VaR should be positive (representing loss)
        assert!(metrics.var_95 > 0.0);
        // CVaR should be >= VaR
        assert!(metrics.cvar_95 >= metrics.var_95);
    }

    #[test]
    fn test_rolling_metrics() {
        let returns: Vec<f64> = (0..100).map(|i| 0.01 * (i as f64 / 100.0).sin()).collect();
        let rolling = RollingMetrics::new(&returns, 20, 252);

        assert!(!rolling.sharpe_ratios.is_empty());
        assert!(rolling.sharpe_stability() >= 0.0);
    }
}
