//! Performance metrics calculation
//!
//! Calculates various performance metrics for factor strategies.

use serde::{Deserialize, Serialize};

/// Factor-specific metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FactorMetrics {
    /// Information Coefficient
    pub ic: f64,
    /// IC Information Ratio
    pub ic_ir: f64,
    /// Rank IC
    pub rank_ic: f64,
    /// Rank IC IR
    pub rank_ic_ir: f64,
    /// Factor autocorrelation
    pub autocorrelation: f64,
    /// Factor turnover
    pub turnover: f64,
    /// Coverage (% of universe with valid values)
    pub coverage: f64,
}

/// Strategy performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annual_return: f64,
    /// Annualized volatility
    pub annual_volatility: f64,
    /// Sharpe ratio (assuming 0 risk-free rate)
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
    pub num_trades: usize,
}

/// Metrics calculator
pub struct MetricsCalculator {
    /// Annualization factor (252 for daily, 365*24 for hourly)
    annualization_factor: f64,
}

impl MetricsCalculator {
    /// Create calculator for daily data
    pub fn daily() -> Self {
        Self {
            annualization_factor: 252.0,
        }
    }

    /// Create calculator for hourly data
    pub fn hourly() -> Self {
        Self {
            annualization_factor: 365.0 * 24.0,
        }
    }

    /// Create with custom annualization factor
    pub fn with_factor(factor: f64) -> Self {
        Self {
            annualization_factor: factor,
        }
    }

    /// Calculate performance metrics from returns
    pub fn calculate(&self, returns: &[f64]) -> PerformanceMetrics {
        if returns.is_empty() {
            return PerformanceMetrics::default();
        }

        let n = returns.len() as f64;

        // Total return (compounded)
        let total_return: f64 = returns.iter().map(|r| 1.0 + r).product::<f64>() - 1.0;

        // Mean and standard deviation
        let mean_return = returns.iter().sum::<f64>() / n;
        let variance: f64 =
            returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / (n - 1.0);
        let std_return = variance.sqrt();

        // Annualized metrics
        let annual_return = mean_return * self.annualization_factor;
        let annual_volatility = std_return * self.annualization_factor.sqrt();

        // Sharpe ratio
        let sharpe_ratio = if annual_volatility > 0.0 {
            annual_return / annual_volatility
        } else {
            0.0
        };

        // Sortino ratio (downside deviation)
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
        let downside_variance = if !downside_returns.is_empty() {
            downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64
        } else {
            0.0
        };
        let downside_std = downside_variance.sqrt() * self.annualization_factor.sqrt();
        let sortino_ratio = if downside_std > 0.0 {
            annual_return / downside_std
        } else {
            0.0
        };

        // Maximum drawdown
        let max_drawdown = self.calculate_max_drawdown(returns);

        // Calmar ratio
        let calmar_ratio = if max_drawdown > 0.0 {
            annual_return / max_drawdown
        } else {
            0.0
        };

        // Win rate
        let winning_trades = returns.iter().filter(|&&r| r > 0.0).count();
        let win_rate = winning_trades as f64 / n;

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

        PerformanceMetrics {
            total_return,
            annual_return,
            annual_volatility,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            win_rate,
            profit_factor,
            num_trades: returns.len(),
        }
    }

    /// Calculate maximum drawdown from returns
    fn calculate_max_drawdown(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        // Build cumulative returns
        let mut cumulative = Vec::with_capacity(returns.len());
        let mut cum = 1.0;
        for r in returns {
            cum *= 1.0 + r;
            cumulative.push(cum);
        }

        // Find maximum drawdown
        let mut max_dd = 0.0;
        let mut peak = cumulative[0];

        for cum in cumulative {
            if cum > peak {
                peak = cum;
            }
            let dd = (peak - cum) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        max_dd
    }

    /// Calculate factor-specific metrics
    pub fn calculate_factor_metrics(
        &self,
        factor_values: &[f64],
        forward_returns: &[f64],
    ) -> FactorMetrics {
        use super::ic::IcCalculator;

        let ic = IcCalculator::pearson_correlation(factor_values, forward_returns).unwrap_or(0.0);
        let rank_ic =
            IcCalculator::spearman_correlation(factor_values, forward_returns).unwrap_or(0.0);

        // Autocorrelation (lag 1)
        let autocorrelation = self.calculate_autocorrelation(factor_values, 1);

        // Turnover
        let turnover = self.calculate_turnover(factor_values);

        // Coverage
        let valid_count = factor_values
            .iter()
            .filter(|x| !x.is_nan() && x.is_finite())
            .count();
        let coverage = valid_count as f64 / factor_values.len() as f64;

        FactorMetrics {
            ic,
            ic_ir: 0.0, // Would need time series to calculate
            rank_ic,
            rank_ic_ir: 0.0,
            autocorrelation,
            turnover,
            coverage,
        }
    }

    /// Calculate autocorrelation at given lag
    fn calculate_autocorrelation(&self, values: &[f64], lag: usize) -> f64 {
        if values.len() <= lag {
            return 0.0;
        }

        let n = values.len() - lag;
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;

        let mut cov = 0.0;
        let mut var = 0.0;

        for i in 0..n {
            let xi = values[i] - mean;
            let yi = values[i + lag] - mean;
            cov += xi * yi;
            var += xi * xi;
        }

        if var > 0.0 {
            cov / var
        } else {
            0.0
        }
    }

    /// Calculate turnover from factor values
    fn calculate_turnover(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mut total_change = 0.0;
        for i in 1..values.len() {
            total_change += (values[i] - values[i - 1]).abs();
        }

        // Normalize by mean absolute value
        let mean_abs: f64 = values.iter().map(|x| x.abs()).sum::<f64>() / values.len() as f64;

        if mean_abs > 0.0 {
            total_change / ((values.len() - 1) as f64 * mean_abs)
        } else {
            0.0
        }
    }
}

impl Default for MetricsCalculator {
    fn default() -> Self {
        Self::daily()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics_positive_returns() {
        let calc = MetricsCalculator::daily();
        let returns = vec![0.01, 0.02, -0.01, 0.015, 0.005, -0.005, 0.02];

        let metrics = calc.calculate(&returns);

        assert!(metrics.total_return > 0.0);
        assert!(metrics.annual_return > 0.0);
        assert!(metrics.sharpe_ratio > 0.0);
        assert!(metrics.max_drawdown > 0.0);
        assert!(metrics.win_rate > 0.5);
    }

    #[test]
    fn test_max_drawdown() {
        let calc = MetricsCalculator::daily();
        // Returns that create a 10% drawdown
        let returns = vec![0.1, 0.1, -0.15, -0.05, 0.1];

        let metrics = calc.calculate(&returns);

        // Should have some drawdown
        assert!(metrics.max_drawdown > 0.0);
    }

    #[test]
    fn test_autocorrelation() {
        let calc = MetricsCalculator::daily();

        // High autocorrelation (trending)
        let trending: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let ac_high = calc.calculate_autocorrelation(&trending, 1);

        // Low autocorrelation (random-like)
        let random: Vec<f64> = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let ac_low = calc.calculate_autocorrelation(&random, 1);

        assert!(ac_high > ac_low);
        assert!(ac_high > 0.9);
        assert!(ac_low < 0.0);
    }
}
