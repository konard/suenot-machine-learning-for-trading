//! Performance metrics for trading strategies
//!
//! Provides comprehensive metrics for evaluating trading performance.

use serde::{Deserialize, Serialize};

/// Trading performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total return percentage
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Sharpe ratio (risk-adjusted return)
    pub sharpe_ratio: f64,
    /// Sortino ratio (downside risk-adjusted return)
    pub sortino_ratio: f64,
    /// Calmar ratio (return / max drawdown)
    pub calmar_ratio: f64,
    /// Win rate
    pub win_rate: f64,
    /// Average win
    pub avg_win: f64,
    /// Average loss
    pub avg_loss: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Number of trades
    pub total_trades: usize,
    /// Number of winning trades
    pub winning_trades: usize,
    /// Number of losing trades
    pub losing_trades: usize,
    /// Average trade duration (in periods)
    pub avg_trade_duration: f64,
    /// Volatility (standard deviation of returns)
    pub volatility: f64,
    /// Skewness of returns
    pub skewness: f64,
    /// Kurtosis of returns
    pub kurtosis: f64,
}

impl PerformanceMetrics {
    /// Create metrics indicating no trades
    pub fn no_trades() -> Self {
        Self::default()
    }

    /// Check if the strategy is profitable
    pub fn is_profitable(&self) -> bool {
        self.total_return > 0.0
    }

    /// Check if the strategy has acceptable risk metrics
    pub fn is_acceptable(&self, min_sharpe: f64, max_drawdown: f64) -> bool {
        self.sharpe_ratio >= min_sharpe && self.max_drawdown <= max_drawdown
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "Return: {:.2}% | Sharpe: {:.2} | MaxDD: {:.2}% | WinRate: {:.1}% | Trades: {}",
            self.total_return * 100.0,
            self.sharpe_ratio,
            self.max_drawdown * 100.0,
            self.win_rate * 100.0,
            self.total_trades
        )
    }
}

/// Calculator for trading performance metrics
pub struct MetricsCalculator {
    risk_free_rate: f64,
    periods_per_year: f64,
}

impl MetricsCalculator {
    /// Create a new metrics calculator
    ///
    /// # Arguments
    /// * `risk_free_rate` - Annual risk-free rate (e.g., 0.04 for 4%)
    /// * `periods_per_year` - Number of trading periods per year (e.g., 252 for daily, 365*24 for hourly)
    pub fn new(risk_free_rate: f64, periods_per_year: f64) -> Self {
        Self {
            risk_free_rate,
            periods_per_year,
        }
    }

    /// Create calculator for daily data with default risk-free rate
    pub fn daily() -> Self {
        Self::new(0.04, 252.0)
    }

    /// Create calculator for hourly data
    pub fn hourly() -> Self {
        Self::new(0.04, 252.0 * 24.0)
    }

    /// Calculate performance metrics from a series of portfolio values
    pub fn calculate(&self, portfolio_values: &[f64]) -> PerformanceMetrics {
        if portfolio_values.len() < 2 {
            return PerformanceMetrics::no_trades();
        }

        let returns = self.calculate_returns(portfolio_values);
        if returns.is_empty() {
            return PerformanceMetrics::no_trades();
        }

        let total_return = (portfolio_values.last().unwrap() / portfolio_values.first().unwrap()) - 1.0;
        let annualized_return = self.annualize_return(total_return, portfolio_values.len());
        let volatility = self.calculate_std(&returns);
        let annualized_volatility = volatility * self.periods_per_year.sqrt();

        let excess_return = annualized_return - self.risk_free_rate;
        let sharpe_ratio = if annualized_volatility > 0.0 {
            excess_return / annualized_volatility
        } else {
            0.0
        };

        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_std = if downside_returns.is_empty() {
            0.0
        } else {
            self.calculate_std(&downside_returns) * self.periods_per_year.sqrt()
        };
        let sortino_ratio = if downside_std > 0.0 {
            excess_return / downside_std
        } else {
            f64::INFINITY
        };

        let max_drawdown = self.calculate_max_drawdown(portfolio_values);
        let calmar_ratio = if max_drawdown > 0.0 {
            annualized_return / max_drawdown
        } else {
            f64::INFINITY
        };

        let skewness = self.calculate_skewness(&returns);
        let kurtosis = self.calculate_kurtosis(&returns);

        PerformanceMetrics {
            total_return,
            annualized_return,
            max_drawdown,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            win_rate: 0.0, // Requires trade data
            avg_win: 0.0,
            avg_loss: 0.0,
            profit_factor: 0.0,
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            avg_trade_duration: 0.0,
            volatility: annualized_volatility,
            skewness,
            kurtosis,
        }
    }

    /// Calculate metrics with trade-level data
    pub fn calculate_with_trades(
        &self,
        portfolio_values: &[f64],
        trade_pnls: &[f64],
        trade_durations: &[f64],
    ) -> PerformanceMetrics {
        let mut metrics = self.calculate(portfolio_values);

        if !trade_pnls.is_empty() {
            let wins: Vec<f64> = trade_pnls.iter().filter(|&&p| p > 0.0).cloned().collect();
            let losses: Vec<f64> = trade_pnls.iter().filter(|&&p| p < 0.0).cloned().collect();

            metrics.total_trades = trade_pnls.len();
            metrics.winning_trades = wins.len();
            metrics.losing_trades = losses.len();
            metrics.win_rate = wins.len() as f64 / trade_pnls.len() as f64;

            metrics.avg_win = if !wins.is_empty() {
                wins.iter().sum::<f64>() / wins.len() as f64
            } else {
                0.0
            };

            metrics.avg_loss = if !losses.is_empty() {
                losses.iter().sum::<f64>() / losses.len() as f64
            } else {
                0.0
            };

            let gross_profit: f64 = wins.iter().sum();
            let gross_loss: f64 = losses.iter().map(|l| l.abs()).sum();
            metrics.profit_factor = if gross_loss > 0.0 {
                gross_profit / gross_loss
            } else if gross_profit > 0.0 {
                f64::INFINITY
            } else {
                0.0
            };

            if !trade_durations.is_empty() {
                metrics.avg_trade_duration =
                    trade_durations.iter().sum::<f64>() / trade_durations.len() as f64;
            }
        }

        metrics
    }

    /// Calculate period returns from portfolio values
    fn calculate_returns(&self, values: &[f64]) -> Vec<f64> {
        values
            .windows(2)
            .map(|w| {
                if w[0] > 0.0 {
                    w[1] / w[0] - 1.0
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Annualize a return
    fn annualize_return(&self, total_return: f64, n_periods: usize) -> f64 {
        let periods_factor = self.periods_per_year / n_periods as f64;
        (1.0 + total_return).powf(periods_factor) - 1.0
    }

    /// Calculate standard deviation
    fn calculate_std(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        variance.sqrt()
    }

    /// Calculate maximum drawdown
    fn calculate_max_drawdown(&self, values: &[f64]) -> f64 {
        let mut max_dd = 0.0;
        let mut peak = values[0];

        for &value in values {
            if value > peak {
                peak = value;
            }
            let dd = (peak - value) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        max_dd
    }

    /// Calculate skewness
    fn calculate_skewness(&self, data: &[f64]) -> f64 {
        if data.len() < 3 {
            return 0.0;
        }

        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let std = self.calculate_std(data);

        if std == 0.0 {
            return 0.0;
        }

        let sum_cubed: f64 = data.iter().map(|x| ((x - mean) / std).powi(3)).sum();
        sum_cubed / n
    }

    /// Calculate kurtosis (excess kurtosis)
    fn calculate_kurtosis(&self, data: &[f64]) -> f64 {
        if data.len() < 4 {
            return 0.0;
        }

        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let std = self.calculate_std(data);

        if std == 0.0 {
            return 0.0;
        }

        let sum_fourth: f64 = data.iter().map(|x| ((x - mean) / std).powi(4)).sum();
        sum_fourth / n - 3.0 // Excess kurtosis (normal = 0)
    }
}

impl Default for MetricsCalculator {
    fn default() -> Self {
        Self::daily()
    }
}

/// Compare two strategies
pub fn compare_strategies(baseline: &PerformanceMetrics, strategy: &PerformanceMetrics) -> StrategyComparison {
    StrategyComparison {
        return_diff: strategy.total_return - baseline.total_return,
        sharpe_diff: strategy.sharpe_ratio - baseline.sharpe_ratio,
        drawdown_diff: strategy.max_drawdown - baseline.max_drawdown,
        win_rate_diff: strategy.win_rate - baseline.win_rate,
        outperforms: strategy.sharpe_ratio > baseline.sharpe_ratio,
    }
}

/// Comparison between two strategies
#[derive(Debug, Clone)]
pub struct StrategyComparison {
    pub return_diff: f64,
    pub sharpe_diff: f64,
    pub drawdown_diff: f64,
    pub win_rate_diff: f64,
    pub outperforms: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_returns_calculation() {
        let calculator = MetricsCalculator::daily();
        let values = vec![100.0, 105.0, 102.0, 108.0];

        let metrics = calculator.calculate(&values);

        assert!((metrics.total_return - 0.08).abs() < 0.001);
    }

    #[test]
    fn test_max_drawdown() {
        let calculator = MetricsCalculator::daily();
        let values = vec![100.0, 110.0, 105.0, 90.0, 95.0, 100.0];

        let metrics = calculator.calculate(&values);

        // Max DD should be (110 - 90) / 110 = 0.1818
        assert!((metrics.max_drawdown - 0.1818).abs() < 0.001);
    }

    #[test]
    fn test_sharpe_ratio() {
        let calculator = MetricsCalculator::new(0.0, 252.0); // 0 risk-free for simpler calculation

        // Create a series with positive returns
        let values: Vec<f64> = (0..253)
            .map(|i| 100.0 * 1.001_f64.powi(i))
            .collect();

        let metrics = calculator.calculate(&values);

        // Sharpe should be positive for consistently positive returns
        assert!(metrics.sharpe_ratio > 0.0);
    }

    #[test]
    fn test_trade_metrics() {
        let calculator = MetricsCalculator::daily();
        let values = vec![100.0, 105.0, 103.0, 108.0];
        let trade_pnls = vec![5.0, -2.0, 5.0];
        let trade_durations = vec![1.0, 1.0, 1.0];

        let metrics = calculator.calculate_with_trades(&values, &trade_pnls, &trade_durations);

        assert_eq!(metrics.total_trades, 3);
        assert_eq!(metrics.winning_trades, 2);
        assert_eq!(metrics.losing_trades, 1);
        assert!((metrics.win_rate - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_empty_data() {
        let calculator = MetricsCalculator::daily();
        let metrics = calculator.calculate(&[]);

        assert_eq!(metrics.total_return, 0.0);
        assert_eq!(metrics.sharpe_ratio, 0.0);
    }

    #[test]
    fn test_strategy_comparison() {
        let baseline = PerformanceMetrics {
            total_return: 0.10,
            sharpe_ratio: 1.0,
            max_drawdown: 0.15,
            win_rate: 0.55,
            ..Default::default()
        };

        let strategy = PerformanceMetrics {
            total_return: 0.15,
            sharpe_ratio: 1.5,
            max_drawdown: 0.10,
            win_rate: 0.60,
            ..Default::default()
        };

        let comparison = compare_strategies(&baseline, &strategy);

        assert!(comparison.outperforms);
        assert!((comparison.return_diff - 0.05).abs() < 0.001);
        assert!((comparison.sharpe_diff - 0.5).abs() < 0.001);
    }
}
