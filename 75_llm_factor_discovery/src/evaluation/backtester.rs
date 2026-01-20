//! Factor backtesting framework
//!
//! Provides tools for historical testing of alpha factors.

use super::ic::{IcCalculator, IcStats};
use super::metrics::PerformanceMetrics;
use crate::parser::FactorExpr;
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Backtest configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Start date for backtest
    pub start_date: Option<DateTime<Utc>>,
    /// End date for backtest
    pub end_date: Option<DateTime<Utc>>,
    /// Forward periods for IC calculation
    pub forward_periods: Vec<usize>,
    /// Universe of symbols
    pub universe: Vec<String>,
    /// Minimum data points required
    pub min_data_points: usize,
    /// Include transaction costs
    pub include_costs: bool,
    /// Transaction cost (bps)
    pub transaction_cost_bps: f64,
    /// Rebalance frequency (days)
    pub rebalance_days: usize,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            start_date: None,
            end_date: None,
            forward_periods: vec![1, 5, 10, 20],
            universe: vec![
                "BTCUSDT".to_string(),
                "ETHUSDT".to_string(),
                "BNBUSDT".to_string(),
                "SOLUSDT".to_string(),
                "XRPUSDT".to_string(),
            ],
            min_data_points: 100,
            include_costs: true,
            transaction_cost_bps: 10.0,
            rebalance_days: 1,
        }
    }
}

/// Backtest result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Factor name
    pub factor_name: String,
    /// Factor expression
    pub expression: String,
    /// Mean IC across all periods
    pub ic: f64,
    /// IC Information Ratio
    pub ic_ir: f64,
    /// Rank IC
    pub rank_ic: f64,
    /// Rank IC IR
    pub rank_ic_ir: f64,
    /// IC by forward period
    pub ic_by_period: Vec<(usize, f64)>,
    /// IC time series statistics
    pub ic_stats: IcStats,
    /// Annual turnover estimate
    pub turnover: f64,
    /// Strategy returns (if simulated)
    pub strategy_returns: Option<f64>,
    /// Sharpe ratio (if simulated)
    pub sharpe_ratio: Option<f64>,
    /// Maximum drawdown
    pub max_drawdown: Option<f64>,
    /// Number of observations
    pub num_observations: usize,
    /// Backtest period
    pub period_start: Option<DateTime<Utc>>,
    pub period_end: Option<DateTime<Utc>>,
}

impl BacktestResult {
    /// Check if factor meets quality thresholds
    pub fn meets_threshold(&self, min_ic: f64, min_ir: f64) -> bool {
        self.ic.abs() >= min_ic && self.ic_ir.abs() >= min_ir
    }

    /// Get overall quality score
    pub fn quality_score(&self) -> f64 {
        let ic_score = self.ic.abs() * 10.0; // Scale IC
        let ir_score = self.ic_ir.abs();
        let consistency_score = self.ic_stats.hit_rate;

        (ic_score + ir_score + consistency_score) / 3.0
    }
}

/// Factor backtester
pub struct Backtester {
    config: BacktestConfig,
    ic_calculator: IcCalculator,
}

impl Backtester {
    /// Create a new backtester
    pub fn new(config: BacktestConfig) -> Self {
        let ic_calculator = IcCalculator::with_periods(config.forward_periods.clone());

        Self {
            config,
            ic_calculator,
        }
    }

    /// Run backtest for a factor expression
    pub async fn run(&self, factor: &FactorExpr) -> Result<BacktestResult> {
        info!("Running backtest for factor: {}", factor);

        // For now, we'll create a simulated backtest
        // In production, this would fetch real data and evaluate the factor

        let result = self.simulate_backtest(factor).await?;

        Ok(result)
    }

    /// Simulate backtest (placeholder for actual implementation)
    async fn simulate_backtest(&self, factor: &FactorExpr) -> Result<BacktestResult> {
        // This is a simulation - in production, you would:
        // 1. Fetch historical data for the universe
        // 2. Evaluate the factor expression on the data
        // 3. Calculate forward returns
        // 4. Compute IC and other metrics

        // For demonstration, we create mock results based on factor complexity
        let complexity = factor.depth() as f64;
        let variables = factor.variables();

        // Simple heuristic: more complex factors tend to have lower IC but might have some signal
        let base_ic = 0.05 - 0.005 * complexity;
        let ic_noise = (variables.len() as f64 * 0.01).min(0.02);

        // Simulate IC values
        let mock_ic_series: Vec<f64> = (0..100)
            .map(|i| {
                let t = i as f64 / 100.0;
                base_ic + ic_noise * (t * 10.0).sin()
            })
            .collect();

        let ic_stats = self.ic_calculator.ic_statistics(&mock_ic_series);

        // Calculate IC by period
        let ic_by_period: Vec<(usize, f64)> = self
            .config
            .forward_periods
            .iter()
            .map(|&period| {
                let period_ic = base_ic * (1.0 - period as f64 * 0.02);
                (period, period_ic)
            })
            .collect();

        // Estimate turnover based on factor type
        let has_short_window = variables.iter().any(|v| v.contains("1") || v.contains("5"));
        let turnover = if has_short_window { 0.8 } else { 0.3 };

        Ok(BacktestResult {
            factor_name: format!("factor_{}", factor.depth()),
            expression: factor.to_string(),
            ic: ic_stats.mean,
            ic_ir: ic_stats.ir,
            rank_ic: ic_stats.mean * 0.9, // Rank IC is typically slightly lower
            rank_ic_ir: ic_stats.ir * 0.9,
            ic_by_period,
            ic_stats,
            turnover,
            strategy_returns: Some(ic_stats.mean * 252.0 * 0.5), // Rough annual estimate
            sharpe_ratio: Some(ic_stats.ir * 2.0), // Rough conversion
            max_drawdown: Some(0.15),
            num_observations: 100,
            period_start: self.config.start_date,
            period_end: self.config.end_date,
        })
    }

    /// Run backtest with actual data
    pub async fn run_with_data(
        &self,
        factor: &FactorExpr,
        factor_values: &[f64],
        prices: &[f64],
    ) -> Result<BacktestResult> {
        if factor_values.len() != prices.len() {
            return Err(anyhow::anyhow!(
                "Factor values and prices must have same length"
            ));
        }

        let n = factor_values.len();

        // Calculate IC for each forward period
        let mut ic_by_period = Vec::new();
        let mut all_ic_values = Vec::new();

        for &period in &self.config.forward_periods {
            if n <= period {
                continue;
            }

            let forward_returns = IcCalculator::calculate_forward_returns(prices, period);
            let factor_slice = &factor_values[..forward_returns.len()];

            if let Some(ic) = self.ic_calculator.calculate_ic(factor_slice, &forward_returns) {
                ic_by_period.push((period, ic));
                all_ic_values.push(ic);
            }
        }

        // Calculate rolling IC for statistics
        let period_1_returns = IcCalculator::calculate_forward_returns(prices, 1);
        let rolling_ic =
            self.ic_calculator
                .rolling_ic(&factor_values[..period_1_returns.len()], &period_1_returns, 20);

        let ic_stats = self.ic_calculator.ic_statistics(&rolling_ic);

        // Calculate turnover
        let turnover = self.calculate_turnover(factor_values);

        // Mean IC across periods
        let mean_ic = if !all_ic_values.is_empty() {
            all_ic_values.iter().sum::<f64>() / all_ic_values.len() as f64
        } else {
            0.0
        };

        Ok(BacktestResult {
            factor_name: "custom_factor".to_string(),
            expression: factor.to_string(),
            ic: mean_ic,
            ic_ir: ic_stats.ir,
            rank_ic: mean_ic * 0.9, // Approximate
            rank_ic_ir: ic_stats.ir * 0.9,
            ic_by_period,
            ic_stats,
            turnover,
            strategy_returns: None, // Would need position simulation
            sharpe_ratio: None,
            max_drawdown: None,
            num_observations: n,
            period_start: self.config.start_date,
            period_end: self.config.end_date,
        })
    }

    /// Calculate factor turnover
    fn calculate_turnover(&self, factor_values: &[f64]) -> f64 {
        if factor_values.len() < 2 {
            return 0.0;
        }

        // Calculate absolute changes as a proxy for turnover
        let changes: f64 = factor_values
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum();

        let mean_abs: f64 = factor_values.iter().map(|x| x.abs()).sum::<f64>()
            / factor_values.len() as f64;

        if mean_abs > 0.0 {
            changes / (factor_values.len() as f64 * mean_abs)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backtest_config_default() {
        let config = BacktestConfig::default();
        assert!(!config.universe.is_empty());
        assert_eq!(config.forward_periods.len(), 4);
    }

    #[test]
    fn test_backtest_result_quality_score() {
        let result = BacktestResult {
            factor_name: "test".to_string(),
            expression: "rank(close)".to_string(),
            ic: 0.03,
            ic_ir: 0.6,
            rank_ic: 0.027,
            rank_ic_ir: 0.54,
            ic_by_period: vec![(1, 0.03)],
            ic_stats: IcStats {
                mean: 0.03,
                std: 0.05,
                ir: 0.6,
                hit_rate: 0.65,
                t_stat: 2.5,
                count: 100,
            },
            turnover: 0.5,
            strategy_returns: Some(0.15),
            sharpe_ratio: Some(1.2),
            max_drawdown: Some(0.1),
            num_observations: 100,
            period_start: None,
            period_end: None,
        };

        let score = result.quality_score();
        assert!(score > 0.0);
        assert!(result.meets_threshold(0.02, 0.5));
    }
}
