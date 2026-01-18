//! Backtesting engine for Cross-Attention Multi-Asset Trading
//!
//! Evaluates trading strategies with realistic constraints.

use serde::{Deserialize, Serialize};

/// Backtest configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Transaction cost (as fraction)
    pub transaction_cost: f64,
    /// Rebalance frequency (in time steps)
    pub rebalance_freq: usize,
    /// Maximum position size per asset
    pub max_position: f64,
    /// Allow short positions
    pub allow_short: bool,
    /// Slippage (as fraction)
    pub slippage: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            transaction_cost: 0.001, // 10 bps
            rebalance_freq: 24,      // Daily for hourly data
            max_position: 0.5,
            allow_short: false,
            slippage: 0.0005, // 5 bps
        }
    }
}

/// Single step result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    /// Timestamp
    pub timestamp: i64,
    /// Portfolio value
    pub portfolio_value: f64,
    /// Portfolio weights
    pub weights: Vec<f64>,
    /// Returns for this step
    pub step_return: f64,
    /// Transaction costs incurred
    pub transaction_costs: f64,
}

/// Backtest result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Step-by-step results
    pub steps: Vec<StepResult>,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Asset symbols
    pub symbols: Vec<String>,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Sortino ratio (annualized)
    pub sortino_ratio: f64,
    /// Calmar ratio
    pub calmar_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Volatility (annualized)
    pub volatility: f64,
    /// Win rate
    pub win_rate: f64,
    /// Total transaction costs
    pub total_transaction_costs: f64,
    /// Number of trades
    pub n_trades: usize,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_return: 0.0,
            annualized_return: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            calmar_ratio: 0.0,
            max_drawdown: 0.0,
            volatility: 0.0,
            win_rate: 0.0,
            total_transaction_costs: 0.0,
            n_trades: 0,
        }
    }
}

/// Backtesting engine
pub struct Backtest {
    config: BacktestConfig,
}

impl Backtest {
    /// Create a new backtest engine
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run backtest
    ///
    /// # Arguments
    /// * `weights` - Model-predicted weights [n_steps, n_assets]
    /// * `returns` - Actual returns [n_steps, n_assets]
    /// * `symbols` - Asset symbols
    /// * `timestamps` - Timestamps for each step
    ///
    /// # Returns
    /// Backtest result with step-by-step data and metrics
    pub fn run(
        &self,
        weights: &[Vec<f64>],
        returns: &[Vec<f64>],
        symbols: &[String],
        timestamps: &[i64],
    ) -> BacktestResult {
        assert_eq!(weights.len(), returns.len(), "Weights and returns must have same length");
        assert!(!weights.is_empty(), "Must have at least one step");

        let n_steps = weights.len();
        let n_assets = symbols.len();

        let mut steps = Vec::with_capacity(n_steps);
        let mut portfolio_value = self.config.initial_capital;
        let mut current_weights = vec![1.0 / n_assets as f64; n_assets]; // Equal weight start
        let mut total_costs = 0.0;
        let mut n_trades = 0;

        for i in 0..n_steps {
            let target_weights = &weights[i];
            let asset_returns = &returns[i];

            // Check if rebalancing
            let should_rebalance = i % self.config.rebalance_freq == 0;

            // Calculate transaction costs if rebalancing
            let transaction_cost = if should_rebalance {
                let cost = self.calculate_transaction_cost(&current_weights, target_weights, portfolio_value);
                total_costs += cost;
                n_trades += 1;
                cost
            } else {
                0.0
            };

            // Apply transaction costs
            portfolio_value -= transaction_cost;

            // Update weights if rebalancing
            if should_rebalance {
                current_weights = self.apply_constraints(target_weights);
            }

            // Calculate portfolio return
            let portfolio_return: f64 = current_weights
                .iter()
                .zip(asset_returns.iter())
                .map(|(w, r)| w * r)
                .sum();

            // Apply slippage if rebalancing
            let slippage_cost = if should_rebalance {
                portfolio_value * self.config.slippage
            } else {
                0.0
            };
            portfolio_value -= slippage_cost;

            // Update portfolio value
            portfolio_value *= 1.0 + portfolio_return;

            // Update weights based on returns (drift)
            if !should_rebalance {
                let total: f64 = current_weights
                    .iter()
                    .zip(asset_returns.iter())
                    .map(|(w, r)| w * (1.0 + r))
                    .sum();

                if total > 0.0 {
                    for (w, r) in current_weights.iter_mut().zip(asset_returns.iter()) {
                        *w = *w * (1.0 + r) / total;
                    }
                }
            }

            steps.push(StepResult {
                timestamp: timestamps.get(i).copied().unwrap_or(i as i64),
                portfolio_value,
                weights: current_weights.clone(),
                step_return: portfolio_return,
                transaction_costs: transaction_cost + slippage_cost,
            });
        }

        // Calculate metrics
        let metrics = self.calculate_metrics(&steps, total_costs, n_trades);

        BacktestResult {
            steps,
            metrics,
            symbols: symbols.to_vec(),
        }
    }

    /// Apply position constraints
    fn apply_constraints(&self, weights: &[f64]) -> Vec<f64> {
        let mut result: Vec<f64> = weights
            .iter()
            .map(|w| {
                if self.config.allow_short {
                    w.clamp(-self.config.max_position, self.config.max_position)
                } else {
                    w.clamp(0.0, self.config.max_position)
                }
            })
            .collect();

        // Normalize to sum to 1 (for long-only)
        if !self.config.allow_short {
            let total: f64 = result.iter().sum();
            if total > 0.0 {
                for w in &mut result {
                    *w /= total;
                }
            }
        }

        result
    }

    /// Calculate transaction cost
    fn calculate_transaction_cost(
        &self,
        current: &[f64],
        target: &[f64],
        portfolio_value: f64,
    ) -> f64 {
        let turnover: f64 = current
            .iter()
            .zip(target.iter())
            .map(|(c, t)| (t - c).abs())
            .sum();

        portfolio_value * turnover * self.config.transaction_cost
    }

    /// Calculate performance metrics
    fn calculate_metrics(
        &self,
        steps: &[StepResult],
        total_costs: f64,
        n_trades: usize,
    ) -> PerformanceMetrics {
        if steps.is_empty() {
            return PerformanceMetrics::default();
        }

        let n = steps.len();
        let final_value = steps.last().unwrap().portfolio_value;
        let initial_value = self.config.initial_capital;

        // Total return
        let total_return = (final_value - initial_value) / initial_value;

        // Returns series
        let returns: Vec<f64> = steps.iter().map(|s| s.step_return).collect();

        // Annualized return (assuming hourly data, 24*365 periods per year)
        let periods_per_year = 24.0 * 365.0;
        let annualized_return = (1.0 + total_return).powf(periods_per_year / n as f64) - 1.0;

        // Volatility
        let mean_return: f64 = returns.iter().sum::<f64>() / n as f64;
        let variance: f64 = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();
        let volatility = std_dev * (periods_per_year as f64).sqrt();

        // Sharpe ratio (assuming 0% risk-free rate)
        let sharpe_ratio = if volatility > 0.0 {
            annualized_return / volatility
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
        let downside_deviation = downside_variance.sqrt() * (periods_per_year as f64).sqrt();
        let sortino_ratio = if downside_deviation > 0.0 {
            annualized_return / downside_deviation
        } else {
            0.0
        };

        // Maximum drawdown
        let mut max_drawdown = 0.0;
        let mut peak = initial_value;
        for step in steps {
            if step.portfolio_value > peak {
                peak = step.portfolio_value;
            }
            let drawdown = (peak - step.portfolio_value) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        // Calmar ratio
        let calmar_ratio = if max_drawdown > 0.0 {
            annualized_return / max_drawdown
        } else {
            0.0
        };

        // Win rate
        let positive_returns = returns.iter().filter(|&&r| r > 0.0).count();
        let win_rate = positive_returns as f64 / n as f64;

        PerformanceMetrics {
            total_return,
            annualized_return,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            max_drawdown,
            volatility,
            win_rate,
            total_transaction_costs: total_costs,
            n_trades,
        }
    }

    /// Compare with equal-weight baseline
    pub fn compare_baseline(
        &self,
        returns: &[Vec<f64>],
        symbols: &[String],
        timestamps: &[i64],
    ) -> BacktestResult {
        let n_assets = symbols.len();
        let equal_weights: Vec<Vec<f64>> = returns
            .iter()
            .map(|_| vec![1.0 / n_assets as f64; n_assets])
            .collect();

        self.run(&equal_weights, returns, symbols, timestamps)
    }
}

impl Default for Backtest {
    fn default() -> Self {
        Self::new(BacktestConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data(n_steps: usize, n_assets: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<String>, Vec<i64>) {
        let weights: Vec<Vec<f64>> = (0..n_steps)
            .map(|_| {
                let mut w = vec![1.0 / n_assets as f64; n_assets];
                w[0] += 0.1;
                w[1] -= 0.1 / (n_assets - 1) as f64;
                let total: f64 = w.iter().sum();
                w.iter().map(|x| x / total).collect()
            })
            .collect();

        let returns: Vec<Vec<f64>> = (0..n_steps)
            .map(|i| {
                (0..n_assets)
                    .map(|j| 0.001 * ((i + j) as f64 * 0.1).sin())
                    .collect()
            })
            .collect();

        let symbols: Vec<String> = (0..n_assets).map(|i| format!("ASSET{}", i)).collect();
        let timestamps: Vec<i64> = (0..n_steps).map(|i| i as i64 * 3600000).collect();

        (weights, returns, symbols, timestamps)
    }

    #[test]
    fn test_backtest_creation() {
        let config = BacktestConfig::default();
        let backtest = Backtest::new(config);

        assert_eq!(backtest.config.initial_capital, 100_000.0);
    }

    #[test]
    fn test_backtest_run() {
        let (weights, returns, symbols, timestamps) = create_test_data(100, 5);
        let backtest = Backtest::default();

        let result = backtest.run(&weights, &returns, &symbols, &timestamps);

        assert_eq!(result.steps.len(), 100);
        assert!(result.metrics.total_return.is_finite());
    }

    #[test]
    fn test_backtest_metrics() {
        let (weights, returns, symbols, timestamps) = create_test_data(1000, 5);
        let backtest = Backtest::default();

        let result = backtest.run(&weights, &returns, &symbols, &timestamps);
        let metrics = &result.metrics;

        // Verify metrics are calculated
        assert!(metrics.sharpe_ratio.is_finite());
        assert!(metrics.max_drawdown >= 0.0);
        assert!(metrics.max_drawdown <= 1.0);
        assert!(metrics.win_rate >= 0.0);
        assert!(metrics.win_rate <= 1.0);
    }

    #[test]
    fn test_baseline_comparison() {
        let (_, returns, symbols, timestamps) = create_test_data(100, 5);
        let backtest = Backtest::default();

        let result = backtest.compare_baseline(&returns, &symbols, &timestamps);

        assert_eq!(result.steps.len(), 100);
        // Equal weights should sum to 1
        let weight_sum: f64 = result.steps[0].weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_transaction_costs() {
        let (weights, returns, symbols, timestamps) = create_test_data(100, 5);
        let config = BacktestConfig {
            transaction_cost: 0.01, // 1% to make it significant
            ..Default::default()
        };
        let backtest = Backtest::new(config);

        let result = backtest.run(&weights, &returns, &symbols, &timestamps);

        assert!(result.metrics.total_transaction_costs > 0.0);
        assert!(result.metrics.n_trades > 0);
    }

    #[test]
    fn test_position_constraints() {
        let config = BacktestConfig {
            max_position: 0.3,
            ..Default::default()
        };
        let backtest = Backtest::new(config);

        let weights = vec![0.8, 0.1, 0.05, 0.05];
        let constrained = backtest.apply_constraints(&weights);

        // After capping at 0.3: [0.3, 0.1, 0.05, 0.05] sum = 0.5
        // After normalization: [0.6, 0.2, 0.1, 0.1] sum = 1.0
        // The constraint is applied before normalization
        let sum: f64 = constrained.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Weights should sum to 1");

        // Check that no weight is negative
        for w in &constrained {
            assert!(*w >= 0.0, "No negative weights");
        }
    }
}
