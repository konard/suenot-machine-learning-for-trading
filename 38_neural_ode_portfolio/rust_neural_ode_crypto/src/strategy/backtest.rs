//! # Backtesting
//!
//! Backtest framework for evaluating portfolio strategies.

use serde::{Deserialize, Serialize};
use tracing::info;

use crate::data::{CandleData, Features, TechnicalIndicators};
use super::rebalancer::ContinuousRebalancer;

/// Backtest configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial portfolio value
    pub initial_value: f64,
    /// Transaction cost (percentage)
    pub transaction_cost: f64,
    /// Rebalance threshold
    pub rebalance_threshold: f64,
    /// Minimum rebalance interval (in candles)
    pub min_rebalance_interval: usize,
    /// Benchmark strategy: "hold" or "equal_weight"
    pub benchmark: String,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_value: 100_000.0,
            transaction_cost: 0.001,
            rebalance_threshold: 0.02,
            min_rebalance_interval: 24, // 24 hours for hourly data
            benchmark: "equal_weight".to_string(),
        }
    }
}

/// Backtest result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Portfolio value over time
    pub portfolio_values: Vec<f64>,
    /// Benchmark values over time
    pub benchmark_values: Vec<f64>,
    /// Total return
    pub total_return: f64,
    /// Benchmark total return
    pub benchmark_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Annualized volatility
    pub annualized_volatility: f64,
    /// Sharpe ratio (assuming 0% risk-free rate)
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Number of rebalances
    pub num_rebalances: usize,
    /// Total transaction costs
    pub total_costs: f64,
    /// Rebalance timestamps
    pub rebalance_times: Vec<usize>,
    /// Weight history
    pub weight_history: Vec<Vec<f64>>,
}

impl BacktestResult {
    /// Calculate Calmar ratio (annualized return / max drawdown)
    pub fn calmar_ratio(&self) -> f64 {
        if self.max_drawdown == 0.0 {
            return 0.0;
        }
        self.annualized_return / self.max_drawdown
    }

    /// Calculate Sortino ratio (using only downside volatility)
    pub fn sortino_ratio(&self, target: f64) -> f64 {
        let returns = self.calculate_returns();
        let downside: Vec<f64> = returns
            .iter()
            .filter(|&&r| r < target)
            .map(|&r| (r - target).powi(2))
            .collect();

        if downside.is_empty() {
            return f64::INFINITY;
        }

        let downside_vol = (downside.iter().sum::<f64>() / downside.len() as f64).sqrt();
        let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;

        if downside_vol == 0.0 {
            return 0.0;
        }

        (mean_return - target) / downside_vol
    }

    fn calculate_returns(&self) -> Vec<f64> {
        self.portfolio_values
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }
}

/// Backtester for portfolio strategies
pub struct Backtester {
    config: BacktestConfig,
    indicators: TechnicalIndicators,
}

impl Backtester {
    /// Create a new backtester
    pub fn new(config: BacktestConfig) -> Self {
        Self {
            config,
            indicators: TechnicalIndicators::default(),
        }
    }

    /// Run backtest with Neural ODE rebalancer
    pub fn run(
        &self,
        rebalancer: &ContinuousRebalancer,
        data: &[CandleData],
    ) -> BacktestResult {
        let n_assets = data.len();
        let n_periods = data.iter().map(|d| d.len()).min().unwrap_or(0);

        if n_periods < 100 {
            return self.empty_result(n_assets);
        }

        let mut portfolio_value = self.config.initial_value;
        let mut benchmark_value = self.config.initial_value;
        let mut weights = vec![1.0 / n_assets as f64; n_assets];
        let benchmark_weights = vec![1.0 / n_assets as f64; n_assets];

        let mut portfolio_values = vec![portfolio_value];
        let mut benchmark_values = vec![benchmark_value];
        let mut weight_history = vec![weights.clone()];
        let mut rebalance_times = Vec::new();
        let mut total_costs = 0.0;
        let mut last_rebalance = 0;

        // Start from period 100 to have enough history for features
        for t in 100..n_periods {
            // Calculate returns for this period
            let returns: Vec<f64> = data
                .iter()
                .map(|d| {
                    let prev = d.candles[t - 1].close;
                    let curr = d.candles[t].close;
                    (curr - prev) / prev
                })
                .collect();

            // Update portfolio value with returns
            let portfolio_return: f64 = weights
                .iter()
                .zip(returns.iter())
                .map(|(w, r)| w * r)
                .sum();
            portfolio_value *= 1.0 + portfolio_return;

            // Update benchmark
            let benchmark_return: f64 = benchmark_weights
                .iter()
                .zip(returns.iter())
                .map(|(w, r)| w * r)
                .sum();
            benchmark_value *= 1.0 + benchmark_return;

            // Update weights due to price changes
            for i in 0..n_assets {
                weights[i] *= 1.0 + returns[i];
            }
            let weight_sum: f64 = weights.iter().sum();
            for w in &mut weights {
                *w /= weight_sum;
            }

            // Check if we should rebalance
            if t - last_rebalance >= self.config.min_rebalance_interval {
                // Calculate features for each asset
                let features = self.calculate_features(data, t);

                let decision = rebalancer.check_rebalance(&weights, &features);

                if decision.should_rebalance {
                    // Calculate transaction costs
                    let turnover: f64 = weights
                        .iter()
                        .zip(decision.target_weights.iter())
                        .map(|(w, t)| (w - t).abs())
                        .sum();
                    let cost = turnover * portfolio_value * self.config.transaction_cost;

                    portfolio_value -= cost;
                    total_costs += cost;

                    // Update to target weights
                    weights = decision.target_weights;
                    rebalance_times.push(t);
                    last_rebalance = t;
                }
            }

            portfolio_values.push(portfolio_value);
            benchmark_values.push(benchmark_value);
            weight_history.push(weights.clone());
        }

        self.calculate_metrics(
            portfolio_values,
            benchmark_values,
            rebalance_times,
            total_costs,
            weight_history,
        )
    }

    /// Calculate features for all assets at time t
    fn calculate_features(&self, data: &[CandleData], t: usize) -> Features {
        let n_assets = data.len();
        let lookback = 100.min(t);

        let mut all_features = Vec::new();
        let mut feature_names = Vec::new();

        for asset_data in data {
            let slice = asset_data.slice(t - lookback, t + 1);
            let asset_features = self.indicators.calculate_all(&slice);

            if feature_names.is_empty() {
                feature_names = asset_features.names.clone();
            }

            all_features.push(asset_features.data[0].clone());
        }

        Features {
            n_assets,
            n_features: feature_names.len(),
            data: all_features,
            names: feature_names,
        }
    }

    /// Calculate performance metrics
    fn calculate_metrics(
        &self,
        portfolio_values: Vec<f64>,
        benchmark_values: Vec<f64>,
        rebalance_times: Vec<usize>,
        total_costs: f64,
        weight_history: Vec<Vec<f64>>,
    ) -> BacktestResult {
        let initial = portfolio_values[0];
        let final_value = *portfolio_values.last().unwrap();

        let total_return = (final_value - initial) / initial;
        let benchmark_return = (benchmark_values.last().unwrap() - benchmark_values[0])
            / benchmark_values[0];

        // Calculate returns
        let returns: Vec<f64> = portfolio_values
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        // Annualized metrics (assuming hourly data)
        let periods_per_year = 365.0 * 24.0;
        let n_periods = returns.len() as f64;

        let mean_return: f64 = returns.iter().sum::<f64>() / n_periods;
        let variance: f64 = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / n_periods;
        let volatility = variance.sqrt();

        let annualized_return = (1.0 + mean_return).powf(periods_per_year) - 1.0;
        let annualized_volatility = volatility * periods_per_year.sqrt();

        let sharpe_ratio = if annualized_volatility > 0.0 {
            annualized_return / annualized_volatility
        } else {
            0.0
        };

        // Maximum drawdown
        let mut max_value = portfolio_values[0];
        let mut max_drawdown = 0.0;

        for &value in &portfolio_values {
            if value > max_value {
                max_value = value;
            }
            let drawdown = (max_value - value) / max_value;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        BacktestResult {
            portfolio_values,
            benchmark_values,
            total_return,
            benchmark_return,
            annualized_return,
            annualized_volatility,
            sharpe_ratio,
            max_drawdown,
            num_rebalances: rebalance_times.len(),
            total_costs,
            rebalance_times,
            weight_history,
        }
    }

    fn empty_result(&self, n_assets: usize) -> BacktestResult {
        BacktestResult {
            portfolio_values: vec![self.config.initial_value],
            benchmark_values: vec![self.config.initial_value],
            total_return: 0.0,
            benchmark_return: 0.0,
            annualized_return: 0.0,
            annualized_volatility: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            num_rebalances: 0,
            total_costs: 0.0,
            rebalance_times: Vec::new(),
            weight_history: vec![vec![1.0 / n_assets as f64; n_assets]],
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Candle, Timeframe};
    use crate::model::NeuralODEPortfolio;

    fn create_test_data(n_candles: usize, base_price: f64) -> CandleData {
        let mut candles = Vec::with_capacity(n_candles);

        for i in 0..n_candles {
            let price = base_price * (1.0 + 0.001 * (i as f64).sin());
            candles.push(Candle::new(
                i as i64 * 3600000,
                price,
                price * 1.01,
                price * 0.99,
                price * 1.005,
                1000.0,
                1000.0 * price,
            ));
        }

        CandleData::new("TEST".to_string(), Timeframe::Hour1, candles)
    }

    #[test]
    fn test_backtest_config() {
        let config = BacktestConfig::default();
        assert_eq!(config.initial_value, 100_000.0);
        assert!((config.transaction_cost - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_backtest_result_metrics() {
        let result = BacktestResult {
            portfolio_values: vec![100.0, 110.0, 105.0, 115.0],
            benchmark_values: vec![100.0, 102.0, 101.0, 103.0],
            total_return: 0.15,
            benchmark_return: 0.03,
            annualized_return: 0.5,
            annualized_volatility: 0.3,
            sharpe_ratio: 1.67,
            max_drawdown: 0.05,
            num_rebalances: 2,
            total_costs: 10.0,
            rebalance_times: vec![50, 150],
            weight_history: vec![vec![0.5, 0.5]],
        };

        let calmar = result.calmar_ratio();
        assert!(calmar > 0.0);
    }

    #[test]
    fn test_backtester() {
        let config = BacktestConfig::default();
        let backtester = Backtester::new(config);

        let model = NeuralODEPortfolio::new(2, 12, 8);
        let rebalancer = ContinuousRebalancer::new(model, 0.02);

        let data = vec![
            create_test_data(200, 100.0),
            create_test_data(200, 50.0),
        ];

        let result = backtester.run(&rebalancer, &data);

        assert!(!result.portfolio_values.is_empty());
        assert!(!result.benchmark_values.is_empty());
    }
}
