//! Backtesting engine for multi-task learning trading strategies.
//!
//! Simulates strategy execution on historical data and computes
//! performance metrics.

use crate::model::network::MultiTaskModel;
use crate::trading::signals::{SignalGenerator, TradingSignal};
use crate::data::features::FeatureRow;

/// Single step in the backtest.
#[derive(Debug, Clone)]
pub struct BacktestStep {
    pub timestamp: i64,
    pub price: f64,
    pub signal: TradingSignal,
    pub position: f64,
    pub position_return: f64,
    pub capital: f64,
}

/// Performance metrics from a backtest.
#[derive(Debug, Clone)]
pub struct BacktestMetrics {
    pub total_return: f64,
    pub annualized_return: f64,
    pub annualized_volatility: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub num_trades: usize,
}

/// Backtesting engine.
pub struct BacktestEngine {
    model: MultiTaskModel,
    signal_generator: SignalGenerator,
    transaction_cost: f64,
}

impl BacktestEngine {
    /// Create a new backtest engine.
    pub fn new(
        model: MultiTaskModel,
        return_threshold: f64,
        direction_threshold: f64,
        volatility_cap: f64,
        transaction_cost: f64,
    ) -> Self {
        Self {
            model,
            signal_generator: SignalGenerator::new(
                return_threshold,
                direction_threshold,
                volatility_cap,
            ),
            transaction_cost,
        }
    }

    /// Run backtest on historical feature data.
    ///
    /// # Arguments
    /// * `features` - Feature rows computed from historical klines
    /// * `prices` - Corresponding close prices
    /// * `initial_capital` - Starting capital
    pub fn run(
        &self,
        features: &[FeatureRow],
        prices: &[f64],
        initial_capital: f64,
    ) -> Vec<BacktestStep> {
        assert_eq!(features.len(), prices.len());
        let n = features.len();
        if n < 2 {
            return Vec::new();
        }

        let inputs: Vec<Vec<f64>> = features.iter().map(|f| f.features.clone()).collect();
        let predictions = self.model.forward_batch(&inputs);
        let signals = self.signal_generator.generate_batch(&predictions);

        let mut results = Vec::with_capacity(n - 1);
        let mut capital = initial_capital;
        let mut position = 0.0_f64;

        for i in 0..n - 1 {
            let signal = signals[i];
            let new_position = signal.position();

            // Transaction cost on position change
            if (new_position - position).abs() > 1e-10 {
                capital *= 1.0 - self.transaction_cost;
            }

            // Compute return
            let actual_return = prices[i + 1] / prices[i] - 1.0;
            let position_return = position * actual_return;
            capital *= 1.0 + position_return;

            results.push(BacktestStep {
                timestamp: features[i].timestamp,
                price: prices[i],
                signal,
                position,
                position_return,
                capital,
            });

            position = new_position;
        }

        results
    }

    /// Compute performance metrics from backtest results.
    pub fn compute_metrics(&self, results: &[BacktestStep]) -> BacktestMetrics {
        if results.is_empty() {
            return BacktestMetrics {
                total_return: 0.0,
                annualized_return: 0.0,
                annualized_volatility: 0.0,
                sharpe_ratio: 0.0,
                sortino_ratio: 0.0,
                max_drawdown: 0.0,
                win_rate: 0.0,
                profit_factor: 0.0,
                num_trades: 0,
            };
        }

        let returns: Vec<f64> = results.iter().map(|s| s.position_return).collect();
        let n = returns.len() as f64;

        let total_return = results.last().unwrap().capital / results[0].capital
            * (1.0 + results[0].position_return) - 1.0;
        let ann_return = (1.0 + total_return).powf(252.0 / n) - 1.0;

        let mean_ret = returns.iter().sum::<f64>() / n;
        let variance = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();
        let ann_volatility = std * (252.0_f64).sqrt();
        let sharpe = (252.0_f64).sqrt() * mean_ret / (std + 1e-10);

        let downside_var = returns.iter()
            .filter(|&&r| r < 0.0)
            .map(|r| r.powi(2))
            .sum::<f64>()
            / n.max(1.0);
        let sortino = (252.0_f64).sqrt() * mean_ret / (downside_var.sqrt() + 1e-10);

        // Max drawdown
        let mut peak = results[0].capital;
        let mut max_dd = 0.0_f64;
        for step in results {
            if step.capital > peak {
                peak = step.capital;
            }
            let dd = step.capital / peak - 1.0;
            if dd < max_dd {
                max_dd = dd;
            }
        }

        let wins = returns.iter().filter(|&&r| r > 0.0).count() as f64;
        let losses = returns.iter().filter(|&&r| r < 0.0).count() as f64;
        let win_rate = if wins + losses > 0.0 { wins / (wins + losses) } else { 0.0 };

        let gross_profits: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
        let gross_losses: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();
        let profit_factor = gross_profits / (gross_losses + 1e-10);

        let num_trades = results.windows(2)
            .filter(|w| (w[1].position - w[0].position).abs() > 1e-10)
            .count();

        BacktestMetrics {
            total_return,
            annualized_return: ann_return,
            annualized_volatility: ann_volatility,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            max_drawdown: max_dd,
            win_rate,
            profit_factor,
            num_trades,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::SimulatedDataGenerator;
    use crate::data::features::FeatureGenerator;

    #[test]
    fn test_backtest_runs() {
        let klines = SimulatedDataGenerator::generate_klines(200, 50000.0, 0.02);
        let gen = FeatureGenerator::new(20);
        let features = gen.generate_features(&klines);
        let n = features.len();

        // Map features to corresponding prices
        let feature_timestamps: Vec<i64> = features.iter().map(|f| f.timestamp).collect();
        let prices: Vec<f64> = klines.iter()
            .filter(|k| feature_timestamps.contains(&k.timestamp))
            .map(|k| k.close)
            .collect();

        let prices = &prices[..n.min(prices.len())];
        let features = &features[..prices.len()];

        let model = MultiTaskModel::new(10, &[16, 8], 4);
        let engine = BacktestEngine::new(model, 0.001, 0.55, 0.05, 0.001);
        let results = engine.run(features, prices, 10000.0);

        assert!(!results.is_empty());
        assert!(results.last().unwrap().capital > 0.0);
    }

    #[test]
    fn test_metrics_computation() {
        let steps = vec![
            BacktestStep {
                timestamp: 0, price: 100.0,
                signal: TradingSignal::Long, position: 1.0,
                position_return: 0.01, capital: 10100.0,
            },
            BacktestStep {
                timestamp: 1, price: 101.0,
                signal: TradingSignal::Long, position: 1.0,
                position_return: -0.005, capital: 10049.5,
            },
            BacktestStep {
                timestamp: 2, price: 100.5,
                signal: TradingSignal::Neutral, position: 0.0,
                position_return: 0.0, capital: 10049.5,
            },
        ];

        let model = MultiTaskModel::new(5, &[8], 4);
        let engine = BacktestEngine::new(model, 0.001, 0.55, 0.05, 0.001);
        let metrics = engine.compute_metrics(&steps);

        assert!(metrics.total_return.is_finite());
        assert!(metrics.sharpe_ratio.is_finite());
        assert!(metrics.max_drawdown <= 0.0);
    }
}
