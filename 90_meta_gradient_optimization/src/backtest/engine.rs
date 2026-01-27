//! Backtesting engine for meta-gradient optimization strategies.
//!
//! Provides tools to evaluate trading performance with
//! adaptive hyperparameter optimization.

use crate::data::bybit::Kline;
use crate::data::features::FeatureGenerator;
use crate::meta_gradient::optimizer::{MetaGradientOptimizer, TaskData};
use crate::trading::signals::TradingSignal;

/// Single backtest result entry
#[derive(Debug, Clone)]
pub struct BacktestEntry {
    pub timestamp: i64,
    pub price: f64,
    pub prediction: f64,
    pub actual_return: f64,
    pub position: f64,
    pub position_return: f64,
    pub capital: f64,
    pub learning_rate: f64,
}

/// Backtest performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
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

/// Backtesting engine for meta-gradient strategies
pub struct BacktestEngine {
    optimizer: MetaGradientOptimizer,
    feature_generator: FeatureGenerator,
    prediction_threshold: f64,
    transaction_cost: f64,
    adaptation_window: usize,
    validation_window: usize,
}

impl BacktestEngine {
    /// Create a new backtest engine
    pub fn new(
        optimizer: MetaGradientOptimizer,
        feature_generator: FeatureGenerator,
        prediction_threshold: f64,
        transaction_cost: f64,
        adaptation_window: usize,
        validation_window: usize,
    ) -> Self {
        Self {
            optimizer,
            feature_generator,
            prediction_threshold,
            transaction_cost,
            adaptation_window,
            validation_window,
        }
    }

    /// Run backtest on historical kline data
    pub fn run(&mut self, klines: &[Kline], initial_capital: f64) -> Vec<BacktestEntry> {
        let features = self.feature_generator.compute_features(klines);
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let timestamps: Vec<i64> = klines.iter().map(|k| k.timestamp).collect();

        if features.is_empty() {
            return Vec::new();
        }

        let total_window = self.adaptation_window + self.validation_window;
        let mut results = Vec::new();
        let mut capital = initial_capital;
        let mut position = 0.0;

        // Compute returns for labels
        let returns: Vec<f64> = closes
            .windows(2)
            .map(|w| (w[1] / w[0]) - 1.0)
            .collect();

        let offset = klines.len() - features.len();

        for i in total_window..features.len().saturating_sub(1) {
            if i >= returns.len() {
                break;
            }

            let train_start = i - total_window;
            let val_start = i - self.validation_window;

            // Create task
            let task = TaskData::new(
                features[train_start..val_start].to_vec(),
                returns[train_start + offset..val_start + offset]
                    .iter()
                    .copied()
                    .collect(),
                features[val_start..i].to_vec(),
                returns[val_start + offset..i + offset]
                    .iter()
                    .copied()
                    .collect(),
            );

            // Meta-gradient update
            self.optimizer.meta_train_step(&[task.clone()]);

            // Adapt and predict
            let adapted = self.optimizer.adapt(
                &task.train_features,
                &task.train_labels,
                None,
            );
            let prediction = adapted.predict(&features[i]);

            // Trading signal
            let signal = TradingSignal::from_prediction(prediction, self.prediction_threshold);
            let new_position = signal.position();

            // Transaction costs
            if (new_position - position).abs() > 1e-10 {
                capital *= 1.0 - self.transaction_cost;
            }

            // Calculate return
            let actual_return = if i + offset + 1 < closes.len() {
                closes[i + offset + 1] / closes[i + offset] - 1.0
            } else {
                0.0
            };
            let position_return = position * actual_return;
            capital *= 1.0 + position_return;

            let lr = self.optimizer.get_learning_rates()[0];

            results.push(BacktestEntry {
                timestamp: if i + offset < timestamps.len() {
                    timestamps[i + offset]
                } else {
                    0
                },
                price: if i + offset < closes.len() {
                    closes[i + offset]
                } else {
                    0.0
                },
                prediction,
                actual_return,
                position,
                position_return,
                capital,
                learning_rate: lr,
            });

            position = new_position;
        }

        results
    }

    /// Calculate performance metrics from backtest results
    pub fn calculate_metrics(results: &[BacktestEntry]) -> PerformanceMetrics {
        if results.is_empty() {
            return PerformanceMetrics {
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

        let returns: Vec<f64> = results.iter().map(|r| r.position_return).collect();
        let n = returns.len() as f64;

        let total_return = results.last().unwrap().capital / results[0].capital - 1.0;
        let ann_return = (1.0 + total_return).powf(252.0 / n) - 1.0;

        let mean_return = returns.iter().sum::<f64>() / n;
        let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / n;
        let ann_volatility = variance.sqrt() * (252.0_f64).sqrt();

        let sharpe_ratio = if ann_volatility > 0.0 {
            (252.0_f64).sqrt() * mean_return / variance.sqrt()
        } else {
            0.0
        };

        let downside_returns: Vec<f64> = returns.iter().filter(|r| **r < 0.0).copied().collect();
        let downside_variance = if !downside_returns.is_empty() {
            downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64
        } else {
            1e-10
        };
        let sortino_ratio = (252.0_f64).sqrt() * mean_return / downside_variance.sqrt();

        // Maximum drawdown
        let mut peak = results[0].capital;
        let mut max_drawdown = 0.0_f64;
        for r in results {
            if r.capital > peak {
                peak = r.capital;
            }
            let drawdown = (peak - r.capital) / peak;
            max_drawdown = max_drawdown.max(drawdown);
        }

        let wins = returns.iter().filter(|r| **r > 0.0).count();
        let losses = returns.iter().filter(|r| **r < 0.0).count();
        let win_rate = if wins + losses > 0 {
            wins as f64 / (wins + losses) as f64
        } else {
            0.0
        };

        let gross_profits: f64 = returns.iter().filter(|r| **r > 0.0).sum();
        let gross_losses: f64 = returns.iter().filter(|r| **r < 0.0).map(|r| r.abs()).sum();
        let profit_factor = if gross_losses > 0.0 {
            gross_profits / gross_losses
        } else {
            0.0
        };

        let num_trades = results
            .iter()
            .filter(|r| r.position.abs() > 1e-10)
            .count();

        PerformanceMetrics {
            total_return,
            annualized_return: ann_return,
            annualized_volatility: ann_volatility,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
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
    use crate::model::network::TradingModel;

    #[test]
    fn test_backtest_engine() {
        let model = TradingModel::new(11, 16, 1);
        let optimizer = MetaGradientOptimizer::new(model, 0.01, 0.001, 2, false, false);
        let feature_gen = FeatureGenerator::new(20);

        let mut engine = BacktestEngine::new(optimizer, feature_gen, 0.001, 0.001, 20, 5);

        let klines = SimulatedDataGenerator::generate_klines(200, 50000.0, 0.02);
        let results = engine.run(&klines, 10000.0);

        // Should have some results
        assert!(!results.is_empty());

        // Capital should be positive
        for r in &results {
            assert!(r.capital > 0.0);
        }
    }

    #[test]
    fn test_performance_metrics() {
        let entries = vec![
            BacktestEntry {
                timestamp: 0,
                price: 100.0,
                prediction: 0.01,
                actual_return: 0.01,
                position: 1.0,
                position_return: 0.01,
                capital: 10100.0,
                learning_rate: 0.01,
            },
            BacktestEntry {
                timestamp: 1,
                price: 101.0,
                prediction: -0.005,
                actual_return: -0.005,
                position: -1.0,
                position_return: 0.005,
                capital: 10150.5,
                learning_rate: 0.01,
            },
        ];

        let metrics = BacktestEngine::calculate_metrics(&entries);
        assert!(metrics.total_return > 0.0);
        assert!(metrics.win_rate > 0.0);
    }
}
