//! Backtesting module for Neural Spline Flow trading strategies
//!
//! This module provides a comprehensive backtesting framework for evaluating
//! trading strategies based on Neural Spline Flows.

use crate::flow::NeuralSplineFlow;
use crate::trading::{SignalGenerator, SignalGeneratorConfig, TradingSignal};
use crate::utils::{extract_features, Candle, FeatureVector};
use chrono::{DateTime, Utc};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Backtesting configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Transaction cost per trade (as fraction)
    pub transaction_cost: f64,
    /// Slippage per trade (as fraction)
    pub slippage: f64,
    /// Lookback period for feature extraction
    pub lookback: usize,
    /// Warmup period (skip first N bars)
    pub warmup: usize,
    /// Maximum position size (as fraction of capital)
    pub max_position: f64,
    /// Signal generator configuration
    pub signal_config: SignalGeneratorConfig,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            transaction_cost: 0.001,
            slippage: 0.0005,
            lookback: 20,
            warmup: 50,
            max_position: 1.0,
            signal_config: SignalGeneratorConfig::default(),
        }
    }
}

/// Single trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
    /// Exit timestamp
    pub exit_time: DateTime<Utc>,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Position size (-1 to 1)
    pub position_size: f64,
    /// Profit/Loss
    pub pnl: f64,
    /// Return percentage
    pub return_pct: f64,
}

/// Bar-by-bar backtest result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarResult {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Close price
    pub price: f64,
    /// Trading signal
    pub signal: f64,
    /// Confidence
    pub confidence: f64,
    /// Log probability
    pub log_prob: f64,
    /// Whether in distribution
    pub in_distribution: bool,
    /// Current position
    pub position: f64,
    /// Bar PnL
    pub pnl: f64,
    /// Cumulative PnL
    pub cumulative_pnl: f64,
    /// Equity
    pub equity: f64,
}

/// Complete backtest result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    /// Bar-by-bar results
    pub bars: Vec<BarResult>,
    /// List of trades
    pub trades: Vec<Trade>,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
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
    /// Number of trades
    pub num_trades: usize,
    /// Average trade return
    pub avg_trade_return: f64,
    /// In-distribution ratio
    pub in_distribution_ratio: f64,
    /// Average log probability
    pub avg_log_prob: f64,
    /// Average confidence
    pub avg_confidence: f64,
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
            profit_factor: 0.0,
            num_trades: 0,
            avg_trade_return: 0.0,
            in_distribution_ratio: 0.0,
            avg_log_prob: 0.0,
            avg_confidence: 0.0,
        }
    }
}

/// Backtesting engine
pub struct BacktestEngine {
    /// NSF model
    model: NeuralSplineFlow,
    /// Configuration
    config: BacktestConfig,
}

impl BacktestEngine {
    /// Create a new backtest engine
    pub fn new(model: NeuralSplineFlow, config: BacktestConfig) -> Self {
        Self { model, config }
    }

    /// Run backtest on candle data
    pub fn run(&self, candles: &[Candle]) -> BacktestResult {
        let signal_generator = SignalGenerator::new(
            self.model.clone(),
            self.config.signal_config.clone(),
        );

        let mut bars = Vec::with_capacity(candles.len());
        let mut trades = Vec::new();

        let mut position = 0.0;
        let mut cumulative_pnl = 0.0;
        let mut equity = self.config.initial_capital;
        let mut entry_price = 0.0;
        let mut entry_time = candles[0].timestamp;

        for i in self.config.warmup..candles.len() {
            // Extract features from lookback window
            let start = i.saturating_sub(self.config.lookback);
            let window = &candles[start..i];

            if window.len() < self.config.lookback {
                continue;
            }

            let features = extract_features(window);
            let feature_array = Array1::from_vec(features.values.clone());

            // Generate signal
            let signal_info = signal_generator.generate_signal(&feature_array);

            // Calculate PnL from previous position
            let daily_return = if i > self.config.warmup {
                candles[i].close / candles[i - 1].close - 1.0
            } else {
                0.0
            };

            let pnl = position * daily_return * equity;

            // Apply transaction costs if position changes
            let position_change = (signal_info.position_size - position).abs();
            let costs = position_change * self.config.transaction_cost * equity;
            let slippage = position_change * self.config.slippage * equity;

            let net_pnl = pnl - costs - slippage;
            cumulative_pnl += net_pnl;
            equity += net_pnl;

            // Record trade if position changed
            if position != 0.0 && signal_info.position_size == 0.0 {
                // Closing position
                trades.push(Trade {
                    entry_time,
                    exit_time: candles[i].timestamp,
                    entry_price,
                    exit_price: candles[i].close,
                    position_size: position,
                    pnl: (candles[i].close / entry_price - 1.0) * position,
                    return_pct: (candles[i].close / entry_price - 1.0) * position * 100.0,
                });
            } else if position == 0.0 && signal_info.position_size != 0.0 {
                // Opening position
                entry_price = candles[i].close;
                entry_time = candles[i].timestamp;
            }

            // Update position
            position = signal_info.position_size;

            // Record bar result
            bars.push(BarResult {
                timestamp: candles[i].timestamp,
                price: candles[i].close,
                signal: signal_info.signal,
                confidence: signal_info.confidence,
                log_prob: signal_info.log_prob,
                in_distribution: signal_info.in_distribution,
                position,
                pnl: net_pnl,
                cumulative_pnl,
                equity,
            });
        }

        // Calculate metrics
        let metrics = self.calculate_metrics(&bars, &trades);

        BacktestResult {
            bars,
            trades,
            metrics,
        }
    }

    /// Calculate performance metrics
    fn calculate_metrics(&self, bars: &[BarResult], trades: &[Trade]) -> PerformanceMetrics {
        if bars.is_empty() {
            return PerformanceMetrics::default();
        }

        // Total return
        let total_return = bars.last().unwrap().cumulative_pnl / self.config.initial_capital;

        // Daily returns for Sharpe/Sortino
        let daily_returns: Vec<f64> = bars
            .iter()
            .map(|b| b.pnl / self.config.initial_capital)
            .collect();

        let mean_return = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
        let variance =
            daily_returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / daily_returns.len() as f64;
        let std = variance.sqrt();

        // Sharpe ratio (annualized)
        let sharpe_ratio = if std > 0.0 {
            mean_return / std * (252.0_f64).sqrt()
        } else {
            0.0
        };

        // Sortino ratio
        let downside_returns: Vec<f64> = daily_returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_std = if !downside_returns.is_empty() {
            let downside_var =
                downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / downside_returns.len() as f64;
            downside_var.sqrt()
        } else {
            0.0
        };

        let sortino_ratio = if downside_std > 0.0 {
            mean_return / downside_std * (252.0_f64).sqrt()
        } else {
            0.0
        };

        // Maximum drawdown
        let mut peak = self.config.initial_capital;
        let mut max_drawdown = 0.0;
        for bar in bars {
            if bar.equity > peak {
                peak = bar.equity;
            }
            let drawdown = (peak - bar.equity) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        // Trade statistics
        let winning_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl < 0.0).collect();

        let win_rate = if !trades.is_empty() {
            winning_trades.len() as f64 / trades.len() as f64
        } else {
            0.0
        };

        let gross_profit: f64 = winning_trades.iter().map(|t| t.pnl).sum();
        let gross_loss: f64 = losing_trades.iter().map(|t| t.pnl.abs()).sum();

        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        let avg_trade_return = if !trades.is_empty() {
            trades.iter().map(|t| t.return_pct).sum::<f64>() / trades.len() as f64
        } else {
            0.0
        };

        // Distribution metrics
        let in_dist_count = bars.iter().filter(|b| b.in_distribution).count();
        let in_distribution_ratio = in_dist_count as f64 / bars.len() as f64;

        let avg_log_prob = bars.iter().map(|b| b.log_prob).sum::<f64>() / bars.len() as f64;
        let avg_confidence = bars.iter().map(|b| b.confidence).sum::<f64>() / bars.len() as f64;

        // Annualized return
        let days = bars.len() as f64;
        let years = days / 252.0;
        let annualized_return = if years > 0.0 {
            (1.0 + total_return).powf(1.0 / years) - 1.0
        } else {
            0.0
        };

        PerformanceMetrics {
            total_return,
            annualized_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            num_trades: trades.len(),
            avg_trade_return,
            in_distribution_ratio,
            avg_log_prob,
            avg_confidence,
        }
    }

    /// Get reference to the model
    pub fn model(&self) -> &NeuralSplineFlow {
        &self.model
    }

    /// Get configuration
    pub fn config(&self) -> &BacktestConfig {
        &self.config
    }
}

/// Print backtest summary to console
pub fn print_summary(result: &BacktestResult) {
    let m = &result.metrics;

    println!("\n=== Backtest Results ===\n");
    println!("Total Return:        {:>10.2}%", m.total_return * 100.0);
    println!("Annualized Return:   {:>10.2}%", m.annualized_return * 100.0);
    println!("Sharpe Ratio:        {:>10.2}", m.sharpe_ratio);
    println!("Sortino Ratio:       {:>10.2}", m.sortino_ratio);
    println!("Max Drawdown:        {:>10.2}%", m.max_drawdown * 100.0);
    println!();
    println!("Number of Trades:    {:>10}", m.num_trades);
    println!("Win Rate:            {:>10.2}%", m.win_rate * 100.0);
    println!("Profit Factor:       {:>10.2}", m.profit_factor);
    println!("Avg Trade Return:    {:>10.2}%", m.avg_trade_return);
    println!();
    println!("In-Distribution:     {:>10.2}%", m.in_distribution_ratio * 100.0);
    println!("Avg Log Probability: {:>10.2}", m.avg_log_prob);
    println!("Avg Confidence:      {:>10.2}", m.avg_confidence);
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::NSFConfig;

    fn create_test_candles(n: usize) -> Vec<Candle> {
        let now = Utc::now();
        (0..n)
            .map(|i| {
                let price = 100.0 + (i as f64 * 0.01).sin() * 10.0;
                Candle {
                    timestamp: now + chrono::Duration::hours(i as i64),
                    open: price - 0.5,
                    high: price + 1.0,
                    low: price - 1.0,
                    close: price,
                    volume: 1000.0 + (i as f64 * 0.1).cos() * 100.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_backtest_engine() {
        let nsf_config = NSFConfig::new(10);
        let model = NeuralSplineFlow::new(nsf_config);

        let config = BacktestConfig {
            warmup: 30,
            lookback: 20,
            ..Default::default()
        };

        let engine = BacktestEngine::new(model, config);
        let candles = create_test_candles(100);

        let result = engine.run(&candles);

        assert!(!result.bars.is_empty());
        assert!(result.metrics.total_return.is_finite());
    }

    #[test]
    fn test_performance_metrics() {
        let nsf_config = NSFConfig::new(10);
        let model = NeuralSplineFlow::new(nsf_config);
        let config = BacktestConfig::default();
        let engine = BacktestEngine::new(model, config);

        let candles = create_test_candles(200);
        let result = engine.run(&candles);

        // Metrics should be finite
        assert!(result.metrics.sharpe_ratio.is_finite());
        assert!(result.metrics.max_drawdown >= 0.0);
        assert!(result.metrics.win_rate >= 0.0 && result.metrics.win_rate <= 1.0);
    }
}
