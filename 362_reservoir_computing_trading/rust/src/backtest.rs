//! # Backtesting Engine
//!
//! Walk-forward backtesting framework for reservoir computing strategies.
//!
//! ## Features
//!
//! - Walk-forward validation
//! - Comprehensive performance metrics
//! - Transaction cost modeling
//! - Drawdown analysis

use crate::bybit::Kline;
use crate::features::{extract_features_batch, FeatureExtractor, FeatureScaler, MarketFeatures};
use crate::reservoir::{EchoStateNetwork, EsnConfig};
use crate::trading::{Position, Signal, Trade, TradingConfig, TradingSystem};
use ndarray::{Array1, Array2, s};
use std::collections::HashMap;

/// Backtesting configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Fraction of data to use for initial training
    pub train_ratio: f64,

    /// Washout period for ESN
    pub washout: usize,

    /// Initial capital
    pub initial_capital: f64,

    /// Whether to retrain periodically
    pub rolling_retrain: bool,

    /// Retrain every N bars (if rolling_retrain is true)
    pub retrain_interval: usize,

    /// ESN configuration
    pub esn_config: EsnConfig,

    /// Trading configuration
    pub trading_config: TradingConfig,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            train_ratio: 0.6,
            washout: 100,
            initial_capital: 10000.0,
            rolling_retrain: false,
            retrain_interval: 500,
            esn_config: EsnConfig::default(),
            trading_config: TradingConfig::default(),
        }
    }
}

/// Performance metrics
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Total return
    pub total_return: f64,

    /// Annualized return (assuming daily data)
    pub annualized_return: f64,

    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,

    /// Sortino ratio (annualized)
    pub sortino_ratio: f64,

    /// Maximum drawdown
    pub max_drawdown: f64,

    /// Win rate
    pub win_rate: f64,

    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,

    /// Number of trades
    pub n_trades: usize,

    /// Average trade return
    pub avg_trade_return: f64,

    /// Calmar ratio (annualized return / max drawdown)
    pub calmar_ratio: f64,

    /// Volatility (annualized)
    pub volatility: f64,

    /// Skewness of returns
    pub skewness: f64,

    /// Kurtosis of returns
    pub kurtosis: f64,
}

impl PerformanceMetrics {
    /// Calculate metrics from returns
    pub fn from_returns(returns: &[f64], n_trades: usize, trades: &[TradeRecord]) -> Self {
        let n = returns.len();
        if n == 0 {
            return Self::default();
        }

        // Total and annualized return
        let cumulative: Vec<f64> = returns
            .iter()
            .scan(1.0, |acc, &r| {
                *acc *= 1.0 + r;
                Some(*acc)
            })
            .collect();

        let total_return = cumulative.last().unwrap_or(&1.0) - 1.0;
        let annualized_return = (1.0 + total_return).powf(252.0 / n as f64) - 1.0;

        // Mean and std
        let mean = returns.iter().sum::<f64>() / n as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64;
        let std = variance.sqrt();
        let volatility = std * (252.0_f64).sqrt();

        // Sharpe ratio
        let sharpe_ratio = if std > 1e-10 {
            mean / std * (252.0_f64).sqrt()
        } else {
            0.0
        };

        // Sortino ratio
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_std = if downside_returns.len() > 1 {
            let ds_var = downside_returns
                .iter()
                .map(|r| r.powi(2))
                .sum::<f64>()
                / downside_returns.len() as f64;
            ds_var.sqrt()
        } else {
            1e-10
        };
        let sortino_ratio = mean / downside_std * (252.0_f64).sqrt();

        // Maximum drawdown
        let mut peak = cumulative[0];
        let mut max_drawdown = 0.0;
        for &val in &cumulative {
            if val > peak {
                peak = val;
            }
            let drawdown = (peak - val) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        // Calmar ratio
        let calmar_ratio = if max_drawdown > 1e-10 {
            annualized_return / max_drawdown
        } else {
            0.0
        };

        // Trade statistics
        let (win_rate, profit_factor, avg_trade_return) = if !trades.is_empty() {
            let winning: Vec<&TradeRecord> = trades.iter().filter(|t| t.pnl > 0.0).collect();
            let losing: Vec<&TradeRecord> = trades.iter().filter(|t| t.pnl < 0.0).collect();

            let win_rate = winning.len() as f64 / trades.len() as f64;

            let gross_profit: f64 = winning.iter().map(|t| t.pnl).sum();
            let gross_loss: f64 = losing.iter().map(|t| -t.pnl).sum();
            let profit_factor = if gross_loss > 1e-10 {
                gross_profit / gross_loss
            } else if gross_profit > 0.0 {
                f64::INFINITY
            } else {
                1.0
            };

            let avg_return = trades.iter().map(|t| t.pnl).sum::<f64>() / trades.len() as f64;

            (win_rate, profit_factor, avg_return)
        } else {
            (0.0, 1.0, 0.0)
        };

        // Higher moments
        let skewness = if n > 2 && std > 1e-10 {
            let m3 = returns.iter().map(|r| ((r - mean) / std).powi(3)).sum::<f64>() / n as f64;
            m3
        } else {
            0.0
        };

        let kurtosis = if n > 3 && std > 1e-10 {
            let m4 = returns.iter().map(|r| ((r - mean) / std).powi(4)).sum::<f64>() / n as f64;
            m4 - 3.0 // Excess kurtosis
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
            profit_factor,
            n_trades,
            avg_trade_return,
            calmar_ratio,
            volatility,
            skewness,
            kurtosis,
        }
    }

    /// Print metrics summary
    pub fn summary(&self) -> String {
        format!(
            "Performance Summary:\n\
             ─────────────────────────────────\n\
             Total Return:      {:>10.2}%\n\
             Annualized Return: {:>10.2}%\n\
             Sharpe Ratio:      {:>10.2}\n\
             Sortino Ratio:     {:>10.2}\n\
             Max Drawdown:      {:>10.2}%\n\
             Win Rate:          {:>10.2}%\n\
             Profit Factor:     {:>10.2}\n\
             Number of Trades:  {:>10}\n\
             Avg Trade Return:  {:>10.4}%\n\
             Calmar Ratio:      {:>10.2}\n\
             Volatility (Ann.): {:>10.2}%\n\
             ─────────────────────────────────",
            self.total_return * 100.0,
            self.annualized_return * 100.0,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown * 100.0,
            self.win_rate * 100.0,
            self.profit_factor,
            self.n_trades,
            self.avg_trade_return * 100.0,
            self.calmar_ratio,
            self.volatility * 100.0,
        )
    }
}

/// Trade record for analysis
#[derive(Debug, Clone)]
pub struct TradeRecord {
    /// Entry bar index
    pub entry_bar: usize,

    /// Exit bar index
    pub exit_bar: usize,

    /// Entry price
    pub entry_price: f64,

    /// Exit price
    pub exit_price: f64,

    /// Position size
    pub size: f64,

    /// Profit/loss
    pub pnl: f64,

    /// Return percentage
    pub return_pct: f64,

    /// Hold duration in bars
    pub duration: usize,
}

/// Backtest result
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Performance metrics
    pub metrics: PerformanceMetrics,

    /// Equity curve
    pub equity_curve: Vec<f64>,

    /// Returns series
    pub returns: Vec<f64>,

    /// Positions over time
    pub positions: Vec<f64>,

    /// Signals over time
    pub signals: Vec<f64>,

    /// Trade records
    pub trades: Vec<TradeRecord>,

    /// Predictions
    pub predictions: Vec<f64>,

    /// Actual returns (for comparison)
    pub actual_returns: Vec<f64>,
}

/// Backtester for reservoir computing strategies
pub struct Backtester {
    config: BacktestConfig,
}

impl Backtester {
    /// Create a new backtester
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run backtest on kline data
    pub fn run(&self, klines: &[Kline]) -> BacktestResult {
        // Extract features
        let (features, targets) = extract_features_batch(klines);
        let n_samples = features.nrows();

        if n_samples < self.config.washout + 10 {
            return BacktestResult {
                metrics: PerformanceMetrics::default(),
                equity_curve: vec![self.config.initial_capital],
                returns: vec![],
                positions: vec![],
                signals: vec![],
                trades: vec![],
                predictions: vec![],
                actual_returns: vec![],
            };
        }

        // Scale features
        let mut scaler = FeatureScaler::new();
        let scaled_features = scaler.fit_transform(&features);

        // Split data
        let train_size = (n_samples as f64 * self.config.train_ratio) as usize;
        let test_start = train_size;

        let train_features = scaled_features.slice(s![..train_size, ..]).to_owned();
        let train_targets = targets.slice(s![..train_size]).to_owned();

        // Train ESN
        let n_features = MarketFeatures::n_features();
        let mut esn = EchoStateNetwork::new(n_features, 1, self.config.esn_config.clone());

        let train_targets_2d = train_targets.clone().insert_axis(ndarray::Axis(1));
        let _ = esn.fit(&train_features, &train_targets_2d, self.config.washout);

        // Initialize trading system
        let mut trading = TradingSystem::new(
            self.config.trading_config.clone(),
            self.config.initial_capital,
        );

        // Run backtest
        let mut equity_curve = vec![self.config.initial_capital];
        let mut returns = Vec::new();
        let mut positions = Vec::new();
        let mut signals = Vec::new();
        let mut predictions = Vec::new();
        let mut actual_returns = Vec::new();
        let mut trades = Vec::new();

        let mut current_trade_entry: Option<(usize, f64, f64)> = None;

        // Reset ESN state
        esn.reset_state();
        // Warm up on training data
        for i in 0..train_size {
            let input = scaled_features.row(i).to_owned();
            let _ = esn.predict_one(&input);
        }

        // Test phase
        for i in test_start..n_samples {
            let input = scaled_features.row(i).to_owned();
            let actual_return = targets[i];

            // Get prediction
            let prediction = esn.predict_one(&input).unwrap_or_else(|_| Array1::zeros(1));
            let pred_value = prediction[0];

            predictions.push(pred_value);
            actual_returns.push(actual_return);

            // Get price (from features or use index)
            let price_idx = i - (n_samples - klines.len());
            let current_price = if price_idx < klines.len() {
                klines[price_idx].close
            } else {
                100.0 // Fallback
            };

            // Execute trading decision
            let prev_position = trading.position().size;

            if let Ok(Some(trade)) = trading.execute(pred_value.tanh(), current_price) {
                // Record trade
                if let Some((entry_bar, entry_price, size)) = current_trade_entry.take() {
                    trades.push(TradeRecord {
                        entry_bar,
                        exit_bar: i,
                        entry_price,
                        exit_price: current_price,
                        size,
                        pnl: size * (current_price - entry_price),
                        return_pct: (current_price / entry_price - 1.0) * size.signum(),
                        duration: i - entry_bar,
                    });
                }

                if !trading.position().is_flat() {
                    current_trade_entry = Some((i, current_price, trading.position().size));
                }
            }

            // Record position and signal
            positions.push(trading.position().size);
            signals.push(pred_value.tanh());

            // Calculate return
            let position_return = prev_position * actual_return;
            returns.push(position_return);

            // Update equity
            let new_equity = equity_curve.last().unwrap() * (1.0 + position_return);
            equity_curve.push(new_equity);
        }

        // Calculate metrics
        let metrics = PerformanceMetrics::from_returns(&returns, trades.len(), &trades);

        BacktestResult {
            metrics,
            equity_curve,
            returns,
            positions,
            signals,
            trades,
            predictions,
            actual_returns,
        }
    }

    /// Run walk-forward backtest with periodic retraining
    pub fn run_walk_forward(&self, klines: &[Kline], window_size: usize) -> BacktestResult {
        let (features, targets) = extract_features_batch(klines);
        let n_samples = features.nrows();

        if n_samples < window_size + self.config.washout + 10 {
            return self.run(klines);
        }

        let mut scaler = FeatureScaler::new();

        let mut all_returns = Vec::new();
        let mut all_positions = Vec::new();
        let mut all_signals = Vec::new();
        let mut all_predictions = Vec::new();
        let mut all_actual_returns = Vec::new();
        let mut all_trades = Vec::new();
        let mut equity_curve = vec![self.config.initial_capital];

        let mut trading = TradingSystem::new(
            self.config.trading_config.clone(),
            self.config.initial_capital,
        );

        let n_features = MarketFeatures::n_features();
        let retrain_interval = self.config.retrain_interval;

        let mut last_train_end = 0;
        let mut esn: Option<EchoStateNetwork> = None;

        for i in window_size..n_samples {
            // Retrain if needed
            if esn.is_none() || (i - last_train_end >= retrain_interval && self.config.rolling_retrain) {
                let train_start = if i > window_size { i - window_size } else { 0 };
                let train_end = i;

                let train_features = features.slice(s![train_start..train_end, ..]).to_owned();
                let train_targets = targets.slice(s![train_start..train_end]).to_owned();

                scaler.fit(&train_features);
                let scaled_train = scaler.transform(&train_features);

                let mut new_esn = EchoStateNetwork::new(n_features, 1, self.config.esn_config.clone());
                let train_targets_2d = train_targets.insert_axis(ndarray::Axis(1));
                let _ = new_esn.fit(&scaled_train, &train_targets_2d, self.config.washout);

                esn = Some(new_esn);
                last_train_end = i;
            }

            let esn_ref = esn.as_mut().unwrap();

            let input = scaler.transform_one(&features.row(i).to_owned());
            let actual_return = targets[i];

            let prediction = esn_ref.predict_one(&input).unwrap_or_else(|_| Array1::zeros(1));
            let pred_value = prediction[0];

            all_predictions.push(pred_value);
            all_actual_returns.push(actual_return);

            let price_idx = i;
            let current_price = if price_idx < klines.len() {
                klines[price_idx].close
            } else {
                100.0
            };

            let prev_position = trading.position().size;
            let _ = trading.execute(pred_value.tanh(), current_price);

            all_positions.push(trading.position().size);
            all_signals.push(pred_value.tanh());

            let position_return = prev_position * actual_return;
            all_returns.push(position_return);

            let new_equity = equity_curve.last().unwrap() * (1.0 + position_return);
            equity_curve.push(new_equity);
        }

        let metrics = PerformanceMetrics::from_returns(&all_returns, all_trades.len(), &all_trades);

        BacktestResult {
            metrics,
            equity_curve,
            returns: all_returns,
            positions: all_positions,
            signals: all_signals,
            trades: all_trades,
            predictions: all_predictions,
            actual_returns: all_actual_returns,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| {
                let base = 100.0 + (i as f64 * 0.01).sin() * 10.0;
                Kline {
                    start_time: i as u64 * 60000,
                    open: base,
                    high: base + 1.0,
                    low: base - 1.0,
                    close: base + 0.5,
                    volume: 1000.0 + (i as f64 * 0.1).cos() * 100.0,
                    turnover: base * 1000.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_performance_metrics() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.003];
        let trades = vec![];
        let metrics = PerformanceMetrics::from_returns(&returns, 0, &trades);

        assert!(metrics.total_return > 0.0);
        assert!(metrics.sharpe_ratio != 0.0);
    }

    #[test]
    fn test_backtester() {
        let klines = generate_test_klines(500);
        let config = BacktestConfig {
            esn_config: EsnConfig {
                reservoir_size: 50,
                ..EsnConfig::default()
            },
            ..BacktestConfig::default()
        };

        let backtester = Backtester::new(config);
        let result = backtester.run(&klines);

        assert!(!result.equity_curve.is_empty());
        assert!(!result.returns.is_empty());
    }

    #[test]
    fn test_metrics_summary() {
        let metrics = PerformanceMetrics {
            total_return: 0.25,
            annualized_return: 0.35,
            sharpe_ratio: 1.5,
            sortino_ratio: 2.1,
            max_drawdown: 0.12,
            win_rate: 0.55,
            profit_factor: 1.8,
            n_trades: 42,
            avg_trade_return: 0.006,
            calmar_ratio: 2.9,
            volatility: 0.15,
            skewness: 0.2,
            kurtosis: 0.5,
        };

        let summary = metrics.summary();
        assert!(summary.contains("25.00%"));
        assert!(summary.contains("1.50"));
    }
}
