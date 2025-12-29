//! Backtesting Engine
//!
//! Run strategy backtests on historical data

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::data::OHLCVSeries;
use crate::features::FeatureEngine;
use crate::nn::NeuralNetwork;
use crate::strategy::{Signal, SignalGenerator, TradingStrategy, StrategyConfig};
use super::metrics::BacktestMetrics;

/// Backtester configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Trading commission (percentage)
    pub commission: f64,
    /// Slippage (percentage)
    pub slippage: f64,
    /// Target horizon for predictions
    pub target_horizon: usize,
    /// Train/test split ratio
    pub train_ratio: f64,
    /// Use walk-forward optimization
    pub walk_forward: bool,
    /// Walk-forward window size
    pub walk_forward_window: usize,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            commission: 0.001,  // 0.1%
            slippage: 0.0005,  // 0.05%
            target_horizon: 1,
            train_ratio: 0.7,
            walk_forward: false,
            walk_forward_window: 100,
        }
    }
}

/// Backtest result
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub metrics: BacktestMetrics,
    pub equity_curve: Vec<f64>,
    pub trade_returns: Vec<f64>,
    pub predictions: Vec<f64>,
    pub signals: Vec<Signal>,
}

/// Backtesting engine
pub struct Backtester {
    pub config: BacktestConfig,
    pub feature_engine: FeatureEngine,
}

impl Backtester {
    /// Create new backtester
    pub fn new(config: BacktestConfig) -> Self {
        Self {
            config,
            feature_engine: FeatureEngine::default_config(),
        }
    }

    /// Run backtest with pre-trained model
    pub fn run(
        &mut self,
        model: &mut NeuralNetwork,
        data: &OHLCVSeries,
        strategy_config: StrategyConfig,
    ) -> BacktestResult {
        // Extract features
        let (features, _targets, valid_indices) = self.feature_engine.extract_features(
            data,
            self.config.target_horizon,
        );

        // Split into train/test
        let split_idx = (valid_indices.len() as f64 * self.config.train_ratio) as usize;
        let test_features = features.slice(ndarray::s![split_idx.., ..]).to_owned();
        let test_indices = &valid_indices[split_idx..];

        // Generate predictions
        let predictions_2d = model.predict(&test_features);
        let predictions: Vec<f64> = predictions_2d.column(0).to_vec();

        // Generate signals
        let signal_generator = SignalGenerator::new(strategy_config.signal_config.clone());
        let signals: Vec<Signal> = predictions
            .iter()
            .map(|&pred| signal_generator.generate(pred, 1.0))
            .collect();

        // Run strategy
        let mut strategy = TradingStrategy::new(strategy_config, self.config.initial_capital);
        let mut equity_curve = vec![self.config.initial_capital];

        for (i, &signal) in signals.iter().enumerate() {
            let data_idx = test_indices[i];
            let candle = &data.data[data_idx];
            let price = candle.close;

            // Apply slippage
            let execution_price = match signal {
                Signal::StrongBuy | Signal::Buy => price * (1.0 + self.config.slippage),
                Signal::StrongSell | Signal::Sell => price * (1.0 - self.config.slippage),
                Signal::Hold => price,
            };

            strategy.process_signal(
                signal,
                execution_price,
                candle.timestamp,
                &data.symbol,
                None,
            );

            // Apply commission to trades
            // (simplified - in reality would track this per trade)

            // Record equity
            let current_equity = strategy.get_capital() +
                strategy.get_position().map_or(0.0, |p| p.unrealized_pnl());
            equity_curve.push(current_equity);
        }

        // Calculate trade returns
        let trade_returns: Vec<f64> = strategy
            .get_trades()
            .iter()
            .map(|t| t.pnl_percent / 100.0)
            .collect();

        // Calculate periods per year based on interval
        let periods_per_year = match data.interval.as_str() {
            "1" => 525600.0,   // 1 minute
            "5" => 105120.0,   // 5 minutes
            "15" => 35040.0,   // 15 minutes
            "60" => 8760.0,    // 1 hour
            "240" => 2190.0,   // 4 hours
            "D" => 365.0,      // 1 day
            _ => 365.0,
        };

        let metrics = BacktestMetrics::calculate(
            &equity_curve,
            &trade_returns,
            self.config.initial_capital,
            periods_per_year,
        );

        BacktestResult {
            metrics,
            equity_curve,
            trade_returns,
            predictions,
            signals,
        }
    }

    /// Run full pipeline: train and backtest
    pub fn train_and_backtest(
        &mut self,
        model: &mut NeuralNetwork,
        data: &OHLCVSeries,
        strategy_config: StrategyConfig,
        epochs: usize,
        batch_size: usize,
    ) -> BacktestResult {
        // Extract features
        let (features, targets, valid_indices) = self.feature_engine.extract_features(
            data,
            self.config.target_horizon,
        );

        // Split into train/test
        let split_idx = (valid_indices.len() as f64 * self.config.train_ratio) as usize;

        let train_features = features.slice(ndarray::s![..split_idx, ..]).to_owned();
        let train_targets = targets.slice(ndarray::s![..split_idx]).to_owned();

        // Reshape targets for training
        let train_targets_2d = train_targets.into_shape((split_idx, 1)).unwrap();

        // Train model
        println!("Training model on {} samples...", split_idx);
        let losses = model.train(&train_features, &train_targets_2d, epochs, batch_size, true);
        println!("Final training loss: {:.6}", losses.last().unwrap_or(&0.0));

        // Run backtest
        self.run(model, data, strategy_config)
    }

    /// Walk-forward backtest
    pub fn walk_forward_backtest(
        &mut self,
        model: &mut NeuralNetwork,
        data: &OHLCVSeries,
        strategy_config: StrategyConfig,
        epochs: usize,
        batch_size: usize,
    ) -> BacktestResult {
        let (features, targets, valid_indices) = self.feature_engine.extract_features(
            data,
            self.config.target_horizon,
        );

        let window = self.config.walk_forward_window;
        let n = valid_indices.len();

        let mut all_predictions = Vec::new();
        let mut all_signals = Vec::new();

        // Walk forward through data
        let mut train_end = (n as f64 * self.config.train_ratio) as usize;

        while train_end < n {
            let test_end = (train_end + window).min(n);

            // Train on data up to train_end
            let train_features = features.slice(ndarray::s![..train_end, ..]).to_owned();
            let train_targets = targets.slice(ndarray::s![..train_end]).to_owned();
            let train_targets_2d = train_targets.into_shape((train_end, 1)).unwrap();

            model.train(&train_features, &train_targets_2d, epochs, batch_size, false);

            // Test on window
            let test_features = features.slice(ndarray::s![train_end..test_end, ..]).to_owned();
            let predictions_2d = model.predict(&test_features);

            let predictions: Vec<f64> = predictions_2d.column(0).to_vec();
            let signal_generator = SignalGenerator::new(strategy_config.signal_config.clone());
            let signals: Vec<Signal> = predictions
                .iter()
                .map(|&pred| signal_generator.generate(pred, 1.0))
                .collect();

            all_predictions.extend(predictions);
            all_signals.extend(signals);

            train_end = test_end;
        }

        // Run strategy on all predictions
        let mut strategy = TradingStrategy::new(strategy_config, self.config.initial_capital);
        let mut equity_curve = vec![self.config.initial_capital];

        let start_idx = (valid_indices.len() as f64 * self.config.train_ratio) as usize;

        for (i, &signal) in all_signals.iter().enumerate() {
            let data_idx = valid_indices[start_idx + i];
            let candle = &data.data[data_idx];

            strategy.process_signal(
                signal,
                candle.close,
                candle.timestamp,
                &data.symbol,
                None,
            );

            let current_equity = strategy.get_capital() +
                strategy.get_position().map_or(0.0, |p| p.unrealized_pnl());
            equity_curve.push(current_equity);
        }

        let trade_returns: Vec<f64> = strategy
            .get_trades()
            .iter()
            .map(|t| t.pnl_percent / 100.0)
            .collect();

        let metrics = BacktestMetrics::calculate(
            &equity_curve,
            &trade_returns,
            self.config.initial_capital,
            365.0,
        );

        BacktestResult {
            metrics,
            equity_curve,
            trade_returns,
            predictions: all_predictions,
            signals: all_signals,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::OHLCV;
    use crate::nn::activation::ActivationType;
    use chrono::{TimeZone, Utc};

    fn create_test_data(n: usize) -> OHLCVSeries {
        let data: Vec<OHLCV> = (0..n)
            .map(|i| {
                let base_price = 50000.0;
                let trend = (i as f64) * 10.0;
                let noise = ((i as f64) * 0.1).sin() * 500.0;
                let price = base_price + trend + noise;

                OHLCV::new(
                    Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap()
                        + chrono::Duration::hours(i as i64),
                    price - 50.0,
                    price + 100.0,
                    price - 100.0,
                    price,
                    1000000.0 + (i as f64) * 1000.0,
                )
            })
            .collect();

        OHLCVSeries::with_data("BTCUSDT".to_string(), "60".to_string(), data)
    }

    #[test]
    fn test_backtester_creation() {
        let backtester = Backtester::new(BacktestConfig::default());
        assert_eq!(backtester.config.initial_capital, 10000.0);
    }

    #[test]
    fn test_simple_backtest() {
        let data = create_test_data(500);
        let mut backtester = Backtester::new(BacktestConfig::default());

        let mut model = NeuralNetwork::regression(
            backtester.feature_engine.num_features(),
            &[32, 16],
            1,
        );

        let result = backtester.train_and_backtest(
            &mut model,
            &data,
            StrategyConfig::default(),
            10,
            32,
        );

        assert!(result.equity_curve.len() > 0);
        assert!(result.predictions.len() > 0);
    }
}
