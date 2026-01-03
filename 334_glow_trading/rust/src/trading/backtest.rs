//! Backtesting framework for GLOW trading strategies

use crate::data::{Candle, FeatureExtractor, MarketFeatures};
use crate::trading::{GLOWTrader, TradingSignal, PerformanceMetrics};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Configuration for backtesting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Transaction cost (percentage)
    pub transaction_cost: f64,
    /// Slippage (percentage)
    pub slippage: f64,
    /// Warmup period (number of candles)
    pub warmup: usize,
    /// Lookback for feature extraction
    pub lookback: usize,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            transaction_cost: 0.001, // 0.1%
            slippage: 0.0005,        // 0.05%
            warmup: 100,
            lookback: 20,
        }
    }
}

/// Single step result in backtest
#[derive(Debug, Clone)]
pub struct BacktestStep {
    pub timestamp: i64,
    pub price: f64,
    pub signal: f64,
    pub log_likelihood: f64,
    pub in_distribution: bool,
    pub regime: usize,
    pub position: f64,
    pub pnl: f64,
    pub cumulative_pnl: f64,
    pub equity: f64,
}

/// Complete backtest results
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Individual step results
    pub steps: Vec<BacktestStep>,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Configuration used
    pub config: BacktestConfig,
}

impl BacktestResult {
    /// Get final equity
    pub fn final_equity(&self) -> f64 {
        self.steps.last().map(|s| s.equity).unwrap_or(0.0)
    }

    /// Get total return
    pub fn total_return(&self) -> f64 {
        self.steps.last().map(|s| s.cumulative_pnl).unwrap_or(0.0)
    }

    /// Get PnL series
    pub fn pnl_series(&self) -> Vec<f64> {
        self.steps.iter().map(|s| s.pnl).collect()
    }

    /// Get equity curve
    pub fn equity_curve(&self) -> Vec<f64> {
        self.steps.iter().map(|s| s.equity).collect()
    }

    /// Get signal series
    pub fn signal_series(&self) -> Vec<f64> {
        self.steps.iter().map(|s| s.signal).collect()
    }
}

/// Backtesting engine
pub struct Backtest {
    config: BacktestConfig,
}

impl Backtest {
    /// Create new backtester
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run backtest on price data
    pub fn run(&self, trader: &mut GLOWTrader, candles: &[Candle]) -> BacktestResult {
        let mut extractor = FeatureExtractor::new(self.config.lookback);
        let mut steps = Vec::new();

        let mut position = 0.0;
        let mut cumulative_pnl = 0.0;
        let mut equity = self.config.initial_capital;
        let mut prev_price = 0.0;

        for (i, candle) in candles.iter().enumerate() {
            // Extract features
            let features = extractor.add_candle(candle.clone());

            // Skip warmup period
            if i < self.config.warmup || features.is_none() {
                prev_price = candle.close;
                continue;
            }

            let features = features.unwrap();

            // Generate signal
            let signal_info = trader.generate_signal(&features);

            // Calculate PnL from previous position
            let price_change = if prev_price > 0.0 {
                (candle.close - prev_price) / prev_price
            } else {
                0.0
            };

            let gross_pnl = position * price_change * equity;

            // Calculate transaction costs for position change
            let position_change = (signal_info.signal - position).abs();
            let transaction_costs = position_change * equity *
                (self.config.transaction_cost + self.config.slippage);

            let net_pnl = gross_pnl - transaction_costs;
            cumulative_pnl += net_pnl;
            equity += net_pnl;

            // Update position
            let prev_position = position;
            position = signal_info.signal;

            steps.push(BacktestStep {
                timestamp: candle.timestamp,
                price: candle.close,
                signal: signal_info.signal,
                log_likelihood: signal_info.log_likelihood,
                in_distribution: signal_info.in_distribution,
                regime: signal_info.regime,
                position: prev_position,
                pnl: net_pnl,
                cumulative_pnl,
                equity,
            });

            prev_price = candle.close;
        }

        // Calculate metrics
        let pnl_series: Vec<f64> = steps.iter().map(|s| s.pnl).collect();
        let metrics = PerformanceMetrics::from_returns(&pnl_series, self.config.initial_capital);

        BacktestResult {
            steps,
            metrics,
            config: self.config.clone(),
        }
    }

    /// Run walk-forward backtest
    pub fn walk_forward(
        &self,
        trader: &mut GLOWTrader,
        candles: &[Candle],
        train_window: usize,
        test_window: usize,
    ) -> Vec<BacktestResult> {
        let mut results = Vec::new();
        let mut start = 0;

        while start + train_window + test_window <= candles.len() {
            let _train_end = start + train_window;
            let test_start = start + train_window;
            let test_end = test_start + test_window;

            // Run backtest on test period
            let test_candles = &candles[test_start..test_end];
            let result = self.run(trader, test_candles);
            results.push(result);

            start += test_window;
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{GLOWModel, GLOWConfig};
    use crate::trading::TraderConfig;

    fn create_test_candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| {
                let price = 100.0 + (i as f64 * 0.1).sin() * 10.0;
                Candle {
                    timestamp: i as i64 * 3600000,
                    open: price - 0.5,
                    high: price + 1.0,
                    low: price - 1.0,
                    close: price,
                    volume: 1000.0 + i as f64 * 10.0,
                    turnover: 100000.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_backtest() {
        let config = GLOWConfig::with_features(16);
        let model = GLOWModel::new(config);
        let trader_config = TraderConfig::default();
        let mut trader = GLOWTrader::new(model, trader_config);

        let candles = create_test_candles(200);
        let backtest_config = BacktestConfig::default();
        let backtest = Backtest::new(backtest_config);

        let result = backtest.run(&mut trader, &candles);

        assert!(!result.steps.is_empty());
        assert!(result.metrics.sharpe_ratio.is_finite());
    }
}
