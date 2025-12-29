//! Backtest Engine
//!
//! Implements backtesting for online learning strategies.

use crate::api::Candle;
use crate::features::MomentumFeatures;
use crate::models::{AdaptiveMomentumWeights, OnlineLinearRegression, OnlineModel};
use crate::streaming::StreamSimulator;

/// Result of a single trade
#[derive(Debug, Clone)]
pub struct TradeResult {
    /// Entry timestamp
    pub entry_time: u64,
    /// Exit timestamp
    pub exit_time: u64,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Position size (-1, 0, 1)
    pub position: f64,
    /// Profit/loss as percentage
    pub pnl_pct: f64,
}

/// Result of backtest
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Total return
    pub total_return: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate
    pub win_rate: f64,
    /// Total number of trades
    pub n_trades: usize,
    /// Average trade PnL
    pub avg_trade_pnl: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Cumulative returns
    pub cumulative_returns: Vec<f64>,
    /// Trade results
    pub trades: Vec<TradeResult>,
}

impl BacktestResult {
    /// Calculate metrics from returns
    pub fn from_returns(returns: &[f64], trades: Vec<TradeResult>) -> Self {
        let n = returns.len() as f64;

        // Total return
        let total_return: f64 = returns.iter().sum();

        // Sharpe ratio (assuming daily/hourly returns, annualize appropriately)
        let mean = total_return / n;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();
        let sharpe_ratio = if std > 0.0 {
            mean / std * (252.0_f64).sqrt() // Annualized assuming daily
        } else {
            0.0
        };

        // Maximum drawdown
        let mut cumulative = 0.0;
        let mut peak = 0.0;
        let mut max_dd = 0.0;
        let mut cumulative_returns = Vec::with_capacity(returns.len());

        for &r in returns {
            cumulative += r;
            cumulative_returns.push(cumulative);
            peak = peak.max(cumulative);
            max_dd = max_dd.max(peak - cumulative);
        }

        // Win rate and trade statistics
        let winning_trades: Vec<&TradeResult> = trades.iter().filter(|t| t.pnl_pct > 0.0).collect();
        let losing_trades: Vec<&TradeResult> = trades.iter().filter(|t| t.pnl_pct < 0.0).collect();

        let win_rate = if !trades.is_empty() {
            winning_trades.len() as f64 / trades.len() as f64
        } else {
            0.0
        };

        let avg_trade_pnl = if !trades.is_empty() {
            trades.iter().map(|t| t.pnl_pct).sum::<f64>() / trades.len() as f64
        } else {
            0.0
        };

        let gross_profit: f64 = winning_trades.iter().map(|t| t.pnl_pct).sum();
        let gross_loss: f64 = losing_trades.iter().map(|t| t.pnl_pct.abs()).sum();

        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        Self {
            total_return,
            sharpe_ratio,
            max_drawdown: max_dd,
            win_rate,
            n_trades: trades.len(),
            avg_trade_pnl,
            profit_factor,
            cumulative_returns,
            trades,
        }
    }
}

/// Comparison results between approaches
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Online learning results
    pub online_sharpe: f64,
    pub online_return: f64,
    pub online_trades: usize,

    /// Static model results
    pub static_sharpe: f64,
    pub static_return: f64,
    pub static_trades: usize,

    /// Monthly retrain results
    pub monthly_sharpe: f64,
    pub monthly_return: f64,
    pub monthly_trades: usize,
}

/// Backtest engine for online learning strategies
pub struct BacktestEngine {
    /// Historical candles
    candles: Vec<Candle>,
    /// Momentum periods
    periods: Vec<usize>,
    /// Transaction cost (as fraction of trade)
    transaction_cost: f64,
    /// Signal threshold for trading
    signal_threshold: f64,
}

impl BacktestEngine {
    /// Create new backtest engine
    pub fn new(candles: Vec<Candle>) -> Self {
        Self {
            candles,
            periods: vec![12, 24, 48, 96],
            transaction_cost: 0.001, // 0.1% per trade
            signal_threshold: 0.001,
        }
    }

    /// Set momentum periods
    pub fn with_periods(mut self, periods: Vec<usize>) -> Self {
        self.periods = periods;
        self
    }

    /// Set transaction cost
    pub fn with_transaction_cost(mut self, cost: f64) -> Self {
        self.transaction_cost = cost;
        self
    }

    /// Set signal threshold
    pub fn with_signal_threshold(mut self, threshold: f64) -> Self {
        self.signal_threshold = threshold;
        self
    }

    /// Run online learning backtest
    pub fn run_online(&self, learning_rate: f64) -> anyhow::Result<BacktestResult> {
        let mut simulator = StreamSimulator::new(self.candles.clone(), self.periods.clone());
        let mut model = AdaptiveMomentumWeights::with_equal_weights(self.periods.len(), learning_rate);

        let mut returns = Vec::new();
        let mut trades = Vec::new();
        let mut prev_position = 0.0;
        let mut entry_price = 0.0;
        let mut entry_time = 0u64;

        while let Some(obs) = simulator.next() {
            // Predict before learning
            let prediction = model.predict(&obs.features);

            // Generate signal
            let signal = if prediction > self.signal_threshold {
                1.0
            } else if prediction < -self.signal_threshold {
                -1.0
            } else {
                0.0
            };

            // Calculate return (with transaction costs)
            let mut pnl = signal * obs.target;

            // Apply transaction cost on position change
            if (signal - prev_position).abs() > 0.5 {
                pnl -= self.transaction_cost;

                // Record trade if closing position
                if prev_position != 0.0 {
                    trades.push(TradeResult {
                        entry_time,
                        exit_time: obs.timestamp,
                        entry_price,
                        exit_price: obs.price,
                        position: prev_position,
                        pnl_pct: prev_position * (obs.price - entry_price) / entry_price - self.transaction_cost,
                    });
                }

                // Record new entry
                if signal != 0.0 {
                    entry_price = obs.price;
                    entry_time = obs.timestamp;
                }
            }

            returns.push(pnl);
            prev_position = signal;

            // Learn after observing result
            model.update(&obs.features, obs.target);
        }

        Ok(BacktestResult::from_returns(&returns, trades))
    }

    /// Run static model backtest (trained once)
    pub fn run_static(&self, train_size: usize) -> anyhow::Result<BacktestResult> {
        if self.candles.len() < train_size + 100 {
            return Err(anyhow::anyhow!("Not enough data for static model"));
        }

        let feature_gen = MomentumFeatures::new(self.periods.clone());
        let warmup = *self.periods.iter().max().unwrap() + 1;

        // Train model on initial period
        let mut model = OnlineLinearRegression::new(self.periods.len(), 0.01);

        for i in warmup..train_size {
            if let Some(features) = feature_gen.compute(&self.candles[..=i]) {
                if i + 1 < self.candles.len() {
                    let target = (self.candles[i + 1].close - self.candles[i].close) / self.candles[i].close;
                    model.learn(&features, target);
                }
            }
        }

        // Test on remaining data (no more learning)
        let mut returns = Vec::new();
        let mut trades = Vec::new();
        let mut prev_position = 0.0;
        let mut entry_price = 0.0;
        let mut entry_time = 0u64;

        for i in train_size..self.candles.len() - 1 {
            if let Some(features) = feature_gen.compute(&self.candles[..=i]) {
                let prediction = model.predict(&features);
                let target = (self.candles[i + 1].close - self.candles[i].close) / self.candles[i].close;

                let signal = if prediction > self.signal_threshold {
                    1.0
                } else if prediction < -self.signal_threshold {
                    -1.0
                } else {
                    0.0
                };

                let mut pnl = signal * target;

                if (signal - prev_position).abs() > 0.5 {
                    pnl -= self.transaction_cost;

                    if prev_position != 0.0 {
                        trades.push(TradeResult {
                            entry_time,
                            exit_time: self.candles[i].timestamp,
                            entry_price,
                            exit_price: self.candles[i].close,
                            position: prev_position,
                            pnl_pct: prev_position * (self.candles[i].close - entry_price) / entry_price - self.transaction_cost,
                        });
                    }

                    if signal != 0.0 {
                        entry_price = self.candles[i].close;
                        entry_time = self.candles[i].timestamp;
                    }
                }

                returns.push(pnl);
                prev_position = signal;
            }
        }

        Ok(BacktestResult::from_returns(&returns, trades))
    }

    /// Run monthly retrain backtest
    pub fn run_monthly_retrain(&self, train_window: usize, retrain_period: usize) -> anyhow::Result<BacktestResult> {
        let feature_gen = MomentumFeatures::new(self.periods.clone());
        let warmup = *self.periods.iter().max().unwrap() + 1;

        let mut model = OnlineLinearRegression::new(self.periods.len(), 0.01);
        let mut returns = Vec::new();
        let mut trades = Vec::new();
        let mut prev_position = 0.0;
        let mut entry_price = 0.0;
        let mut entry_time = 0u64;
        let mut last_retrain = 0;

        for i in warmup..self.candles.len() - 1 {
            // Retrain periodically
            if i - last_retrain >= retrain_period || last_retrain == 0 {
                model.reset();

                // Train on window
                let start = if i > train_window { i - train_window } else { warmup };
                for j in start..i {
                    if let Some(features) = feature_gen.compute(&self.candles[..=j]) {
                        if j + 1 < self.candles.len() {
                            let target = (self.candles[j + 1].close - self.candles[j].close) / self.candles[j].close;
                            model.learn(&features, target);
                        }
                    }
                }

                last_retrain = i;
            }

            if i < train_window {
                continue;
            }

            // Predict and trade
            if let Some(features) = feature_gen.compute(&self.candles[..=i]) {
                let prediction = model.predict(&features);
                let target = (self.candles[i + 1].close - self.candles[i].close) / self.candles[i].close;

                let signal = if prediction > self.signal_threshold {
                    1.0
                } else if prediction < -self.signal_threshold {
                    -1.0
                } else {
                    0.0
                };

                let mut pnl = signal * target;

                if (signal - prev_position).abs() > 0.5 {
                    pnl -= self.transaction_cost;

                    if prev_position != 0.0 {
                        trades.push(TradeResult {
                            entry_time,
                            exit_time: self.candles[i].timestamp,
                            entry_price,
                            exit_price: self.candles[i].close,
                            position: prev_position,
                            pnl_pct: prev_position * (self.candles[i].close - entry_price) / entry_price - self.transaction_cost,
                        });
                    }

                    if signal != 0.0 {
                        entry_price = self.candles[i].close;
                        entry_time = self.candles[i].timestamp;
                    }
                }

                returns.push(pnl);
                prev_position = signal;
            }
        }

        Ok(BacktestResult::from_returns(&returns, trades))
    }

    /// Compare all approaches
    pub fn compare_approaches(&self, learning_rate: f64) -> anyhow::Result<ComparisonResult> {
        let online = self.run_online(learning_rate)?;
        let train_size = self.candles.len() / 4;
        let static_result = self.run_static(train_size)?;
        let monthly = self.run_monthly_retrain(100, 720)?; // 720 hours = ~30 days

        Ok(ComparisonResult {
            online_sharpe: online.sharpe_ratio,
            online_return: online.total_return,
            online_trades: online.n_trades,
            static_sharpe: static_result.sharpe_ratio,
            static_return: static_result.total_return,
            static_trades: static_result.n_trades,
            monthly_sharpe: monthly.sharpe_ratio,
            monthly_return: monthly.total_return,
            monthly_trades: monthly.n_trades,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| {
                let trend = (i as f64 * 0.01).sin() * 10.0;
                Candle {
                    timestamp: (i * 3600000) as u64,
                    open: 100.0 + trend,
                    high: 102.0 + trend,
                    low: 98.0 + trend,
                    close: 100.0 + trend + (i as f64 * 0.001),
                    volume: 1000.0,
                    turnover: 100000.0,
                }
            })
            .collect()
    }

    #[test]
    fn test_backtest_online() {
        let candles = create_test_candles(500);
        let engine = BacktestEngine::new(candles);

        let result = engine.run_online(0.01);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.n_trades > 0);
    }

    #[test]
    fn test_backtest_static() {
        let candles = create_test_candles(500);
        let engine = BacktestEngine::new(candles);

        let result = engine.run_static(200);
        assert!(result.is_ok());
    }

    #[test]
    fn test_comparison() {
        let candles = create_test_candles(1000);
        let engine = BacktestEngine::new(candles);

        let comparison = engine.compare_approaches(0.01);
        assert!(comparison.is_ok());
    }

    #[test]
    fn test_backtest_result_metrics() {
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015];
        let trades = vec![];

        let result = BacktestResult::from_returns(&returns, trades);

        assert!(result.total_return > 0.0);
        assert!(result.max_drawdown >= 0.0);
    }
}
