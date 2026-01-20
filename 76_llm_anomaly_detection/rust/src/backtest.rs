//! Backtesting framework for anomaly-based trading strategies.

use crate::data_loader::FeatureCalculator;
use crate::detector::AnomalyDetector;
use crate::signals::{SignalGenerator, SignalStrategy};
use crate::types::{BacktestResult, Candle, Features, SignalType, Trade};
use anyhow::Result;
use chrono::Utc;

/// Backtester configuration.
pub struct BacktesterConfig {
    pub initial_capital: f64,
    pub position_size: f64,
    pub max_positions: usize,
    pub transaction_cost: f64,
    pub slippage: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
}

impl Default for BacktesterConfig {
    fn default() -> Self {
        Self {
            initial_capital: 100_000.0,
            position_size: 0.1,
            max_positions: 5,
            transaction_cost: 0.001,
            slippage: 0.0005,
            stop_loss: 0.05,
            take_profit: 0.10,
        }
    }
}

/// Backtester for anomaly-based strategies.
pub struct Backtester<D: AnomalyDetector> {
    detector: D,
    signal_generator: SignalGenerator,
    config: BacktesterConfig,
    feature_calculator: FeatureCalculator,
}

impl<D: AnomalyDetector> Backtester<D> {
    /// Create a new backtester.
    pub fn new(detector: D) -> Self {
        Self {
            detector,
            signal_generator: SignalGenerator::new(),
            config: BacktesterConfig::default(),
            feature_calculator: FeatureCalculator::default(),
        }
    }

    /// Set the signal generator.
    pub fn with_signal_generator(mut self, generator: SignalGenerator) -> Self {
        self.signal_generator = generator;
        self
    }

    /// Set the configuration.
    pub fn with_config(mut self, config: BacktesterConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the signal strategy.
    pub fn with_strategy(mut self, strategy: SignalStrategy) -> Self {
        self.signal_generator = SignalGenerator::with_strategy(strategy);
        self
    }

    /// Apply transaction costs and slippage.
    fn apply_costs(&self, price: f64, direction: i32) -> f64 {
        let cost_factor = 1.0 + self.config.transaction_cost + self.config.slippage;
        if direction > 0 {
            price * cost_factor
        } else {
            price / cost_factor
        }
    }

    /// Check exit conditions for a trade.
    fn should_exit(
        &self,
        trade: &Trade,
        current_price: f64,
        signal_type: SignalType,
    ) -> bool {
        // Calculate unrealized PnL
        let unrealized_pct = (current_price / trade.entry_price - 1.0) * trade.direction as f64;

        // Stop loss
        if unrealized_pct <= -self.config.stop_loss {
            return true;
        }

        // Take profit
        if unrealized_pct >= self.config.take_profit {
            return true;
        }

        // Exit signals
        matches!(
            (signal_type, trade.direction),
            (SignalType::ExitLong, 1) | (SignalType::ExitShort, -1)
        )
    }

    /// Calculate equity from capital and open positions.
    fn calculate_equity(&self, capital: f64, open_trades: &[Trade], current_price: f64) -> f64 {
        let mut equity = capital;

        for trade in open_trades {
            let unrealized_pnl =
                (current_price - trade.entry_price) * trade.direction as f64 * trade.size;
            equity += trade.entry_price * trade.size + unrealized_pnl;
        }

        equity
    }

    /// Run backtest on historical data.
    pub fn run(&mut self, candles: &[Candle], train_period: usize) -> Result<BacktestResult> {
        if candles.len() < train_period + 10 {
            return Err(anyhow::anyhow!("Not enough data for backtest"));
        }

        // Calculate features for all data
        let all_features = self.feature_calculator.calculate_features(candles);

        // Train detector on initial period
        let train_features: Vec<Features> = all_features.iter().take(train_period).cloned().collect();
        self.detector.fit(&train_features)?;

        // Initialize state
        let mut capital = self.config.initial_capital;
        let mut trades: Vec<Trade> = Vec::new();
        let mut open_trades: Vec<Trade> = Vec::new();
        let mut equity_curve: Vec<f64> = Vec::new();
        let mut anomaly_count = 0;

        // Run through data
        for i in train_period..candles.len() {
            let candle = &candles[i];
            let features = &all_features[i];
            let current_price = candle.close;

            // Detect anomaly
            let anomaly_result = self.detector.detect(features)?;

            if anomaly_result.is_anomaly {
                anomaly_count += 1;
            }

            // Calculate current position
            let current_position: f64 = open_trades.iter().map(|t| t.direction as f64 * t.size).sum();

            // Generate signal
            let signal = self.signal_generator.generate(&anomaly_result, features, current_position);

            // Check exit conditions for open trades
            let mut trades_to_close: Vec<usize> = Vec::new();
            for (idx, trade) in open_trades.iter().enumerate() {
                if self.should_exit(trade, current_price, signal.signal_type) {
                    trades_to_close.push(idx);
                }
            }

            // Close trades (in reverse order to maintain indices)
            for idx in trades_to_close.into_iter().rev() {
                let mut trade = open_trades.remove(idx);
                let exit_price = self.apply_costs(current_price, -trade.direction);
                trade.close(candle.timestamp, exit_price);

                // Return capital + PnL
                let position_value = trade.entry_price * trade.size;
                capital += position_value + trade.pnl;

                trades.push(trade);
            }

            // Open new positions
            if open_trades.len() < self.config.max_positions {
                let direction = match signal.signal_type {
                    SignalType::Buy => Some(1),
                    SignalType::Sell => Some(-1),
                    _ => None,
                };

                if let Some(dir) = direction {
                    let position_value = capital * self.config.position_size;
                    let entry_price = self.apply_costs(current_price, dir);
                    let size = position_value / entry_price;

                    let trade = Trade::new(
                        candle.timestamp,
                        entry_price,
                        dir,
                        size,
                        &signal.reason,
                    );

                    capital -= position_value;
                    open_trades.push(trade);
                }
            }

            // Record equity
            equity_curve.push(self.calculate_equity(capital, &open_trades, current_price));
        }

        // Close remaining positions
        if !open_trades.is_empty() {
            let last_candle = &candles[candles.len() - 1];
            let last_price = last_candle.close;

            for mut trade in open_trades {
                let exit_price = self.apply_costs(last_price, -trade.direction);
                trade.close(last_candle.timestamp, exit_price);
                trades.push(trade);
            }
        }

        // Calculate metrics
        self.calculate_metrics(trades, equity_curve, anomaly_count)
    }

    /// Calculate backtest performance metrics.
    fn calculate_metrics(
        &self,
        trades: Vec<Trade>,
        equity_curve: Vec<f64>,
        anomaly_count: usize,
    ) -> Result<BacktestResult> {
        let num_trades = trades.len();

        // Total return
        let final_equity = equity_curve.last().copied().unwrap_or(self.config.initial_capital);
        let total_return = final_equity - self.config.initial_capital;
        let total_return_pct = (total_return / self.config.initial_capital) * 100.0;

        // Daily returns (approximate)
        let returns: Vec<f64> = equity_curve
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        // Sharpe ratio
        let avg_return = if returns.is_empty() {
            0.0
        } else {
            returns.iter().sum::<f64>() / returns.len() as f64
        };

        let std_return = if returns.len() > 1 {
            let variance: f64 = returns.iter().map(|r| (r - avg_return).powi(2)).sum::<f64>()
                / (returns.len() - 1) as f64;
            variance.sqrt()
        } else {
            1.0
        };

        let sharpe_ratio = if std_return > 0.0 {
            (252.0_f64).sqrt() * avg_return / std_return
        } else {
            0.0
        };

        // Sortino ratio
        let negative_returns: Vec<f64> = returns.iter().filter(|r| **r < 0.0).copied().collect();
        let downside_deviation = if negative_returns.len() > 1 {
            let neg_variance: f64 = negative_returns.iter().map(|r| r.powi(2)).sum::<f64>()
                / negative_returns.len() as f64;
            neg_variance.sqrt()
        } else {
            std_return
        };

        let sortino_ratio = if downside_deviation > 0.0 {
            (252.0_f64).sqrt() * avg_return / downside_deviation
        } else {
            sharpe_ratio
        };

        // Maximum drawdown
        let mut peak = equity_curve.first().copied().unwrap_or(self.config.initial_capital);
        let mut max_drawdown = 0.0_f64;

        for equity in &equity_curve {
            if *equity > peak {
                peak = *equity;
            }
            let drawdown = peak - equity;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        let max_drawdown_pct = if peak > 0.0 {
            (max_drawdown / peak) * 100.0
        } else {
            0.0
        };

        // Trade statistics
        let winning_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl <= 0.0).collect();

        let win_rate = if num_trades > 0 {
            (winning_trades.len() as f64 / num_trades as f64) * 100.0
        } else {
            0.0
        };

        let avg_win = if !winning_trades.is_empty() {
            winning_trades.iter().map(|t| t.pnl).sum::<f64>() / winning_trades.len() as f64
        } else {
            0.0
        };

        let avg_loss = if !losing_trades.is_empty() {
            losing_trades.iter().map(|t| t.pnl).sum::<f64>() / losing_trades.len() as f64
        } else {
            0.0
        };

        let gross_profit: f64 = winning_trades.iter().map(|t| t.pnl).sum();
        let gross_loss: f64 = losing_trades.iter().map(|t| t.pnl.abs()).sum();

        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else {
            f64::INFINITY
        };

        Ok(BacktestResult {
            total_return,
            total_return_pct,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            max_drawdown_pct,
            num_trades,
            win_rate,
            avg_win,
            avg_loss,
            profit_factor,
            trades,
            equity_curve,
            anomaly_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::StatisticalDetector;

    fn create_test_candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| Candle {
                timestamp: Utc::now(),
                open: 100.0 + (i as f64 * 0.1),
                high: 101.0 + (i as f64 * 0.1),
                low: 99.0 + (i as f64 * 0.1),
                close: 100.5 + (i as f64 * 0.1),
                volume: 1000.0 + (i as f64 * 10.0),
            })
            .collect()
    }

    #[test]
    fn test_backtest_basic() {
        let candles = create_test_candles(200);
        let detector = StatisticalDetector::new(2.5);

        let mut backtester = Backtester::new(detector)
            .with_config(BacktesterConfig {
                initial_capital: 10000.0,
                position_size: 0.1,
                ..Default::default()
            });

        let result = backtester.run(&candles, 100).unwrap();

        assert!(result.equity_curve.len() > 0);
        assert_eq!(result.equity_curve.len(), candles.len() - 100);
    }
}
