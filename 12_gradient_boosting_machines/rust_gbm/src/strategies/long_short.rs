//! Long-Short trading strategy based on GBM predictions
//!
//! This module implements a simple long-short strategy that:
//! - Goes long when the model predicts positive returns
//! - Goes short when the model predicts negative returns

use crate::data::{Candle, Dataset};
use crate::models::{GbmRegressor, ModelError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trading signal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Signal {
    /// Buy signal
    Long,
    /// Sell signal
    Short,
    /// No position
    Neutral,
}

impl Signal {
    /// Convert to position multiplier (-1, 0, or 1)
    pub fn to_multiplier(&self) -> f64 {
        match self {
            Signal::Long => 1.0,
            Signal::Short => -1.0,
            Signal::Neutral => 0.0,
        }
    }
}

/// Position in a trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Symbol
    pub symbol: String,
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
    /// Entry price
    pub entry_price: f64,
    /// Position size (positive for long, negative for short)
    pub size: f64,
    /// Current unrealized PnL
    pub unrealized_pnl: f64,
    /// Current signal
    pub signal: Signal,
}

/// Trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Symbol
    pub symbol: String,
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
    /// Exit timestamp
    pub exit_time: DateTime<Utc>,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Position size
    pub size: f64,
    /// Trade signal (long or short)
    pub signal: Signal,
    /// Realized PnL
    pub pnl: f64,
    /// Return percentage
    pub return_pct: f64,
}

/// Strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Threshold for generating long signal (predicted return > threshold)
    pub long_threshold: f64,
    /// Threshold for generating short signal (predicted return < -threshold)
    pub short_threshold: f64,
    /// Initial capital
    pub initial_capital: f64,
    /// Position size as fraction of capital (0.0 to 1.0)
    pub position_size: f64,
    /// Trading fee rate (e.g., 0.001 for 0.1%)
    pub fee_rate: f64,
    /// Maximum number of open positions
    pub max_positions: usize,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            long_threshold: 0.1,
            short_threshold: 0.1,
            initial_capital: 10000.0,
            position_size: 0.1,
            fee_rate: 0.001,
            max_positions: 1,
        }
    }
}

/// Strategy performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total return percentage
    pub total_return: f64,
    /// Annualized return (assuming 365 trading days)
    pub annualized_return: f64,
    /// Maximum drawdown percentage
    pub max_drawdown: f64,
    /// Sharpe ratio (assuming 0 risk-free rate)
    pub sharpe_ratio: f64,
    /// Win rate (percentage of profitable trades)
    pub win_rate: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Total number of trades
    pub num_trades: usize,
    /// Average trade return
    pub avg_trade_return: f64,
    /// Total fees paid
    pub total_fees: f64,
}

/// Long-Short trading strategy
pub struct LongShortStrategy {
    config: StrategyConfig,
    model: Option<GbmRegressor>,
    positions: HashMap<String, Position>,
    trades: Vec<Trade>,
    equity_curve: Vec<(DateTime<Utc>, f64)>,
    current_capital: f64,
}

impl LongShortStrategy {
    /// Create a new long-short strategy with default configuration
    pub fn new() -> Self {
        Self::with_config(StrategyConfig::default())
    }

    /// Create a new long-short strategy with custom configuration
    pub fn with_config(config: StrategyConfig) -> Self {
        Self {
            current_capital: config.initial_capital,
            config,
            model: None,
            positions: HashMap::new(),
            trades: Vec::new(),
            equity_curve: Vec::new(),
        }
    }

    /// Set the prediction model
    pub fn set_model(&mut self, model: GbmRegressor) {
        self.model = Some(model);
    }

    /// Generate signal from prediction
    pub fn generate_signal(&self, prediction: f64) -> Signal {
        if prediction > self.config.long_threshold {
            Signal::Long
        } else if prediction < -self.config.short_threshold {
            Signal::Short
        } else {
            Signal::Neutral
        }
    }

    /// Generate signals for a dataset
    pub fn generate_signals(&self, dataset: &Dataset) -> Result<Vec<Signal>, ModelError> {
        let model = self.model.as_ref().ok_or(ModelError::NotTrained)?;
        let predictions = model.predict_dataset(dataset)?;

        Ok(predictions.iter().map(|&p| self.generate_signal(p)).collect())
    }

    /// Run backtest on historical data
    pub fn backtest(&mut self, dataset: &Dataset, prices: &[f64]) -> Result<PerformanceMetrics, ModelError> {
        if dataset.len() != prices.len() {
            return Err(ModelError::InvalidData(
                "Dataset and prices length mismatch".to_string(),
            ));
        }

        let signals = self.generate_signals(dataset)?;

        // Reset state
        self.current_capital = self.config.initial_capital;
        self.positions.clear();
        self.trades.clear();
        self.equity_curve.clear();

        let mut position: Option<Position> = None;
        let symbol = dataset.symbol.clone();

        for i in 0..signals.len() {
            let signal = signals[i];
            let price = prices[i];
            let timestamp = dataset.timestamps[i];

            // Close existing position if signal changes
            if let Some(ref pos) = position {
                let should_close = match signal {
                    Signal::Long => pos.signal != Signal::Long,
                    Signal::Short => pos.signal != Signal::Short,
                    Signal::Neutral => true,
                };

                if should_close {
                    // Close position
                    let pnl = (price - pos.entry_price) * pos.size;
                    let fee = price * pos.size.abs() * self.config.fee_rate;
                    let net_pnl = pnl - fee;

                    let return_pct = if pos.entry_price != 0.0 {
                        (price - pos.entry_price) / pos.entry_price * pos.size.signum() * 100.0
                    } else {
                        0.0
                    };

                    self.trades.push(Trade {
                        symbol: symbol.clone(),
                        entry_time: pos.entry_time,
                        exit_time: timestamp,
                        entry_price: pos.entry_price,
                        exit_price: price,
                        size: pos.size,
                        signal: pos.signal,
                        pnl: net_pnl,
                        return_pct,
                    });

                    self.current_capital += net_pnl;
                    position = None;
                }
            }

            // Open new position if no current position and signal is not neutral
            if position.is_none() && signal != Signal::Neutral {
                let position_value = self.current_capital * self.config.position_size;
                let size = if signal == Signal::Long {
                    position_value / price
                } else {
                    -position_value / price
                };

                let fee = price * size.abs() * self.config.fee_rate;
                self.current_capital -= fee;

                position = Some(Position {
                    symbol: symbol.clone(),
                    entry_time: timestamp,
                    entry_price: price,
                    size,
                    unrealized_pnl: 0.0,
                    signal,
                });
            }

            // Update equity curve
            let unrealized = position
                .as_ref()
                .map(|p| (price - p.entry_price) * p.size)
                .unwrap_or(0.0);

            self.equity_curve
                .push((timestamp, self.current_capital + unrealized));
        }

        // Close any remaining position at the end
        if let Some(pos) = position {
            let price = prices[prices.len() - 1];
            let pnl = (price - pos.entry_price) * pos.size;
            let fee = price * pos.size.abs() * self.config.fee_rate;
            let net_pnl = pnl - fee;

            let return_pct = if pos.entry_price != 0.0 {
                (price - pos.entry_price) / pos.entry_price * pos.size.signum() * 100.0
            } else {
                0.0
            };

            self.trades.push(Trade {
                symbol: symbol.clone(),
                entry_time: pos.entry_time,
                exit_time: dataset.timestamps[dataset.len() - 1],
                entry_price: pos.entry_price,
                exit_price: price,
                size: pos.size,
                signal: pos.signal,
                pnl: net_pnl,
                return_pct,
            });

            self.current_capital += net_pnl;
        }

        Ok(self.calculate_metrics())
    }

    /// Calculate performance metrics
    fn calculate_metrics(&self) -> PerformanceMetrics {
        let total_return =
            (self.current_capital - self.config.initial_capital) / self.config.initial_capital * 100.0;

        // Calculate maximum drawdown
        let mut max_equity = self.config.initial_capital;
        let mut max_drawdown = 0.0;
        for (_, equity) in &self.equity_curve {
            if *equity > max_equity {
                max_equity = *equity;
            }
            let drawdown = (max_equity - equity) / max_equity * 100.0;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        // Calculate returns for Sharpe ratio
        let returns: Vec<f64> = self.trades.iter().map(|t| t.return_pct).collect();
        let avg_return = if !returns.is_empty() {
            returns.iter().sum::<f64>() / returns.len() as f64
        } else {
            0.0
        };

        let std_return = if returns.len() > 1 {
            let variance: f64 = returns.iter().map(|r| (r - avg_return).powi(2)).sum::<f64>()
                / (returns.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        let sharpe_ratio = if std_return > 0.0 {
            avg_return / std_return * (252_f64).sqrt() // Annualized
        } else {
            0.0
        };

        // Win rate and profit factor
        let winning_trades: Vec<&Trade> = self.trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing_trades: Vec<&Trade> = self.trades.iter().filter(|t| t.pnl < 0.0).collect();

        let win_rate = if !self.trades.is_empty() {
            winning_trades.len() as f64 / self.trades.len() as f64 * 100.0
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

        // Calculate annualized return
        let trading_days = if self.equity_curve.len() > 1 {
            let first_date = self.equity_curve.first().unwrap().0;
            let last_date = self.equity_curve.last().unwrap().0;
            (last_date - first_date).num_days() as f64
        } else {
            1.0
        };

        let annualized_return = if trading_days > 0.0 {
            ((1.0 + total_return / 100.0).powf(365.0 / trading_days) - 1.0) * 100.0
        } else {
            0.0
        };

        // Total fees
        let total_fees: f64 = self.trades.iter().map(|t| {
            (t.entry_price * t.size.abs() + t.exit_price * t.size.abs()) * self.config.fee_rate
        }).sum();

        PerformanceMetrics {
            total_return,
            annualized_return,
            max_drawdown,
            sharpe_ratio,
            win_rate,
            profit_factor,
            num_trades: self.trades.len(),
            avg_trade_return: avg_return,
            total_fees,
        }
    }

    /// Get all trades
    pub fn trades(&self) -> &[Trade] {
        &self.trades
    }

    /// Get equity curve
    pub fn equity_curve(&self) -> &[(DateTime<Utc>, f64)] {
        &self.equity_curve
    }

    /// Get current capital
    pub fn current_capital(&self) -> f64 {
        self.current_capital
    }

    /// Get configuration
    pub fn config(&self) -> &StrategyConfig {
        &self.config
    }
}

impl Default for LongShortStrategy {
    fn default() -> Self {
        Self::new()
    }
}

/// Print a summary of the backtest results
pub fn print_backtest_summary(metrics: &PerformanceMetrics) {
    println!("\n{'='} Backtest Results {}", "=".repeat(40));
    println!("Total Return:        {:>10.2}%", metrics.total_return);
    println!("Annualized Return:   {:>10.2}%", metrics.annualized_return);
    println!("Maximum Drawdown:    {:>10.2}%", metrics.max_drawdown);
    println!("Sharpe Ratio:        {:>10.2}", metrics.sharpe_ratio);
    println!("Win Rate:            {:>10.2}%", metrics.win_rate);
    println!("Profit Factor:       {:>10.2}", metrics.profit_factor);
    println!("Number of Trades:    {:>10}", metrics.num_trades);
    println!("Avg Trade Return:    {:>10.2}%", metrics.avg_trade_return);
    println!("Total Fees:          {:>10.2}", metrics.total_fees);
    println!("{}", "=".repeat(60));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_generation() {
        let strategy = LongShortStrategy::with_config(StrategyConfig {
            long_threshold: 0.5,
            short_threshold: 0.5,
            ..Default::default()
        });

        assert_eq!(strategy.generate_signal(1.0), Signal::Long);
        assert_eq!(strategy.generate_signal(-1.0), Signal::Short);
        assert_eq!(strategy.generate_signal(0.0), Signal::Neutral);
        assert_eq!(strategy.generate_signal(0.3), Signal::Neutral);
    }

    #[test]
    fn test_signal_multiplier() {
        assert_eq!(Signal::Long.to_multiplier(), 1.0);
        assert_eq!(Signal::Short.to_multiplier(), -1.0);
        assert_eq!(Signal::Neutral.to_multiplier(), 0.0);
    }
}
