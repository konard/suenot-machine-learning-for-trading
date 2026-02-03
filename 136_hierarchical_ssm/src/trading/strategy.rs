//! Trading strategy with risk management.

use super::signals::{Signal, SignalGenerator};
use crate::model::Prediction;

/// Configuration for the trading strategy.
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    pub max_position_pct: f64,
    pub stop_loss_multiplier: f64,
    pub take_profit_multiplier: f64,
    pub max_drawdown_limit: f64,
    pub transaction_cost: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            max_position_pct: 0.2,
            stop_loss_multiplier: 2.0,
            take_profit_multiplier: 3.0,
            max_drawdown_limit: 0.15,
            transaction_cost: 0.001,
        }
    }
}

/// Active trading position.
#[derive(Debug, Clone)]
pub struct Position {
    pub direction: Signal,
    pub entry_price: f64,
    pub size: f64,
    pub entry_volatility: f64,
}

/// Trading strategy that manages positions based on HiSS predictions.
#[derive(Debug)]
pub struct TradingStrategy {
    config: StrategyConfig,
    signal_gen: SignalGenerator,
    position: Option<Position>,
    portfolio_value: f64,
    peak_value: f64,
}

impl TradingStrategy {
    pub fn new(config: StrategyConfig, initial_capital: f64) -> Self {
        Self {
            signal_gen: SignalGenerator::default(),
            config,
            position: None,
            portfolio_value: initial_capital,
            peak_value: initial_capital,
        }
    }

    /// Process a new prediction and price, returning any realized PnL.
    pub fn step(&mut self, prediction: &Prediction, current_price: f64) -> f64 {
        let mut realized_pnl = 0.0;

        // Check stop-loss / take-profit
        if let Some(ref pos) = self.position {
            let dir_mult = match pos.direction {
                Signal::Long => 1.0,
                Signal::Short => -1.0,
                _ => 0.0,
            };
            let price_change = dir_mult * (current_price - pos.entry_price);
            let sl = self.config.stop_loss_multiplier * pos.entry_volatility * pos.entry_price;
            let tp = self.config.take_profit_multiplier * pos.entry_volatility * pos.entry_price;

            if price_change <= -sl || price_change >= tp {
                realized_pnl = price_change * pos.size;
                let cost = current_price * pos.size * self.config.transaction_cost;
                self.portfolio_value += realized_pnl - cost;
                self.position = None;
            }
        }

        let signal = self.signal_gen.generate(prediction);

        // Close on signal reversal
        if let Some(ref pos) = self.position {
            let should_close = match (pos.direction, signal) {
                (Signal::Long, Signal::Short) | (Signal::Short, Signal::Long) => true,
                _ => false,
            };
            if should_close {
                let dir_mult = match pos.direction {
                    Signal::Long => 1.0,
                    Signal::Short => -1.0,
                    _ => 0.0,
                };
                let pnl = dir_mult * (current_price - pos.entry_price) * pos.size;
                let cost = current_price * pos.size * self.config.transaction_cost;
                self.portfolio_value += pnl - cost;
                realized_pnl += pnl;
                self.position = None;
            }
        }

        // Open new position
        if self.position.is_none() && signal != Signal::Flat {
            let drawdown = (self.peak_value - self.portfolio_value) / self.peak_value;
            if drawdown < self.config.max_drawdown_limit {
                let size_frac =
                    self.signal_gen
                        .position_size(prediction, self.config.max_position_pct);
                let size = size_frac * self.portfolio_value / current_price;
                let cost = current_price * size * self.config.transaction_cost;
                self.portfolio_value -= cost;

                self.position = Some(Position {
                    direction: signal,
                    entry_price: current_price,
                    size,
                    entry_volatility: prediction.predicted_volatility,
                });
            }
        }

        self.peak_value = self.peak_value.max(self.portfolio_value);
        realized_pnl
    }

    pub fn portfolio_value(&self) -> f64 {
        self.portfolio_value
    }

    pub fn has_position(&self) -> bool {
        self.position.is_some()
    }
}
