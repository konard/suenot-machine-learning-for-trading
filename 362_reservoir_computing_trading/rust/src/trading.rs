//! # Trading System
//!
//! Trading signal generation and position management using reservoir computing.
//!
//! ## Components
//!
//! - Signal generation from ESN predictions
//! - Position sizing with risk management
//! - Trade execution logic

use crate::features::MarketFeatures;
use crate::reservoir::EchoStateNetwork;
use ndarray::Array1;
use std::collections::VecDeque;
use thiserror::Error;

/// Trading errors
#[derive(Error, Debug)]
pub enum TradingError {
    #[error("Model not ready for predictions")]
    ModelNotReady,

    #[error("Invalid position size: {0}")]
    InvalidPositionSize(f64),

    #[error("Risk limit exceeded")]
    RiskLimitExceeded,
}

/// Trading signal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    /// Strong buy signal
    StrongBuy,

    /// Buy signal
    Buy,

    /// Hold / no action
    Hold,

    /// Sell signal
    Sell,

    /// Strong sell signal
    StrongSell,
}

impl Signal {
    /// Convert signal to position direction multiplier
    pub fn direction(&self) -> f64 {
        match self {
            Signal::StrongBuy => 1.0,
            Signal::Buy => 0.5,
            Signal::Hold => 0.0,
            Signal::Sell => -0.5,
            Signal::StrongSell => -1.0,
        }
    }

    /// Create signal from prediction value
    pub fn from_prediction(pred: f64, thresholds: &SignalThresholds) -> Self {
        if pred > thresholds.strong_buy {
            Signal::StrongBuy
        } else if pred > thresholds.buy {
            Signal::Buy
        } else if pred < thresholds.strong_sell {
            Signal::StrongSell
        } else if pred < thresholds.sell {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }
}

/// Signal thresholds
#[derive(Debug, Clone)]
pub struct SignalThresholds {
    pub strong_buy: f64,
    pub buy: f64,
    pub sell: f64,
    pub strong_sell: f64,
}

impl Default for SignalThresholds {
    fn default() -> Self {
        Self {
            strong_buy: 0.5,
            buy: 0.2,
            sell: -0.2,
            strong_sell: -0.5,
        }
    }
}

/// Current position
#[derive(Debug, Clone, Default)]
pub struct Position {
    /// Position size (positive = long, negative = short, 0 = flat)
    pub size: f64,

    /// Entry price (average)
    pub entry_price: f64,

    /// Unrealized PnL
    pub unrealized_pnl: f64,

    /// Realized PnL
    pub realized_pnl: f64,

    /// Number of trades
    pub n_trades: usize,
}

impl Position {
    /// Check if position is flat
    pub fn is_flat(&self) -> bool {
        self.size.abs() < 1e-10
    }

    /// Check if position is long
    pub fn is_long(&self) -> bool {
        self.size > 1e-10
    }

    /// Check if position is short
    pub fn is_short(&self) -> bool {
        self.size < -1e-10
    }

    /// Update unrealized PnL based on current price
    pub fn update_pnl(&mut self, current_price: f64) {
        if !self.is_flat() {
            self.unrealized_pnl = self.size * (current_price - self.entry_price);
        } else {
            self.unrealized_pnl = 0.0;
        }
    }

    /// Close position and realize PnL
    pub fn close(&mut self, exit_price: f64) -> f64 {
        let pnl = self.size * (exit_price - self.entry_price);
        self.realized_pnl += pnl;
        self.unrealized_pnl = 0.0;
        self.size = 0.0;
        self.entry_price = 0.0;
        pnl
    }
}

/// Trading configuration
#[derive(Debug, Clone)]
pub struct TradingConfig {
    /// Maximum position size (units)
    pub max_position: f64,

    /// Position sizing factor (0-1)
    pub position_scale: f64,

    /// Transaction cost (as fraction)
    pub transaction_cost: f64,

    /// Stop loss percentage
    pub stop_loss: Option<f64>,

    /// Take profit percentage
    pub take_profit: Option<f64>,

    /// Maximum drawdown before stopping
    pub max_drawdown: f64,

    /// Signal thresholds
    pub thresholds: SignalThresholds,
}

impl Default for TradingConfig {
    fn default() -> Self {
        Self {
            max_position: 1.0,
            position_scale: 0.5,
            transaction_cost: 0.001,
            stop_loss: Some(0.02),
            take_profit: Some(0.04),
            max_drawdown: 0.10,
            thresholds: SignalThresholds::default(),
        }
    }
}

/// Trading system using reservoir computing
pub struct TradingSystem {
    /// ESN model reference (borrowed during trading)
    config: TradingConfig,

    /// Current position
    position: Position,

    /// Equity curve
    equity: Vec<f64>,

    /// Peak equity for drawdown calculation
    peak_equity: f64,

    /// Signal history for debugging
    signal_history: VecDeque<Signal>,

    /// Maximum signal history length
    max_history: usize,

    /// Trading enabled flag
    enabled: bool,

    /// Current equity
    current_equity: f64,
}

impl TradingSystem {
    /// Create a new trading system
    pub fn new(config: TradingConfig, initial_equity: f64) -> Self {
        Self {
            config,
            position: Position::default(),
            equity: vec![initial_equity],
            peak_equity: initial_equity,
            signal_history: VecDeque::with_capacity(1000),
            max_history: 1000,
            enabled: true,
            current_equity: initial_equity,
        }
    }

    /// Generate trading signal from ESN prediction
    pub fn generate_signal(&self, prediction: f64) -> Signal {
        Signal::from_prediction(prediction, &self.config.thresholds)
    }

    /// Calculate target position size from signal
    pub fn calculate_position_size(&self, signal: Signal, current_price: f64) -> f64 {
        let direction = signal.direction();
        let base_size = self.config.max_position * self.config.position_scale;

        direction * base_size
    }

    /// Execute trading decision
    ///
    /// # Arguments
    ///
    /// * `prediction` - Model prediction
    /// * `current_price` - Current market price
    ///
    /// # Returns
    ///
    /// Trade details if a trade was executed
    pub fn execute(
        &mut self,
        prediction: f64,
        current_price: f64,
    ) -> Result<Option<Trade>, TradingError> {
        if !self.enabled {
            return Ok(None);
        }

        // Generate signal
        let signal = self.generate_signal(prediction);
        self.record_signal(signal);

        // Check stop loss / take profit
        self.position.update_pnl(current_price);

        if self.should_stop_loss(current_price) {
            let trade = self.close_position(current_price, TradeReason::StopLoss)?;
            return Ok(Some(trade));
        }

        if self.should_take_profit(current_price) {
            let trade = self.close_position(current_price, TradeReason::TakeProfit)?;
            return Ok(Some(trade));
        }

        // Check max drawdown
        self.update_equity(current_price);
        if self.current_drawdown() > self.config.max_drawdown {
            self.enabled = false;
            if !self.position.is_flat() {
                let trade = self.close_position(current_price, TradeReason::MaxDrawdown)?;
                return Ok(Some(trade));
            }
            return Ok(None);
        }

        // Calculate target position
        let target_position = self.calculate_position_size(signal, current_price);

        // Execute trade if position change needed
        let position_delta = target_position - self.position.size;

        if position_delta.abs() > 0.01 {
            let trade = self.execute_trade(position_delta, current_price, signal)?;
            return Ok(Some(trade));
        }

        Ok(None)
    }

    /// Execute a trade
    fn execute_trade(
        &mut self,
        size: f64,
        price: f64,
        signal: Signal,
    ) -> Result<Trade, TradingError> {
        let transaction_cost = size.abs() * price * self.config.transaction_cost;

        // Update position
        if self.position.is_flat() {
            // Opening new position
            self.position.size = size;
            self.position.entry_price = price;
        } else if (self.position.size > 0.0 && size < 0.0)
            || (self.position.size < 0.0 && size > 0.0)
        {
            // Closing/reversing position
            if size.abs() >= self.position.size.abs() {
                // Full close + possible reversal
                let pnl = self.position.close(price);
                let remaining = size + self.position.size;

                if remaining.abs() > 0.01 {
                    self.position.size = remaining;
                    self.position.entry_price = price;
                }

                self.position.realized_pnl += pnl - transaction_cost;
            } else {
                // Partial close
                let closed_size = size.abs();
                let pnl =
                    closed_size * (price - self.position.entry_price) * self.position.size.signum();
                self.position.realized_pnl += pnl - transaction_cost;
                self.position.size += size;
            }
        } else {
            // Adding to position
            let total_cost =
                self.position.size * self.position.entry_price + size * price;
            self.position.size += size;
            self.position.entry_price = total_cost / self.position.size;
        }

        self.position.n_trades += 1;

        Ok(Trade {
            size,
            price,
            signal,
            reason: TradeReason::Signal,
            transaction_cost,
            timestamp: std::time::SystemTime::now(),
        })
    }

    /// Close current position
    fn close_position(
        &mut self,
        price: f64,
        reason: TradeReason,
    ) -> Result<Trade, TradingError> {
        let size = -self.position.size;
        let transaction_cost = size.abs() * price * self.config.transaction_cost;

        let pnl = self.position.close(price) - transaction_cost;
        self.position.n_trades += 1;

        Ok(Trade {
            size,
            price,
            signal: if size > 0.0 { Signal::Buy } else { Signal::Sell },
            reason,
            transaction_cost,
            timestamp: std::time::SystemTime::now(),
        })
    }

    /// Check if stop loss should trigger
    fn should_stop_loss(&self, current_price: f64) -> bool {
        if let Some(stop_loss) = self.config.stop_loss {
            if !self.position.is_flat() {
                let pct_change = if self.position.is_long() {
                    (current_price - self.position.entry_price) / self.position.entry_price
                } else {
                    (self.position.entry_price - current_price) / self.position.entry_price
                };

                return pct_change < -stop_loss;
            }
        }
        false
    }

    /// Check if take profit should trigger
    fn should_take_profit(&self, current_price: f64) -> bool {
        if let Some(take_profit) = self.config.take_profit {
            if !self.position.is_flat() {
                let pct_change = if self.position.is_long() {
                    (current_price - self.position.entry_price) / self.position.entry_price
                } else {
                    (self.position.entry_price - current_price) / self.position.entry_price
                };

                return pct_change > take_profit;
            }
        }
        false
    }

    /// Update equity tracking
    fn update_equity(&mut self, current_price: f64) {
        self.position.update_pnl(current_price);
        self.current_equity = self.equity[0] + self.position.realized_pnl + self.position.unrealized_pnl;
        self.equity.push(self.current_equity);

        if self.current_equity > self.peak_equity {
            self.peak_equity = self.current_equity;
        }
    }

    /// Calculate current drawdown
    fn current_drawdown(&self) -> f64 {
        if self.peak_equity > 0.0 {
            (self.peak_equity - self.current_equity) / self.peak_equity
        } else {
            0.0
        }
    }

    /// Record signal to history
    fn record_signal(&mut self, signal: Signal) {
        self.signal_history.push_back(signal);
        if self.signal_history.len() > self.max_history {
            self.signal_history.pop_front();
        }
    }

    /// Get current position
    pub fn position(&self) -> &Position {
        &self.position
    }

    /// Get equity curve
    pub fn equity_curve(&self) -> &[f64] {
        &self.equity
    }

    /// Get current equity
    pub fn current_equity(&self) -> f64 {
        self.current_equity
    }

    /// Get total return
    pub fn total_return(&self) -> f64 {
        if self.equity[0] > 0.0 {
            (self.current_equity - self.equity[0]) / self.equity[0]
        } else {
            0.0
        }
    }

    /// Check if trading is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Re-enable trading
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Reset the trading system
    pub fn reset(&mut self, initial_equity: f64) {
        self.position = Position::default();
        self.equity = vec![initial_equity];
        self.peak_equity = initial_equity;
        self.signal_history.clear();
        self.enabled = true;
        self.current_equity = initial_equity;
    }
}

/// Trade reason
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradeReason {
    Signal,
    StopLoss,
    TakeProfit,
    MaxDrawdown,
}

/// Trade record
#[derive(Debug, Clone)]
pub struct Trade {
    /// Trade size (positive = buy, negative = sell)
    pub size: f64,

    /// Execution price
    pub price: f64,

    /// Signal that triggered the trade
    pub signal: Signal,

    /// Reason for the trade
    pub reason: TradeReason,

    /// Transaction cost
    pub transaction_cost: f64,

    /// Timestamp
    pub timestamp: std::time::SystemTime,
}

impl Trade {
    /// Get trade value
    pub fn value(&self) -> f64 {
        self.size * self.price
    }

    /// Check if this is a buy
    pub fn is_buy(&self) -> bool {
        self.size > 0.0
    }

    /// Check if this is a sell
    pub fn is_sell(&self) -> bool {
        self.size < 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_from_prediction() {
        let thresholds = SignalThresholds::default();

        assert_eq!(Signal::from_prediction(0.6, &thresholds), Signal::StrongBuy);
        assert_eq!(Signal::from_prediction(0.3, &thresholds), Signal::Buy);
        assert_eq!(Signal::from_prediction(0.0, &thresholds), Signal::Hold);
        assert_eq!(Signal::from_prediction(-0.3, &thresholds), Signal::Sell);
        assert_eq!(Signal::from_prediction(-0.6, &thresholds), Signal::StrongSell);
    }

    #[test]
    fn test_position() {
        let mut position = Position::default();
        assert!(position.is_flat());

        position.size = 1.0;
        position.entry_price = 100.0;
        assert!(position.is_long());

        position.update_pnl(110.0);
        assert!((position.unrealized_pnl - 10.0).abs() < 0.01);

        let pnl = position.close(110.0);
        assert!((pnl - 10.0).abs() < 0.01);
        assert!(position.is_flat());
    }

    #[test]
    fn test_trading_system() {
        let config = TradingConfig::default();
        let mut system = TradingSystem::new(config, 10000.0);

        // Strong buy signal
        let result = system.execute(0.6, 100.0).unwrap();
        assert!(result.is_some());

        let trade = result.unwrap();
        assert!(trade.is_buy());
        assert!(system.position().is_long());

        // Hold signal - no trade
        let result = system.execute(0.0, 101.0).unwrap();
        assert!(result.is_none());

        // Strong sell - close and reverse
        let result = system.execute(-0.6, 102.0).unwrap();
        assert!(result.is_some());
    }
}
