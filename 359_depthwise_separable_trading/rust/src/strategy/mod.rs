//! Trading strategy module
//!
//! Provides trading signal generation and backtesting framework.

mod backtest;
mod signals;

pub use backtest::Backtest;
pub use signals::{Signal, SignalGenerator, TradingStrategy};

use ndarray::Array1;
use thiserror::Error;

/// Strategy errors
#[derive(Error, Debug)]
pub enum StrategyError {
    #[error("Insufficient data: need {needed}, got {got}")]
    InsufficientData { needed: usize, got: usize },

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Computation error: {0}")]
    ComputationError(String),
}

/// Position in the market
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Position {
    /// No position
    Flat,
    /// Long position
    Long(f64),
    /// Short position
    Short(f64),
}

impl Position {
    /// Get position size (positive for long, negative for short)
    pub fn size(&self) -> f64 {
        match self {
            Position::Flat => 0.0,
            Position::Long(size) => *size,
            Position::Short(size) => -*size,
        }
    }

    /// Check if position is flat
    pub fn is_flat(&self) -> bool {
        matches!(self, Position::Flat)
    }

    /// Check if position is long
    pub fn is_long(&self) -> bool {
        matches!(self, Position::Long(_))
    }

    /// Check if position is short
    pub fn is_short(&self) -> bool {
        matches!(self, Position::Short(_))
    }
}

/// Trade record
#[derive(Debug, Clone)]
pub struct Trade {
    /// Entry timestamp index
    pub entry_idx: usize,
    /// Exit timestamp index
    pub exit_idx: usize,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: f64,
    /// Position size
    pub size: f64,
    /// Trade direction (true = long, false = short)
    pub is_long: bool,
    /// Profit/Loss in absolute terms
    pub pnl: f64,
    /// Profit/Loss in percentage
    pub pnl_pct: f64,
}

impl Trade {
    /// Calculate trade metrics
    pub fn new(
        entry_idx: usize,
        exit_idx: usize,
        entry_price: f64,
        exit_price: f64,
        size: f64,
        is_long: bool,
    ) -> Self {
        let pnl = if is_long {
            (exit_price - entry_price) * size
        } else {
            (entry_price - exit_price) * size
        };

        let pnl_pct = if is_long {
            (exit_price - entry_price) / entry_price * 100.0
        } else {
            (entry_price - exit_price) / entry_price * 100.0
        };

        Self {
            entry_idx,
            exit_idx,
            entry_price,
            exit_price,
            size,
            is_long,
            pnl,
            pnl_pct,
        }
    }

    /// Get trade duration in bars
    pub fn duration(&self) -> usize {
        self.exit_idx - self.entry_idx
    }

    /// Check if trade is profitable
    pub fn is_profitable(&self) -> bool {
        self.pnl > 0.0
    }
}

/// Portfolio state
#[derive(Debug, Clone)]
pub struct Portfolio {
    /// Current cash
    pub cash: f64,
    /// Current position
    pub position: Position,
    /// Initial capital
    pub initial_capital: f64,
    /// Commission rate (e.g., 0.001 = 0.1%)
    pub commission: f64,
    /// Equity history
    pub equity_history: Vec<f64>,
    /// Trade history
    pub trades: Vec<Trade>,
}

impl Portfolio {
    /// Create new portfolio
    pub fn new(initial_capital: f64, commission: f64) -> Self {
        Self {
            cash: initial_capital,
            position: Position::Flat,
            initial_capital,
            commission,
            equity_history: vec![initial_capital],
            trades: Vec::new(),
        }
    }

    /// Get current equity (cash + position value)
    pub fn equity(&self, current_price: f64) -> f64 {
        self.cash + self.position.size() * current_price
    }

    /// Open a long position
    pub fn open_long(&mut self, price: f64, size: f64, idx: usize) {
        let cost = size * price * (1.0 + self.commission);
        if cost <= self.cash {
            self.cash -= cost;
            self.position = Position::Long(size);
        }
    }

    /// Open a short position
    pub fn open_short(&mut self, price: f64, size: f64, idx: usize) {
        let margin = size * price * (1.0 + self.commission);
        if margin <= self.cash {
            self.cash -= margin;
            self.position = Position::Short(size);
        }
    }

    /// Close current position
    pub fn close_position(&mut self, price: f64, entry_idx: usize, exit_idx: usize, entry_price: f64) {
        match self.position {
            Position::Long(size) => {
                let proceeds = size * price * (1.0 - self.commission);
                self.cash += proceeds;

                let trade = Trade::new(entry_idx, exit_idx, entry_price, price, size, true);
                self.trades.push(trade);
            }
            Position::Short(size) => {
                let pnl = (entry_price - price) * size;
                let cost = size * price * self.commission;
                self.cash += size * entry_price + pnl - cost;

                let trade = Trade::new(entry_idx, exit_idx, entry_price, price, size, false);
                self.trades.push(trade);
            }
            Position::Flat => {}
        }
        self.position = Position::Flat;
    }

    /// Update equity history
    pub fn update_equity(&mut self, current_price: f64) {
        self.equity_history.push(self.equity(current_price));
    }

    /// Get total return
    pub fn total_return(&self) -> f64 {
        if let Some(&last_equity) = self.equity_history.last() {
            (last_equity - self.initial_capital) / self.initial_capital
        } else {
            0.0
        }
    }

    /// Get equity curve as array
    pub fn equity_curve(&self) -> Array1<f64> {
        Array1::from_vec(self.equity_history.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position() {
        assert!(Position::Flat.is_flat());
        assert!(Position::Long(100.0).is_long());
        assert!(Position::Short(100.0).is_short());
        assert_eq!(Position::Long(100.0).size(), 100.0);
        assert_eq!(Position::Short(100.0).size(), -100.0);
    }

    #[test]
    fn test_trade() {
        let trade = Trade::new(0, 10, 100.0, 110.0, 1.0, true);
        assert!(trade.is_profitable());
        assert_eq!(trade.pnl, 10.0);
        assert_eq!(trade.duration(), 10);
    }

    #[test]
    fn test_portfolio() {
        let mut portfolio = Portfolio::new(10000.0, 0.001);

        assert_eq!(portfolio.cash, 10000.0);
        assert!(portfolio.position.is_flat());

        portfolio.open_long(100.0, 10.0, 0);
        assert!(portfolio.position.is_long());

        portfolio.close_position(110.0, 0, 5, 100.0);
        assert!(portfolio.position.is_flat());
        assert!(portfolio.cash > 10000.0); // Should have profit
    }
}
