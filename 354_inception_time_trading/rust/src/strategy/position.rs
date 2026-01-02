//! Position management
//!
//! This module handles position sizing and tracking.

use serde::{Deserialize, Serialize};

use super::signals::Signal;

/// Position state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionState {
    /// No position
    Flat,
    /// Long position
    Long,
    /// Short position
    Short,
}

/// Current position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub state: PositionState,
    pub size: f64,
    pub entry_price: f64,
    pub entry_time: i64,
    pub unrealized_pnl: f64,
}

impl Default for Position {
    fn default() -> Self {
        Self {
            state: PositionState::Flat,
            size: 0.0,
            entry_price: 0.0,
            entry_time: 0,
            unrealized_pnl: 0.0,
        }
    }
}

impl Position {
    /// Update unrealized PnL
    pub fn update_pnl(&mut self, current_price: f64) {
        match self.state {
            PositionState::Long => {
                self.unrealized_pnl = (current_price - self.entry_price) * self.size;
            }
            PositionState::Short => {
                self.unrealized_pnl = (self.entry_price - current_price) * self.size;
            }
            PositionState::Flat => {
                self.unrealized_pnl = 0.0;
            }
        }
    }

    /// Calculate return percentage
    pub fn return_pct(&self) -> f64 {
        if self.entry_price > 0.0 && self.size > 0.0 {
            match self.state {
                PositionState::Long | PositionState::Short => {
                    self.unrealized_pnl / (self.entry_price * self.size) * 100.0
                }
                PositionState::Flat => 0.0,
            }
        } else {
            0.0
        }
    }
}

/// Position manager
#[derive(Debug, Clone)]
pub struct PositionManager {
    /// Current position
    position: Position,
    /// Maximum position size (as fraction of capital)
    max_position_size: f64,
    /// Current capital
    capital: f64,
    /// Initial capital
    initial_capital: f64,
    /// Realized PnL
    realized_pnl: f64,
}

impl PositionManager {
    /// Create a new position manager
    pub fn new(initial_capital: f64, max_position_size: f64) -> Self {
        Self {
            position: Position::default(),
            max_position_size,
            capital: initial_capital,
            initial_capital,
            realized_pnl: 0.0,
        }
    }

    /// Get current position
    pub fn position(&self) -> &Position {
        &self.position
    }

    /// Get current capital
    pub fn capital(&self) -> f64 {
        self.capital
    }

    /// Get total equity (capital + unrealized PnL)
    pub fn equity(&self) -> f64 {
        self.capital + self.position.unrealized_pnl
    }

    /// Calculate position size
    pub fn calculate_size(&self, price: f64, confidence: f64) -> f64 {
        let max_value = self.capital * self.max_position_size;
        let adjusted_value = max_value * confidence;
        adjusted_value / price
    }

    /// Open a position
    pub fn open_position(
        &mut self,
        signal: Signal,
        price: f64,
        size: f64,
        timestamp: i64,
    ) -> bool {
        if self.position.state != PositionState::Flat {
            return false;
        }

        let state = match signal {
            Signal::Buy => PositionState::Long,
            Signal::Sell => PositionState::Short,
            Signal::Hold => return false,
        };

        let cost = price * size;
        if cost > self.capital {
            return false;
        }

        self.position = Position {
            state,
            size,
            entry_price: price,
            entry_time: timestamp,
            unrealized_pnl: 0.0,
        };

        true
    }

    /// Close current position
    pub fn close_position(&mut self, price: f64) -> f64 {
        if self.position.state == PositionState::Flat {
            return 0.0;
        }

        self.position.update_pnl(price);
        let pnl = self.position.unrealized_pnl;

        self.capital += pnl;
        self.realized_pnl += pnl;

        self.position = Position::default();

        pnl
    }

    /// Update position with current price
    pub fn update(&mut self, current_price: f64) {
        self.position.update_pnl(current_price);
    }

    /// Get realized PnL
    pub fn realized_pnl(&self) -> f64 {
        self.realized_pnl
    }

    /// Get total return percentage
    pub fn total_return_pct(&self) -> f64 {
        (self.equity() - self.initial_capital) / self.initial_capital * 100.0
    }

    /// Process a signal
    pub fn process_signal(
        &mut self,
        signal: Signal,
        price: f64,
        confidence: f64,
        timestamp: i64,
    ) -> Option<f64> {
        match (self.position.state, signal) {
            // No position, open new
            (PositionState::Flat, Signal::Buy) | (PositionState::Flat, Signal::Sell) => {
                let size = self.calculate_size(price, confidence);
                if self.open_position(signal, price, size, timestamp) {
                    None
                } else {
                    None
                }
            }
            // Long position, sell signal -> close
            (PositionState::Long, Signal::Sell) => {
                let pnl = self.close_position(price);
                Some(pnl)
            }
            // Short position, buy signal -> close
            (PositionState::Short, Signal::Buy) => {
                let pnl = self.close_position(price);
                Some(pnl)
            }
            // Hold or same direction
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_manager() {
        let mut manager = PositionManager::new(100000.0, 0.2);

        // Open long position
        assert!(manager.open_position(Signal::Buy, 50000.0, 0.4, 1000));
        assert_eq!(manager.position().state, PositionState::Long);

        // Update with higher price
        manager.update(51000.0);
        assert!(manager.position().unrealized_pnl > 0.0);

        // Close position
        let pnl = manager.close_position(51000.0);
        assert!(pnl > 0.0);
        assert_eq!(manager.position().state, PositionState::Flat);
    }

    #[test]
    fn test_position_sizing() {
        let manager = PositionManager::new(100000.0, 0.2);

        // Max 20% of capital
        let size = manager.calculate_size(50000.0, 1.0);
        assert!((size - 0.4).abs() < 0.01); // 20000 / 50000 = 0.4

        // With 50% confidence
        let size = manager.calculate_size(50000.0, 0.5);
        assert!((size - 0.2).abs() < 0.01); // 10000 / 50000 = 0.2
    }
}
