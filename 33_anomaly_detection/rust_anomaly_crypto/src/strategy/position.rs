//! Position management based on anomaly signals

use super::{Signal, TradingSignal};

/// Position side
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PositionSide {
    Long,
    Short,
    Flat,
}

/// Current position information
#[derive(Clone, Debug)]
pub struct Position {
    /// Position side
    pub side: PositionSide,
    /// Position size (absolute value)
    pub size: f64,
    /// Entry price
    pub entry_price: f64,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
    /// Maximum position size allowed
    pub max_size: f64,
}

impl Position {
    /// Create a flat position
    pub fn flat(max_size: f64) -> Self {
        Self {
            side: PositionSide::Flat,
            size: 0.0,
            entry_price: 0.0,
            unrealized_pnl: 0.0,
            max_size,
        }
    }

    /// Check if position is flat
    pub fn is_flat(&self) -> bool {
        self.side == PositionSide::Flat || self.size == 0.0
    }

    /// Get signed position size (negative for short)
    pub fn signed_size(&self) -> f64 {
        match self.side {
            PositionSide::Long => self.size,
            PositionSide::Short => -self.size,
            PositionSide::Flat => 0.0,
        }
    }

    /// Update unrealized PnL based on current price
    pub fn update_pnl(&mut self, current_price: f64) {
        if self.is_flat() {
            self.unrealized_pnl = 0.0;
            return;
        }

        let price_change = current_price - self.entry_price;
        self.unrealized_pnl = match self.side {
            PositionSide::Long => self.size * price_change,
            PositionSide::Short => -self.size * price_change,
            PositionSide::Flat => 0.0,
        };
    }
}

/// Position manager with anomaly-based risk management
pub struct PositionManager {
    /// Current position
    pub position: Position,
    /// Base position size (before anomaly adjustments)
    base_size: f64,
    /// Current anomaly-adjusted size multiplier (0-1)
    size_multiplier: f64,
    /// Realized PnL
    realized_pnl: f64,
    /// Number of trades
    trade_count: usize,
}

impl PositionManager {
    /// Create a new position manager
    pub fn new(max_position_size: f64) -> Self {
        Self {
            position: Position::flat(max_position_size),
            base_size: max_position_size,
            size_multiplier: 1.0,
            realized_pnl: 0.0,
            trade_count: 0,
        }
    }

    /// Get current size multiplier
    pub fn size_multiplier(&self) -> f64 {
        self.size_multiplier
    }

    /// Get realized PnL
    pub fn realized_pnl(&self) -> f64 {
        self.realized_pnl
    }

    /// Get total PnL (realized + unrealized)
    pub fn total_pnl(&self) -> f64 {
        self.realized_pnl + self.position.unrealized_pnl
    }

    /// Get number of trades
    pub fn trade_count(&self) -> usize {
        self.trade_count
    }

    /// Process a trading signal
    ///
    /// Returns: (executed_action, new_position_size)
    pub fn process_signal(
        &mut self,
        signal: &TradingSignal,
        current_price: f64,
    ) -> (String, f64) {
        self.position.update_pnl(current_price);

        match signal.signal {
            Signal::Hold => {
                // Gradually restore size multiplier if below 1.0
                if self.size_multiplier < 1.0 {
                    self.size_multiplier = (self.size_multiplier + 0.1).min(1.0);
                }
                ("HOLD".to_string(), self.position.size)
            }

            Signal::ReducePosition => {
                // Reduce size multiplier
                self.size_multiplier =
                    (self.size_multiplier + signal.position_adjustment).max(0.1);

                if !self.position.is_flat() {
                    let new_size = self.base_size * self.size_multiplier;
                    let reduction = self.position.size - new_size;

                    if reduction > 0.0 {
                        // Close partial position
                        self.realized_pnl +=
                            self.position.unrealized_pnl * (reduction / self.position.size);
                        self.position.size = new_size;
                        self.trade_count += 1;

                        return (format!("REDUCE by {:.2}", reduction), new_size);
                    }
                }

                ("REDUCE (no position)".to_string(), self.position.size)
            }

            Signal::ExitAll => {
                if !self.position.is_flat() {
                    self.realized_pnl += self.position.unrealized_pnl;
                    self.position.size = 0.0;
                    self.position.side = PositionSide::Flat;
                    self.position.unrealized_pnl = 0.0;
                    self.size_multiplier = 0.1; // Very reduced for new positions
                    self.trade_count += 1;

                    return ("EXIT ALL".to_string(), 0.0);
                }

                ("EXIT (no position)".to_string(), 0.0)
            }

            Signal::EntryLong => {
                if self.position.is_flat() {
                    let size = self.base_size * self.size_multiplier * signal.confidence;
                    self.position.side = PositionSide::Long;
                    self.position.size = size;
                    self.position.entry_price = current_price;
                    self.trade_count += 1;

                    return (format!("ENTRY LONG {:.4}", size), size);
                } else if self.position.side == PositionSide::Short {
                    // Close short and go long
                    self.realized_pnl += self.position.unrealized_pnl;
                    let size = self.base_size * self.size_multiplier * signal.confidence;
                    self.position.side = PositionSide::Long;
                    self.position.size = size;
                    self.position.entry_price = current_price;
                    self.position.unrealized_pnl = 0.0;
                    self.trade_count += 2; // Close + open

                    return (format!("FLIP TO LONG {:.4}", size), size);
                }

                ("ENTRY LONG (already long)".to_string(), self.position.size)
            }

            Signal::EntryShort => {
                if self.position.is_flat() {
                    let size = self.base_size * self.size_multiplier * signal.confidence;
                    self.position.side = PositionSide::Short;
                    self.position.size = size;
                    self.position.entry_price = current_price;
                    self.trade_count += 1;

                    return (format!("ENTRY SHORT {:.4}", size), size);
                } else if self.position.side == PositionSide::Long {
                    // Close long and go short
                    self.realized_pnl += self.position.unrealized_pnl;
                    let size = self.base_size * self.size_multiplier * signal.confidence;
                    self.position.side = PositionSide::Short;
                    self.position.size = size;
                    self.position.entry_price = current_price;
                    self.position.unrealized_pnl = 0.0;
                    self.trade_count += 2;

                    return (format!("FLIP TO SHORT {:.4}", size), size);
                }

                ("ENTRY SHORT (already short)".to_string(), self.position.size)
            }
        }
    }

    /// Close all positions
    pub fn close_all(&mut self, current_price: f64) {
        self.position.update_pnl(current_price);

        if !self.position.is_flat() {
            self.realized_pnl += self.position.unrealized_pnl;
            self.position.size = 0.0;
            self.position.side = PositionSide::Flat;
            self.position.unrealized_pnl = 0.0;
            self.trade_count += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_flat() {
        let pos = Position::flat(1.0);
        assert!(pos.is_flat());
        assert_eq!(pos.signed_size(), 0.0);
    }

    #[test]
    fn test_position_pnl() {
        let mut pos = Position::flat(1.0);
        pos.side = PositionSide::Long;
        pos.size = 1.0;
        pos.entry_price = 100.0;

        pos.update_pnl(110.0);
        assert_eq!(pos.unrealized_pnl, 10.0);

        pos.side = PositionSide::Short;
        pos.update_pnl(110.0);
        assert_eq!(pos.unrealized_pnl, -10.0);
    }

    #[test]
    fn test_position_manager_entry_exit() {
        let mut manager = PositionManager::new(1.0);

        // Entry long
        let signal = TradingSignal::entry_long(0.3, 0.8);
        let (action, size) = manager.process_signal(&signal, 100.0);
        assert!(action.contains("ENTRY LONG"));
        assert!(size > 0.0);

        // Exit all
        let signal = TradingSignal::exit_all(2.0);
        let (action, size) = manager.process_signal(&signal, 105.0);
        assert!(action.contains("EXIT"));
        assert_eq!(size, 0.0);
        assert!(manager.realized_pnl > 0.0);
    }
}
