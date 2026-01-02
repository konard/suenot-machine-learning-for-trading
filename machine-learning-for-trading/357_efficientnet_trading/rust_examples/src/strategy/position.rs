//! Position management

use crate::strategy::{Signal, SignalType};

/// Position side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionSide {
    Long,
    Short,
    Flat,
}

/// Trading position
#[derive(Debug, Clone)]
pub struct Position {
    pub side: PositionSide,
    pub size: f64,
    pub entry_price: f64,
    pub entry_time: u64,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub unrealized_pnl: f64,
}

impl Position {
    /// Create a flat (no position) state
    pub fn flat() -> Self {
        Self {
            side: PositionSide::Flat,
            size: 0.0,
            entry_price: 0.0,
            entry_time: 0,
            stop_loss: None,
            take_profit: None,
            unrealized_pnl: 0.0,
        }
    }

    /// Create a long position
    pub fn long(size: f64, entry_price: f64, entry_time: u64) -> Self {
        Self {
            side: PositionSide::Long,
            size,
            entry_price,
            entry_time,
            stop_loss: None,
            take_profit: None,
            unrealized_pnl: 0.0,
        }
    }

    /// Create a short position
    pub fn short(size: f64, entry_price: f64, entry_time: u64) -> Self {
        Self {
            side: PositionSide::Short,
            size,
            entry_price,
            entry_time,
            stop_loss: None,
            take_profit: None,
            unrealized_pnl: 0.0,
        }
    }

    /// Check if position is open
    pub fn is_open(&self) -> bool {
        self.side != PositionSide::Flat && self.size > 0.0
    }

    /// Update unrealized PnL
    pub fn update_pnl(&mut self, current_price: f64) {
        self.unrealized_pnl = match self.side {
            PositionSide::Long => (current_price - self.entry_price) * self.size,
            PositionSide::Short => (self.entry_price - current_price) * self.size,
            PositionSide::Flat => 0.0,
        };
    }

    /// Calculate PnL percentage
    pub fn pnl_percent(&self, current_price: f64) -> f64 {
        if self.entry_price == 0.0 {
            return 0.0;
        }

        match self.side {
            PositionSide::Long => (current_price / self.entry_price - 1.0) * 100.0,
            PositionSide::Short => (1.0 - current_price / self.entry_price) * 100.0,
            PositionSide::Flat => 0.0,
        }
    }

    /// Check if stop loss hit
    pub fn is_stop_hit(&self, current_price: f64) -> bool {
        match (self.side, self.stop_loss) {
            (PositionSide::Long, Some(sl)) => current_price <= sl,
            (PositionSide::Short, Some(sl)) => current_price >= sl,
            _ => false,
        }
    }

    /// Check if take profit hit
    pub fn is_tp_hit(&self, current_price: f64) -> bool {
        match (self.side, self.take_profit) {
            (PositionSide::Long, Some(tp)) => current_price >= tp,
            (PositionSide::Short, Some(tp)) => current_price <= tp,
            _ => false,
        }
    }

    /// Set stop loss
    pub fn with_stop_loss(mut self, stop_loss: f64) -> Self {
        self.stop_loss = Some(stop_loss);
        self
    }

    /// Set take profit
    pub fn with_take_profit(mut self, take_profit: f64) -> Self {
        self.take_profit = Some(take_profit);
        self
    }
}

/// Position manager for handling entries and exits
pub struct PositionManager {
    pub position: Position,
    pub max_position_size: f64,
    pub risk_per_trade: f64,
    pub stop_loss_pct: f64,
    pub take_profit_pct: f64,
}

impl PositionManager {
    pub fn new(max_position_size: f64) -> Self {
        Self {
            position: Position::flat(),
            max_position_size,
            risk_per_trade: 0.02, // 2% risk per trade
            stop_loss_pct: 0.02,   // 2% stop loss
            take_profit_pct: 0.04, // 4% take profit
        }
    }

    /// Process a trading signal
    pub fn process_signal(&mut self, signal: &Signal, capital: f64) -> Option<TradeAction> {
        if !signal.is_actionable(0.6) {
            return None;
        }

        match (&self.position.side, signal.signal_type) {
            // No position, open new
            (PositionSide::Flat, SignalType::Buy) => {
                let size = self.calculate_position_size(capital, signal.price);
                let position = Position::long(size, signal.price, signal.timestamp)
                    .with_stop_loss(signal.price * (1.0 - self.stop_loss_pct))
                    .with_take_profit(signal.price * (1.0 + self.take_profit_pct));
                self.position = position;

                Some(TradeAction::Open {
                    side: PositionSide::Long,
                    size,
                    price: signal.price,
                })
            }
            (PositionSide::Flat, SignalType::Sell) => {
                let size = self.calculate_position_size(capital, signal.price);
                let position = Position::short(size, signal.price, signal.timestamp)
                    .with_stop_loss(signal.price * (1.0 + self.stop_loss_pct))
                    .with_take_profit(signal.price * (1.0 - self.take_profit_pct));
                self.position = position;

                Some(TradeAction::Open {
                    side: PositionSide::Short,
                    size,
                    price: signal.price,
                })
            }
            // Close opposite position
            (PositionSide::Long, SignalType::Sell) => {
                let pnl = self.position.pnl_percent(signal.price);
                let size = self.position.size;
                self.position = Position::flat();

                Some(TradeAction::Close {
                    size,
                    price: signal.price,
                    pnl,
                })
            }
            (PositionSide::Short, SignalType::Buy) => {
                let pnl = self.position.pnl_percent(signal.price);
                let size = self.position.size;
                self.position = Position::flat();

                Some(TradeAction::Close {
                    size,
                    price: signal.price,
                    pnl,
                })
            }
            _ => None,
        }
    }

    /// Check and execute stop loss / take profit
    pub fn check_exit(&mut self, current_price: f64) -> Option<TradeAction> {
        if !self.position.is_open() {
            return None;
        }

        if self.position.is_stop_hit(current_price) {
            let pnl = self.position.pnl_percent(current_price);
            let size = self.position.size;
            self.position = Position::flat();

            return Some(TradeAction::StopLoss {
                size,
                price: current_price,
                pnl,
            });
        }

        if self.position.is_tp_hit(current_price) {
            let pnl = self.position.pnl_percent(current_price);
            let size = self.position.size;
            self.position = Position::flat();

            return Some(TradeAction::TakeProfit {
                size,
                price: current_price,
                pnl,
            });
        }

        None
    }

    /// Calculate position size based on risk
    fn calculate_position_size(&self, capital: f64, price: f64) -> f64 {
        let risk_amount = capital * self.risk_per_trade;
        let size_from_risk = risk_amount / (price * self.stop_loss_pct);
        size_from_risk.min(self.max_position_size)
    }

    /// Update position with current price
    pub fn update(&mut self, current_price: f64) {
        self.position.update_pnl(current_price);
    }
}

/// Trade action
#[derive(Debug, Clone)]
pub enum TradeAction {
    Open {
        side: PositionSide,
        size: f64,
        price: f64,
    },
    Close {
        size: f64,
        price: f64,
        pnl: f64,
    },
    StopLoss {
        size: f64,
        price: f64,
        pnl: f64,
    },
    TakeProfit {
        size: f64,
        price: f64,
        pnl: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_pnl() {
        let mut pos = Position::long(1.0, 100.0, 0);
        pos.update_pnl(110.0);
        assert_eq!(pos.unrealized_pnl, 10.0);

        let pct = pos.pnl_percent(110.0);
        assert!((pct - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_stop_loss() {
        let pos = Position::long(1.0, 100.0, 0).with_stop_loss(95.0);
        assert!(pos.is_stop_hit(94.0));
        assert!(!pos.is_stop_hit(96.0));
    }

    #[test]
    fn test_position_manager() {
        let mut manager = PositionManager::new(10.0);
        let signal = Signal::new(SignalType::Buy, 0.8, 0, 100.0);

        let action = manager.process_signal(&signal, 10000.0);
        assert!(action.is_some());
        assert!(manager.position.is_open());
    }
}
