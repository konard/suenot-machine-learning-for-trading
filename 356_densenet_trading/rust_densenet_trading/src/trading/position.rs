//! Position Management
//!
//! Track and manage trading positions.

use serde::{Deserialize, Serialize};
use super::signal::SignalDirection;

/// Position side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionSide {
    Long,
    Short,
    Flat,
}

impl From<SignalDirection> for PositionSide {
    fn from(dir: SignalDirection) -> Self {
        match dir {
            SignalDirection::Long => PositionSide::Long,
            SignalDirection::Short => PositionSide::Short,
            SignalDirection::Neutral => PositionSide::Flat,
        }
    }
}

/// A trading position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Position side
    pub side: PositionSide,

    /// Entry price
    pub entry_price: f64,

    /// Position size (in base currency)
    pub size: f64,

    /// Entry timestamp
    pub entry_time: u64,

    /// Stop loss price
    pub stop_loss: Option<f64>,

    /// Take profit price
    pub take_profit: Option<f64>,

    /// Maximum adverse excursion (worst unrealized P&L)
    pub max_adverse: f64,

    /// Maximum favorable excursion (best unrealized P&L)
    pub max_favorable: f64,
}

impl Position {
    /// Create a new position
    pub fn new(side: PositionSide, entry_price: f64, size: f64, entry_time: u64) -> Self {
        Self {
            side,
            entry_price,
            size,
            entry_time,
            stop_loss: None,
            take_profit: None,
            max_adverse: 0.0,
            max_favorable: 0.0,
        }
    }

    /// Open a long position
    pub fn long(entry_price: f64, size: f64, entry_time: u64) -> Self {
        Self::new(PositionSide::Long, entry_price, size, entry_time)
    }

    /// Open a short position
    pub fn short(entry_price: f64, size: f64, entry_time: u64) -> Self {
        Self::new(PositionSide::Short, entry_price, size, entry_time)
    }

    /// Set stop loss
    pub fn with_stop_loss(mut self, price: f64) -> Self {
        self.stop_loss = Some(price);
        self
    }

    /// Set take profit
    pub fn with_take_profit(mut self, price: f64) -> Self {
        self.take_profit = Some(price);
        self
    }

    /// Check if position is flat
    pub fn is_flat(&self) -> bool {
        self.side == PositionSide::Flat || self.size == 0.0
    }

    /// Calculate unrealized P&L at current price
    pub fn unrealized_pnl(&self, current_price: f64) -> f64 {
        if self.is_flat() {
            return 0.0;
        }

        let price_change = current_price - self.entry_price;
        let multiplier = match self.side {
            PositionSide::Long => 1.0,
            PositionSide::Short => -1.0,
            PositionSide::Flat => 0.0,
        };

        price_change * self.size * multiplier
    }

    /// Calculate unrealized P&L as percentage
    pub fn unrealized_pnl_percent(&self, current_price: f64) -> f64 {
        if self.is_flat() || self.entry_price == 0.0 {
            return 0.0;
        }

        let price_change_pct = (current_price - self.entry_price) / self.entry_price;
        let multiplier = match self.side {
            PositionSide::Long => 1.0,
            PositionSide::Short => -1.0,
            PositionSide::Flat => 0.0,
        };

        price_change_pct * 100.0 * multiplier
    }

    /// Update max adverse/favorable excursions
    pub fn update_excursions(&mut self, current_price: f64) {
        let pnl = self.unrealized_pnl_percent(current_price);

        if pnl < self.max_adverse {
            self.max_adverse = pnl;
        }
        if pnl > self.max_favorable {
            self.max_favorable = pnl;
        }
    }

    /// Check if stop loss is hit
    pub fn is_stop_hit(&self, current_price: f64) -> bool {
        match (self.stop_loss, self.side) {
            (Some(sl), PositionSide::Long) => current_price <= sl,
            (Some(sl), PositionSide::Short) => current_price >= sl,
            _ => false,
        }
    }

    /// Check if take profit is hit
    pub fn is_target_hit(&self, current_price: f64) -> bool {
        match (self.take_profit, self.side) {
            (Some(tp), PositionSide::Long) => current_price >= tp,
            (Some(tp), PositionSide::Short) => current_price <= tp,
            _ => false,
        }
    }

    /// Calculate position value at current price
    pub fn value(&self, current_price: f64) -> f64 {
        self.size * current_price
    }

    /// Calculate notional value at entry
    pub fn notional(&self) -> f64 {
        self.size * self.entry_price
    }
}

/// Closed position with realized P&L
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosedPosition {
    /// Original position
    pub position: Position,

    /// Exit price
    pub exit_price: f64,

    /// Exit timestamp
    pub exit_time: u64,

    /// Realized P&L
    pub realized_pnl: f64,

    /// Realized P&L percentage
    pub realized_pnl_percent: f64,

    /// Trading fees paid
    pub fees: f64,

    /// Exit reason
    pub exit_reason: ExitReason,
}

/// Reason for closing a position
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExitReason {
    Signal,
    StopLoss,
    TakeProfit,
    Manual,
    Timeout,
}

impl ClosedPosition {
    /// Create from closing a position
    pub fn close(
        position: Position,
        exit_price: f64,
        exit_time: u64,
        fee_rate: f64,
        reason: ExitReason,
    ) -> Self {
        let pnl = position.unrealized_pnl(exit_price);
        let pnl_pct = position.unrealized_pnl_percent(exit_price);

        // Calculate fees (entry + exit)
        let fees = position.notional() * fee_rate + position.size * exit_price * fee_rate;

        Self {
            position,
            exit_price,
            exit_time,
            realized_pnl: pnl - fees,
            realized_pnl_percent: pnl_pct - (fees / position.notional() * 100.0),
            fees,
            exit_reason: reason,
        }
    }

    /// Check if trade was profitable
    pub fn is_winner(&self) -> bool {
        self.realized_pnl > 0.0
    }

    /// Get holding period in milliseconds
    pub fn holding_period(&self) -> u64 {
        self.exit_time.saturating_sub(self.position.entry_time)
    }

    /// Get R-multiple (profit in terms of risk)
    pub fn r_multiple(&self) -> Option<f64> {
        self.position.stop_loss.map(|sl| {
            let risk = (self.position.entry_price - sl).abs();
            if risk > 0.0 {
                self.realized_pnl / (risk * self.position.size)
            } else {
                0.0
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_long_position() {
        let pos = Position::long(100.0, 1.0, 1000);

        assert_eq!(pos.side, PositionSide::Long);
        assert!(!pos.is_flat());

        // Price goes up
        assert!((pos.unrealized_pnl(110.0) - 10.0).abs() < 0.001);
        assert!((pos.unrealized_pnl_percent(110.0) - 10.0).abs() < 0.001);

        // Price goes down
        assert!((pos.unrealized_pnl(90.0) - (-10.0)).abs() < 0.001);
    }

    #[test]
    fn test_short_position() {
        let pos = Position::short(100.0, 1.0, 1000);

        // Price goes down (profit for short)
        assert!((pos.unrealized_pnl(90.0) - 10.0).abs() < 0.001);

        // Price goes up (loss for short)
        assert!((pos.unrealized_pnl(110.0) - (-10.0)).abs() < 0.001);
    }

    #[test]
    fn test_stop_loss() {
        let pos = Position::long(100.0, 1.0, 1000)
            .with_stop_loss(95.0);

        assert!(!pos.is_stop_hit(96.0));
        assert!(pos.is_stop_hit(95.0));
        assert!(pos.is_stop_hit(90.0));
    }

    #[test]
    fn test_take_profit() {
        let pos = Position::long(100.0, 1.0, 1000)
            .with_take_profit(110.0);

        assert!(!pos.is_target_hit(105.0));
        assert!(pos.is_target_hit(110.0));
        assert!(pos.is_target_hit(115.0));
    }

    #[test]
    fn test_close_position() {
        let pos = Position::long(100.0, 1.0, 1000);
        let closed = ClosedPosition::close(pos, 110.0, 2000, 0.001, ExitReason::Signal);

        assert!(closed.is_winner());
        assert!(closed.realized_pnl > 0.0);
        assert!(closed.fees > 0.0);
        assert_eq!(closed.holding_period(), 1000);
    }
}
