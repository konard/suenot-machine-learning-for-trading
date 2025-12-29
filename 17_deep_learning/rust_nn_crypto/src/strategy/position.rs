//! Position Management
//!
//! Track and manage trading positions

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Position side
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PositionSide {
    Long,
    Short,
    Flat,
}

impl PositionSide {
    pub fn to_multiplier(&self) -> f64 {
        match self {
            PositionSide::Long => 1.0,
            PositionSide::Short => -1.0,
            PositionSide::Flat => 0.0,
        }
    }
}

/// Trading position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub side: PositionSide,
    pub size: f64,
    pub entry_price: f64,
    pub entry_time: DateTime<Utc>,
    pub current_price: f64,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub pnl: f64,
    pub pnl_percent: f64,
}

impl Position {
    /// Create a new position
    pub fn new(
        symbol: String,
        side: PositionSide,
        size: f64,
        entry_price: f64,
        entry_time: DateTime<Utc>,
    ) -> Self {
        Self {
            symbol,
            side,
            size,
            entry_price,
            entry_time,
            current_price: entry_price,
            stop_loss: None,
            take_profit: None,
            pnl: 0.0,
            pnl_percent: 0.0,
        }
    }

    /// Create a flat (no position) state
    pub fn flat(symbol: String) -> Self {
        Self {
            symbol,
            side: PositionSide::Flat,
            size: 0.0,
            entry_price: 0.0,
            entry_time: Utc::now(),
            current_price: 0.0,
            stop_loss: None,
            take_profit: None,
            pnl: 0.0,
            pnl_percent: 0.0,
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

    /// Update position with current price
    pub fn update_price(&mut self, current_price: f64) {
        self.current_price = current_price;
        self.calculate_pnl();
    }

    /// Calculate profit/loss
    fn calculate_pnl(&mut self) {
        if self.side == PositionSide::Flat || self.entry_price == 0.0 {
            self.pnl = 0.0;
            self.pnl_percent = 0.0;
            return;
        }

        let price_change = self.current_price - self.entry_price;
        self.pnl = price_change * self.size * self.side.to_multiplier();
        self.pnl_percent = (price_change / self.entry_price) * 100.0 * self.side.to_multiplier();
    }

    /// Check if stop loss is hit
    pub fn is_stop_loss_hit(&self) -> bool {
        if let Some(sl) = self.stop_loss {
            match self.side {
                PositionSide::Long => self.current_price <= sl,
                PositionSide::Short => self.current_price >= sl,
                PositionSide::Flat => false,
            }
        } else {
            false
        }
    }

    /// Check if take profit is hit
    pub fn is_take_profit_hit(&self) -> bool {
        if let Some(tp) = self.take_profit {
            match self.side {
                PositionSide::Long => self.current_price >= tp,
                PositionSide::Short => self.current_price <= tp,
                PositionSide::Flat => false,
            }
        } else {
            false
        }
    }

    /// Check if position should be closed
    pub fn should_close(&self) -> bool {
        self.is_stop_loss_hit() || self.is_take_profit_hit()
    }

    /// Get position value
    pub fn value(&self) -> f64 {
        self.size * self.current_price
    }

    /// Get unrealized PnL
    pub fn unrealized_pnl(&self) -> f64 {
        self.pnl
    }

    /// Is position open
    pub fn is_open(&self) -> bool {
        self.side != PositionSide::Flat && self.size > 0.0
    }
}

/// Position sizing methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionSizing {
    /// Fixed size in base currency
    Fixed(f64),
    /// Percentage of capital
    PercentOfCapital(f64),
    /// Kelly criterion based on win rate and payoff ratio
    Kelly { win_rate: f64, payoff_ratio: f64 },
    /// Volatility-based sizing
    VolatilityBased { target_volatility: f64 },
}

impl PositionSizing {
    /// Calculate position size
    pub fn calculate(&self, capital: f64, price: f64, volatility: Option<f64>) -> f64 {
        match self {
            PositionSizing::Fixed(size) => *size,
            PositionSizing::PercentOfCapital(pct) => (capital * pct) / price,
            PositionSizing::Kelly { win_rate, payoff_ratio } => {
                let kelly = win_rate - (1.0 - win_rate) / payoff_ratio;
                let kelly = kelly.max(0.0).min(0.25); // Cap at 25%
                (capital * kelly) / price
            }
            PositionSizing::VolatilityBased { target_volatility } => {
                if let Some(vol) = volatility {
                    if vol > 0.0 {
                        let risk_factor = target_volatility / vol;
                        let risk_factor = risk_factor.min(2.0); // Cap at 2x
                        (capital * risk_factor * 0.1) / price
                    } else {
                        (capital * 0.1) / price
                    }
                } else {
                    (capital * 0.1) / price
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_pnl() {
        let mut position = Position::new(
            "BTCUSDT".to_string(),
            PositionSide::Long,
            1.0,
            50000.0,
            Utc::now(),
        );

        position.update_price(51000.0);
        assert_eq!(position.pnl, 1000.0);
        assert!((position.pnl_percent - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_short_position() {
        let mut position = Position::new(
            "BTCUSDT".to_string(),
            PositionSide::Short,
            1.0,
            50000.0,
            Utc::now(),
        );

        position.update_price(49000.0);
        assert_eq!(position.pnl, 1000.0); // Profit on short when price goes down
    }

    #[test]
    fn test_stop_loss() {
        let mut position = Position::new(
            "BTCUSDT".to_string(),
            PositionSide::Long,
            1.0,
            50000.0,
            Utc::now(),
        )
        .with_stop_loss(48000.0);

        position.update_price(48000.0);
        assert!(position.is_stop_loss_hit());
    }

    #[test]
    fn test_position_sizing() {
        let sizing = PositionSizing::PercentOfCapital(0.1);
        let size = sizing.calculate(100000.0, 50000.0, None);
        assert!((size - 0.2).abs() < 0.001); // 10% of 100k / 50k price = 0.2
    }
}
