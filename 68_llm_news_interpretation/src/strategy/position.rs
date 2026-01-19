//! Position management for trading

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Position side (long or short)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PositionSide {
    /// Long position (profit when price goes up)
    Long,
    /// Short position (profit when price goes down)
    Short,
}

impl PositionSide {
    /// Get multiplier for P&L calculations
    pub fn multiplier(&self) -> f64 {
        match self {
            PositionSide::Long => 1.0,
            PositionSide::Short => -1.0,
        }
    }
}

/// A trading position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Unique position ID
    pub id: String,
    /// Symbol being traded
    pub symbol: String,
    /// Position side
    pub side: PositionSide,
    /// Entry price
    pub entry_price: f64,
    /// Position size in base currency
    pub size: f64,
    /// Position size in quote currency (notional)
    pub notional: f64,
    /// Leverage used (1.0 = no leverage)
    pub leverage: f64,
    /// Stop loss price
    pub stop_loss: Option<f64>,
    /// Take profit price
    pub take_profit: Option<f64>,
    /// Trailing stop price
    pub trailing_stop: Option<f64>,
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
    /// Reason for opening position
    pub entry_reason: String,
    /// News IDs that triggered the position
    pub trigger_news_ids: Vec<String>,
}

impl Position {
    /// Create a new position
    pub fn new(
        id: String,
        symbol: String,
        side: PositionSide,
        entry_price: f64,
        size: f64,
        leverage: f64,
    ) -> Self {
        Self {
            id,
            symbol,
            side,
            entry_price,
            size,
            notional: entry_price * size,
            leverage,
            stop_loss: None,
            take_profit: None,
            trailing_stop: None,
            entry_time: Utc::now(),
            entry_reason: String::new(),
            trigger_news_ids: vec![],
        }
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

    /// Set entry reason
    pub fn with_reason(mut self, reason: &str) -> Self {
        self.entry_reason = reason.to_string();
        self
    }

    /// Set trigger news IDs
    pub fn with_news_ids(mut self, ids: Vec<String>) -> Self {
        self.trigger_news_ids = ids;
        self
    }

    /// Calculate unrealized P&L
    pub fn unrealized_pnl(&self, current_price: f64) -> f64 {
        let price_change = current_price - self.entry_price;
        price_change * self.size * self.side.multiplier()
    }

    /// Calculate unrealized P&L percentage
    pub fn unrealized_pnl_pct(&self, current_price: f64) -> f64 {
        let pnl = self.unrealized_pnl(current_price);
        (pnl / self.notional) * 100.0 * self.leverage
    }

    /// Check if stop loss is hit
    pub fn is_stop_loss_hit(&self, current_price: f64) -> bool {
        if let Some(sl) = self.stop_loss {
            match self.side {
                PositionSide::Long => current_price <= sl,
                PositionSide::Short => current_price >= sl,
            }
        } else {
            false
        }
    }

    /// Check if take profit is hit
    pub fn is_take_profit_hit(&self, current_price: f64) -> bool {
        if let Some(tp) = self.take_profit {
            match self.side {
                PositionSide::Long => current_price >= tp,
                PositionSide::Short => current_price <= tp,
            }
        } else {
            false
        }
    }

    /// Update trailing stop based on current price
    pub fn update_trailing_stop(&mut self, current_price: f64, distance_pct: f64) {
        match self.side {
            PositionSide::Long => {
                let new_stop = current_price * (1.0 - distance_pct);
                if let Some(current_stop) = self.trailing_stop {
                    if new_stop > current_stop {
                        self.trailing_stop = Some(new_stop);
                    }
                } else if current_price > self.entry_price {
                    self.trailing_stop = Some(new_stop);
                }
            }
            PositionSide::Short => {
                let new_stop = current_price * (1.0 + distance_pct);
                if let Some(current_stop) = self.trailing_stop {
                    if new_stop < current_stop {
                        self.trailing_stop = Some(new_stop);
                    }
                } else if current_price < self.entry_price {
                    self.trailing_stop = Some(new_stop);
                }
            }
        }
    }

    /// Check if trailing stop is hit
    pub fn is_trailing_stop_hit(&self, current_price: f64) -> bool {
        if let Some(ts) = self.trailing_stop {
            match self.side {
                PositionSide::Long => current_price <= ts,
                PositionSide::Short => current_price >= ts,
            }
        } else {
            false
        }
    }

    /// Get position duration
    pub fn duration(&self) -> chrono::Duration {
        Utc::now() - self.entry_time
    }
}

/// Position sizer for calculating optimal position sizes
#[derive(Debug, Clone)]
pub struct PositionSizer {
    /// Account equity
    equity: f64,
    /// Risk per trade (as fraction)
    risk_per_trade: f64,
    /// Maximum position size (as fraction of equity)
    max_position_size: f64,
}

impl PositionSizer {
    /// Create a new position sizer
    pub fn new(equity: f64) -> Self {
        Self {
            equity,
            risk_per_trade: 0.02,      // 2% risk per trade
            max_position_size: 0.25,   // 25% max position
        }
    }

    /// Set risk per trade
    pub fn with_risk_per_trade(mut self, risk: f64) -> Self {
        self.risk_per_trade = risk.clamp(0.001, 0.1);
        self
    }

    /// Set maximum position size
    pub fn with_max_position(mut self, max: f64) -> Self {
        self.max_position_size = max.clamp(0.01, 1.0);
        self
    }

    /// Update equity
    pub fn update_equity(&mut self, equity: f64) {
        self.equity = equity;
    }

    /// Calculate position size based on risk
    pub fn calculate_size(
        &self,
        entry_price: f64,
        stop_loss_price: f64,
        signal_confidence: f64,
    ) -> PositionSize {
        let risk_amount = self.equity * self.risk_per_trade * signal_confidence;
        let stop_distance = (entry_price - stop_loss_price).abs();

        if stop_distance == 0.0 {
            return PositionSize {
                size: 0.0,
                notional: 0.0,
                risk_amount: 0.0,
            };
        }

        let size_from_risk = risk_amount / stop_distance;
        let max_notional = self.equity * self.max_position_size;
        let max_size = max_notional / entry_price;

        let final_size = size_from_risk.min(max_size);
        let notional = final_size * entry_price;
        let actual_risk = final_size * stop_distance;

        PositionSize {
            size: final_size,
            notional,
            risk_amount: actual_risk,
        }
    }

    /// Calculate size for a fixed percentage of equity
    pub fn calculate_fixed_size(&self, entry_price: f64, equity_pct: f64) -> PositionSize {
        let pct = equity_pct.min(self.max_position_size);
        let notional = self.equity * pct;
        let size = notional / entry_price;

        PositionSize {
            size,
            notional,
            risk_amount: notional, // Full notional at risk without stop loss
        }
    }
}

/// Calculated position size
#[derive(Debug, Clone)]
pub struct PositionSize {
    /// Size in base currency
    pub size: f64,
    /// Notional value in quote currency
    pub notional: f64,
    /// Risk amount in quote currency
    pub risk_amount: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_pnl_long() {
        let pos = Position::new(
            "test".to_string(),
            "BTC".to_string(),
            PositionSide::Long,
            50000.0,
            1.0,
            1.0,
        );

        assert!((pos.unrealized_pnl(55000.0) - 5000.0).abs() < 0.01);
        assert!((pos.unrealized_pnl(45000.0) - (-5000.0)).abs() < 0.01);
    }

    #[test]
    fn test_position_pnl_short() {
        let pos = Position::new(
            "test".to_string(),
            "BTC".to_string(),
            PositionSide::Short,
            50000.0,
            1.0,
            1.0,
        );

        assert!((pos.unrealized_pnl(45000.0) - 5000.0).abs() < 0.01);
        assert!((pos.unrealized_pnl(55000.0) - (-5000.0)).abs() < 0.01);
    }

    #[test]
    fn test_stop_loss_hit() {
        let pos = Position::new(
            "test".to_string(),
            "BTC".to_string(),
            PositionSide::Long,
            50000.0,
            1.0,
            1.0,
        )
        .with_stop_loss(48000.0);

        assert!(!pos.is_stop_loss_hit(49000.0));
        assert!(pos.is_stop_loss_hit(47000.0));
    }

    #[test]
    fn test_take_profit_hit() {
        let pos = Position::new(
            "test".to_string(),
            "BTC".to_string(),
            PositionSide::Long,
            50000.0,
            1.0,
            1.0,
        )
        .with_take_profit(55000.0);

        assert!(!pos.is_take_profit_hit(54000.0));
        assert!(pos.is_take_profit_hit(56000.0));
    }

    #[test]
    fn test_position_sizer() {
        let sizer = PositionSizer::new(10000.0).with_risk_per_trade(0.02);

        let size = sizer.calculate_size(100.0, 95.0, 1.0);

        // Risk = $200, stop distance = $5, size = 40 units
        assert!((size.size - 40.0).abs() < 0.01);
        assert!((size.notional - 4000.0).abs() < 0.01);
    }

    #[test]
    fn test_trailing_stop_update() {
        let mut pos = Position::new(
            "test".to_string(),
            "BTC".to_string(),
            PositionSide::Long,
            50000.0,
            1.0,
            1.0,
        );

        // Price rises, trailing stop should be set
        pos.update_trailing_stop(55000.0, 0.05);
        assert!(pos.trailing_stop.is_some());

        let stop = pos.trailing_stop.unwrap();
        assert!((stop - 52250.0).abs() < 0.01); // 55000 * 0.95

        // Price rises more, trailing stop should update
        pos.update_trailing_stop(60000.0, 0.05);
        let new_stop = pos.trailing_stop.unwrap();
        assert!((new_stop - 57000.0).abs() < 0.01); // 60000 * 0.95
    }
}
