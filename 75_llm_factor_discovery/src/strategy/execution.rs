//! Trade execution
//!
//! Handles conversion of signals to trade actions.

use super::signals::TradingSignal;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Trade action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradeAction {
    /// Open a new position
    Open {
        symbol: String,
        side: Side,
        size: f64,
        limit_price: Option<f64>,
    },
    /// Close an existing position
    Close {
        symbol: String,
        reason: CloseReason,
    },
    /// Adjust position size
    Adjust {
        symbol: String,
        new_size: f64,
    },
    /// No action
    Hold,
}

/// Trade side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Long,
    Short,
}

/// Reason for closing a position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CloseReason {
    SignalReversal,
    StopLoss,
    TakeProfit,
    MaxHoldingPeriod,
    RiskLimit,
    Manual,
}

/// Execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    /// Maximum position size (fraction of capital)
    pub max_position_size: f64,
    /// Minimum position size to open
    pub min_position_size: f64,
    /// Stop loss percentage
    pub stop_loss_pct: f64,
    /// Take profit percentage
    pub take_profit_pct: f64,
    /// Maximum holding period (hours)
    pub max_holding_hours: u32,
    /// Use limit orders
    pub use_limit_orders: bool,
    /// Limit order offset (bps from mid)
    pub limit_offset_bps: f64,
    /// Minimum signal change to adjust position
    pub min_adjustment_threshold: f64,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            max_position_size: 0.1,
            min_position_size: 0.01,
            stop_loss_pct: 0.02,
            take_profit_pct: 0.05,
            max_holding_hours: 24,
            use_limit_orders: true,
            limit_offset_bps: 5.0,
            min_adjustment_threshold: 0.2,
        }
    }
}

/// Current position state
#[derive(Debug, Clone, Default)]
pub struct Position {
    pub symbol: String,
    pub side: Option<Side>,
    pub size: f64,
    pub entry_price: f64,
    pub entry_time: Option<DateTime<Utc>>,
    pub unrealized_pnl: f64,
}

impl Position {
    /// Check if position is open
    pub fn is_open(&self) -> bool {
        self.size > 0.0 && self.side.is_some()
    }

    /// Calculate holding duration
    pub fn holding_duration(&self) -> Option<chrono::Duration> {
        self.entry_time.map(|t| Utc::now() - t)
    }

    /// Update unrealized PnL
    pub fn update_pnl(&mut self, current_price: f64) {
        if !self.is_open() {
            self.unrealized_pnl = 0.0;
            return;
        }

        let price_change = (current_price - self.entry_price) / self.entry_price;
        self.unrealized_pnl = match self.side {
            Some(Side::Long) => price_change * self.size,
            Some(Side::Short) => -price_change * self.size,
            None => 0.0,
        };
    }
}

/// Trade executor
pub struct TradeExecutor {
    config: ExecutionConfig,
}

impl TradeExecutor {
    /// Create new executor
    pub fn new(config: ExecutionConfig) -> Self {
        Self { config }
    }

    /// Determine trade action from signal and current position
    pub fn determine_action(
        &self,
        signal: &TradingSignal,
        position: &Position,
        current_price: f64,
    ) -> TradeAction {
        // Check for stop loss or take profit on existing position
        if position.is_open() {
            if let Some(action) = self.check_exit_conditions(position, current_price) {
                return action;
            }
        }

        // Determine target position from signal
        let target_side = if signal.direction > 0.0 {
            Some(Side::Long)
        } else if signal.direction < 0.0 {
            Some(Side::Short)
        } else {
            None
        };

        let target_size = signal.suggested_position_size(self.config.max_position_size).abs();

        // No position -> potentially open
        if !position.is_open() {
            if target_size >= self.config.min_position_size {
                if let Some(side) = target_side {
                    return TradeAction::Open {
                        symbol: signal.symbol.clone(),
                        side,
                        size: target_size,
                        limit_price: if self.config.use_limit_orders {
                            Some(self.calculate_limit_price(current_price, side))
                        } else {
                            None
                        },
                    };
                }
            }
            return TradeAction::Hold;
        }

        // Has position -> check for reversal or adjustment
        if target_side != position.side {
            // Signal reversal - close position
            return TradeAction::Close {
                symbol: signal.symbol.clone(),
                reason: CloseReason::SignalReversal,
            };
        }

        // Same direction - check if adjustment needed
        let size_diff = (target_size - position.size).abs();
        if size_diff > self.config.min_adjustment_threshold * position.size {
            return TradeAction::Adjust {
                symbol: signal.symbol.clone(),
                new_size: target_size,
            };
        }

        TradeAction::Hold
    }

    /// Check exit conditions for position
    fn check_exit_conditions(
        &self,
        position: &Position,
        current_price: f64,
    ) -> Option<TradeAction> {
        // Calculate current PnL percentage
        let pnl_pct = match position.side {
            Some(Side::Long) => (current_price - position.entry_price) / position.entry_price,
            Some(Side::Short) => (position.entry_price - current_price) / position.entry_price,
            None => return None,
        };

        // Stop loss
        if pnl_pct <= -self.config.stop_loss_pct {
            return Some(TradeAction::Close {
                symbol: position.symbol.clone(),
                reason: CloseReason::StopLoss,
            });
        }

        // Take profit
        if pnl_pct >= self.config.take_profit_pct {
            return Some(TradeAction::Close {
                symbol: position.symbol.clone(),
                reason: CloseReason::TakeProfit,
            });
        }

        // Max holding period
        if let Some(duration) = position.holding_duration() {
            if duration.num_hours() > self.config.max_holding_hours as i64 {
                return Some(TradeAction::Close {
                    symbol: position.symbol.clone(),
                    reason: CloseReason::MaxHoldingPeriod,
                });
            }
        }

        None
    }

    /// Calculate limit price with offset
    fn calculate_limit_price(&self, current_price: f64, side: Side) -> f64 {
        let offset = current_price * self.config.limit_offset_bps / 10000.0;
        match side {
            Side::Long => current_price - offset, // Buy below market
            Side::Short => current_price + offset, // Sell above market
        }
    }
}

impl Default for TradeExecutor {
    fn default() -> Self {
        Self::new(ExecutionConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_signal(direction: f64, strength: f64) -> TradingSignal {
        TradingSignal {
            symbol: "BTCUSDT".to_string(),
            direction,
            strength,
            confidence: 0.8,
            timestamp: Utc::now(),
            source_factors: vec!["test".to_string()],
        }
    }

    #[test]
    fn test_open_long_position() {
        let executor = TradeExecutor::default();
        let signal = create_test_signal(1.0, 0.7);
        let position = Position::default();

        let action = executor.determine_action(&signal, &position, 50000.0);

        assert!(matches!(
            action,
            TradeAction::Open {
                side: Side::Long,
                ..
            }
        ));
    }

    #[test]
    fn test_open_short_position() {
        let executor = TradeExecutor::default();
        let signal = create_test_signal(-1.0, 0.7);
        let position = Position::default();

        let action = executor.determine_action(&signal, &position, 50000.0);

        assert!(matches!(
            action,
            TradeAction::Open {
                side: Side::Short,
                ..
            }
        ));
    }

    #[test]
    fn test_signal_reversal() {
        let executor = TradeExecutor::default();
        let signal = create_test_signal(-1.0, 0.7);
        let position = Position {
            symbol: "BTCUSDT".to_string(),
            side: Some(Side::Long),
            size: 0.05,
            entry_price: 50000.0,
            entry_time: Some(Utc::now()),
            unrealized_pnl: 0.0,
        };

        let action = executor.determine_action(&signal, &position, 50000.0);

        assert!(matches!(
            action,
            TradeAction::Close {
                reason: CloseReason::SignalReversal,
                ..
            }
        ));
    }

    #[test]
    fn test_stop_loss() {
        let executor = TradeExecutor::default();
        let signal = create_test_signal(1.0, 0.7);
        let position = Position {
            symbol: "BTCUSDT".to_string(),
            side: Some(Side::Long),
            size: 0.05,
            entry_price: 50000.0,
            entry_time: Some(Utc::now()),
            unrealized_pnl: 0.0,
        };

        // Price dropped 3% (below 2% stop loss)
        let action = executor.determine_action(&signal, &position, 48500.0);

        assert!(matches!(
            action,
            TradeAction::Close {
                reason: CloseReason::StopLoss,
                ..
            }
        ));
    }
}
