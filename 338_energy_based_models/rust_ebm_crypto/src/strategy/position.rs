//! Position management for EBM-based trading

use super::signals::{SignalType, TradingSignal};

/// Position side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionSide {
    /// Long position (profit from price increase)
    Long,
    /// Short position (profit from price decrease)
    Short,
    /// No position
    Flat,
}

impl PositionSide {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            PositionSide::Long => "LONG",
            PositionSide::Short => "SHORT",
            PositionSide::Flat => "FLAT",
        }
    }

    /// Get multiplier for PnL calculation
    pub fn multiplier(&self) -> f64 {
        match self {
            PositionSide::Long => 1.0,
            PositionSide::Short => -1.0,
            PositionSide::Flat => 0.0,
        }
    }
}

/// Current position state
#[derive(Debug, Clone)]
pub struct Position {
    /// Position side
    pub side: PositionSide,
    /// Position size (0-1 as fraction of capital)
    pub size: f64,
    /// Entry price
    pub entry_price: f64,
    /// Entry timestamp
    pub entry_time: i64,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
    /// Stop loss price
    pub stop_loss: Option<f64>,
    /// Take profit price
    pub take_profit: Option<f64>,
}

impl Default for Position {
    fn default() -> Self {
        Self {
            side: PositionSide::Flat,
            size: 0.0,
            entry_price: 0.0,
            entry_time: 0,
            unrealized_pnl: 0.0,
            stop_loss: None,
            take_profit: None,
        }
    }
}

impl Position {
    /// Create a new flat position
    pub fn flat() -> Self {
        Self::default()
    }

    /// Create a new long position
    pub fn long(size: f64, entry_price: f64, entry_time: i64) -> Self {
        Self {
            side: PositionSide::Long,
            size,
            entry_price,
            entry_time,
            unrealized_pnl: 0.0,
            stop_loss: None,
            take_profit: None,
        }
    }

    /// Create a new short position
    pub fn short(size: f64, entry_price: f64, entry_time: i64) -> Self {
        Self {
            side: PositionSide::Short,
            size,
            entry_price,
            entry_time,
            unrealized_pnl: 0.0,
            stop_loss: None,
            take_profit: None,
        }
    }

    /// Check if position is open
    pub fn is_open(&self) -> bool {
        self.side != PositionSide::Flat && self.size > 0.0
    }

    /// Update unrealized PnL
    pub fn update_pnl(&mut self, current_price: f64) {
        if !self.is_open() || self.entry_price == 0.0 {
            self.unrealized_pnl = 0.0;
            return;
        }

        let price_change = (current_price - self.entry_price) / self.entry_price;
        self.unrealized_pnl = price_change * self.side.multiplier() * self.size;
    }

    /// Check if stop loss is triggered
    pub fn is_stopped_out(&self, current_price: f64) -> bool {
        match (self.side, self.stop_loss) {
            (PositionSide::Long, Some(sl)) => current_price <= sl,
            (PositionSide::Short, Some(sl)) => current_price >= sl,
            _ => false,
        }
    }

    /// Check if take profit is triggered
    pub fn is_take_profit(&self, current_price: f64) -> bool {
        match (self.side, self.take_profit) {
            (PositionSide::Long, Some(tp)) => current_price >= tp,
            (PositionSide::Short, Some(tp)) => current_price <= tp,
            _ => false,
        }
    }

    /// Set stop loss
    pub fn set_stop_loss(&mut self, price: f64) {
        self.stop_loss = Some(price);
    }

    /// Set take profit
    pub fn set_take_profit(&mut self, price: f64) {
        self.take_profit = Some(price);
    }

    /// Set ATR-based stops
    pub fn set_atr_stops(&mut self, atr: f64, sl_multiplier: f64, tp_multiplier: f64) {
        match self.side {
            PositionSide::Long => {
                self.stop_loss = Some(self.entry_price - atr * sl_multiplier);
                self.take_profit = Some(self.entry_price + atr * tp_multiplier);
            }
            PositionSide::Short => {
                self.stop_loss = Some(self.entry_price + atr * sl_multiplier);
                self.take_profit = Some(self.entry_price - atr * tp_multiplier);
            }
            PositionSide::Flat => {}
        }
    }
}

/// Position manager configuration
#[derive(Debug, Clone)]
pub struct PositionConfig {
    /// Maximum position size (0-1)
    pub max_position_size: f64,
    /// Minimum position size (0-1)
    pub min_position_size: f64,
    /// Default stop loss percentage
    pub default_stop_loss_pct: f64,
    /// Default take profit percentage
    pub default_take_profit_pct: f64,
    /// Whether to use trailing stops
    pub use_trailing_stop: bool,
    /// Trailing stop activation threshold
    pub trailing_stop_activation: f64,
    /// Trailing stop distance
    pub trailing_stop_distance: f64,
}

impl Default for PositionConfig {
    fn default() -> Self {
        Self {
            max_position_size: 1.0,
            min_position_size: 0.1,
            default_stop_loss_pct: 0.02, // 2%
            default_take_profit_pct: 0.04, // 4%
            use_trailing_stop: true,
            trailing_stop_activation: 0.02, // 2% profit
            trailing_stop_distance: 0.01, // 1% trail
        }
    }
}

/// Position manager for handling entries, exits, and sizing
#[derive(Debug, Clone)]
pub struct PositionManager {
    /// Configuration
    pub config: PositionConfig,
    /// Current position
    pub position: Position,
    /// Highest price since entry (for trailing stop)
    highest_since_entry: f64,
    /// Lowest price since entry (for trailing stop)
    lowest_since_entry: f64,
}

impl PositionManager {
    /// Create a new position manager
    pub fn new(config: PositionConfig) -> Self {
        Self {
            config,
            position: Position::flat(),
            highest_since_entry: 0.0,
            lowest_since_entry: f64::INFINITY,
        }
    }

    /// Process a trading signal
    pub fn process_signal(
        &mut self,
        signal: &TradingSignal,
        current_price: f64,
    ) -> Option<TradeAction> {
        // Update position PnL
        self.position.update_pnl(current_price);

        // Update trailing stop levels
        self.update_trailing_levels(current_price);

        // Check stop loss / take profit
        if self.position.is_open() {
            if self.position.is_stopped_out(current_price) {
                return Some(self.close_position(current_price, "Stop loss triggered"));
            }
            if self.position.is_take_profit(current_price) {
                return Some(self.close_position(current_price, "Take profit triggered"));
            }

            // Check trailing stop
            if self.config.use_trailing_stop {
                if let Some(action) = self.check_trailing_stop(current_price) {
                    return Some(action);
                }
            }
        }

        // Process signal
        match signal.signal_type {
            SignalType::Exit => {
                if self.position.is_open() {
                    return Some(self.close_position(current_price, &signal.reason));
                }
            }
            SignalType::ReducePosition => {
                if self.position.is_open() {
                    return Some(self.reduce_position(signal.position_scale, current_price));
                }
            }
            SignalType::Long => {
                if self.position.side != PositionSide::Long {
                    // Close any short position first
                    if self.position.side == PositionSide::Short {
                        self.close_position(current_price, "Reversing to long");
                    }
                    return Some(self.open_long(signal.position_scale, current_price, signal.timestamp));
                }
            }
            SignalType::Short => {
                if self.position.side != PositionSide::Short {
                    // Close any long position first
                    if self.position.side == PositionSide::Long {
                        self.close_position(current_price, "Reversing to short");
                    }
                    return Some(self.open_short(signal.position_scale, current_price, signal.timestamp));
                }
            }
            SignalType::Hold => {}
        }

        None
    }

    /// Open a long position
    fn open_long(&mut self, scale: f64, price: f64, timestamp: i64) -> TradeAction {
        let size = (scale * self.config.max_position_size)
            .max(self.config.min_position_size)
            .min(self.config.max_position_size);

        self.position = Position::long(size, price, timestamp);
        self.position.set_stop_loss(price * (1.0 - self.config.default_stop_loss_pct));
        self.position.set_take_profit(price * (1.0 + self.config.default_take_profit_pct));

        self.highest_since_entry = price;
        self.lowest_since_entry = price;

        TradeAction {
            action_type: ActionType::Open,
            side: PositionSide::Long,
            size,
            price,
            timestamp,
            reason: "Long entry".to_string(),
        }
    }

    /// Open a short position
    fn open_short(&mut self, scale: f64, price: f64, timestamp: i64) -> TradeAction {
        let size = (scale * self.config.max_position_size)
            .max(self.config.min_position_size)
            .min(self.config.max_position_size);

        self.position = Position::short(size, price, timestamp);
        self.position.set_stop_loss(price * (1.0 + self.config.default_stop_loss_pct));
        self.position.set_take_profit(price * (1.0 - self.config.default_take_profit_pct));

        self.highest_since_entry = price;
        self.lowest_since_entry = price;

        TradeAction {
            action_type: ActionType::Open,
            side: PositionSide::Short,
            size,
            price,
            timestamp,
            reason: "Short entry".to_string(),
        }
    }

    /// Close the current position
    fn close_position(&mut self, price: f64, reason: &str) -> TradeAction {
        let action = TradeAction {
            action_type: ActionType::Close,
            side: self.position.side,
            size: self.position.size,
            price,
            timestamp: 0,
            reason: reason.to_string(),
        };

        self.position = Position::flat();
        self.highest_since_entry = 0.0;
        self.lowest_since_entry = f64::INFINITY;

        action
    }

    /// Reduce position size
    fn reduce_position(&mut self, target_scale: f64, price: f64) -> TradeAction {
        let new_size = (target_scale * self.position.size)
            .max(self.config.min_position_size)
            .min(self.position.size);

        let reduced_amount = self.position.size - new_size;
        self.position.size = new_size;

        TradeAction {
            action_type: ActionType::Reduce,
            side: self.position.side,
            size: reduced_amount,
            price,
            timestamp: 0,
            reason: format!("Position reduced to {:.2}%", new_size * 100.0),
        }
    }

    /// Update trailing stop levels
    fn update_trailing_levels(&mut self, price: f64) {
        if price > self.highest_since_entry {
            self.highest_since_entry = price;
        }
        if price < self.lowest_since_entry {
            self.lowest_since_entry = price;
        }
    }

    /// Check trailing stop condition
    fn check_trailing_stop(&self, price: f64) -> Option<TradeAction> {
        if !self.position.is_open() {
            return None;
        }

        match self.position.side {
            PositionSide::Long => {
                let profit_pct =
                    (self.highest_since_entry - self.position.entry_price) / self.position.entry_price;

                if profit_pct >= self.config.trailing_stop_activation {
                    let trailing_stop =
                        self.highest_since_entry * (1.0 - self.config.trailing_stop_distance);
                    if price <= trailing_stop {
                        return Some(TradeAction {
                            action_type: ActionType::Close,
                            side: PositionSide::Long,
                            size: self.position.size,
                            price,
                            timestamp: 0,
                            reason: format!(
                                "Trailing stop triggered at {:.4}",
                                trailing_stop
                            ),
                        });
                    }
                }
            }
            PositionSide::Short => {
                let profit_pct =
                    (self.position.entry_price - self.lowest_since_entry) / self.position.entry_price;

                if profit_pct >= self.config.trailing_stop_activation {
                    let trailing_stop =
                        self.lowest_since_entry * (1.0 + self.config.trailing_stop_distance);
                    if price >= trailing_stop {
                        return Some(TradeAction {
                            action_type: ActionType::Close,
                            side: PositionSide::Short,
                            size: self.position.size,
                            price,
                            timestamp: 0,
                            reason: format!(
                                "Trailing stop triggered at {:.4}",
                                trailing_stop
                            ),
                        });
                    }
                }
            }
            _ => {}
        }

        None
    }
}

/// Trade action type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionType {
    /// Open new position
    Open,
    /// Close position
    Close,
    /// Reduce position size
    Reduce,
    /// Add to position
    Add,
}

/// Trade action to be executed
#[derive(Debug, Clone)]
pub struct TradeAction {
    /// Action type
    pub action_type: ActionType,
    /// Position side
    pub side: PositionSide,
    /// Size (fraction of capital or amount to change)
    pub size: f64,
    /// Execution price
    pub price: f64,
    /// Timestamp
    pub timestamp: i64,
    /// Reason for trade
    pub reason: String,
}

impl TradeAction {
    /// Get string representation
    pub fn as_str(&self) -> String {
        format!(
            "{:?} {} {:.2}% at {:.4} - {}",
            self.action_type,
            self.side.as_str(),
            self.size * 100.0,
            self.price,
            self.reason
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_pnl() {
        let mut pos = Position::long(1.0, 100.0, 0);

        pos.update_pnl(110.0); // 10% up
        assert!((pos.unrealized_pnl - 0.1).abs() < 1e-10);

        pos.update_pnl(95.0); // 5% down
        assert!((pos.unrealized_pnl - (-0.05)).abs() < 1e-10);
    }

    #[test]
    fn test_short_position_pnl() {
        let mut pos = Position::short(1.0, 100.0, 0);

        pos.update_pnl(90.0); // 10% down = profit
        assert!((pos.unrealized_pnl - 0.1).abs() < 1e-10);

        pos.update_pnl(105.0); // 5% up = loss
        assert!((pos.unrealized_pnl - (-0.05)).abs() < 1e-10);
    }

    #[test]
    fn test_stop_loss() {
        let mut pos = Position::long(1.0, 100.0, 0);
        pos.set_stop_loss(95.0);

        assert!(!pos.is_stopped_out(96.0));
        assert!(pos.is_stopped_out(94.0));
    }
}
