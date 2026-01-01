//! Position Management

use super::signals::{Prediction, Signal};

/// Trading position
#[derive(Debug, Clone)]
pub struct Position {
    /// Symbol
    pub symbol: String,
    /// Position size (positive = long, negative = short)
    pub size: f64,
    /// Entry price
    pub entry_price: f64,
    /// Current unrealized PnL
    pub unrealized_pnl: f64,
    /// Timestamp of position entry
    pub entry_time: i64,
}

impl Position {
    /// Create a new position
    pub fn new(symbol: impl Into<String>, size: f64, entry_price: f64, entry_time: i64) -> Self {
        Self {
            symbol: symbol.into(),
            size,
            entry_price,
            unrealized_pnl: 0.0,
            entry_time,
        }
    }

    /// Check if position is long
    pub fn is_long(&self) -> bool {
        self.size > 0.0
    }

    /// Check if position is short
    pub fn is_short(&self) -> bool {
        self.size < 0.0
    }

    /// Check if position is flat
    pub fn is_flat(&self) -> bool {
        self.size.abs() < 1e-10
    }

    /// Update unrealized PnL
    pub fn update_pnl(&mut self, current_price: f64) {
        self.unrealized_pnl = self.size * (current_price - self.entry_price);
    }

    /// Calculate return percentage
    pub fn return_pct(&self, current_price: f64) -> f64 {
        if self.entry_price > 0.0 {
            (current_price - self.entry_price) / self.entry_price * self.size.signum()
        } else {
            0.0
        }
    }
}

/// Position sizing strategy
#[derive(Debug, Clone)]
pub struct PositionSizer {
    /// Maximum position size as fraction of capital
    max_position: f64,
    /// Risk per trade as fraction of capital
    risk_per_trade: f64,
    /// Use volatility-adjusted sizing
    volatility_adjusted: bool,
}

impl Default for PositionSizer {
    fn default() -> Self {
        Self::new()
    }
}

impl PositionSizer {
    /// Create a new position sizer with default settings
    pub fn new() -> Self {
        Self {
            max_position: 0.1,      // 10% max position
            risk_per_trade: 0.02,  // 2% risk per trade
            volatility_adjusted: true,
        }
    }

    /// Create with custom settings
    pub fn with_settings(max_position: f64, risk_per_trade: f64, volatility_adjusted: bool) -> Self {
        Self {
            max_position: max_position.clamp(0.01, 1.0),
            risk_per_trade: risk_per_trade.clamp(0.001, 0.1),
            volatility_adjusted,
        }
    }

    /// Calculate position size based on prediction
    ///
    /// # Arguments
    /// - `prediction` - Model prediction with direction, magnitude, volatility
    /// - `capital` - Available capital
    /// - `current_price` - Current asset price
    ///
    /// # Returns
    /// - Position size in units of the asset
    pub fn calculate_size(
        &self,
        prediction: &Prediction,
        capital: f64,
        current_price: f64,
    ) -> f64 {
        if current_price <= 0.0 || capital <= 0.0 {
            return 0.0;
        }

        // Base position based on signal
        let base_fraction = match prediction.signal {
            Signal::StrongBuy | Signal::StrongSell => self.max_position,
            Signal::Buy | Signal::Sell => self.max_position * 0.5,
            Signal::Hold => 0.0,
        };

        let position_value = if self.volatility_adjusted && prediction.volatility > 0.0 {
            // Volatility-adjusted: smaller positions for higher volatility
            let vol_factor = (self.risk_per_trade / prediction.volatility).min(1.0);
            capital * base_fraction * vol_factor * prediction.confidence
        } else {
            capital * base_fraction * prediction.confidence
        };

        // Calculate units
        let units = position_value / current_price;

        // Apply direction
        units * prediction.direction.signum()
    }

    /// Calculate position size with stop loss
    ///
    /// # Arguments
    /// - `signal` - Trading signal
    /// - `capital` - Available capital
    /// - `current_price` - Current asset price
    /// - `stop_loss_pct` - Stop loss percentage (e.g., 0.02 for 2%)
    ///
    /// # Returns
    /// - Position size in units
    pub fn calculate_size_with_stop(
        &self,
        signal: Signal,
        capital: f64,
        current_price: f64,
        stop_loss_pct: f64,
    ) -> f64 {
        if current_price <= 0.0 || capital <= 0.0 || stop_loss_pct <= 0.0 {
            return 0.0;
        }

        // Risk amount
        let risk_amount = capital * self.risk_per_trade;

        // Position size = risk / stop loss distance
        let position_value = risk_amount / stop_loss_pct;

        // Cap at max position
        let capped_value = position_value.min(capital * self.max_position);

        // Convert to units
        let units = capped_value / current_price;

        // Apply direction
        units * signal.to_multiplier()
    }
}

/// Kelly criterion for optimal position sizing
#[derive(Debug, Clone)]
pub struct KellyCriterion {
    /// Fraction of Kelly to use (e.g., 0.25 for quarter Kelly)
    kelly_fraction: f64,
    /// Maximum position size
    max_position: f64,
}

impl Default for KellyCriterion {
    fn default() -> Self {
        Self {
            kelly_fraction: 0.25, // Quarter Kelly is common
            max_position: 0.2,
        }
    }
}

impl KellyCriterion {
    /// Create a new Kelly criterion calculator
    pub fn new(kelly_fraction: f64, max_position: f64) -> Self {
        Self {
            kelly_fraction: kelly_fraction.clamp(0.0, 1.0),
            max_position: max_position.clamp(0.01, 1.0),
        }
    }

    /// Calculate optimal position size using Kelly criterion
    ///
    /// # Arguments
    /// - `win_rate` - Historical win rate (0 to 1)
    /// - `avg_win` - Average winning return
    /// - `avg_loss` - Average losing return (positive number)
    /// - `capital` - Available capital
    /// - `current_price` - Current asset price
    ///
    /// # Returns
    /// - Optimal position size in units
    pub fn calculate(
        &self,
        win_rate: f64,
        avg_win: f64,
        avg_loss: f64,
        capital: f64,
        current_price: f64,
    ) -> f64 {
        if avg_loss <= 0.0 || current_price <= 0.0 || capital <= 0.0 {
            return 0.0;
        }

        // Kelly formula: f* = (p*b - q) / b
        // where p = win rate, q = 1-p, b = win/loss ratio
        let p = win_rate.clamp(0.0, 1.0);
        let q = 1.0 - p;
        let b = avg_win / avg_loss;

        let kelly = (p * b - q) / b;

        // Apply fraction and max
        let fraction = (kelly * self.kelly_fraction).clamp(0.0, self.max_position);

        // Calculate units
        (capital * fraction) / current_price
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_pnl() {
        let mut pos = Position::new("BTCUSDT", 1.0, 100.0, 0);
        pos.update_pnl(110.0);
        assert!((pos.unrealized_pnl - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_position_return() {
        let pos = Position::new("BTCUSDT", 1.0, 100.0, 0);
        let ret = pos.return_pct(110.0);
        assert!((ret - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_short_position() {
        let pos = Position::new("BTCUSDT", -1.0, 100.0, 0);
        assert!(pos.is_short());
        let ret = pos.return_pct(90.0);
        assert!((ret - 0.1).abs() < 1e-10); // 10% gain on short
    }

    #[test]
    fn test_position_sizer() {
        let sizer = PositionSizer::new();
        let pred = Prediction::new(1.0, 0.02, 0.01);
        let size = sizer.calculate_size(&pred, 10000.0, 100.0);
        assert!(size > 0.0);
        assert!(size <= 10.0); // Max 10% of 10000 / 100 = 10 units
    }

    #[test]
    fn test_kelly_criterion() {
        let kelly = KellyCriterion::default();
        let size = kelly.calculate(0.6, 0.02, 0.01, 10000.0, 100.0);
        assert!(size > 0.0);
    }
}
