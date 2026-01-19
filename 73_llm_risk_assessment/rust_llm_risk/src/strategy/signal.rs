//! Trading signals and positions.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Type of trading signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    /// Go long (buy).
    Long,
    /// Go short (sell).
    Short,
    /// Close position.
    Close,
    /// No action / stay neutral.
    Neutral,
}

impl fmt::Display for SignalType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SignalType::Long => write!(f, "LONG"),
            SignalType::Short => write!(f, "SHORT"),
            SignalType::Close => write!(f, "CLOSE"),
            SignalType::Neutral => write!(f, "NEUTRAL"),
        }
    }
}

/// A trading signal with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    /// Type of signal.
    pub signal_type: SignalType,
    /// Recommended position size (0.0 to 1.0).
    pub position_size: f64,
    /// Confidence in the signal (0.0 to 1.0).
    pub confidence: f64,
    /// Reason for the signal.
    pub reason: String,
    /// Risk score that generated this signal.
    pub risk_score: f64,
    /// Timestamp of signal generation.
    pub timestamp: i64,
    /// Symbol this signal is for.
    pub symbol: Option<String>,
    /// Stop loss price (optional).
    pub stop_loss: Option<f64>,
    /// Take profit price (optional).
    pub take_profit: Option<f64>,
}

impl TradingSignal {
    /// Create a new trading signal.
    pub fn new(signal_type: SignalType, position_size: f64, reason: &str) -> Self {
        Self {
            signal_type,
            position_size: position_size.clamp(0.0, 1.0),
            confidence: 0.5,
            reason: reason.to_string(),
            risk_score: 5.0,
            timestamp: chrono::Utc::now().timestamp_millis(),
            symbol: None,
            stop_loss: None,
            take_profit: None,
        }
    }

    /// Create a long signal.
    pub fn long(position_size: f64, reason: &str) -> Self {
        Self::new(SignalType::Long, position_size, reason)
    }

    /// Create a short signal.
    pub fn short(position_size: f64, reason: &str) -> Self {
        Self::new(SignalType::Short, position_size, reason)
    }

    /// Create a neutral signal.
    pub fn neutral(reason: &str) -> Self {
        Self::new(SignalType::Neutral, 0.0, reason)
    }

    /// Create a close signal.
    pub fn close(reason: &str) -> Self {
        Self::new(SignalType::Close, 0.0, reason)
    }

    /// Set confidence level.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set risk score.
    pub fn with_risk_score(mut self, score: f64) -> Self {
        self.risk_score = score;
        self
    }

    /// Set symbol.
    pub fn with_symbol(mut self, symbol: String) -> Self {
        self.symbol = Some(symbol);
        self
    }

    /// Set stop loss.
    pub fn with_stop_loss(mut self, price: f64) -> Self {
        self.stop_loss = Some(price);
        self
    }

    /// Set take profit.
    pub fn with_take_profit(mut self, price: f64) -> Self {
        self.take_profit = Some(price);
        self
    }

    /// Check if the signal is actionable (not neutral).
    pub fn is_actionable(&self) -> bool {
        !matches!(self.signal_type, SignalType::Neutral)
    }

    /// Check if this is a buy signal.
    pub fn is_buy(&self) -> bool {
        matches!(self.signal_type, SignalType::Long)
    }

    /// Check if this is a sell signal.
    pub fn is_sell(&self) -> bool {
        matches!(self.signal_type, SignalType::Short | SignalType::Close)
    }
}

impl fmt::Display for TradingSignal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} size={:.0}% conf={:.0}% - {}",
            self.symbol.as_deref().unwrap_or("???"),
            self.signal_type,
            self.position_size * 100.0,
            self.confidence * 100.0,
            self.reason
        )
    }
}

/// Current position in an asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Symbol of the asset.
    pub symbol: String,
    /// Position size (negative for short).
    pub size: f64,
    /// Entry price.
    pub entry_price: f64,
    /// Entry timestamp.
    pub entry_time: i64,
    /// Current unrealized PnL.
    pub unrealized_pnl: f64,
    /// Stop loss price.
    pub stop_loss: Option<f64>,
    /// Take profit price.
    pub take_profit: Option<f64>,
}

impl Position {
    /// Create a new position.
    pub fn new(symbol: String, size: f64, entry_price: f64) -> Self {
        Self {
            symbol,
            size,
            entry_price,
            entry_time: chrono::Utc::now().timestamp_millis(),
            unrealized_pnl: 0.0,
            stop_loss: None,
            take_profit: None,
        }
    }

    /// Check if this is a long position.
    pub fn is_long(&self) -> bool {
        self.size > 0.0
    }

    /// Check if this is a short position.
    pub fn is_short(&self) -> bool {
        self.size < 0.0
    }

    /// Update the unrealized PnL based on current price.
    pub fn update_pnl(&mut self, current_price: f64) {
        let price_change = current_price - self.entry_price;
        self.unrealized_pnl = price_change * self.size;
    }

    /// Calculate return percentage.
    pub fn return_pct(&self, current_price: f64) -> f64 {
        if self.entry_price > 0.0 {
            let price_change = (current_price - self.entry_price) / self.entry_price;
            if self.is_long() {
                price_change * 100.0
            } else {
                -price_change * 100.0
            }
        } else {
            0.0
        }
    }

    /// Check if stop loss has been hit.
    pub fn stop_loss_hit(&self, current_price: f64) -> bool {
        if let Some(sl) = self.stop_loss {
            if self.is_long() {
                current_price <= sl
            } else {
                current_price >= sl
            }
        } else {
            false
        }
    }

    /// Check if take profit has been hit.
    pub fn take_profit_hit(&self, current_price: f64) -> bool {
        if let Some(tp) = self.take_profit {
            if self.is_long() {
                current_price >= tp
            } else {
                current_price <= tp
            }
        } else {
            false
        }
    }

    /// Set stop loss price.
    pub fn with_stop_loss(mut self, price: f64) -> Self {
        self.stop_loss = Some(price);
        self
    }

    /// Set take profit price.
    pub fn with_take_profit(mut self, price: f64) -> Self {
        self.take_profit = Some(price);
        self
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let direction = if self.is_long() { "LONG" } else { "SHORT" };
        write!(
            f,
            "{} {} @ {:.2} (size: {:.4}, PnL: {:.2})",
            direction, self.symbol, self.entry_price, self.size.abs(), self.unrealized_pnl
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_creation() {
        let signal = TradingSignal::long(0.5, "Low risk detected")
            .with_confidence(0.8)
            .with_symbol("BTCUSDT".to_string());

        assert!(signal.is_buy());
        assert!(signal.is_actionable());
        assert_eq!(signal.position_size, 0.5);
    }

    #[test]
    fn test_neutral_signal() {
        let signal = TradingSignal::neutral("Waiting for better opportunity");

        assert!(!signal.is_actionable());
        assert!(!signal.is_buy());
        assert!(!signal.is_sell());
    }

    #[test]
    fn test_position_pnl() {
        let mut position = Position::new("BTCUSDT".to_string(), 1.0, 50000.0);

        position.update_pnl(52000.0);
        assert_eq!(position.unrealized_pnl, 2000.0);

        let return_pct = position.return_pct(52000.0);
        assert!((return_pct - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_position_stop_loss() {
        let position = Position::new("BTCUSDT".to_string(), 1.0, 50000.0)
            .with_stop_loss(48000.0);

        assert!(!position.stop_loss_hit(49000.0));
        assert!(position.stop_loss_hit(47000.0));
    }

    #[test]
    fn test_short_position() {
        let position = Position::new("BTCUSDT".to_string(), -1.0, 50000.0);

        assert!(position.is_short());
        assert!(!position.is_long());

        // For short, price going down is profitable
        let return_pct = position.return_pct(48000.0);
        assert!(return_pct > 0.0);
    }
}
