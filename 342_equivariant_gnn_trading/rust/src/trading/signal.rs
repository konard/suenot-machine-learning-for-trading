//! Trading Signal Module

use serde::{Deserialize, Serialize};

/// Trade direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeDirection {
    Long,
    Short,
    Hold,
}

impl TradeDirection {
    /// Convert from integer signal
    pub fn from_signal(signal: i32) -> Self {
        match signal {
            1 => TradeDirection::Long,
            -1 => TradeDirection::Short,
            _ => TradeDirection::Hold,
        }
    }

    /// Convert to integer signal
    pub fn to_signal(&self) -> i32 {
        match self {
            TradeDirection::Long => 1,
            TradeDirection::Short => -1,
            TradeDirection::Hold => 0,
        }
    }
}

/// Trading signal with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    /// Asset symbol
    pub symbol: String,
    /// Trade direction
    pub direction: TradeDirection,
    /// Position size (0.0 to 1.0)
    pub size: f64,
    /// Confidence score
    pub confidence: f64,
    /// Predicted volatility
    pub volatility: f64,
    /// Timestamp
    pub timestamp: u64,
}

impl TradingSignal {
    /// Create a new trading signal
    pub fn new(
        symbol: String,
        direction: TradeDirection,
        size: f64,
        confidence: f64,
        volatility: f64,
        timestamp: u64,
    ) -> Self {
        Self { symbol, direction, size, confidence, volatility, timestamp }
    }

    /// Check if signal is actionable (not hold)
    pub fn is_actionable(&self) -> bool {
        self.direction != TradeDirection::Hold && self.size > 0.01
    }
}
