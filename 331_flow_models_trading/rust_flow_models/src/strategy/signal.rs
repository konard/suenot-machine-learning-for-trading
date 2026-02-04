//! Trading signal types and generation

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Type of trading signal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    /// Go long / buy
    Long,
    /// Go short / sell
    Short,
    /// No position / neutral
    Neutral,
    /// Reduce exposure due to anomaly
    ReduceExposure,
}

impl SignalType {
    /// Check if signal is actionable (not neutral)
    pub fn is_actionable(&self) -> bool {
        matches!(self, SignalType::Long | SignalType::Short | SignalType::ReduceExposure)
    }

    /// Get direction multiplier (-1, 0, or 1)
    pub fn direction(&self) -> i32 {
        match self {
            SignalType::Long => 1,
            SignalType::Short => -1,
            _ => 0,
        }
    }
}

/// Trading signal with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Signal type
    pub signal_type: SignalType,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Reason for the signal
    pub reason: String,
    /// Timestamp of signal generation
    pub timestamp: Option<DateTime<Utc>>,
    /// Log-likelihood from flow model
    pub log_likelihood: Option<f64>,
    /// Detected market regime
    pub regime: Option<String>,
    /// Symbol this signal is for
    pub symbol: Option<String>,
}

impl Signal {
    /// Create a new signal
    pub fn new(signal_type: SignalType, confidence: f64, reason: impl Into<String>) -> Self {
        Self {
            signal_type,
            confidence,
            reason: reason.into(),
            timestamp: Some(Utc::now()),
            log_likelihood: None,
            regime: None,
            symbol: None,
        }
    }

    /// Create a long signal
    pub fn long(confidence: f64, reason: impl Into<String>) -> Self {
        Self::new(SignalType::Long, confidence, reason)
    }

    /// Create a short signal
    pub fn short(confidence: f64, reason: impl Into<String>) -> Self {
        Self::new(SignalType::Short, confidence, reason)
    }

    /// Create a neutral signal
    pub fn neutral(reason: impl Into<String>) -> Self {
        Self::new(SignalType::Neutral, 0.5, reason)
    }

    /// Create a reduce exposure signal
    pub fn reduce_exposure(confidence: f64, reason: impl Into<String>) -> Self {
        Self::new(SignalType::ReduceExposure, confidence, reason)
    }

    /// Set log likelihood
    pub fn with_log_likelihood(mut self, log_likelihood: f64) -> Self {
        self.log_likelihood = Some(log_likelihood);
        self
    }

    /// Set regime
    pub fn with_regime(mut self, regime: impl Into<String>) -> Self {
        self.regime = Some(regime.into());
        self
    }

    /// Set symbol
    pub fn with_symbol(mut self, symbol: impl Into<String>) -> Self {
        self.symbol = Some(symbol.into());
        self
    }

    /// Check if this is a bullish signal
    pub fn is_bullish(&self) -> bool {
        matches!(self.signal_type, SignalType::Long)
    }

    /// Check if this is a bearish signal
    pub fn is_bearish(&self) -> bool {
        matches!(self.signal_type, SignalType::Short)
    }

    /// Check if high confidence (> 0.7)
    pub fn is_high_confidence(&self) -> bool {
        self.confidence > 0.7
    }
}

/// Signal generator interface
pub trait SignalGenerator {
    /// Generate signal from features
    fn generate(&mut self, features: &ndarray::Array1<f64>) -> Signal;
}

impl Default for Signal {
    fn default() -> Self {
        Self::neutral("No signal")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_creation() {
        let signal = Signal::long(0.8, "Bullish regime")
            .with_regime("High Vol Bull")
            .with_symbol("BTCUSDT");

        assert_eq!(signal.signal_type, SignalType::Long);
        assert_eq!(signal.confidence, 0.8);
        assert!(signal.is_bullish());
        assert!(signal.is_high_confidence());
        assert_eq!(signal.regime, Some("High Vol Bull".to_string()));
    }

    #[test]
    fn test_signal_type_direction() {
        assert_eq!(SignalType::Long.direction(), 1);
        assert_eq!(SignalType::Short.direction(), -1);
        assert_eq!(SignalType::Neutral.direction(), 0);
    }
}
