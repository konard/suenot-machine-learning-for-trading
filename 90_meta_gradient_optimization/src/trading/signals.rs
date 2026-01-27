//! Trading signal generation.
//!
//! Converts model predictions into actionable trading signals.

/// Trading signal types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradingSignal {
    /// Buy / Go long
    Long,
    /// Sell / Go short
    Short,
    /// No position
    Neutral,
}

impl TradingSignal {
    /// Convert prediction to signal based on threshold
    pub fn from_prediction(prediction: f64, threshold: f64) -> Self {
        if prediction > threshold {
            TradingSignal::Long
        } else if prediction < -threshold {
            TradingSignal::Short
        } else {
            TradingSignal::Neutral
        }
    }

    /// Get position size (-1, 0, or 1)
    pub fn position(&self) -> f64 {
        match self {
            TradingSignal::Long => 1.0,
            TradingSignal::Short => -1.0,
            TradingSignal::Neutral => 0.0,
        }
    }

    /// Check if signal represents an active position
    pub fn is_active(&self) -> bool {
        !matches!(self, TradingSignal::Neutral)
    }
}

/// Signal with confidence score
#[derive(Debug, Clone)]
pub struct ConfidenceSignal {
    pub signal: TradingSignal,
    pub confidence: f64,
    pub prediction: f64,
}

impl ConfidenceSignal {
    /// Create a new confidence signal from a prediction
    pub fn from_prediction(prediction: f64, threshold: f64) -> Self {
        let signal = TradingSignal::from_prediction(prediction, threshold);
        let confidence = prediction.abs() / (threshold + 1e-10);

        Self {
            signal,
            confidence: confidence.min(1.0),
            prediction,
        }
    }

    /// Get position size scaled by confidence
    pub fn scaled_position(&self) -> f64 {
        self.signal.position() * self.confidence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_from_prediction() {
        assert_eq!(
            TradingSignal::from_prediction(0.005, 0.001),
            TradingSignal::Long
        );
        assert_eq!(
            TradingSignal::from_prediction(-0.005, 0.001),
            TradingSignal::Short
        );
        assert_eq!(
            TradingSignal::from_prediction(0.0005, 0.001),
            TradingSignal::Neutral
        );
    }

    #[test]
    fn test_position() {
        assert_eq!(TradingSignal::Long.position(), 1.0);
        assert_eq!(TradingSignal::Short.position(), -1.0);
        assert_eq!(TradingSignal::Neutral.position(), 0.0);
    }

    #[test]
    fn test_confidence_signal() {
        let sig = ConfidenceSignal::from_prediction(0.005, 0.001);
        assert_eq!(sig.signal, TradingSignal::Long);
        assert!(sig.confidence > 0.0);
        assert!(sig.scaled_position() > 0.0);
    }
}
