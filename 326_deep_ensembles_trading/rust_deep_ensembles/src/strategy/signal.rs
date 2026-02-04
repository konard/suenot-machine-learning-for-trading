//! Trading signals

use serde::{Deserialize, Serialize};

/// Signal type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    Long,
    Short,
    Hold,
}

/// Trading signal with uncertainty information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Symbol
    pub symbol: String,
    /// Signal type
    pub signal_type: SignalType,
    /// Confidence (0 to 1)
    pub confidence: f64,
    /// Predicted return
    pub predicted_return: f64,
    /// Total uncertainty
    pub total_uncertainty: f64,
    /// Epistemic uncertainty
    pub epistemic_uncertainty: f64,
    /// Aleatoric uncertainty
    pub aleatoric_uncertainty: f64,
}

impl Signal {
    /// Create a new signal
    pub fn new(
        symbol: &str,
        signal_type: SignalType,
        confidence: f64,
        predicted_return: f64,
        total_uncertainty: f64,
        epistemic_uncertainty: f64,
        aleatoric_uncertainty: f64,
    ) -> Self {
        Self {
            symbol: symbol.to_string(),
            signal_type,
            confidence,
            predicted_return,
            total_uncertainty,
            epistemic_uncertainty,
            aleatoric_uncertainty,
        }
    }

    /// Create a hold signal
    pub fn hold(symbol: &str) -> Self {
        Self::new(symbol, SignalType::Hold, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    /// Check if signal is actionable (not hold)
    pub fn is_actionable(&self) -> bool {
        self.signal_type != SignalType::Hold
    }

    /// Get signal direction as integer (-1, 0, 1)
    pub fn direction(&self) -> i32 {
        match self.signal_type {
            SignalType::Long => 1,
            SignalType::Short => -1,
            SignalType::Hold => 0,
        }
    }
}

/// Signal generator
pub struct SignalGenerator {
    /// Confidence threshold for generating signals
    pub confidence_threshold: f64,
    /// Maximum epistemic ratio threshold
    pub epistemic_threshold: f64,
    /// Minimum signal strength (return / std)
    pub signal_strength_threshold: f64,
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new(0.6, 0.5, 1.5)
    }
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new(
        confidence_threshold: f64,
        epistemic_threshold: f64,
        signal_strength_threshold: f64,
    ) -> Self {
        Self {
            confidence_threshold,
            epistemic_threshold,
            signal_strength_threshold,
        }
    }

    /// Generate signal from prediction
    pub fn generate(
        &self,
        symbol: &str,
        mean: f64,
        total_std: f64,
        epistemic_std: f64,
        aleatoric_std: f64,
    ) -> Signal {
        // Calculate epistemic ratio
        let epistemic_ratio = if total_std > 0.0 {
            epistemic_std / total_std
        } else {
            1.0
        };

        // Calculate signal strength
        let signal_strength = if total_std > 0.0 {
            mean / total_std
        } else {
            0.0
        };

        // Determine signal type
        let (signal_type, confidence) = if epistemic_ratio > self.epistemic_threshold {
            // High model uncertainty - don't trade
            (SignalType::Hold, 0.0)
        } else if signal_strength > self.signal_strength_threshold {
            let conf = (1.0 - epistemic_ratio).min(0.95);
            (SignalType::Long, conf)
        } else if signal_strength < -self.signal_strength_threshold {
            let conf = (1.0 - epistemic_ratio).min(0.95);
            (SignalType::Short, conf)
        } else {
            (SignalType::Hold, 0.0)
        };

        Signal::new(
            symbol,
            signal_type,
            confidence,
            mean,
            total_std,
            epistemic_std,
            aleatoric_std,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_creation() {
        let signal = Signal::new(
            "BTCUSDT",
            SignalType::Long,
            0.8,
            0.05,
            0.02,
            0.01,
            0.015,
        );

        assert_eq!(signal.symbol, "BTCUSDT");
        assert!(signal.is_actionable());
        assert_eq!(signal.direction(), 1);
    }

    #[test]
    fn test_signal_generator() {
        let generator = SignalGenerator::default();

        // Strong long signal
        let signal = generator.generate("BTCUSDT", 0.05, 0.02, 0.005, 0.015);
        assert_eq!(signal.signal_type, SignalType::Long);

        // Strong short signal
        let signal = generator.generate("BTCUSDT", -0.05, 0.02, 0.005, 0.015);
        assert_eq!(signal.signal_type, SignalType::Short);

        // High epistemic uncertainty - hold
        let signal = generator.generate("BTCUSDT", 0.05, 0.02, 0.015, 0.005);
        assert_eq!(signal.signal_type, SignalType::Hold);
    }
}
