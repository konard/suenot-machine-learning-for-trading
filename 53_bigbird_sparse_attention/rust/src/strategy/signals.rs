//! Signal generation
//!
//! Convert model predictions to trading signals.

use serde::{Deserialize, Serialize};

/// Type of trading signal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    /// Long (buy) signal
    Long,
    /// Short (sell) signal
    Short,
    /// No position (neutral)
    Neutral,
}

impl std::fmt::Display for SignalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalType::Long => write!(f, "LONG"),
            SignalType::Short => write!(f, "SHORT"),
            SignalType::Neutral => write!(f, "NEUTRAL"),
        }
    }
}

/// A trading signal with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Signal type
    pub signal_type: SignalType,
    /// Raw prediction value
    pub prediction: f64,
    /// Confidence/strength of signal (0-1)
    pub confidence: f64,
    /// Timestamp (if applicable)
    pub timestamp: Option<i64>,
}

impl Signal {
    /// Create a new signal
    pub fn new(signal_type: SignalType, prediction: f64, confidence: f64) -> Self {
        Self {
            signal_type,
            prediction,
            confidence,
            timestamp: None,
        }
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, timestamp: i64) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Check if this is a long signal
    pub fn is_long(&self) -> bool {
        matches!(self.signal_type, SignalType::Long)
    }

    /// Check if this is a short signal
    pub fn is_short(&self) -> bool {
        matches!(self.signal_type, SignalType::Short)
    }

    /// Check if this is neutral
    pub fn is_neutral(&self) -> bool {
        matches!(self.signal_type, SignalType::Neutral)
    }
}

/// Signal generator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalConfig {
    /// Threshold for long signals (prediction must be above this)
    pub long_threshold: f64,
    /// Threshold for short signals (prediction must be below this)
    pub short_threshold: f64,
    /// Minimum confidence for signal generation
    pub min_confidence: f64,
    /// Whether to allow short positions
    pub allow_short: bool,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            long_threshold: 0.001,  // 0.1% predicted return
            short_threshold: -0.001,
            min_confidence: 0.5,
            allow_short: true,
        }
    }
}

/// Signal generator from model predictions
pub struct SignalGenerator {
    config: SignalConfig,
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new(SignalConfig::default())
    }
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new(config: SignalConfig) -> Self {
        Self { config }
    }

    /// Generate a signal from a prediction
    pub fn generate(&self, prediction: f64, confidence: Option<f64>) -> Signal {
        let confidence = confidence.unwrap_or(1.0);

        // Check confidence threshold
        if confidence < self.config.min_confidence {
            return Signal::new(SignalType::Neutral, prediction, confidence);
        }

        // Determine signal type
        let signal_type = if prediction > self.config.long_threshold {
            SignalType::Long
        } else if prediction < self.config.short_threshold && self.config.allow_short {
            SignalType::Short
        } else {
            SignalType::Neutral
        };

        Signal::new(signal_type, prediction, confidence)
    }

    /// Generate signals for a batch of predictions
    pub fn generate_batch(&self, predictions: &[f64], confidences: Option<&[f64]>) -> Vec<Signal> {
        predictions
            .iter()
            .enumerate()
            .map(|(i, &pred)| {
                let conf = confidences.map(|c| c[i]);
                self.generate(pred, conf)
            })
            .collect()
    }

    /// Generate signals with timestamps
    pub fn generate_with_timestamps(
        &self,
        predictions: &[f64],
        timestamps: &[i64],
    ) -> Vec<Signal> {
        predictions
            .iter()
            .zip(timestamps)
            .map(|(&pred, &ts)| self.generate(pred, None).with_timestamp(ts))
            .collect()
    }

    /// Calculate signal statistics
    pub fn signal_stats(&self, signals: &[Signal]) -> SignalStats {
        let total = signals.len();
        let long_count = signals.iter().filter(|s| s.is_long()).count();
        let short_count = signals.iter().filter(|s| s.is_short()).count();
        let neutral_count = signals.iter().filter(|s| s.is_neutral()).count();

        let avg_confidence = if total > 0 {
            signals.iter().map(|s| s.confidence).sum::<f64>() / total as f64
        } else {
            0.0
        };

        SignalStats {
            total,
            long_count,
            short_count,
            neutral_count,
            long_ratio: long_count as f64 / total.max(1) as f64,
            short_ratio: short_count as f64 / total.max(1) as f64,
            avg_confidence,
        }
    }
}

/// Signal statistics
#[derive(Debug, Clone)]
pub struct SignalStats {
    pub total: usize,
    pub long_count: usize,
    pub short_count: usize,
    pub neutral_count: usize,
    pub long_ratio: f64,
    pub short_ratio: f64,
    pub avg_confidence: f64,
}

impl std::fmt::Display for SignalStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SignalStats {{ total: {}, long: {} ({:.1}%), short: {} ({:.1}%), neutral: {}, avg_conf: {:.2} }}",
            self.total,
            self.long_count,
            self.long_ratio * 100.0,
            self.short_count,
            self.short_ratio * 100.0,
            self.neutral_count,
            self.avg_confidence
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_generation() {
        let generator = SignalGenerator::default();

        // Test long signal
        let signal = generator.generate(0.01, None);
        assert!(signal.is_long());

        // Test short signal
        let signal = generator.generate(-0.01, None);
        assert!(signal.is_short());

        // Test neutral signal
        let signal = generator.generate(0.0001, None);
        assert!(signal.is_neutral());
    }

    #[test]
    fn test_confidence_threshold() {
        let config = SignalConfig {
            min_confidence: 0.7,
            ..Default::default()
        };
        let generator = SignalGenerator::new(config);

        // Should be neutral due to low confidence
        let signal = generator.generate(0.01, Some(0.5));
        assert!(signal.is_neutral());

        // Should be long with high confidence
        let signal = generator.generate(0.01, Some(0.8));
        assert!(signal.is_long());
    }

    #[test]
    fn test_no_short() {
        let config = SignalConfig {
            allow_short: false,
            ..Default::default()
        };
        let generator = SignalGenerator::new(config);

        // Should be neutral instead of short
        let signal = generator.generate(-0.01, None);
        assert!(signal.is_neutral());
    }

    #[test]
    fn test_batch_generation() {
        let generator = SignalGenerator::default();
        let predictions = vec![0.01, -0.01, 0.0, 0.005, -0.005];

        let signals = generator.generate_batch(&predictions, None);
        assert_eq!(signals.len(), 5);

        let stats = generator.signal_stats(&signals);
        println!("{}", stats);
    }
}
