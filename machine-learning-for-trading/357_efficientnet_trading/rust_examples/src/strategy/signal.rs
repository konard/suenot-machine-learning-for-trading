//! Trading signal generation

use crate::model::PredictionResult;

/// Signal type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    Buy,
    Sell,
    Hold,
}

/// Trading signal
#[derive(Debug, Clone)]
pub struct Signal {
    pub signal_type: SignalType,
    pub confidence: f64,
    pub timestamp: u64,
    pub price: f64,
    pub metadata: SignalMetadata,
}

/// Additional signal metadata
#[derive(Debug, Clone, Default)]
pub struct SignalMetadata {
    pub model_version: Option<String>,
    pub timeframe: Option<String>,
    pub features: Option<Vec<f64>>,
}

impl Signal {
    /// Create a new signal
    pub fn new(signal_type: SignalType, confidence: f64, timestamp: u64, price: f64) -> Self {
        Self {
            signal_type,
            confidence,
            timestamp,
            price,
            metadata: SignalMetadata::default(),
        }
    }

    /// Create from prediction result
    pub fn from_prediction(result: &PredictionResult, timestamp: u64, price: f64) -> Self {
        Self {
            signal_type: result.signal,
            confidence: result.confidence,
            timestamp,
            price,
            metadata: SignalMetadata::default(),
        }
    }

    /// Check if signal is actionable (not Hold and confident enough)
    pub fn is_actionable(&self, min_confidence: f64) -> bool {
        self.signal_type != SignalType::Hold && self.confidence >= min_confidence
    }

    /// Get signal strength (-1 to 1)
    pub fn strength(&self) -> f64 {
        match self.signal_type {
            SignalType::Buy => self.confidence,
            SignalType::Sell => -self.confidence,
            SignalType::Hold => 0.0,
        }
    }
}

/// Signal generator combining multiple signals
pub struct SignalGenerator {
    confidence_threshold: f64,
    ensemble_weights: Vec<f64>,
}

impl SignalGenerator {
    pub fn new(confidence_threshold: f64) -> Self {
        Self {
            confidence_threshold,
            ensemble_weights: Vec::new(),
        }
    }

    /// Set ensemble weights for combining signals
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        self.ensemble_weights = weights;
        self
    }

    /// Generate signal from single prediction
    pub fn from_prediction(&self, result: &PredictionResult, timestamp: u64, price: f64) -> Signal {
        Signal::from_prediction(result, timestamp, price)
    }

    /// Combine multiple signals using weighted ensemble
    pub fn ensemble(&self, signals: &[Signal]) -> Option<Signal> {
        if signals.is_empty() {
            return None;
        }

        let weights = if self.ensemble_weights.len() == signals.len() {
            &self.ensemble_weights
        } else {
            // Equal weights
            &vec![1.0 / signals.len() as f64; signals.len()]
        };

        // Calculate weighted average signal strength
        let mut weighted_strength = 0.0;
        let mut total_weight = 0.0;
        let mut latest_timestamp = 0u64;
        let mut latest_price = 0.0;

        for (signal, &weight) in signals.iter().zip(weights.iter()) {
            weighted_strength += signal.strength() * weight;
            total_weight += weight;

            if signal.timestamp > latest_timestamp {
                latest_timestamp = signal.timestamp;
                latest_price = signal.price;
            }
        }

        if total_weight == 0.0 {
            return None;
        }

        let avg_strength = weighted_strength / total_weight;
        let confidence = avg_strength.abs();

        let signal_type = if avg_strength > self.confidence_threshold {
            SignalType::Buy
        } else if avg_strength < -self.confidence_threshold {
            SignalType::Sell
        } else {
            SignalType::Hold
        };

        Some(Signal::new(signal_type, confidence, latest_timestamp, latest_price))
    }

    /// Apply confirmation logic (require N out of M signals to agree)
    pub fn confirm(&self, signals: &[Signal], required: usize) -> Option<Signal> {
        if signals.len() < required {
            return None;
        }

        let buy_count = signals.iter().filter(|s| s.signal_type == SignalType::Buy).count();
        let sell_count = signals.iter().filter(|s| s.signal_type == SignalType::Sell).count();

        let latest = signals.iter().max_by_key(|s| s.timestamp)?;

        if buy_count >= required {
            let avg_confidence: f64 = signals
                .iter()
                .filter(|s| s.signal_type == SignalType::Buy)
                .map(|s| s.confidence)
                .sum::<f64>() / buy_count as f64;

            Some(Signal::new(SignalType::Buy, avg_confidence, latest.timestamp, latest.price))
        } else if sell_count >= required {
            let avg_confidence: f64 = signals
                .iter()
                .filter(|s| s.signal_type == SignalType::Sell)
                .map(|s| s.confidence)
                .sum::<f64>() / sell_count as f64;

            Some(Signal::new(SignalType::Sell, avg_confidence, latest.timestamp, latest.price))
        } else {
            Some(Signal::new(SignalType::Hold, 0.5, latest.timestamp, latest.price))
        }
    }
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new(0.6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_strength() {
        let buy = Signal::new(SignalType::Buy, 0.8, 0, 100.0);
        assert!((buy.strength() - 0.8).abs() < 0.001);

        let sell = Signal::new(SignalType::Sell, 0.7, 0, 100.0);
        assert!((sell.strength() - (-0.7)).abs() < 0.001);

        let hold = Signal::new(SignalType::Hold, 0.5, 0, 100.0);
        assert_eq!(hold.strength(), 0.0);
    }

    #[test]
    fn test_ensemble() {
        let gen = SignalGenerator::new(0.3);

        let signals = vec![
            Signal::new(SignalType::Buy, 0.8, 0, 100.0),
            Signal::new(SignalType::Buy, 0.7, 1, 101.0),
            Signal::new(SignalType::Hold, 0.5, 2, 102.0),
        ];

        let result = gen.ensemble(&signals).unwrap();
        assert_eq!(result.signal_type, SignalType::Buy);
    }

    #[test]
    fn test_confirmation() {
        let gen = SignalGenerator::new(0.5);

        let signals = vec![
            Signal::new(SignalType::Buy, 0.8, 0, 100.0),
            Signal::new(SignalType::Buy, 0.7, 1, 101.0),
            Signal::new(SignalType::Sell, 0.6, 2, 102.0),
        ];

        let result = gen.confirm(&signals, 2).unwrap();
        assert_eq!(result.signal_type, SignalType::Buy);
    }
}
