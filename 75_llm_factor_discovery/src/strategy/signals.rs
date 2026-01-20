//! Signal generation from factors
//!
//! Converts factor values into trading signals.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A trading signal generated from factor analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    /// Symbol
    pub symbol: String,
    /// Signal direction (-1 to 1)
    pub direction: f64,
    /// Signal strength (0 to 1)
    pub strength: f64,
    /// Confidence in the signal (0 to 1)
    pub confidence: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Source factor(s)
    pub source_factors: Vec<String>,
}

impl TradingSignal {
    /// Check if signal is long
    pub fn is_long(&self) -> bool {
        self.direction > 0.0
    }

    /// Check if signal is short
    pub fn is_short(&self) -> bool {
        self.direction < 0.0
    }

    /// Check if signal is strong enough to trade
    pub fn is_tradeable(&self, min_strength: f64, min_confidence: f64) -> bool {
        self.strength.abs() >= min_strength && self.confidence >= min_confidence
    }

    /// Get suggested position size (as fraction of capital)
    pub fn suggested_position_size(&self, max_position: f64) -> f64 {
        self.strength * self.confidence * max_position * self.direction.signum()
    }
}

/// Factor signal (raw factor value with metadata)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorSignal {
    /// Factor name
    pub factor_name: String,
    /// Factor value (typically normalized)
    pub value: f64,
    /// Z-score of the factor value
    pub z_score: f64,
    /// Percentile rank
    pub percentile: f64,
    /// Historical average
    pub historical_mean: f64,
    /// Historical standard deviation
    pub historical_std: f64,
}

impl FactorSignal {
    /// Create from raw value with historical context
    pub fn new(
        factor_name: String,
        value: f64,
        historical_mean: f64,
        historical_std: f64,
    ) -> Self {
        let z_score = if historical_std > 0.0 {
            (value - historical_mean) / historical_std
        } else {
            0.0
        };

        Self {
            factor_name,
            value,
            z_score,
            percentile: 0.5, // Would need full distribution to calculate
            historical_mean,
            historical_std,
        }
    }

    /// Check if signal is extreme (|z| > threshold)
    pub fn is_extreme(&self, threshold: f64) -> bool {
        self.z_score.abs() > threshold
    }
}

/// Signal generator from factor values
pub struct SignalGenerator {
    /// Minimum z-score to generate signal
    min_z_score: f64,
    /// Confidence decay for older signals
    confidence_decay: f64,
}

impl SignalGenerator {
    /// Create new signal generator
    pub fn new() -> Self {
        Self {
            min_z_score: 1.0,
            confidence_decay: 0.95,
        }
    }

    /// Create with custom parameters
    pub fn with_params(min_z_score: f64, confidence_decay: f64) -> Self {
        Self {
            min_z_score,
            confidence_decay,
        }
    }

    /// Generate trading signal from factor signals
    pub fn generate(
        &self,
        symbol: &str,
        factor_signals: &[FactorSignal],
        factor_weights: &[f64],
    ) -> Option<TradingSignal> {
        if factor_signals.is_empty() || factor_signals.len() != factor_weights.len() {
            return None;
        }

        // Calculate weighted signal
        let total_weight: f64 = factor_weights.iter().map(|w| w.abs()).sum();
        if total_weight == 0.0 {
            return None;
        }

        let weighted_z: f64 = factor_signals
            .iter()
            .zip(factor_weights.iter())
            .map(|(s, w)| s.z_score * w)
            .sum::<f64>()
            / total_weight;

        // Only generate signal if z-score is significant
        if weighted_z.abs() < self.min_z_score {
            return None;
        }

        // Direction from sign of z-score
        let direction = weighted_z.signum();

        // Strength from magnitude (capped at 3 std)
        let strength = (weighted_z.abs() / 3.0).min(1.0);

        // Confidence from agreement of factors
        let agreements = factor_signals
            .iter()
            .filter(|s| s.z_score.signum() == direction)
            .count();
        let confidence = agreements as f64 / factor_signals.len() as f64;

        let source_factors: Vec<String> = factor_signals
            .iter()
            .map(|s| s.factor_name.clone())
            .collect();

        Some(TradingSignal {
            symbol: symbol.to_string(),
            direction,
            strength,
            confidence,
            timestamp: Utc::now(),
            source_factors,
        })
    }
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Factor-based trading strategy
pub struct FactorStrategy {
    /// Strategy name
    pub name: String,
    /// Factor expressions
    pub factors: Vec<String>,
    /// Factor weights
    pub weights: Vec<f64>,
    /// Minimum signal strength to trade
    pub min_strength: f64,
    /// Minimum confidence to trade
    pub min_confidence: f64,
    /// Maximum position size
    pub max_position: f64,
}

impl FactorStrategy {
    /// Create a new strategy
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            factors: Vec::new(),
            weights: Vec::new(),
            min_strength: 0.3,
            min_confidence: 0.5,
            max_position: 0.1,
        }
    }

    /// Add a factor to the strategy
    pub fn add_factor(&mut self, expression: &str, weight: f64) {
        self.factors.push(expression.to_string());
        self.weights.push(weight);
    }

    /// Set trading thresholds
    pub fn set_thresholds(&mut self, min_strength: f64, min_confidence: f64) {
        self.min_strength = min_strength;
        self.min_confidence = min_confidence;
    }

    /// Check if a signal should be traded
    pub fn should_trade(&self, signal: &TradingSignal) -> bool {
        signal.is_tradeable(self.min_strength, self.min_confidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trading_signal() {
        let signal = TradingSignal {
            symbol: "BTCUSDT".to_string(),
            direction: 1.0,
            strength: 0.8,
            confidence: 0.7,
            timestamp: Utc::now(),
            source_factors: vec!["momentum".to_string()],
        };

        assert!(signal.is_long());
        assert!(!signal.is_short());
        assert!(signal.is_tradeable(0.5, 0.5));
        assert!(!signal.is_tradeable(0.9, 0.5));
    }

    #[test]
    fn test_factor_signal() {
        let signal = FactorSignal::new("momentum".to_string(), 2.5, 0.0, 1.0);

        assert!((signal.z_score - 2.5).abs() < 1e-10);
        assert!(signal.is_extreme(2.0));
        assert!(!signal.is_extreme(3.0));
    }

    #[test]
    fn test_signal_generator() {
        let generator = SignalGenerator::new();

        let factors = vec![
            FactorSignal::new("mom".to_string(), 2.0, 0.0, 1.0),
            FactorSignal::new("rev".to_string(), 1.5, 0.0, 1.0),
        ];
        let weights = vec![1.0, 0.5];

        let signal = generator.generate("BTCUSDT", &factors, &weights);
        assert!(signal.is_some());

        let signal = signal.unwrap();
        assert!(signal.is_long());
        assert!(signal.confidence > 0.0);
    }
}
