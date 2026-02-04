//! Trading signal types and generation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Trading signal type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    Long,
    Short,
    Hold,
}

impl std::fmt::Display for SignalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalType::Long => write!(f, "LONG"),
            SignalType::Short => write!(f, "SHORT"),
            SignalType::Hold => write!(f, "HOLD"),
        }
    }
}

/// Trading signal with uncertainty information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub symbol: String,
    pub signal_type: SignalType,
    pub prediction: f64,
    pub uncertainty: f64,
    pub confidence: f64,
    pub position_size: f64,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub timestamp: Option<i64>,
}

impl Signal {
    /// Create a new signal
    pub fn new(
        symbol: String,
        signal_type: SignalType,
        prediction: f64,
        uncertainty: f64,
        confidence: f64,
        position_size: f64,
    ) -> Self {
        Self {
            symbol,
            signal_type,
            prediction,
            uncertainty,
            confidence,
            position_size,
            stop_loss: None,
            take_profit: None,
            timestamp: None,
        }
    }

    /// Set stop loss price
    pub fn with_stop_loss(mut self, stop_loss: f64) -> Self {
        self.stop_loss = Some(stop_loss);
        self
    }

    /// Set take profit price
    pub fn with_take_profit(mut self, take_profit: f64) -> Self {
        self.take_profit = Some(take_profit);
        self
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, timestamp: i64) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Check if this is an actionable signal (not hold)
    pub fn is_actionable(&self) -> bool {
        self.signal_type != SignalType::Hold
    }
}

/// Signal generator from ensemble predictions
pub struct SignalGenerator {
    min_prediction_threshold: f64,
    max_uncertainty_threshold: f64,
    min_confidence_threshold: f64,
    base_position_size: f64,
}

impl SignalGenerator {
    /// Create new signal generator
    pub fn new(
        min_prediction_threshold: f64,
        max_uncertainty_threshold: f64,
        min_confidence_threshold: f64,
    ) -> Self {
        Self {
            min_prediction_threshold,
            max_uncertainty_threshold,
            min_confidence_threshold,
            base_position_size: 1.0,
        }
    }

    /// Set base position size
    pub fn with_base_position_size(mut self, size: f64) -> Self {
        self.base_position_size = size;
        self
    }

    /// Calculate confidence from prediction and uncertainty
    pub fn calculate_confidence(&self, prediction: f64, uncertainty: f64) -> f64 {
        if uncertainty >= self.max_uncertainty_threshold {
            return 0.0;
        }

        // Base confidence from uncertainty level
        let uncertainty_conf = 1.0 - (uncertainty / self.max_uncertainty_threshold);

        // Boost for stronger predictions
        let pred_strength = (prediction.abs() / 0.02).min(1.0);

        // Combined confidence
        let confidence = uncertainty_conf * (0.5 + 0.5 * pred_strength);

        confidence.clamp(0.0, 1.0)
    }

    /// Calculate position size based on confidence
    pub fn calculate_position_size(&self, prediction: f64, confidence: f64) -> f64 {
        let mut size = self.base_position_size * confidence;

        // Scale by prediction strength
        let pred_factor = (prediction.abs() / self.min_prediction_threshold).min(2.0) / 2.0;
        size *= pred_factor;

        size.clamp(0.0, self.base_position_size)
    }

    /// Generate a single signal
    pub fn generate_signal(
        &self,
        symbol: &str,
        prediction: f64,
        uncertainty: f64,
        current_price: Option<f64>,
    ) -> Signal {
        let confidence = self.calculate_confidence(prediction, uncertainty);

        let signal_type = if confidence < self.min_confidence_threshold {
            SignalType::Hold
        } else if prediction > self.min_prediction_threshold {
            SignalType::Long
        } else if prediction < -self.min_prediction_threshold {
            SignalType::Short
        } else {
            SignalType::Hold
        };

        let position_size = if signal_type == SignalType::Hold {
            0.0
        } else {
            self.calculate_position_size(prediction, confidence)
        };

        let mut signal = Signal::new(
            symbol.to_string(),
            signal_type,
            prediction,
            uncertainty,
            confidence,
            position_size,
        );

        // Calculate stop loss and take profit if price provided
        if let Some(price) = current_price {
            if signal_type != SignalType::Hold {
                let stop_distance = price * uncertainty * 2.0;
                let tp_distance = price * prediction.abs() * 1.5;

                match signal_type {
                    SignalType::Long => {
                        signal = signal
                            .with_stop_loss(price - stop_distance)
                            .with_take_profit(price + tp_distance);
                    }
                    SignalType::Short => {
                        signal = signal
                            .with_stop_loss(price + stop_distance)
                            .with_take_profit(price - tp_distance);
                    }
                    _ => {}
                }
            }
        }

        signal
    }

    /// Generate signals for multiple predictions
    pub fn generate_signals(
        &self,
        symbols: &[&str],
        predictions: &[f64],
        uncertainties: &[f64],
        current_prices: Option<&[f64]>,
    ) -> Vec<Signal> {
        symbols
            .iter()
            .enumerate()
            .map(|(i, symbol)| {
                let price = current_prices.map(|p| p[i]);
                self.generate_signal(symbol, predictions[i], uncertainties[i], price)
            })
            .collect()
    }

    /// Filter signals by confidence
    pub fn filter_signals(&self, signals: Vec<Signal>, max_positions: Option<usize>) -> Vec<Signal> {
        let mut filtered: Vec<Signal> = signals
            .into_iter()
            .filter(|s| s.confidence >= self.min_confidence_threshold)
            .filter(|s| s.signal_type != SignalType::Hold)
            .collect();

        // Sort by confidence descending
        filtered.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

        // Take top N if specified
        if let Some(max) = max_positions {
            filtered.truncate(max);
        }

        filtered
    }
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new(0.001, 0.05, 0.6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_generation() {
        let generator = SignalGenerator::new(0.001, 0.05, 0.6);

        let signal = generator.generate_signal("BTCUSDT", 0.02, 0.01, Some(50000.0));
        assert_eq!(signal.signal_type, SignalType::Long);
        assert!(signal.confidence > 0.6);
        assert!(signal.position_size > 0.0);
        assert!(signal.stop_loss.is_some());
        assert!(signal.take_profit.is_some());
    }

    #[test]
    fn test_hold_signal() {
        let generator = SignalGenerator::new(0.001, 0.05, 0.6);

        // High uncertainty should result in hold
        let signal = generator.generate_signal("BTCUSDT", 0.02, 0.06, Some(50000.0));
        assert_eq!(signal.signal_type, SignalType::Hold);
        assert_eq!(signal.position_size, 0.0);
    }
}
