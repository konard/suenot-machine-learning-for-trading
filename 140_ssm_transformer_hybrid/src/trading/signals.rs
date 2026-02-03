//! Trading signal generation from hybrid model predictions.

use crate::model::hybrid::HybridPredictions;

/// Trading signal direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalDirection {
    Long,
    Short,
    Flat,
}

/// A trading signal with direction and position size.
#[derive(Debug, Clone)]
pub struct TradingSignal {
    pub direction: SignalDirection,
    pub position_size: f64,
    pub confidence: f64,
    pub predicted_volatility: f64,
    pub predicted_return: f64,
}

impl TradingSignal {
    /// Generate a trading signal from model predictions.
    ///
    /// # Arguments
    /// * `preds` - Model predictions
    /// * `direction_threshold` - Threshold for taking a position (default 0.55)
    /// * `max_position` - Maximum position size (default 1.0)
    /// * `base_size` - Base position size scaling factor (default 0.1)
    pub fn from_predictions(
        preds: &HybridPredictions,
        direction_threshold: f64,
        max_position: f64,
        base_size: f64,
    ) -> Self {
        let vol = preds.volatility.max(1e-6);

        if preds.direction > direction_threshold {
            let confidence = (preds.direction - 0.5) * 2.0;
            let size = (confidence * base_size / vol).min(max_position);
            TradingSignal {
                direction: SignalDirection::Long,
                position_size: size,
                confidence,
                predicted_volatility: preds.volatility,
                predicted_return: preds.return_magnitude,
            }
        } else if preds.direction < (1.0 - direction_threshold) {
            let confidence = (0.5 - preds.direction) * 2.0;
            let size = (confidence * base_size / vol).min(max_position);
            TradingSignal {
                direction: SignalDirection::Short,
                position_size: size,
                confidence,
                predicted_volatility: preds.volatility,
                predicted_return: preds.return_magnitude,
            }
        } else {
            TradingSignal {
                direction: SignalDirection::Flat,
                position_size: 0.0,
                confidence: 0.0,
                predicted_volatility: preds.volatility,
                predicted_return: preds.return_magnitude,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::hybrid::HybridPredictions;

    #[test]
    fn test_long_signal() {
        let preds = HybridPredictions {
            direction: 0.8,
            volatility: 0.02,
            return_magnitude: 0.01,
        };
        let signal = TradingSignal::from_predictions(&preds, 0.55, 1.0, 0.1);
        assert_eq!(signal.direction, SignalDirection::Long);
        assert!(signal.position_size > 0.0);
    }

    #[test]
    fn test_short_signal() {
        let preds = HybridPredictions {
            direction: 0.2,
            volatility: 0.02,
            return_magnitude: -0.01,
        };
        let signal = TradingSignal::from_predictions(&preds, 0.55, 1.0, 0.1);
        assert_eq!(signal.direction, SignalDirection::Short);
        assert!(signal.position_size > 0.0);
    }

    #[test]
    fn test_flat_signal() {
        let preds = HybridPredictions {
            direction: 0.5,
            volatility: 0.02,
            return_magnitude: 0.0,
        };
        let signal = TradingSignal::from_predictions(&preds, 0.55, 1.0, 0.1);
        assert_eq!(signal.direction, SignalDirection::Flat);
        assert_eq!(signal.position_size, 0.0);
    }
}
