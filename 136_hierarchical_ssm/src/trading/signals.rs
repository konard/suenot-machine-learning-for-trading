//! Signal generation from HiSS model predictions.

use crate::model::Prediction;

/// Trading signal type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    Long,
    Short,
    Flat,
}

/// Generates trading signals from model predictions.
#[derive(Debug, Clone)]
pub struct SignalGenerator {
    pub direction_threshold_long: f64,
    pub direction_threshold_short: f64,
    pub return_threshold: f64,
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self {
            direction_threshold_long: 0.6,
            direction_threshold_short: 0.4,
            return_threshold: 0.001,
        }
    }
}

impl SignalGenerator {
    /// Generate a trading signal from a model prediction.
    pub fn generate(&self, prediction: &Prediction) -> Signal {
        if prediction.direction_probability > self.direction_threshold_long
            && prediction.predicted_return > self.return_threshold
        {
            Signal::Long
        } else if prediction.direction_probability < self.direction_threshold_short
            && prediction.predicted_return < -self.return_threshold
        {
            Signal::Short
        } else {
            Signal::Flat
        }
    }

    /// Compute position size using Kelly-inspired formula.
    ///
    /// Returns fraction of portfolio to allocate (0.0 to max_position_pct).
    pub fn position_size(
        &self,
        prediction: &Prediction,
        max_position_pct: f64,
    ) -> f64 {
        if prediction.predicted_volatility <= 0.0 {
            return 0.0;
        }
        let kelly =
            prediction.predicted_return.abs() / (prediction.predicted_volatility.powi(2) + 1e-8);
        (kelly * 0.25).min(max_position_pct).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_long() {
        let gen = SignalGenerator::default();
        let pred = Prediction {
            predicted_return: 0.01,
            direction_probability: 0.75,
            predicted_volatility: 0.02,
        };
        assert_eq!(gen.generate(&pred), Signal::Long);
    }

    #[test]
    fn test_signal_short() {
        let gen = SignalGenerator::default();
        let pred = Prediction {
            predicted_return: -0.01,
            direction_probability: 0.25,
            predicted_volatility: 0.02,
        };
        assert_eq!(gen.generate(&pred), Signal::Short);
    }

    #[test]
    fn test_signal_flat() {
        let gen = SignalGenerator::default();
        let pred = Prediction {
            predicted_return: 0.0001,
            direction_probability: 0.5,
            predicted_volatility: 0.02,
        };
        assert_eq!(gen.generate(&pred), Signal::Flat);
    }
}
