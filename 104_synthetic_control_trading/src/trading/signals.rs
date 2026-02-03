//! Signal generation for SCM-based trading strategies.

use crate::model::SyntheticControlModel;
use anyhow::Result;

/// Trading signal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    /// Long position
    Long,
    /// Short position
    Short,
    /// No action
    Hold,
}

/// Signal with confidence and metadata.
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Signal direction
    pub signal: Signal,
    /// Confidence level (0-1)
    pub confidence: f64,
    /// Cumulative abnormal return
    pub car: f64,
    /// Pre-treatment fit quality (RMSE)
    pub fit_quality: f64,
    /// Suggested position size
    pub position_size: f64,
}

/// Signal generator configuration.
#[derive(Debug, Clone)]
pub struct SignalConfig {
    /// CAR threshold for long signal
    pub long_threshold: f64,
    /// CAR threshold for short signal
    pub short_threshold: f64,
    /// Minimum fit quality (max RMSE)
    pub max_rmse: f64,
    /// Minimum confidence for signal
    pub min_confidence: f64,
    /// Base position size
    pub base_position_size: f64,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            long_threshold: 0.02,
            short_threshold: -0.02,
            max_rmse: 0.05,
            min_confidence: 0.5,
            base_position_size: 0.1,
        }
    }
}

/// Generator for SCM-based trading signals.
#[derive(Debug, Clone)]
pub struct SignalGenerator {
    config: SignalConfig,
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self::new(SignalConfig::default())
    }
}

impl SignalGenerator {
    /// Create a new signal generator.
    pub fn new(config: SignalConfig) -> Self {
        Self { config }
    }

    /// Generate trading signal from SCM analysis.
    ///
    /// # Arguments
    ///
    /// * `model` - Fitted SCM model
    /// * `treated_post` - Post-treatment returns for treated unit
    /// * `donors_post` - Post-treatment returns for donors
    ///
    /// # Returns
    ///
    /// Trading signal with metadata
    pub fn generate(
        &self,
        model: &SyntheticControlModel,
        treated_post: &[f64],
        donors_post: &[Vec<f64>],
    ) -> Result<TradingSignal> {
        // Get effects
        let effects = model.estimate_effects(treated_post, donors_post)?;
        let car = effects.car;

        // Get fit quality
        let rmse = model.pre_treatment_rmse().unwrap_or(f64::MAX);

        // Check fit quality
        if rmse > self.config.max_rmse {
            return Ok(TradingSignal {
                signal: Signal::Hold,
                confidence: 0.0,
                car,
                fit_quality: rmse,
                position_size: 0.0,
            });
        }

        // Calculate confidence based on fit quality and CAR magnitude
        let fit_confidence = 1.0 - (rmse / self.config.max_rmse).min(1.0);
        let car_magnitude = car.abs();
        let threshold = self.config.long_threshold.abs();
        let magnitude_confidence = (car_magnitude / (2.0 * threshold)).min(1.0);
        let confidence = fit_confidence * magnitude_confidence;

        // Determine signal
        let signal = if car > self.config.long_threshold && confidence >= self.config.min_confidence
        {
            Signal::Long
        } else if car < self.config.short_threshold
            && confidence >= self.config.min_confidence
        {
            Signal::Short
        } else {
            Signal::Hold
        };

        // Calculate position size
        let position_size = if signal != Signal::Hold {
            self.config.base_position_size * confidence
        } else {
            0.0
        };

        Ok(TradingSignal {
            signal,
            confidence,
            car,
            fit_quality: rmse,
            position_size,
        })
    }

    /// Generate signal from raw data.
    ///
    /// Fits model and generates signal in one step.
    pub fn generate_from_data(
        &self,
        treated_pre: &[f64],
        donors_pre: &[Vec<f64>],
        treated_post: &[f64],
        donors_post: &[Vec<f64>],
    ) -> Result<TradingSignal> {
        let mut model = SyntheticControlModel::new();
        model.fit(treated_pre, donors_pre)?;
        self.generate(&model, treated_post, donors_post)
    }
}

/// Generate simple binary signal based on CAR.
pub fn simple_signal(car: f64, threshold: f64) -> Signal {
    if car > threshold {
        Signal::Long
    } else if car < -threshold {
        Signal::Short
    } else {
        Signal::Hold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_signal() {
        assert_eq!(simple_signal(0.05, 0.02), Signal::Long);
        assert_eq!(simple_signal(-0.05, 0.02), Signal::Short);
        assert_eq!(simple_signal(0.01, 0.02), Signal::Hold);
    }

    #[test]
    fn test_signal_generator() {
        let config = SignalConfig {
            long_threshold: 0.02,
            short_threshold: -0.02,
            max_rmse: 0.1,
            min_confidence: 0.0,
            base_position_size: 0.1,
        };

        let generator = SignalGenerator::new(config);

        // Create simple test data
        let treated_pre = vec![0.01, 0.02, 0.01, 0.02];
        let donors_pre = vec![vec![0.01, 0.02, 0.01, 0.02]];
        let treated_post = vec![0.05, 0.03];
        let donors_post = vec![vec![0.01, 0.01]];

        let signal = generator
            .generate_from_data(&treated_pre, &donors_pre, &treated_post, &donors_post)
            .unwrap();

        // Should generate a long signal due to positive CAR
        assert_eq!(signal.signal, Signal::Long);
        assert!(signal.car > 0.0);
    }
}
