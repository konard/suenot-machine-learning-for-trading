//! Uncertainty-aware trading strategy

use super::signal::{Signal, SignalGenerator, SignalType};
use crate::ensemble::model::DeepEnsemble;
use crate::ensemble::prediction::EnsemblePrediction;
use ndarray::Array2;

/// Uncertainty-aware trading strategy
pub struct UncertaintyStrategy {
    /// Signal generator
    pub signal_generator: SignalGenerator,
    /// Maximum position size as fraction of capital
    pub max_position_size: f64,
}

impl Default for UncertaintyStrategy {
    fn default() -> Self {
        Self::new(0.6, 0.5, 1.5, 0.1)
    }
}

impl UncertaintyStrategy {
    /// Create a new uncertainty strategy
    pub fn new(
        confidence_threshold: f64,
        epistemic_threshold: f64,
        signal_strength_threshold: f64,
        max_position_size: f64,
    ) -> Self {
        Self {
            signal_generator: SignalGenerator::new(
                confidence_threshold,
                epistemic_threshold,
                signal_strength_threshold,
            ),
            max_position_size,
        }
    }

    /// Generate signals from ensemble predictions
    pub fn generate_signals(
        &self,
        predictions: &EnsemblePrediction,
        symbols: &[String],
    ) -> Vec<Signal> {
        let mut signals = Vec::new();

        for (i, symbol) in symbols.iter().enumerate() {
            let signal = self.signal_generator.generate(
                symbol,
                predictions.mean[i],
                predictions.total_std[i],
                predictions.epistemic_std[i],
                predictions.aleatoric_std[i],
            );
            signals.push(signal);
        }

        signals
    }

    /// Generate signals using ensemble model
    pub fn generate_from_model(
        &self,
        ensemble: &mut DeepEnsemble,
        features: &Array2<f64>,
        symbols: &[String],
    ) -> Vec<Signal> {
        let predictions = ensemble.predict(features);
        self.generate_signals(&predictions, symbols)
    }

    /// Calculate position size based on uncertainty
    ///
    /// Uses Kelly-inspired sizing with uncertainty penalty.
    pub fn calculate_position_size(&self, signal: &Signal, capital: f64) -> f64 {
        if signal.signal_type == SignalType::Hold {
            return 0.0;
        }

        // Base Kelly fraction
        let kelly = signal.confidence / (1.0 + signal.total_uncertainty);

        // Epistemic penalty
        let epistemic_ratio = if signal.total_uncertainty > 0.0 {
            signal.epistemic_uncertainty / signal.total_uncertainty
        } else {
            1.0
        };
        let epistemic_penalty = 1.0 - epistemic_ratio.min(1.0);

        // Final position
        let position_fraction = kelly * epistemic_penalty * self.max_position_size;
        let position = capital * position_fraction;

        match signal.signal_type {
            SignalType::Long => position,
            SignalType::Short => -position,
            SignalType::Hold => 0.0,
        }
    }
}

/// Market regime detector using uncertainty dynamics
pub struct RegimeDetector {
    /// Window size for trend calculation
    pub window_size: usize,
    /// Threshold for epistemic trend
    pub epistemic_threshold: f64,
    /// Threshold for aleatoric trend
    pub aleatoric_threshold: f64,
    /// History of epistemic uncertainty
    epistemic_history: Vec<f64>,
    /// History of aleatoric uncertainty
    aleatoric_history: Vec<f64>,
}

impl Default for RegimeDetector {
    fn default() -> Self {
        Self::new(20, 0.1, 0.1)
    }
}

impl RegimeDetector {
    /// Create a new regime detector
    pub fn new(window_size: usize, epistemic_threshold: f64, aleatoric_threshold: f64) -> Self {
        Self {
            window_size,
            epistemic_threshold,
            aleatoric_threshold,
            epistemic_history: Vec::new(),
            aleatoric_history: Vec::new(),
        }
    }

    /// Update with new predictions
    pub fn update(&mut self, predictions: &EnsemblePrediction) {
        let avg_epistemic: f64 =
            predictions.epistemic_std.iter().sum::<f64>() / predictions.num_samples() as f64;
        let avg_aleatoric: f64 =
            predictions.aleatoric_std.iter().sum::<f64>() / predictions.num_samples() as f64;

        self.epistemic_history.push(avg_epistemic);
        self.aleatoric_history.push(avg_aleatoric);

        // Keep limited history
        let max_history = self.window_size * 2;
        if self.epistemic_history.len() > max_history {
            self.epistemic_history = self.epistemic_history[max_history / 2..].to_vec();
            self.aleatoric_history = self.aleatoric_history[max_history / 2..].to_vec();
        }
    }

    /// Detect current market regime
    pub fn detect(&self) -> MarketRegime {
        if self.epistemic_history.len() < self.window_size {
            return MarketRegime::Unknown;
        }

        let n = self.epistemic_history.len();
        let recent_epistemic = &self.epistemic_history[n - self.window_size..];
        let recent_aleatoric = &self.aleatoric_history[n - self.window_size..];

        // Calculate trends using simple linear regression
        let epistemic_trend = Self::calculate_trend(recent_epistemic);
        let aleatoric_trend = Self::calculate_trend(recent_aleatoric);

        let avg_epistemic: f64 = recent_epistemic.iter().sum::<f64>() / self.window_size as f64;
        let avg_aleatoric: f64 = recent_aleatoric.iter().sum::<f64>() / self.window_size as f64;

        if epistemic_trend > self.epistemic_threshold {
            MarketRegime::RegimeChange
        } else if aleatoric_trend > self.aleatoric_threshold {
            MarketRegime::VolatilitySpike
        } else if avg_epistemic < 0.1 && avg_aleatoric < 0.1 {
            MarketRegime::Stable
        } else {
            MarketRegime::Normal
        }
    }

    /// Calculate trend (slope of linear regression)
    fn calculate_trend(data: &[f64]) -> f64 {
        let n = data.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean: f64 = data.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in data.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }
}

/// Market regime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    /// Unknown (insufficient data)
    Unknown,
    /// Stable market
    Stable,
    /// Normal conditions
    Normal,
    /// Regime change detected
    RegimeChange,
    /// Volatility spike
    VolatilitySpike,
}

impl std::fmt::Display for MarketRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarketRegime::Unknown => write!(f, "UNKNOWN"),
            MarketRegime::Stable => write!(f, "STABLE"),
            MarketRegime::Normal => write!(f, "NORMAL"),
            MarketRegime::RegimeChange => write!(f, "REGIME_CHANGE"),
            MarketRegime::VolatilitySpike => write!(f, "VOLATILITY_SPIKE"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_position_sizing() {
        let strategy = UncertaintyStrategy::default();

        // High confidence long signal
        let signal = Signal::new(
            "BTCUSDT",
            SignalType::Long,
            0.8,
            0.05,
            0.02,
            0.005,
            0.015,
        );

        let position = strategy.calculate_position_size(&signal, 100000.0);
        assert!(position > 0.0);
        assert!(position <= 10000.0); // Max 10% of capital

        // Hold signal should have zero position
        let hold_signal = Signal::hold("BTCUSDT");
        let hold_position = strategy.calculate_position_size(&hold_signal, 100000.0);
        assert_eq!(hold_position, 0.0);
    }

    #[test]
    fn test_regime_detector() {
        let mut detector = RegimeDetector::new(5, 0.1, 0.1);

        // Add stable predictions
        for _ in 0..10 {
            let pred = EnsemblePrediction::new(
                vec![Array1::from_vec(vec![0.0; 5]); 3],
                vec![Array1::from_vec(vec![0.05; 5]); 3],
            );
            detector.update(&pred);
        }

        let regime = detector.detect();
        assert!(regime == MarketRegime::Stable || regime == MarketRegime::Normal);
    }
}
