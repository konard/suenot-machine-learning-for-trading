//! Flow-based trading strategy

use crate::flow::anomaly::AnomalyDetector;
use crate::flow::model::NormalizingFlow;
use crate::flow::regime::RegimeDetector;
use crate::strategy::signal::{Signal, SignalType};
use ndarray::{Array1, Array2};

/// Flow-based trading strategy
///
/// Uses normalizing flow for:
/// - Anomaly detection (reduce exposure when unusual)
/// - Regime detection (adapt to market state)
/// - Density-based confidence estimation
pub struct FlowTradingStrategy {
    /// Flow model for density estimation
    pub model: NormalizingFlow,
    /// Anomaly detector
    pub anomaly_detector: AnomalyDetector,
    /// Regime detector
    pub regime_detector: RegimeDetector,
    /// Confidence threshold for signals
    pub confidence_threshold: f64,
    /// Whether strategy has been fitted
    pub is_fitted: bool,
}

impl FlowTradingStrategy {
    /// Create new flow trading strategy
    pub fn new(model: NormalizingFlow) -> Self {
        Self {
            model,
            anomaly_detector: AnomalyDetector::new(5.0),
            regime_detector: RegimeDetector::new(4),
            confidence_threshold: 0.6,
            is_fitted: false,
        }
    }

    /// Set confidence threshold
    pub fn with_confidence_threshold(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Set number of regimes
    pub fn with_n_regimes(mut self, n: usize) -> Self {
        self.regime_detector = RegimeDetector::new(n);
        self
    }

    /// Set anomaly threshold percentile
    pub fn with_anomaly_percentile(mut self, percentile: f64) -> Self {
        self.anomaly_detector = AnomalyDetector::new(percentile);
        self
    }

    /// Fit strategy components on training data
    pub fn fit(&mut self, data: &Array2<f64>, returns: Option<&Array1<f64>>) {
        log::info!("Fitting flow trading strategy...");

        // Fit anomaly detector
        self.anomaly_detector.fit(&mut self.model, data);

        // Fit regime detector
        match returns {
            Some(rets) => self.regime_detector.fit_with_returns(&mut self.model, data, rets),
            None => self.regime_detector.fit(&mut self.model, data),
        }

        self.is_fitted = true;
        log::info!("Strategy fitting complete");
    }

    /// Generate trading signal from current market data
    pub fn generate_signal(&mut self, features: &Array1<f64>) -> Signal {
        if !self.is_fitted {
            return Signal::neutral("Strategy not fitted");
        }

        // Convert to 2D array
        let features_2d = features.clone().insert_axis(ndarray::Axis(0));

        // 1. Compute likelihood for anomaly detection
        let log_likelihood = self.model.log_prob(&features_2d)[0];

        // 2. Check for anomaly
        let (is_anomaly, _) = self.anomaly_detector.detect(&mut self.model, &features_2d);

        if is_anomaly[0] {
            return Signal::reduce_exposure(
                0.9,
                format!("Anomalous market state detected (log_p={:.2})", log_likelihood),
            )
            .with_log_likelihood(log_likelihood);
        }

        // 3. Detect current regime
        let (regimes, distances) = self.regime_detector.detect(&mut self.model, &features_2d);
        let current_regime = regimes[0];
        let regime_label = self.regime_detector.get_label(current_regime).to_string();

        // Compute regime confidence
        let regime_confidence = self.regime_detector.confidence(&distances.row(0).to_owned());

        // 4. Generate signal based on regime
        let signal = if regime_label.contains("Bull") && regime_confidence > self.confidence_threshold {
            Signal::long(
                regime_confidence * 0.8,
                format!("Bullish regime detected ({})", regime_label),
            )
        } else if regime_label.contains("Bear") && regime_confidence > self.confidence_threshold {
            Signal::short(
                regime_confidence * 0.8,
                format!("Bearish regime detected ({})", regime_label),
            )
        } else {
            Signal::neutral(format!(
                "Uncertain regime ({}, conf={:.2})",
                regime_label, regime_confidence
            ))
        };

        signal
            .with_log_likelihood(log_likelihood)
            .with_regime(regime_label)
    }

    /// Generate signals for multiple samples
    pub fn generate_signals(&mut self, features: &Array2<f64>) -> Vec<Signal> {
        features
            .rows()
            .into_iter()
            .map(|row| self.generate_signal(&row.to_owned()))
            .collect()
    }

    /// Get current regime for data
    pub fn get_regime(&mut self, features: &Array2<f64>) -> Vec<String> {
        let (regimes, _) = self.regime_detector.detect(&mut self.model, features);
        regimes
            .iter()
            .map(|&r| self.regime_detector.get_label(r).to_string())
            .collect()
    }

    /// Get anomaly flags for data
    pub fn get_anomalies(&mut self, features: &Array2<f64>) -> Vec<bool> {
        let (is_anomaly, _) = self.anomaly_detector.detect(&mut self.model, features);
        is_anomaly
    }

    /// Get log likelihoods for data
    pub fn get_log_likelihoods(&mut self, features: &Array2<f64>) -> Array1<f64> {
        self.model.log_prob(features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::config::FlowConfig;

    #[test]
    fn test_flow_strategy() {
        let config = FlowConfig::default().with_input_dim(4).with_num_layers(2);
        let model = NormalizingFlow::new(config);
        let mut strategy = FlowTradingStrategy::new(model);

        // Create training data
        let train_data = Array2::from_shape_fn((100, 4), |(i, j)| {
            ((i + j) as f64) * 0.01
        });
        let returns = Array1::from_shape_fn(100, |i| {
            if i % 2 == 0 { 0.01 } else { -0.01 }
        });

        // Fit strategy
        strategy.fit(&train_data, Some(&returns));
        assert!(strategy.is_fitted);

        // Generate signal
        let sample = Array1::from_vec(vec![0.01, 0.02, 0.03, 0.04]);
        let signal = strategy.generate_signal(&sample);

        assert!(signal.log_likelihood.is_some());
        assert!(signal.regime.is_some());
    }
}
