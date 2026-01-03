//! Trading signal generation based on Neural Spline Flows
//!
//! This module provides signal generation utilities using learned distributions.

use crate::flow::NeuralSplineFlow;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Trading signal with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    /// Signal direction and strength (-1.0 to 1.0)
    pub signal: f64,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Log probability of current state
    pub log_prob: f64,
    /// Whether state is within learned distribution
    pub in_distribution: bool,
    /// Expected return (if computed)
    pub expected_return: Option<f64>,
    /// Latent space return component
    pub latent_return: Option<f64>,
    /// Recommended position size (-1.0 to 1.0)
    pub position_size: f64,
    /// Reason for signal (for logging)
    pub reason: String,
}

impl TradingSignal {
    /// Create a no-trade signal
    pub fn no_trade(reason: &str, log_prob: f64) -> Self {
        Self {
            signal: 0.0,
            confidence: 0.0,
            log_prob,
            in_distribution: false,
            expected_return: None,
            latent_return: None,
            position_size: 0.0,
            reason: reason.to_string(),
        }
    }

    /// Check if this is a buy signal
    pub fn is_buy(&self) -> bool {
        self.signal > 0.0 && self.confidence > 0.0
    }

    /// Check if this is a sell signal
    pub fn is_sell(&self) -> bool {
        self.signal < 0.0 && self.confidence > 0.0
    }

    /// Check if this is a no-trade signal
    pub fn is_no_trade(&self) -> bool {
        self.signal == 0.0 || self.confidence == 0.0
    }
}

/// Signal generator configuration
#[derive(Debug, Clone)]
pub struct SignalGeneratorConfig {
    /// Index of return feature in feature vector
    pub return_feature_idx: usize,
    /// Log probability threshold for in-distribution
    pub density_threshold: f64,
    /// Minimum confidence to generate signal
    pub confidence_threshold: f64,
    /// Z-score threshold for signal generation
    pub z_threshold: f64,
    /// Number of samples for expected return estimation
    pub num_samples: usize,
}

impl Default for SignalGeneratorConfig {
    fn default() -> Self {
        Self {
            return_feature_idx: 0,
            density_threshold: -15.0,
            confidence_threshold: 0.3,
            z_threshold: 0.5,
            num_samples: 1000,
        }
    }
}

/// Signal generator using Neural Spline Flow
pub struct SignalGenerator {
    /// NSF model
    model: NeuralSplineFlow,
    /// Configuration
    config: SignalGeneratorConfig,
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new(model: NeuralSplineFlow, config: SignalGeneratorConfig) -> Self {
        Self { model, config }
    }

    /// Create with default configuration
    pub fn with_defaults(model: NeuralSplineFlow) -> Self {
        Self::new(model, SignalGeneratorConfig::default())
    }

    /// Generate trading signal from market state
    pub fn generate_signal(&self, state: &Array1<f64>) -> TradingSignal {
        // Compute log probability
        let log_prob = self.model.log_prob(state);

        // Check if in distribution
        let in_distribution = log_prob > self.config.density_threshold;

        if !in_distribution {
            return TradingSignal::no_trade("Out of distribution", log_prob);
        }

        // Transform to latent space
        let (z, _) = self.model.forward(state);

        // Get return component in latent space
        let return_z = z[self.config.return_feature_idx];

        // Estimate expected return from samples
        let samples = self.model.sample(self.config.num_samples);
        let returns: Vec<f64> = samples
            .column(self.config.return_feature_idx)
            .iter()
            .cloned()
            .collect();

        let expected_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let return_std = (returns.iter().map(|r| (r - expected_return).powi(2)).sum::<f64>()
            / returns.len() as f64)
            .sqrt();

        // Compute confidence based on z-score magnitude
        let confidence = (return_z.abs() / 2.0).min(1.0);

        // Check confidence threshold
        if confidence < self.config.confidence_threshold {
            return TradingSignal {
                signal: 0.0,
                confidence,
                log_prob,
                in_distribution: true,
                expected_return: Some(expected_return),
                latent_return: Some(return_z),
                position_size: 0.0,
                reason: "Low confidence".to_string(),
            };
        }

        // Generate signal based on latent return direction
        let signal = if expected_return > 0.0 && return_z > self.config.z_threshold {
            confidence
        } else if expected_return < 0.0 && return_z < -self.config.z_threshold {
            -confidence
        } else {
            0.0
        };

        // Compute position size (can be scaled by risk manager)
        let position_size = signal;

        let reason = if signal > 0.0 {
            "Bullish signal from latent space".to_string()
        } else if signal < 0.0 {
            "Bearish signal from latent space".to_string()
        } else {
            "No clear signal".to_string()
        };

        TradingSignal {
            signal,
            confidence,
            log_prob,
            in_distribution: true,
            expected_return: Some(expected_return),
            latent_return: Some(return_z),
            position_size,
            reason,
        }
    }

    /// Generate signals for a batch of states
    pub fn generate_signals_batch(&self, states: &[Array1<f64>]) -> Vec<TradingSignal> {
        states.iter().map(|s| self.generate_signal(s)).collect()
    }

    /// Get reference to the underlying model
    pub fn model(&self) -> &NeuralSplineFlow {
        &self.model
    }

    /// Get configuration
    pub fn config(&self) -> &SignalGeneratorConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::NSFConfig;

    #[test]
    fn test_trading_signal() {
        let signal = TradingSignal::no_trade("Test", -10.0);
        assert!(signal.is_no_trade());
        assert!(!signal.is_buy());
        assert!(!signal.is_sell());
    }

    #[test]
    fn test_signal_generator() {
        let config = NSFConfig::new(8);
        let model = NeuralSplineFlow::new(config);
        let generator = SignalGenerator::with_defaults(model);

        let state = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]);
        let signal = generator.generate_signal(&state);

        assert!(signal.log_prob.is_finite());
    }
}
