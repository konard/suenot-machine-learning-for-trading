//! # Trading Signals
//!
//! Trading signal generation using CNF.

use ndarray::Array1;

use crate::cnf::ContinuousNormalizingFlow;
use crate::utils::{Candle, compute_market_features, apply_normalization};

/// Trading signal type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalType {
    /// Long position (buy)
    Long,
    /// Short position (sell)
    Short,
    /// No trade (stay flat)
    Neutral,
}

impl SignalType {
    /// Convert to numeric value
    pub fn to_value(&self) -> f64 {
        match self {
            SignalType::Long => 1.0,
            SignalType::Short => -1.0,
            SignalType::Neutral => 0.0,
        }
    }
}

/// Trading signal with metadata
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Signal type
    pub signal: SignalType,
    /// Confidence level [0, 1]
    pub confidence: f64,
    /// Log-likelihood of current market state
    pub log_likelihood: f64,
    /// Expected return prediction
    pub expected_return: f64,
    /// Return standard deviation
    pub return_std: f64,
    /// Whether a regime change was detected
    pub regime_change: bool,
}

impl TradingSignal {
    /// Get position size based on signal and confidence
    pub fn position_size(&self) -> f64 {
        self.signal.to_value() * self.confidence
    }

    /// Check if signal is actionable
    pub fn is_actionable(&self) -> bool {
        self.signal != SignalType::Neutral && !self.regime_change
    }
}

/// CNF-based trader
#[derive(Debug)]
pub struct CNFTrader {
    /// CNF model
    cnf: ContinuousNormalizingFlow,
    /// Feature means for normalization
    feature_means: Array1<f64>,
    /// Feature stds for normalization
    feature_stds: Array1<f64>,
    /// Index of return feature in feature vector
    return_idx: usize,
    /// Likelihood threshold for trading
    likelihood_threshold: f64,
    /// Confidence threshold for trading
    confidence_threshold: f64,
    /// Lookback period for features
    lookback: usize,
    /// Likelihood history for regime detection
    likelihood_history: Vec<f64>,
    /// Moving average of likelihood
    likelihood_ma: Option<f64>,
}

impl CNFTrader {
    /// Create a new CNF trader
    pub fn new(cnf: ContinuousNormalizingFlow) -> Self {
        let dim = cnf.dim;
        Self {
            cnf,
            feature_means: Array1::zeros(dim),
            feature_stds: Array1::ones(dim),
            return_idx: 0,
            likelihood_threshold: -10.0,
            confidence_threshold: 0.6,
            lookback: 20,
            likelihood_history: Vec::new(),
            likelihood_ma: None,
        }
    }

    /// Set normalization parameters
    pub fn with_normalization(mut self, means: Array1<f64>, stds: Array1<f64>) -> Self {
        self.feature_means = means;
        self.feature_stds = stds;
        self
    }

    /// Set likelihood threshold
    pub fn with_likelihood_threshold(mut self, threshold: f64) -> Self {
        self.likelihood_threshold = threshold;
        self
    }

    /// Set confidence threshold
    pub fn with_confidence_threshold(mut self, threshold: f64) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Set lookback period
    pub fn with_lookback(mut self, lookback: usize) -> Self {
        self.lookback = lookback;
        self
    }

    /// Generate trading signal from candle data
    pub fn generate_signal(&mut self, candles: &[Candle]) -> TradingSignal {
        // Compute features
        let features = compute_market_features(candles, self.lookback);

        // Normalize
        let normalized = apply_normalization(&features, &self.feature_means, &self.feature_stds);

        self.generate_signal_from_features(&normalized)
    }

    /// Generate signal from pre-computed features
    pub fn generate_signal_from_features(&mut self, features: &Array1<f64>) -> TradingSignal {
        // Compute log-likelihood
        let log_prob = self.cnf.log_prob(features);

        // Update regime tracking
        self.update_likelihood_tracking(log_prob);
        let regime_change = self.detect_regime_change();

        // Check if in distribution
        if log_prob < self.likelihood_threshold {
            return TradingSignal {
                signal: SignalType::Neutral,
                confidence: 0.0,
                log_likelihood: log_prob,
                expected_return: 0.0,
                return_std: 1.0,
                regime_change,
            };
        }

        // Estimate conditional return distribution
        let (expected_return, return_std) = self.estimate_conditional_return(features);

        // Compute confidence
        let z_score = expected_return.abs() / (return_std + 1e-8);
        let confidence = (z_score / 3.0).min(1.0);

        // Determine signal
        let signal = if confidence < self.confidence_threshold {
            SignalType::Neutral
        } else if expected_return > 0.0 {
            SignalType::Long
        } else {
            SignalType::Short
        };

        TradingSignal {
            signal,
            confidence,
            log_likelihood: log_prob,
            expected_return,
            return_std,
            regime_change,
        }
    }

    /// Estimate expected return and std via perturbation sampling
    fn estimate_conditional_return(&self, features: &Array1<f64>) -> (f64, f64) {
        let num_samples = 50;
        let perturbation_scale = 0.1;

        let mut returns = Vec::with_capacity(num_samples);

        // Encode to latent space
        let (z, _) = self.cnf.encode(features);

        // Sample perturbed latent codes
        let mut rng = rand::thread_rng();
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.0, perturbation_scale).unwrap();

        for _ in 0..num_samples {
            let mut z_perturbed = z.clone();
            for i in 0..z.len() {
                z_perturbed[i] += normal.sample(&mut rng);
            }

            // Decode
            let (x_sample, _) = self.cnf.decode(&z_perturbed);
            returns.push(x_sample[self.return_idx]);
        }

        // Compute mean and std
        let mean = returns.iter().sum::<f64>() / num_samples as f64;
        let variance = returns.iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f64>() / num_samples as f64;
        let std = variance.sqrt();

        (mean, std)
    }

    /// Update likelihood tracking for regime detection
    fn update_likelihood_tracking(&mut self, log_prob: f64) {
        self.likelihood_history.push(log_prob);

        // Keep last 50 values
        if self.likelihood_history.len() > 50 {
            self.likelihood_history.remove(0);
        }

        // Update moving average
        if self.likelihood_history.len() >= 10 {
            let recent: f64 = self.likelihood_history.iter()
                .rev()
                .take(10)
                .sum::<f64>() / 10.0;
            self.likelihood_ma = Some(recent);
        }
    }

    /// Detect regime change via likelihood drop
    fn detect_regime_change(&self) -> bool {
        if self.likelihood_ma.is_none() || self.likelihood_history.len() < 20 {
            return false;
        }

        let recent = *self.likelihood_history.last().unwrap();
        let baseline: f64 = self.likelihood_history.iter()
            .rev()
            .skip(10)
            .take(10)
            .sum::<f64>() / 10.0;

        // Regime change if recent likelihood is significantly below baseline
        recent < baseline - 2.0
    }

    /// Reset regime tracking
    pub fn reset(&mut self) {
        self.likelihood_history.clear();
        self.likelihood_ma = None;
    }

    /// Get model reference
    pub fn model(&self) -> &ContinuousNormalizingFlow {
        &self.cnf
    }

    /// Get mutable model reference
    pub fn model_mut(&mut self) -> &mut ContinuousNormalizingFlow {
        &mut self.cnf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::generate_synthetic_candles;

    #[test]
    fn test_signal_type() {
        assert_eq!(SignalType::Long.to_value(), 1.0);
        assert_eq!(SignalType::Short.to_value(), -1.0);
        assert_eq!(SignalType::Neutral.to_value(), 0.0);
    }

    #[test]
    fn test_trader_creation() {
        let cnf = ContinuousNormalizingFlow::new(9, 32, 2);
        let trader = CNFTrader::new(cnf)
            .with_likelihood_threshold(-15.0)
            .with_confidence_threshold(0.5);

        assert_eq!(trader.likelihood_threshold, -15.0);
        assert_eq!(trader.confidence_threshold, 0.5);
    }

    #[test]
    fn test_generate_signal() {
        let cnf = ContinuousNormalizingFlow::new(9, 32, 2);
        let mut trader = CNFTrader::new(cnf);

        let candles = generate_synthetic_candles(50, 100.0);
        let signal = trader.generate_signal(&candles);

        assert!(signal.log_likelihood.is_finite());
        assert!(signal.confidence >= 0.0 && signal.confidence <= 1.0);
    }

    #[test]
    fn test_regime_detection() {
        let cnf = ContinuousNormalizingFlow::new(9, 32, 2);
        let mut trader = CNFTrader::new(cnf);

        // Add normal likelihood values
        for _ in 0..20 {
            trader.update_likelihood_tracking(-5.0);
        }
        assert!(!trader.detect_regime_change());

        // Add sudden drop
        trader.update_likelihood_tracking(-15.0);
        assert!(trader.detect_regime_change());
    }
}
