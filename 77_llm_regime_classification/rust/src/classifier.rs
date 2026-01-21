//! Market regime classification module.
//!
//! Provides various methods for classifying market regimes:
//! - Statistical classifier (based on returns and volatility)
//! - HMM-based detector (Hidden Markov Model)
//! - Hybrid classifier (combining multiple signals)

use crate::data::OHLCVData;
use std::collections::HashMap;

/// Market regime types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarketRegime {
    /// Upward trending market with positive momentum
    Bull,
    /// Downward trending market with negative momentum
    Bear,
    /// Range-bound market with no clear direction
    Sideways,
    /// High volatility market
    HighVolatility,
    /// Crisis/extreme market stress
    Crisis,
}

impl MarketRegime {
    /// Get string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            MarketRegime::Bull => "bull",
            MarketRegime::Bear => "bear",
            MarketRegime::Sideways => "sideways",
            MarketRegime::HighVolatility => "high_volatility",
            MarketRegime::Crisis => "crisis",
        }
    }

    /// Get all regimes.
    pub fn all() -> Vec<MarketRegime> {
        vec![
            MarketRegime::Bull,
            MarketRegime::Bear,
            MarketRegime::Sideways,
            MarketRegime::HighVolatility,
            MarketRegime::Crisis,
        ]
    }
}

/// Result of regime classification.
#[derive(Debug, Clone)]
pub struct RegimeResult {
    /// Detected market regime
    pub regime: MarketRegime,
    /// Probability of the detected regime (0-1)
    pub probability: f64,
    /// Confidence in the classification (0-1)
    pub confidence: f64,
    /// Human-readable explanation
    pub explanation: String,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

impl RegimeResult {
    /// Create a new regime result.
    pub fn new(regime: MarketRegime, probability: f64, confidence: f64, explanation: &str) -> Self {
        Self {
            regime,
            probability,
            confidence,
            explanation: explanation.to_string(),
            metrics: HashMap::new(),
        }
    }

    /// Add a metric to the result.
    pub fn with_metric(mut self, key: &str, value: f64) -> Self {
        self.metrics.insert(key.to_string(), value);
        self
    }
}

/// Statistical regime classifier based on returns and volatility.
pub struct StatisticalClassifier {
    /// Window size for calculations
    window_size: usize,
    /// Volatility threshold for high volatility regime
    vol_threshold: f64,
    /// Trend threshold for bull/bear classification
    trend_threshold: f64,
    /// Crisis volatility multiplier
    crisis_multiplier: f64,
}

impl StatisticalClassifier {
    /// Create a new statistical classifier.
    ///
    /// # Arguments
    /// * `window_size` - Rolling window size for calculations
    /// * `vol_threshold` - Daily volatility threshold (e.g., 0.02 = 2%)
    /// * `trend_threshold` - Daily return threshold for trend (e.g., 0.001 = 0.1%)
    pub fn new(window_size: usize, vol_threshold: f64, trend_threshold: f64) -> Self {
        Self {
            window_size,
            vol_threshold,
            trend_threshold,
            crisis_multiplier: 2.5,
        }
    }

    /// Create a crypto-specific classifier with adjusted thresholds.
    pub fn for_crypto() -> Self {
        Self {
            window_size: 24,
            vol_threshold: 0.05,
            trend_threshold: 0.003,
            crisis_multiplier: 2.0,
        }
    }

    /// Classify the current market regime.
    pub fn classify(&self, data: &OHLCVData) -> RegimeResult {
        let returns = data.returns();

        if returns.len() < self.window_size {
            return RegimeResult::new(
                MarketRegime::Sideways,
                0.5,
                0.3,
                "Insufficient data for classification",
            );
        }

        // Get recent window
        let recent_returns: Vec<f64> = returns
            .iter()
            .rev()
            .take(self.window_size)
            .copied()
            .collect();

        // Calculate statistics
        let mean_return = Self::mean(&recent_returns);
        let volatility = Self::std(&recent_returns);
        let skewness = Self::skewness(&recent_returns);

        // Calculate trend strength
        let trend_strength = mean_return / volatility.max(0.0001);

        // Determine regime
        let (regime, confidence, explanation) = if volatility > self.vol_threshold * self.crisis_multiplier {
            // Crisis: Extremely high volatility
            let conf = ((volatility / (self.vol_threshold * self.crisis_multiplier)) - 1.0)
                .min(1.0)
                .max(0.5);
            (
                MarketRegime::Crisis,
                conf,
                format!(
                    "Crisis: Extreme volatility ({:.1}%) with skewness {:.2}",
                    volatility * 100.0,
                    skewness
                ),
            )
        } else if volatility > self.vol_threshold {
            // High volatility
            let conf = ((volatility / self.vol_threshold) - 1.0).min(1.0).max(0.5);
            (
                MarketRegime::HighVolatility,
                conf,
                format!(
                    "High volatility: {:.1}% daily vol, unclear direction",
                    volatility * 100.0
                ),
            )
        } else if mean_return > self.trend_threshold && trend_strength > 0.1 {
            // Bull market
            let conf = (trend_strength * 2.0).min(1.0).max(0.5);
            (
                MarketRegime::Bull,
                conf,
                format!(
                    "Bull market: +{:.2}% avg return, {:.1}% volatility",
                    mean_return * 100.0,
                    volatility * 100.0
                ),
            )
        } else if mean_return < -self.trend_threshold && trend_strength < -0.1 {
            // Bear market
            let conf = (-trend_strength * 2.0).min(1.0).max(0.5);
            (
                MarketRegime::Bear,
                conf,
                format!(
                    "Bear market: {:.2}% avg return, {:.1}% volatility",
                    mean_return * 100.0,
                    volatility * 100.0
                ),
            )
        } else {
            // Sideways
            let conf = (1.0 - trend_strength.abs() * 5.0).max(0.5);
            (
                MarketRegime::Sideways,
                conf,
                format!(
                    "Sideways: {:.2}% avg return, no clear trend",
                    mean_return * 100.0
                ),
            )
        };

        RegimeResult::new(regime, confidence, confidence, &explanation)
            .with_metric("mean_return", mean_return)
            .with_metric("volatility", volatility)
            .with_metric("skewness", skewness)
            .with_metric("trend_strength", trend_strength)
    }

    /// Calculate mean of values.
    fn mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }

    /// Calculate standard deviation.
    fn std(values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean = Self::mean(values);
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (values.len() - 1) as f64;
        variance.sqrt()
    }

    /// Calculate skewness.
    fn skewness(values: &[f64]) -> f64 {
        if values.len() < 3 {
            return 0.0;
        }
        let mean = Self::mean(values);
        let std = Self::std(values);
        if std == 0.0 {
            return 0.0;
        }
        let n = values.len() as f64;
        let m3 = values.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n;
        m3
    }
}

impl Default for StatisticalClassifier {
    fn default() -> Self {
        Self::new(20, 0.02, 0.001)
    }
}

/// Hybrid classifier combining multiple signals.
#[allow(dead_code)]
pub struct HybridClassifier {
    /// Statistical classifier component
    stat_classifier: StatisticalClassifier,
    /// Weight for statistical signals (reserved for future text-based classification)
    stat_weight: f64,
}

impl HybridClassifier {
    /// Create a new hybrid classifier.
    pub fn new(stat_classifier: StatisticalClassifier, stat_weight: f64) -> Self {
        Self {
            stat_classifier,
            stat_weight,
        }
    }

    /// Classify using hybrid approach.
    pub fn classify(&self, data: &OHLCVData) -> RegimeResult {
        // Get statistical classification
        let stat_result = self.stat_classifier.classify(data);

        // For now, just use statistical result
        // In a full implementation, this would combine with text analysis
        stat_result
    }
}

/// Regime transition detector with hysteresis.
pub struct TransitionDetector {
    /// Number of periods to confirm transition
    confirmation_periods: usize,
    /// Current regime
    current_regime: MarketRegime,
    /// Count of consecutive different regime detections
    regime_count: usize,
    /// History of results
    history: Vec<RegimeResult>,
}

impl TransitionDetector {
    /// Create a new transition detector.
    pub fn new(confirmation_periods: usize) -> Self {
        Self {
            confirmation_periods,
            current_regime: MarketRegime::Sideways,
            regime_count: 0,
            history: Vec::new(),
        }
    }

    /// Update with new result and check for transition.
    ///
    /// Returns (transition_detected, new_regime) if a transition occurred.
    pub fn update(&mut self, result: RegimeResult) -> (bool, Option<MarketRegime>) {
        self.history.push(result.clone());

        if result.regime != self.current_regime {
            self.regime_count += 1;

            if self.regime_count >= self.confirmation_periods {
                let _old_regime = self.current_regime;
                self.current_regime = result.regime;
                self.regime_count = 0;
                return (true, Some(result.regime));
            }
        } else {
            self.regime_count = 0;
        }

        (false, None)
    }

    /// Get current regime.
    pub fn current_regime(&self) -> MarketRegime {
        self.current_regime
    }

    /// Get transition history.
    pub fn get_transitions(&self) -> Vec<(usize, MarketRegime, MarketRegime)> {
        let mut transitions = Vec::new();
        let mut current = None;

        for (i, result) in self.history.iter().enumerate() {
            if current != Some(result.regime) {
                if let Some(prev) = current {
                    transitions.push((i, prev, result.regime));
                }
                current = Some(result.regime);
            }
        }

        transitions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::YahooFinanceLoader;

    #[test]
    fn test_statistical_classifier() {
        let loader = YahooFinanceLoader::new();
        let data = loader.generate_mock_data("SPY", 100);

        let classifier = StatisticalClassifier::default();
        let result = classifier.classify(&data);

        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(!result.explanation.is_empty());
    }

    #[test]
    fn test_crypto_classifier() {
        let classifier = StatisticalClassifier::for_crypto();
        assert_eq!(classifier.window_size, 24);
        assert!(classifier.vol_threshold > 0.02); // Higher than stock threshold
    }

    #[test]
    fn test_transition_detector() {
        let mut detector = TransitionDetector::new(3);

        // Initial state
        assert_eq!(detector.current_regime(), MarketRegime::Sideways);

        // Simulate bull detection
        for _ in 0..3 {
            let result = RegimeResult::new(MarketRegime::Bull, 0.8, 0.8, "Bull");
            let (transition, new_regime) = detector.update(result);
            if transition {
                assert_eq!(new_regime, Some(MarketRegime::Bull));
            }
        }

        assert_eq!(detector.current_regime(), MarketRegime::Bull);
    }
}
