//! Market regime classification for few-shot learning.
//!
//! This module provides market regime detection using few-shot learning
//! to identify market conditions with minimal historical examples.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Market regime types that can be classified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Strong upward price movement with high momentum
    StrongUptrend,
    /// Moderate upward movement
    WeakUptrend,
    /// Range-bound market with no clear direction
    Sideways,
    /// Moderate downward movement
    WeakDowntrend,
    /// Strong downward price movement
    StrongDowntrend,
    /// High volatility with rapid price swings
    Volatile,
    /// Sharp market decline (panic selling)
    Crash,
    /// Recovery from a crash or downtrend
    Recovery,
}

impl MarketRegime {
    /// Get all possible regime values.
    pub fn all() -> Vec<MarketRegime> {
        vec![
            MarketRegime::StrongUptrend,
            MarketRegime::WeakUptrend,
            MarketRegime::Sideways,
            MarketRegime::WeakDowntrend,
            MarketRegime::StrongDowntrend,
            MarketRegime::Volatile,
            MarketRegime::Crash,
            MarketRegime::Recovery,
        ]
    }

    /// Get the number of regime types.
    pub fn count() -> usize {
        8
    }

    /// Convert regime to class index.
    pub fn to_index(&self) -> usize {
        match self {
            MarketRegime::StrongUptrend => 0,
            MarketRegime::WeakUptrend => 1,
            MarketRegime::Sideways => 2,
            MarketRegime::WeakDowntrend => 3,
            MarketRegime::StrongDowntrend => 4,
            MarketRegime::Volatile => 5,
            MarketRegime::Crash => 6,
            MarketRegime::Recovery => 7,
        }
    }

    /// Convert class index to regime.
    pub fn from_index(index: usize) -> Option<MarketRegime> {
        match index {
            0 => Some(MarketRegime::StrongUptrend),
            1 => Some(MarketRegime::WeakUptrend),
            2 => Some(MarketRegime::Sideways),
            3 => Some(MarketRegime::WeakDowntrend),
            4 => Some(MarketRegime::StrongDowntrend),
            5 => Some(MarketRegime::Volatile),
            6 => Some(MarketRegime::Crash),
            7 => Some(MarketRegime::Recovery),
            _ => None,
        }
    }

    /// Check if the regime is bullish.
    pub fn is_bullish(&self) -> bool {
        matches!(self, MarketRegime::StrongUptrend | MarketRegime::WeakUptrend | MarketRegime::Recovery)
    }

    /// Check if the regime is bearish.
    pub fn is_bearish(&self) -> bool {
        matches!(self, MarketRegime::StrongDowntrend | MarketRegime::WeakDowntrend | MarketRegime::Crash)
    }

    /// Check if the regime is neutral/sideways.
    pub fn is_neutral(&self) -> bool {
        matches!(self, MarketRegime::Sideways)
    }

    /// Check if the regime indicates high volatility.
    pub fn is_volatile(&self) -> bool {
        matches!(self, MarketRegime::Volatile | MarketRegime::Crash)
    }

    /// Get recommended position bias for this regime.
    pub fn position_bias(&self) -> f64 {
        match self {
            MarketRegime::StrongUptrend => 1.0,
            MarketRegime::WeakUptrend => 0.5,
            MarketRegime::Sideways => 0.0,
            MarketRegime::WeakDowntrend => -0.5,
            MarketRegime::StrongDowntrend => -1.0,
            MarketRegime::Volatile => 0.0,
            MarketRegime::Crash => -0.8,
            MarketRegime::Recovery => 0.6,
        }
    }
}

impl fmt::Display for MarketRegime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            MarketRegime::StrongUptrend => "Strong Uptrend",
            MarketRegime::WeakUptrend => "Weak Uptrend",
            MarketRegime::Sideways => "Sideways",
            MarketRegime::WeakDowntrend => "Weak Downtrend",
            MarketRegime::StrongDowntrend => "Strong Downtrend",
            MarketRegime::Volatile => "Volatile",
            MarketRegime::Crash => "Crash",
            MarketRegime::Recovery => "Recovery",
        };
        write!(f, "{}", name)
    }
}

/// Classifier for detecting market regimes using few-shot learning.
pub struct RegimeClassifier {
    /// Confidence threshold for regime classification
    confidence_threshold: f64,
    /// Minimum samples required for classification
    min_samples: usize,
    /// Historical regime observations for reference
    regime_history: Vec<(MarketRegime, f64)>,
}

impl RegimeClassifier {
    /// Create a new regime classifier.
    pub fn new(confidence_threshold: f64) -> Self {
        Self {
            confidence_threshold,
            min_samples: 3,
            regime_history: Vec::new(),
        }
    }

    /// Classify market regime based on features and probabilities.
    ///
    /// # Arguments
    /// * `class_probabilities` - Probability distribution over regime classes
    /// * `features` - Market features used for classification
    ///
    /// # Returns
    /// Tuple of (predicted regime, confidence)
    pub fn classify(&self, class_probabilities: &[f64]) -> (MarketRegime, f64) {
        if class_probabilities.len() != MarketRegime::count() {
            return (MarketRegime::Sideways, 0.0);
        }

        // Find the class with highest probability
        let (best_idx, &best_prob) = class_probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((2, &0.0)); // Default to Sideways

        let regime = MarketRegime::from_index(best_idx).unwrap_or(MarketRegime::Sideways);

        (regime, best_prob)
    }

    /// Classify with uncertainty quantification.
    pub fn classify_with_uncertainty(&self, class_probabilities: &[f64]) -> RegimeClassification {
        let (regime, confidence) = self.classify(class_probabilities);

        // Calculate entropy as uncertainty measure
        let entropy = self.calculate_entropy(class_probabilities);
        let max_entropy = (MarketRegime::count() as f64).ln();
        let normalized_uncertainty = entropy / max_entropy;

        // Find second best regime
        let mut sorted_probs: Vec<(usize, f64)> = class_probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        sorted_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let second_best = if sorted_probs.len() > 1 {
            MarketRegime::from_index(sorted_probs[1].0)
        } else {
            None
        };

        RegimeClassification {
            regime,
            confidence,
            uncertainty: normalized_uncertainty,
            second_best,
            is_confident: confidence >= self.confidence_threshold,
        }
    }

    /// Calculate entropy of probability distribution.
    fn calculate_entropy(&self, probs: &[f64]) -> f64 {
        probs
            .iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Update regime history with new observation.
    pub fn update_history(&mut self, regime: MarketRegime, confidence: f64) {
        self.regime_history.push((regime, confidence));

        // Keep only recent history
        if self.regime_history.len() > 100 {
            self.regime_history.remove(0);
        }
    }

    /// Get regime transition probability from history.
    pub fn transition_probability(&self, from: MarketRegime, to: MarketRegime) -> f64 {
        if self.regime_history.len() < 2 {
            return 1.0 / MarketRegime::count() as f64;
        }

        let mut transitions = 0;
        let mut from_count = 0;

        for i in 0..self.regime_history.len() - 1 {
            if self.regime_history[i].0 == from {
                from_count += 1;
                if self.regime_history[i + 1].0 == to {
                    transitions += 1;
                }
            }
        }

        if from_count == 0 {
            1.0 / MarketRegime::count() as f64
        } else {
            transitions as f64 / from_count as f64
        }
    }

    /// Check if regime transition is likely based on history.
    pub fn is_transition_likely(&self, from: MarketRegime, to: MarketRegime) -> bool {
        let prob = self.transition_probability(from, to);
        prob > 0.1 // At least 10% historical probability
    }

    /// Get the dominant regime from recent history.
    pub fn dominant_regime(&self, lookback: usize) -> Option<MarketRegime> {
        if self.regime_history.is_empty() {
            return None;
        }

        let start = self.regime_history.len().saturating_sub(lookback);
        let mut counts = [0usize; 8];

        for (regime, _) in &self.regime_history[start..] {
            counts[regime.to_index()] += 1;
        }

        counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .and_then(|(idx, _)| MarketRegime::from_index(idx))
    }
}

impl Default for RegimeClassifier {
    fn default() -> Self {
        Self::new(0.6)
    }
}

/// Result of regime classification with uncertainty.
#[derive(Debug, Clone)]
pub struct RegimeClassification {
    /// Predicted market regime
    pub regime: MarketRegime,
    /// Confidence in the prediction (0-1)
    pub confidence: f64,
    /// Uncertainty measure (normalized entropy, 0-1)
    pub uncertainty: f64,
    /// Second most likely regime
    pub second_best: Option<MarketRegime>,
    /// Whether prediction meets confidence threshold
    pub is_confident: bool,
}

impl RegimeClassification {
    /// Check if we should act on this classification.
    pub fn should_act(&self) -> bool {
        self.is_confident && self.uncertainty < 0.5
    }

    /// Get a description of the classification.
    pub fn description(&self) -> String {
        if self.is_confident {
            format!(
                "{} (confidence: {:.1}%, uncertainty: {:.1}%)",
                self.regime,
                self.confidence * 100.0,
                self.uncertainty * 100.0
            )
        } else {
            format!(
                "Uncertain - possibly {} or {} (confidence: {:.1}%)",
                self.regime,
                self.second_best.map(|r| r.to_string()).unwrap_or_else(|| "unknown".to_string()),
                self.confidence * 100.0
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_index_conversion() {
        for regime in MarketRegime::all() {
            let idx = regime.to_index();
            let recovered = MarketRegime::from_index(idx).unwrap();
            assert_eq!(regime, recovered);
        }
    }

    #[test]
    fn test_regime_properties() {
        assert!(MarketRegime::StrongUptrend.is_bullish());
        assert!(MarketRegime::StrongDowntrend.is_bearish());
        assert!(MarketRegime::Sideways.is_neutral());
        assert!(MarketRegime::Volatile.is_volatile());
    }

    #[test]
    fn test_classifier() {
        let classifier = RegimeClassifier::new(0.6);

        // Strong prediction for uptrend
        let probs = vec![0.8, 0.1, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005];
        let (regime, confidence) = classifier.classify(&probs);

        assert_eq!(regime, MarketRegime::StrongUptrend);
        assert!((confidence - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_classification_with_uncertainty() {
        let classifier = RegimeClassifier::new(0.6);

        // Certain prediction
        let probs = vec![0.9, 0.05, 0.02, 0.01, 0.01, 0.005, 0.0025, 0.0025];
        let result = classifier.classify_with_uncertainty(&probs);

        assert!(result.is_confident);
        assert!(result.uncertainty < 0.5);
    }
}
