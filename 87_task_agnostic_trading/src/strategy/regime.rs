//! Market regime classification using task-agnostic representations

use serde::{Deserialize, Serialize};

/// Market regimes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Strong directional movement
    Trending,
    /// Price oscillating around a mean
    MeanReverting,
    /// High uncertainty with large price swings
    Volatile,
    /// Low volatility sideways movement
    Calm,
}

impl MarketRegime {
    /// Human-readable label
    pub fn label(&self) -> &str {
        match self {
            MarketRegime::Trending => "Trending",
            MarketRegime::MeanReverting => "Mean-Reverting",
            MarketRegime::Volatile => "Volatile",
            MarketRegime::Calm => "Calm",
        }
    }

    /// Suggested position sizing multiplier
    pub fn position_multiplier(&self) -> f64 {
        match self {
            MarketRegime::Trending => 1.2,
            MarketRegime::MeanReverting => 0.8,
            MarketRegime::Volatile => 0.5,
            MarketRegime::Calm => 1.0,
        }
    }

    /// From index (used with model output)
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => MarketRegime::Trending,
            1 => MarketRegime::MeanReverting,
            2 => MarketRegime::Volatile,
            _ => MarketRegime::Calm,
        }
    }
}

/// Configuration for regime classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeConfig {
    /// Volatility threshold for regime boundaries
    pub volatility_threshold: f64,
    /// Trend strength threshold
    pub trend_threshold: f64,
    /// Mean reversion half-life threshold
    pub mean_reversion_threshold: f64,
    /// Smoothing window for regime transitions
    pub smoothing_window: usize,
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            volatility_threshold: 0.02,
            trend_threshold: 0.5,
            mean_reversion_threshold: 10.0,
            smoothing_window: 5,
        }
    }
}

/// Result of regime classification
#[derive(Debug, Clone)]
pub struct RegimeClassification {
    /// Detected regime
    pub regime: MarketRegime,
    /// Confidence in the classification
    pub confidence: f64,
    /// Probability distribution over regimes
    pub probabilities: Vec<(MarketRegime, f64)>,
    /// Regime stability (how long current regime has lasted)
    pub stability: f64,
}

/// Regime classifier using multi-task model features
pub struct RegimeClassifier {
    config: RegimeConfig,
    /// History of classified regimes for smoothing
    history: Vec<MarketRegime>,
}

impl RegimeClassifier {
    /// Create a new regime classifier
    pub fn new(config: RegimeConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
        }
    }

    /// Classify regime from regime detection head probabilities
    pub fn classify(&mut self, regime_probs: &[f64]) -> RegimeClassification {
        // Find the most probable regime
        let best_idx = regime_probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(3);

        let regime = MarketRegime::from_index(best_idx);
        let confidence = regime_probs.get(best_idx).copied().unwrap_or(0.25);

        // Build probability distribution
        let probabilities = vec![
            (MarketRegime::Trending, regime_probs.get(0).copied().unwrap_or(0.25)),
            (MarketRegime::MeanReverting, regime_probs.get(1).copied().unwrap_or(0.25)),
            (MarketRegime::Volatile, regime_probs.get(2).copied().unwrap_or(0.25)),
            (MarketRegime::Calm, regime_probs.get(3).copied().unwrap_or(0.25)),
        ];

        // Track history for smoothing
        self.history.push(regime.clone());
        if self.history.len() > self.config.smoothing_window * 2 {
            self.history.drain(..self.config.smoothing_window);
        }

        // Compute stability
        let stability = if self.history.len() >= 2 {
            let current = &self.history[self.history.len() - 1];
            let consecutive = self
                .history
                .iter()
                .rev()
                .take_while(|r| *r == current)
                .count();
            consecutive as f64 / self.config.smoothing_window as f64
        } else {
            0.0
        };

        RegimeClassification {
            regime,
            confidence,
            probabilities,
            stability,
        }
    }

    /// Classify from raw features (volatility, trend strength, etc.)
    pub fn classify_from_features(
        &mut self,
        volatility: f64,
        trend_strength: f64,
        mean_reversion_speed: f64,
    ) -> RegimeClassification {
        let mut probs = vec![0.0; 4];

        // Heuristic regime classification
        if trend_strength.abs() > self.config.trend_threshold {
            probs[0] = 0.6 + 0.3 * trend_strength.abs().min(1.0);
        }
        if mean_reversion_speed > 0.0 && mean_reversion_speed < self.config.mean_reversion_threshold {
            probs[1] = 0.5 + 0.3 * (1.0 - mean_reversion_speed / self.config.mean_reversion_threshold);
        }
        if volatility > self.config.volatility_threshold {
            probs[2] = 0.5 + 0.4 * (volatility / self.config.volatility_threshold).min(2.0) / 2.0;
        }
        if volatility <= self.config.volatility_threshold && trend_strength.abs() < self.config.trend_threshold {
            probs[3] = 0.6;
        }

        // Normalize
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        } else {
            probs = vec![0.25; 4];
        }

        self.classify(&probs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_classification() {
        let config = RegimeConfig::default();
        let mut classifier = RegimeClassifier::new(config);

        let probs = [0.6, 0.2, 0.1, 0.1];
        let result = classifier.classify(&probs);
        assert_eq!(result.regime, MarketRegime::Trending);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_regime_stability() {
        let config = RegimeConfig::default();
        let mut classifier = RegimeClassifier::new(config);

        // Classify several times with same regime
        for _ in 0..5 {
            classifier.classify(&[0.7, 0.1, 0.1, 0.1]);
        }

        let result = classifier.classify(&[0.7, 0.1, 0.1, 0.1]);
        assert!(result.stability > 0.0);
    }
}
