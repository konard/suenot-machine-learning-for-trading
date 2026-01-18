//! Ensemble anomaly detection
//!
//! Combines multiple anomaly detection methods for robust detection

use super::{
    AnomalyDetector, AnomalyLevel, AnomalyResult, RollingMahalanobisDetector,
    SimpleIsolationDetector, ZScoreDetector,
};
use crate::data::OHLCVSeries;
use crate::features::RiskFeatures;

/// Weights for ensemble combination
#[derive(Debug, Clone)]
pub struct EnsembleWeights {
    pub zscore: f64,
    pub isolation: f64,
    pub mahalanobis: f64,
}

impl Default for EnsembleWeights {
    fn default() -> Self {
        Self {
            zscore: 0.4,
            isolation: 0.3,
            mahalanobis: 0.3,
        }
    }
}

impl EnsembleWeights {
    /// Normalize weights to sum to 1
    pub fn normalize(&mut self) {
        let sum = self.zscore + self.isolation + self.mahalanobis;
        if sum > 0.0 {
            self.zscore /= sum;
            self.isolation /= sum;
            self.mahalanobis /= sum;
        }
    }
}

/// Ensemble detector combining multiple methods
#[derive(Debug, Clone)]
pub struct EnsembleDetector {
    zscore: ZScoreDetector,
    isolation: SimpleIsolationDetector,
    mahalanobis: RollingMahalanobisDetector,
    weights: EnsembleWeights,
}

impl EnsembleDetector {
    /// Create new ensemble detector with custom parameters
    pub fn new(
        zscore_window: usize,
        zscore_threshold: f64,
        isolation_window: usize,
        mahalanobis_window: usize,
        weights: EnsembleWeights,
    ) -> Self {
        Self {
            zscore: ZScoreDetector::new(zscore_window, zscore_threshold),
            isolation: SimpleIsolationDetector::new(isolation_window, 10),
            mahalanobis: RollingMahalanobisDetector::new(mahalanobis_window, 3.0),
            weights,
        }
    }

    /// Set custom weights
    pub fn with_weights(mut self, weights: EnsembleWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Detect anomalies from raw price data
    pub fn detect_from_prices(&self, prices: &[f64]) -> Vec<f64> {
        let zscore_scores = self.zscore.detect(prices);
        let isolation_scores = self.isolation.detect(prices);
        let mahalanobis_scores = self.mahalanobis.detect(prices);

        self.combine_scores(&zscore_scores, &isolation_scores, &mahalanobis_scores)
    }

    /// Detect anomalies from OHLCV data
    pub fn detect_from_ohlcv(&self, data: &OHLCVSeries) -> Vec<AnomalyResult> {
        let returns = data.returns();
        if returns.is_empty() {
            return Vec::new();
        }

        let zscore_scores = self.zscore.detect(&returns);
        let isolation_scores = self.isolation.detect(&returns);
        let mahalanobis_scores = self.mahalanobis.detect(&returns);

        let combined = self.combine_scores(&zscore_scores, &isolation_scores, &mahalanobis_scores);

        combined
            .iter()
            .enumerate()
            .map(|(i, &score)| {
                AnomalyResult::new(
                    score,
                    zscore_scores.get(i).copied().unwrap_or(0.0),
                    isolation_scores.get(i).copied().unwrap_or(0.0),
                    mahalanobis_scores.get(i).copied().unwrap_or(0.0),
                )
            })
            .collect()
    }

    /// Detect anomalies using multiple features
    pub fn detect_from_features(&self, features: &RiskFeatures) -> AnomalyResult {
        // Use return for Z-score
        let return_score = if features.returns.len() > 20 {
            let scores = self.zscore.detect(&features.returns);
            *scores.last().unwrap_or(&0.0)
        } else {
            0.0
        };

        // Use volatility for isolation
        let vol_score = if features.volatility.len() > 20 {
            let scores = self.isolation.detect(&features.volatility);
            *scores.last().unwrap_or(&0.0)
        } else {
            0.0
        };

        // Use volume for Mahalanobis
        let volume_score = if features.volume_change.len() > 20 {
            let scores = self.mahalanobis.detect(&features.volume_change);
            *scores.last().unwrap_or(&0.0)
        } else {
            0.0
        };

        // Combine with weights
        let combined = self.weights.zscore * return_score
            + self.weights.isolation * vol_score
            + self.weights.mahalanobis * volume_score;

        AnomalyResult::new(combined, return_score, vol_score, volume_score)
    }

    /// Combine scores from multiple detectors
    fn combine_scores(
        &self,
        zscore: &[f64],
        isolation: &[f64],
        mahalanobis: &[f64],
    ) -> Vec<f64> {
        let len = zscore.len().min(isolation.len()).min(mahalanobis.len());

        (0..len)
            .map(|i| {
                self.weights.zscore * zscore[i]
                    + self.weights.isolation * isolation[i]
                    + self.weights.mahalanobis * mahalanobis[i]
            })
            .collect()
    }

    /// Get the current risk level based on latest score
    pub fn current_risk_level(&self, data: &OHLCVSeries) -> AnomalyLevel {
        let results = self.detect_from_ohlcv(data);
        results
            .last()
            .map(|r| r.level)
            .unwrap_or(AnomalyLevel::Normal)
    }

    /// Check if immediate action is needed
    pub fn requires_action(&self, data: &OHLCVSeries) -> bool {
        let level = self.current_risk_level(data);
        matches!(level, AnomalyLevel::High | AnomalyLevel::Extreme)
    }
}

impl Default for EnsembleDetector {
    fn default() -> Self {
        Self::new(20, 3.0, 50, 30, EnsembleWeights::default())
    }
}

/// Voting ensemble - uses majority voting instead of weighted average
#[derive(Debug, Clone)]
pub struct VotingEnsemble {
    detectors: Vec<Box<dyn AnomalyDetectorClone>>,
    threshold: f64,
}

/// Trait for cloneable detectors
pub trait AnomalyDetectorClone: AnomalyDetector + Send + Sync {
    fn clone_box(&self) -> Box<dyn AnomalyDetectorClone>;
}

impl<T: AnomalyDetector + Clone + Send + Sync + 'static> AnomalyDetectorClone for T {
    fn clone_box(&self) -> Box<dyn AnomalyDetectorClone> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn AnomalyDetectorClone> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl VotingEnsemble {
    /// Create new voting ensemble
    pub fn new(threshold: f64) -> Self {
        Self {
            detectors: Vec::new(),
            threshold,
        }
    }

    /// Add a detector to the ensemble
    pub fn add_detector<D: AnomalyDetectorClone + 'static>(&mut self, detector: D) {
        self.detectors.push(Box::new(detector));
    }

    /// Detect using majority voting
    pub fn detect_voting(&self, data: &[f64]) -> Vec<f64> {
        if self.detectors.is_empty() {
            return vec![0.0; data.len()];
        }

        let all_scores: Vec<Vec<f64>> = self.detectors.iter().map(|d| d.detect(data)).collect();

        let len = all_scores.iter().map(|s| s.len()).min().unwrap_or(0);

        (0..len)
            .map(|i| {
                let votes: usize = all_scores
                    .iter()
                    .filter(|scores| scores[i] > self.threshold)
                    .count();
                votes as f64 / self.detectors.len() as f64
            })
            .collect()
    }
}

impl Default for VotingEnsemble {
    fn default() -> Self {
        let mut ensemble = Self::new(0.7);
        ensemble.add_detector(ZScoreDetector::default());
        ensemble.add_detector(SimpleIsolationDetector::default());
        ensemble.add_detector(RollingMahalanobisDetector::default());
        ensemble
    }
}

/// Adaptive ensemble that adjusts weights based on recent performance
#[derive(Debug, Clone)]
pub struct AdaptiveEnsemble {
    base: EnsembleDetector,
    adaptation_rate: f64,
    recent_scores: Vec<(f64, f64, f64, f64)>, // (zscore, isolation, mahalanobis, actual)
    window_size: usize,
}

impl AdaptiveEnsemble {
    /// Create new adaptive ensemble
    pub fn new(adaptation_rate: f64, window_size: usize) -> Self {
        Self {
            base: EnsembleDetector::default(),
            adaptation_rate: adaptation_rate.clamp(0.0, 1.0),
            recent_scores: Vec::new(),
            window_size,
        }
    }

    /// Update with actual outcome (did anomaly occur?)
    pub fn update(&mut self, zscore: f64, isolation: f64, mahalanobis: f64, actual_anomaly: bool) {
        let actual = if actual_anomaly { 1.0 } else { 0.0 };
        self.recent_scores.push((zscore, isolation, mahalanobis, actual));

        if self.recent_scores.len() > self.window_size {
            self.recent_scores.remove(0);
        }

        // Adjust weights based on correlation with actual outcomes
        if self.recent_scores.len() >= self.window_size / 2 {
            self.adapt_weights();
        }
    }

    /// Adapt weights based on recent performance
    fn adapt_weights(&mut self) {
        if self.recent_scores.is_empty() {
            return;
        }

        // Calculate correlation of each detector with actual outcomes
        let n = self.recent_scores.len() as f64;

        let mean_actual: f64 = self.recent_scores.iter().map(|s| s.3).sum::<f64>() / n;
        let mean_z: f64 = self.recent_scores.iter().map(|s| s.0).sum::<f64>() / n;
        let mean_i: f64 = self.recent_scores.iter().map(|s| s.1).sum::<f64>() / n;
        let mean_m: f64 = self.recent_scores.iter().map(|s| s.2).sum::<f64>() / n;

        // Simple correlation calculation
        let corr_z = self.calculate_correlation(
            &self.recent_scores.iter().map(|s| s.0).collect::<Vec<_>>(),
            &self.recent_scores.iter().map(|s| s.3).collect::<Vec<_>>(),
            mean_z,
            mean_actual,
        );

        let corr_i = self.calculate_correlation(
            &self.recent_scores.iter().map(|s| s.1).collect::<Vec<_>>(),
            &self.recent_scores.iter().map(|s| s.3).collect::<Vec<_>>(),
            mean_i,
            mean_actual,
        );

        let corr_m = self.calculate_correlation(
            &self.recent_scores.iter().map(|s| s.2).collect::<Vec<_>>(),
            &self.recent_scores.iter().map(|s| s.3).collect::<Vec<_>>(),
            mean_m,
            mean_actual,
        );

        // Update weights using exponential moving average
        let new_z = corr_z.abs().max(0.1);
        let new_i = corr_i.abs().max(0.1);
        let new_m = corr_m.abs().max(0.1);

        self.base.weights.zscore = (1.0 - self.adaptation_rate) * self.base.weights.zscore
            + self.adaptation_rate * new_z;
        self.base.weights.isolation = (1.0 - self.adaptation_rate) * self.base.weights.isolation
            + self.adaptation_rate * new_i;
        self.base.weights.mahalanobis = (1.0 - self.adaptation_rate)
            * self.base.weights.mahalanobis
            + self.adaptation_rate * new_m;

        self.base.weights.normalize();
    }

    fn calculate_correlation(
        &self,
        x: &[f64],
        y: &[f64],
        mean_x: f64,
        mean_y: f64,
    ) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let cov: f64 = x
            .iter()
            .zip(y)
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>()
            / x.len() as f64;

        let var_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum::<f64>() / x.len() as f64;
        let var_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>() / y.len() as f64;

        let std_x = var_x.sqrt();
        let std_y = var_y.sqrt();

        if std_x < 1e-10 || std_y < 1e-10 {
            return 0.0;
        }

        cov / (std_x * std_y)
    }

    /// Detect using the adaptive ensemble
    pub fn detect(&self, data: &OHLCVSeries) -> Vec<AnomalyResult> {
        self.base.detect_from_ohlcv(data)
    }

    /// Get current weights
    pub fn current_weights(&self) -> &EnsembleWeights {
        &self.base.weights
    }
}

impl Default for AdaptiveEnsemble {
    fn default() -> Self {
        Self::new(0.1, 100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_detector() {
        let detector = EnsembleDetector::default();
        let mut data: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin()).collect();
        data.push(200.0); // Anomaly

        let scores = detector.detect_from_prices(&data);

        // Last score should be elevated
        assert!(scores.last().unwrap() > &0.5);
    }

    #[test]
    fn test_weights_normalization() {
        let mut weights = EnsembleWeights {
            zscore: 2.0,
            isolation: 3.0,
            mahalanobis: 5.0,
        };

        weights.normalize();

        let sum = weights.zscore + weights.isolation + weights.mahalanobis;
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
