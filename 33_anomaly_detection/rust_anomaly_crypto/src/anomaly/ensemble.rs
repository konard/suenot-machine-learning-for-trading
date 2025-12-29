//! Ensemble anomaly detection
//!
//! Combines multiple anomaly detectors for more robust detection.

use super::{AnomalyDetector, AnomalyResult, GlobalMADDetector, GlobalZScoreDetector, IQRDetector};

/// Method for combining detector scores
#[derive(Clone, Copy, Debug)]
pub enum CombineMethod {
    /// Average of all detector scores
    Mean,
    /// Maximum score across detectors
    Max,
    /// Weighted average
    Weighted,
    /// Majority voting
    Vote,
}

/// Configuration for a detector in the ensemble
#[derive(Clone)]
pub struct DetectorConfig {
    pub name: String,
    pub weight: f64,
    pub enabled: bool,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            weight: 1.0,
            enabled: true,
        }
    }
}

/// Ensemble anomaly detector
pub struct EnsembleDetector {
    /// How to combine detector scores
    pub combine_method: CombineMethod,
    /// Threshold for final anomaly decision
    pub threshold: f64,
    /// Z-Score detector
    zscore: GlobalZScoreDetector,
    zscore_config: DetectorConfig,
    /// MAD detector
    mad: GlobalMADDetector,
    mad_config: DetectorConfig,
    /// IQR detector
    iqr: IQRDetector,
    iqr_config: DetectorConfig,
}

impl EnsembleDetector {
    /// Create a new ensemble detector with default settings
    pub fn new() -> Self {
        Self {
            combine_method: CombineMethod::Mean,
            threshold: 0.5,
            zscore: GlobalZScoreDetector::new(3.0),
            zscore_config: DetectorConfig {
                name: "ZScore".to_string(),
                weight: 0.35,
                enabled: true,
            },
            mad: GlobalMADDetector::new(3.5),
            mad_config: DetectorConfig {
                name: "MAD".to_string(),
                weight: 0.35,
                enabled: true,
            },
            iqr: IQRDetector::new(1.5, 0),
            iqr_config: DetectorConfig {
                name: "IQR".to_string(),
                weight: 0.30,
                enabled: true,
            },
        }
    }

    /// Set the combination method
    pub fn with_method(mut self, method: CombineMethod) -> Self {
        self.combine_method = method;
        self
    }

    /// Set the threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set Z-Score detector weight
    pub fn set_zscore_weight(&mut self, weight: f64) {
        self.zscore_config.weight = weight;
    }

    /// Set MAD detector weight
    pub fn set_mad_weight(&mut self, weight: f64) {
        self.mad_config.weight = weight;
    }

    /// Set IQR detector weight
    pub fn set_iqr_weight(&mut self, weight: f64) {
        self.iqr_config.weight = weight;
    }

    /// Enable or disable Z-Score detector
    pub fn enable_zscore(&mut self, enabled: bool) {
        self.zscore_config.enabled = enabled;
    }

    /// Enable or disable MAD detector
    pub fn enable_mad(&mut self, enabled: bool) {
        self.mad_config.enabled = enabled;
    }

    /// Enable or disable IQR detector
    pub fn enable_iqr(&mut self, enabled: bool) {
        self.iqr_config.enabled = enabled;
    }

    /// Fit the ensemble to data
    pub fn fit(&mut self, data: &[f64]) {
        if self.zscore_config.enabled {
            self.zscore.fit(data);
        }
        if self.mad_config.enabled {
            self.mad.fit(data);
        }
        // IQR doesn't need fitting
    }

    /// Get results from all enabled detectors
    fn get_detector_results(&self, data: &[f64]) -> Vec<(DetectorConfig, AnomalyResult)> {
        let mut results = Vec::new();

        if self.zscore_config.enabled {
            results.push((self.zscore_config.clone(), self.zscore.detect(data)));
        }
        if self.mad_config.enabled {
            results.push((self.mad_config.clone(), self.mad.detect(data)));
        }
        if self.iqr_config.enabled {
            results.push((self.iqr_config.clone(), self.iqr.detect(data)));
        }

        results
    }

    /// Combine scores using the configured method
    fn combine_scores(&self, detector_results: &[(DetectorConfig, AnomalyResult)]) -> Vec<f64> {
        if detector_results.is_empty() {
            return vec![];
        }

        let n = detector_results[0].1.scores.len();
        let mut combined = vec![0.0; n];

        match self.combine_method {
            CombineMethod::Mean => {
                let total_weight: f64 = detector_results
                    .iter()
                    .map(|(config, _)| config.weight)
                    .sum();

                for (config, result) in detector_results {
                    for (i, &score) in result.normalized_scores.iter().enumerate() {
                        combined[i] += score * config.weight / total_weight;
                    }
                }
            }
            CombineMethod::Max => {
                for i in 0..n {
                    combined[i] = detector_results
                        .iter()
                        .map(|(_, result)| result.normalized_scores[i])
                        .fold(f64::NEG_INFINITY, f64::max);
                }
            }
            CombineMethod::Weighted => {
                let total_weight: f64 = detector_results
                    .iter()
                    .map(|(config, _)| config.weight)
                    .sum();

                for (config, result) in detector_results {
                    for (i, &score) in result.normalized_scores.iter().enumerate() {
                        combined[i] += score * config.weight / total_weight;
                    }
                }
            }
            CombineMethod::Vote => {
                let n_detectors = detector_results.len() as f64;
                for i in 0..n {
                    let votes: usize = detector_results
                        .iter()
                        .filter(|(_, result)| result.is_anomaly[i])
                        .count();
                    combined[i] = votes as f64 / n_detectors;
                }
            }
        }

        combined
    }
}

impl Default for EnsembleDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl AnomalyDetector for EnsembleDetector {
    fn detect(&self, data: &[f64]) -> AnomalyResult {
        let detector_results = self.get_detector_results(data);

        if detector_results.is_empty() {
            return AnomalyResult::new(vec![], vec![], vec![]);
        }

        let combined_scores = self.combine_scores(&detector_results);

        let is_anomaly: Vec<bool> = combined_scores
            .iter()
            .map(|&s| s > self.threshold)
            .collect();

        let normalized_scores: Vec<f64> = combined_scores
            .iter()
            .map(|&s| s / self.threshold)
            .collect();

        AnomalyResult::new(is_anomaly, combined_scores, normalized_scores)
    }

    fn name(&self) -> &str {
        "Ensemble"
    }
}

/// Simple voting ensemble
pub struct VotingEnsemble {
    /// Minimum proportion of detectors that must agree
    pub min_agreement: f64,
}

impl VotingEnsemble {
    /// Create a new voting ensemble
    pub fn new(min_agreement: f64) -> Self {
        Self { min_agreement }
    }

    /// Detect anomalies using multiple detector results
    pub fn combine(&self, results: &[AnomalyResult]) -> AnomalyResult {
        if results.is_empty() {
            return AnomalyResult::new(vec![], vec![], vec![]);
        }

        let n = results[0].is_anomaly.len();
        let n_detectors = results.len() as f64;

        let votes: Vec<f64> = (0..n)
            .map(|i| {
                results
                    .iter()
                    .filter(|r| i < r.is_anomaly.len() && r.is_anomaly[i])
                    .count() as f64
                    / n_detectors
            })
            .collect();

        let is_anomaly: Vec<bool> = votes.iter().map(|&v| v >= self.min_agreement).collect();

        let scores: Vec<f64> = (0..n)
            .map(|i| {
                results
                    .iter()
                    .filter_map(|r| r.scores.get(i))
                    .sum::<f64>()
                    / n_detectors
            })
            .collect();

        AnomalyResult::new(is_anomaly, scores, votes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_detector() {
        let mut data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        data[50] = 100.0; // Outlier

        let mut ensemble = EnsembleDetector::new();
        ensemble.fit(&data);
        let result = ensemble.detect(&data);

        assert!(result.is_anomaly[50]);
    }

    #[test]
    fn test_voting_ensemble() {
        let result1 = AnomalyResult::new(
            vec![false, true, true, false],
            vec![0.1, 0.9, 0.8, 0.2],
            vec![0.1, 0.9, 0.8, 0.2],
        );

        let result2 = AnomalyResult::new(
            vec![false, true, false, false],
            vec![0.2, 0.8, 0.3, 0.1],
            vec![0.2, 0.8, 0.3, 0.1],
        );

        let result3 = AnomalyResult::new(
            vec![false, true, true, true],
            vec![0.1, 0.9, 0.7, 0.6],
            vec![0.1, 0.9, 0.7, 0.6],
        );

        let voting = VotingEnsemble::new(0.5); // Need at least 50% agreement
        let combined = voting.combine(&[result1, result2, result3]);

        // Index 1: 3/3 agree = 100%
        assert!(combined.is_anomaly[1]);

        // Index 0: 0/3 agree = 0%
        assert!(!combined.is_anomaly[0]);

        // Index 2: 2/3 agree = 66%
        assert!(combined.is_anomaly[2]);
    }

    #[test]
    fn test_combine_methods() {
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();

        let mut ensemble_mean = EnsembleDetector::new().with_method(CombineMethod::Mean);
        let mut ensemble_max = EnsembleDetector::new().with_method(CombineMethod::Max);
        let mut ensemble_vote = EnsembleDetector::new().with_method(CombineMethod::Vote);

        ensemble_mean.fit(&data);
        ensemble_max.fit(&data);
        ensemble_vote.fit(&data);

        let result_mean = ensemble_mean.detect(&data);
        let result_max = ensemble_max.detect(&data);
        let result_vote = ensemble_vote.detect(&data);

        // All should produce results of the same length
        assert_eq!(result_mean.scores.len(), data.len());
        assert_eq!(result_max.scores.len(), data.len());
        assert_eq!(result_vote.scores.len(), data.len());
    }
}
