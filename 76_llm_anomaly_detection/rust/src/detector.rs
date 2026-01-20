//! Anomaly detection algorithms.
//!
//! Provides multiple approaches to detecting anomalies in financial data:
//! - Statistical methods (Z-score, Mahalanobis distance)
//! - Isolation-based methods
//! - Ensemble methods

use crate::types::{AnomalyResult, AnomalyType, Features};
use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Trait for anomaly detectors.
pub trait AnomalyDetector {
    /// Fit the detector on training data.
    fn fit(&mut self, features: &[Features]) -> Result<()>;

    /// Detect anomaly in a single observation.
    fn detect(&self, features: &Features) -> Result<AnomalyResult>;

    /// Detect anomalies in multiple observations.
    fn detect_batch(&self, features: &[Features]) -> Result<Vec<AnomalyResult>> {
        features.iter().map(|f| self.detect(f)).collect()
    }
}

/// Statistical anomaly detector using Z-scores and other methods.
pub struct StatisticalDetector {
    z_threshold: f64,
    contamination: f64,

    // Fitted parameters
    means: Option<Vec<f64>>,
    stds: Option<Vec<f64>>,
    is_fitted: bool,
}

impl StatisticalDetector {
    /// Create a new statistical detector.
    pub fn new(z_threshold: f64) -> Self {
        Self {
            z_threshold,
            contamination: 0.05,
            means: None,
            stds: None,
            is_fitted: false,
        }
    }

    /// Set contamination rate (expected proportion of anomalies).
    pub fn with_contamination(mut self, contamination: f64) -> Self {
        self.contamination = contamination;
        self
    }

    /// Calculate Z-score for a value.
    fn zscore(value: f64, mean: f64, std: f64) -> f64 {
        if std > 0.0 {
            (value - mean).abs() / std
        } else {
            0.0
        }
    }

    /// Determine anomaly type based on which feature is most anomalous.
    fn determine_type(&self, features: &Features, z_scores: &[f64]) -> AnomalyType {
        let max_idx = z_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        match max_idx {
            0 => AnomalyType::PriceSpike,  // returns
            1 => AnomalyType::VolumeAnomaly,  // volume_ratio
            2 => AnomalyType::PatternBreak,  // range_ratio
            _ => AnomalyType::Unknown,
        }
    }
}

impl AnomalyDetector for StatisticalDetector {
    fn fit(&mut self, features: &[Features]) -> Result<()> {
        if features.is_empty() {
            return Err(anyhow!("No data to fit"));
        }

        let n = features.len() as f64;

        // Extract feature vectors
        let feature_vecs: Vec<Vec<f64>> = features.iter().map(|f| f.to_vec()).collect();

        let num_features = feature_vecs[0].len();

        // Calculate means
        let mut means = vec![0.0; num_features];
        for fv in &feature_vecs {
            for (i, v) in fv.iter().enumerate() {
                if v.is_finite() {
                    means[i] += v;
                }
            }
        }
        for m in &mut means {
            *m /= n;
        }

        // Calculate standard deviations
        let mut stds = vec![0.0; num_features];
        for fv in &feature_vecs {
            for (i, v) in fv.iter().enumerate() {
                if v.is_finite() {
                    stds[i] += (v - means[i]).powi(2);
                }
            }
        }
        for s in &mut stds {
            *s = (*s / n).sqrt();
            if *s == 0.0 {
                *s = 1.0;  // Prevent division by zero
            }
        }

        self.means = Some(means);
        self.stds = Some(stds);
        self.is_fitted = true;

        Ok(())
    }

    fn detect(&self, features: &Features) -> Result<AnomalyResult> {
        if !self.is_fitted {
            return Err(anyhow!("Detector not fitted"));
        }

        let means = self.means.as_ref().unwrap();
        let stds = self.stds.as_ref().unwrap();

        let fv = features.to_vec();

        // Calculate Z-scores for each feature
        let z_scores: Vec<f64> = fv
            .iter()
            .enumerate()
            .map(|(i, v)| {
                if v.is_finite() {
                    Self::zscore(*v, means[i], stds[i])
                } else {
                    0.0
                }
            })
            .collect();

        // Find maximum Z-score
        let max_zscore = z_scores
            .iter()
            .filter(|z| z.is_finite())
            .cloned()
            .fold(0.0_f64, f64::max);

        // Determine if anomaly
        let is_anomaly = max_zscore > self.z_threshold;

        if is_anomaly {
            let anomaly_type = self.determine_type(features, &z_scores);
            let score = (max_zscore / self.z_threshold).min(2.0) / 2.0;
            let confidence = score.min(1.0);

            let explanation = format!(
                "Z-score {:.2} exceeds threshold {:.1}",
                max_zscore, self.z_threshold
            );

            let mut result = AnomalyResult::anomaly(score, anomaly_type, confidence, explanation);

            // Add details
            result = result.with_detail("max_zscore", max_zscore);
            result = result.with_detail("returns_zscore", z_scores[0]);
            result = result.with_detail("volume_zscore", z_scores.get(1).copied().unwrap_or(0.0));

            Ok(result)
        } else {
            Ok(AnomalyResult::normal())
        }
    }
}

/// Isolation-based anomaly detector.
///
/// Uses a simplified isolation forest concept where anomalies
/// are expected to have extreme feature values.
pub struct IsolationDetector {
    threshold_percentile: f64,
    thresholds: Option<Vec<(f64, f64)>>, // (lower, upper) for each feature
    is_fitted: bool,
}

impl IsolationDetector {
    /// Create a new isolation detector.
    pub fn new(threshold_percentile: f64) -> Self {
        Self {
            threshold_percentile,
            thresholds: None,
            is_fitted: false,
        }
    }

    /// Calculate percentile.
    fn percentile(data: &mut [f64], p: f64) -> f64 {
        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((data.len() as f64 - 1.0) * p / 100.0).round() as usize;
        data[idx.min(data.len() - 1)]
    }
}

impl AnomalyDetector for IsolationDetector {
    fn fit(&mut self, features: &[Features]) -> Result<()> {
        if features.is_empty() {
            return Err(anyhow!("No data to fit"));
        }

        let feature_vecs: Vec<Vec<f64>> = features.iter().map(|f| f.to_vec()).collect();
        let num_features = feature_vecs[0].len();

        let mut thresholds = Vec::with_capacity(num_features);

        for i in 0..num_features {
            let mut values: Vec<f64> = feature_vecs
                .iter()
                .map(|fv| fv[i])
                .filter(|v| v.is_finite())
                .collect();

            if values.is_empty() {
                thresholds.push((f64::NEG_INFINITY, f64::INFINITY));
                continue;
            }

            let lower = Self::percentile(&mut values.clone(), 100.0 - self.threshold_percentile);
            let upper = Self::percentile(&mut values, self.threshold_percentile);

            thresholds.push((lower, upper));
        }

        self.thresholds = Some(thresholds);
        self.is_fitted = true;

        Ok(())
    }

    fn detect(&self, features: &Features) -> Result<AnomalyResult> {
        if !self.is_fitted {
            return Err(anyhow!("Detector not fitted"));
        }

        let thresholds = self.thresholds.as_ref().unwrap();
        let fv = features.to_vec();

        // Check how many features are outside thresholds
        let mut anomaly_count = 0;
        let mut max_deviation = 0.0;
        let mut max_idx = 0;

        for (i, (v, (lower, upper))) in fv.iter().zip(thresholds.iter()).enumerate() {
            if !v.is_finite() {
                continue;
            }

            let deviation = if *v < *lower {
                (lower - v) / lower.abs().max(1.0)
            } else if *v > *upper {
                (v - upper) / upper.abs().max(1.0)
            } else {
                0.0
            };

            if deviation > 0.0 {
                anomaly_count += 1;
                if deviation > max_deviation {
                    max_deviation = deviation;
                    max_idx = i;
                }
            }
        }

        let is_anomaly = anomaly_count > 0;

        if is_anomaly {
            let anomaly_type = match max_idx {
                0 => AnomalyType::PriceSpike,
                1 => AnomalyType::VolumeAnomaly,
                _ => AnomalyType::PatternBreak,
            };

            let score = (anomaly_count as f64 / fv.len() as f64).min(1.0);
            let confidence = max_deviation.min(1.0);

            Ok(AnomalyResult::anomaly(
                score,
                anomaly_type,
                confidence,
                format!("{} features outside normal range", anomaly_count),
            ))
        } else {
            Ok(AnomalyResult::normal())
        }
    }
}

/// Ensemble detector combining multiple detection methods.
pub struct EnsembleDetector {
    detectors: Vec<Box<dyn AnomalyDetector + Send + Sync>>,
    voting: VotingMethod,
    threshold: f64,
}

/// Voting method for ensemble.
#[derive(Debug, Clone, Copy)]
pub enum VotingMethod {
    /// Average scores (soft voting)
    Soft,
    /// Majority vote (hard voting)
    Hard,
}

impl EnsembleDetector {
    /// Create a new ensemble detector.
    pub fn new(voting: VotingMethod, threshold: f64) -> Self {
        Self {
            detectors: Vec::new(),
            voting,
            threshold,
        }
    }

    /// Add a detector to the ensemble.
    pub fn add_detector<D: AnomalyDetector + Send + Sync + 'static>(mut self, detector: D) -> Self {
        self.detectors.push(Box::new(detector));
        self
    }
}

impl AnomalyDetector for EnsembleDetector {
    fn fit(&mut self, features: &[Features]) -> Result<()> {
        for detector in &mut self.detectors {
            detector.fit(features)?;
        }
        Ok(())
    }

    fn detect(&self, features: &Features) -> Result<AnomalyResult> {
        if self.detectors.is_empty() {
            return Err(anyhow!("No detectors in ensemble"));
        }

        let results: Vec<AnomalyResult> = self
            .detectors
            .iter()
            .filter_map(|d| d.detect(features).ok())
            .collect();

        if results.is_empty() {
            return Ok(AnomalyResult::normal());
        }

        let (is_anomaly, score) = match self.voting {
            VotingMethod::Soft => {
                let avg_score: f64 = results.iter().map(|r| r.score).sum::<f64>() / results.len() as f64;
                (avg_score > self.threshold, avg_score)
            }
            VotingMethod::Hard => {
                let votes = results.iter().filter(|r| r.is_anomaly).count();
                let is_anomaly = votes as f64 > results.len() as f64 / 2.0;
                let score = votes as f64 / results.len() as f64;
                (is_anomaly, score)
            }
        };

        if is_anomaly {
            // Get most common anomaly type
            let mut type_counts: HashMap<AnomalyType, usize> = HashMap::new();
            for r in &results {
                if r.is_anomaly {
                    *type_counts.entry(r.anomaly_type).or_insert(0) += 1;
                }
            }

            let anomaly_type = type_counts
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(t, _)| t)
                .unwrap_or(AnomalyType::Unknown);

            let confidence = results.iter().map(|r| r.confidence).sum::<f64>() / results.len() as f64;

            let explanations: Vec<&str> = results
                .iter()
                .filter(|r| r.is_anomaly)
                .map(|r| r.explanation.as_str())
                .collect();

            Ok(AnomalyResult::anomaly(
                score,
                anomaly_type,
                confidence,
                explanations.join(" | "),
            ))
        } else {
            Ok(AnomalyResult::normal())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_features() -> Vec<Features> {
        (0..100)
            .map(|i| Features {
                returns: 0.001 * (i as f64 % 10.0 - 5.0),
                log_returns: 0.001 * (i as f64 % 10.0 - 5.0),
                volatility: 0.02,
                volume_ratio: 1.0 + 0.1 * (i as f64 % 5.0 - 2.5),
                range_ratio: 1.0 + 0.05 * (i as f64 % 4.0 - 2.0),
                returns_zscore: (i as f64 % 10.0 - 5.0) / 2.0,
                volume_zscore: (i as f64 % 5.0 - 2.5) / 2.0,
            })
            .collect()
    }

    #[test]
    fn test_statistical_detector() {
        let features = create_test_features();

        let mut detector = StatisticalDetector::new(2.5);
        detector.fit(&features).unwrap();

        // Test normal observation
        let normal = Features {
            returns: 0.001,
            log_returns: 0.001,
            volatility: 0.02,
            volume_ratio: 1.0,
            range_ratio: 1.0,
            returns_zscore: 0.5,
            volume_zscore: 0.5,
        };

        let result = detector.detect(&normal).unwrap();
        assert!(!result.is_anomaly);

        // Test anomalous observation
        let anomaly = Features {
            returns: 0.10,  // 10% return - very unusual
            log_returns: 0.095,
            volatility: 0.02,
            volume_ratio: 5.0,  // 5x normal volume
            range_ratio: 3.0,
            returns_zscore: 8.0,
            volume_zscore: 6.0,
        };

        let result = detector.detect(&anomaly).unwrap();
        assert!(result.is_anomaly);
    }

    #[test]
    fn test_isolation_detector() {
        let features = create_test_features();

        let mut detector = IsolationDetector::new(95.0);
        detector.fit(&features).unwrap();

        // Test with extreme values
        let extreme = Features {
            returns: 0.5,  // Way outside normal
            log_returns: 0.4,
            volatility: 0.1,
            volume_ratio: 10.0,
            range_ratio: 5.0,
            returns_zscore: 10.0,
            volume_zscore: 8.0,
        };

        let result = detector.detect(&extreme).unwrap();
        assert!(result.is_anomaly);
    }

    #[test]
    fn test_ensemble_detector() {
        let features = create_test_features();

        let mut ensemble = EnsembleDetector::new(VotingMethod::Soft, 0.5)
            .add_detector(StatisticalDetector::new(2.5))
            .add_detector(IsolationDetector::new(95.0));

        ensemble.fit(&features).unwrap();

        let normal = Features::default();
        let result = ensemble.detect(&normal).unwrap();
        assert!(!result.is_anomaly);
    }
}
