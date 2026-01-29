//! Decision fusion for combining multi-task outputs into trading signals
//!
//! Fuses predictions from trend, volatility, regime, and risk heads
//! into a unified trading decision.

use super::{TaskOutput, FusedPrediction, TaskType};
use serde::{Deserialize, Serialize};

/// Method for fusing task outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionMethod {
    /// Simple weighted average
    WeightedAverage,
    /// Attention-based fusion
    Attention,
    /// Voting-based fusion
    MajorityVote,
}

/// Configuration for decision fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// Fusion method
    pub method: FusionMethod,
    /// Weights for each task (trend, volatility, regime, risk)
    pub task_weights: Vec<f64>,
    /// Confidence threshold for signal generation
    pub confidence_threshold: f64,
    /// Risk tolerance (0.0 = conservative, 1.0 = aggressive)
    pub risk_tolerance: f64,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            method: FusionMethod::WeightedAverage,
            task_weights: vec![0.35, 0.20, 0.25, 0.20],
            confidence_threshold: 0.6,
            risk_tolerance: 0.5,
        }
    }
}

/// Decision fusion module
pub struct DecisionFusion {
    /// Configuration
    pub config: FusionConfig,
}

impl DecisionFusion {
    /// Create a new decision fusion module
    pub fn new(config: FusionConfig) -> Self {
        Self { config }
    }

    /// Fuse multiple task outputs into a unified prediction
    pub fn fuse(&self, task_outputs: &[TaskOutput]) -> FusedPrediction {
        match self.config.method {
            FusionMethod::WeightedAverage => self.weighted_average_fusion(task_outputs),
            FusionMethod::Attention => self.attention_fusion(task_outputs),
            FusionMethod::MajorityVote => self.majority_vote_fusion(task_outputs),
        }
    }

    /// Weighted average fusion
    fn weighted_average_fusion(&self, task_outputs: &[TaskOutput]) -> FusedPrediction {
        let n_samples = task_outputs
            .first()
            .map(|o| o.predictions.nrows())
            .unwrap_or(0);

        let mut signals = vec![0.0f64; n_samples];
        let mut confidences = vec![0.0f64; n_samples];
        let mut regimes = vec![String::from("Unknown"); n_samples];
        let mut risk_levels = vec![0.5f64; n_samples];
        let mut task_contributions = Vec::new();

        for (i, output) in task_outputs.iter().enumerate() {
            let weight = self.config.task_weights.get(i).copied().unwrap_or(0.25);

            match output.task_type {
                TaskType::TrendPrediction => {
                    for (j, row) in output.predictions.rows().into_iter().enumerate() {
                        if j < n_samples {
                            // Convert 3-class probabilities to signal: up(+1) - down(-1)
                            let up_prob = row.get(0).copied().unwrap_or(0.33);
                            let down_prob = row.get(2).copied().unwrap_or(0.33);
                            signals[j] += weight * (up_prob - down_prob);
                            confidences[j] += weight * (up_prob.max(down_prob));
                        }
                    }
                    task_contributions.push(("Trend".to_string(), weight));
                }
                TaskType::VolatilityForecast => {
                    for (j, row) in output.predictions.rows().into_iter().enumerate() {
                        if j < n_samples {
                            let vol = row.get(0).copied().unwrap_or(0.5);
                            // High volatility reduces confidence
                            confidences[j] *= 1.0 - 0.3 * vol;
                            risk_levels[j] = vol;
                        }
                    }
                    task_contributions.push(("Volatility".to_string(), weight));
                }
                TaskType::RegimeDetection => {
                    let regime_labels = ["Trending", "Mean-Reverting", "Volatile", "Calm"];
                    for (j, row) in output.predictions.rows().into_iter().enumerate() {
                        if j < n_samples {
                            let best_regime = row
                                .iter()
                                .enumerate()
                                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                                .map(|(idx, _)| idx)
                                .unwrap_or(0);
                            regimes[j] = regime_labels
                                .get(best_regime)
                                .unwrap_or(&"Unknown")
                                .to_string();

                            // Regime affects signal strength
                            let regime_multiplier = match best_regime {
                                0 => 1.2,  // Trending: boost signal
                                1 => 0.8,  // Mean-reverting: reduce signal
                                2 => 0.6,  // Volatile: reduce signal
                                3 => 1.0,  // Calm: neutral
                                _ => 1.0,
                            };
                            signals[j] *= regime_multiplier;
                        }
                    }
                    task_contributions.push(("Regime".to_string(), weight));
                }
                TaskType::RiskAssessment => {
                    for (j, row) in output.predictions.rows().into_iter().enumerate() {
                        if j < n_samples {
                            // Risk classes: Low(0), Medium(1), High(2)
                            let high_risk = row.get(2).copied().unwrap_or(0.33);
                            let risk_adj = 1.0 - high_risk * (1.0 - self.config.risk_tolerance);
                            signals[j] *= risk_adj;
                            risk_levels[j] = risk_levels[j].max(high_risk);
                        }
                    }
                    task_contributions.push(("Risk".to_string(), weight));
                }
            }
        }

        // Clamp signals to [-1, 1]
        for s in signals.iter_mut() {
            *s = s.clamp(-1.0, 1.0);
        }

        // Apply confidence threshold
        for (i, conf) in confidences.iter_mut().enumerate() {
            *conf = conf.clamp(0.0, 1.0);
            if *conf < self.config.confidence_threshold {
                signals[i] = 0.0; // Neutral signal if low confidence
            }
        }

        FusedPrediction {
            signal: signals,
            confidence: confidences,
            regime: regimes,
            risk_level: risk_levels,
            task_contributions,
        }
    }

    /// Attention-based fusion: use learned attention weights
    fn attention_fusion(&self, task_outputs: &[TaskOutput]) -> FusedPrediction {
        // Compute attention scores based on prediction entropy
        let n_samples = task_outputs
            .first()
            .map(|o| o.predictions.nrows())
            .unwrap_or(0);

        let mut attention_weights: Vec<f64> = task_outputs
            .iter()
            .map(|output| {
                // Lower entropy = higher confidence = higher attention weight
                let avg_entropy = output
                    .predictions
                    .rows()
                    .into_iter()
                    .map(|row| {
                        -row.iter()
                            .map(|&p| {
                                let p_clamped = p.max(1e-10);
                                p_clamped * p_clamped.ln()
                            })
                            .sum::<f64>()
                    })
                    .sum::<f64>()
                    / n_samples.max(1) as f64;
                (-avg_entropy).exp()
            })
            .collect();

        // Normalize attention weights
        let sum: f64 = attention_weights.iter().sum();
        if sum > 0.0 {
            for w in attention_weights.iter_mut() {
                *w /= sum;
            }
        }

        // Apply attention-weighted fusion (reuse weighted average logic)
        let mut config = self.config.clone();
        config.task_weights = attention_weights;
        let fusion = DecisionFusion::new(config);
        fusion.weighted_average_fusion(task_outputs)
    }

    /// Majority vote fusion
    fn majority_vote_fusion(&self, task_outputs: &[TaskOutput]) -> FusedPrediction {
        let n_samples = task_outputs
            .first()
            .map(|o| o.predictions.nrows())
            .unwrap_or(0);

        let mut signals = vec![0.0f64; n_samples];
        let mut confidences = vec![0.0f64; n_samples];
        let mut regimes = vec![String::from("Unknown"); n_samples];
        let mut risk_levels = vec![0.5f64; n_samples];

        for j in 0..n_samples {
            let mut bullish_votes = 0;
            let mut bearish_votes = 0;
            let mut total_votes = 0;

            for output in task_outputs {
                if output.task_type == TaskType::TrendPrediction {
                    let row = output.predictions.row(j);
                    let up_prob = row.get(0).copied().unwrap_or(0.33);
                    let down_prob = row.get(2).copied().unwrap_or(0.33);
                    if up_prob > 0.4 {
                        bullish_votes += 1;
                    }
                    if down_prob > 0.4 {
                        bearish_votes += 1;
                    }
                    total_votes += 1;
                } else if output.task_type == TaskType::RegimeDetection {
                    let row = output.predictions.row(j);
                    let best = row
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    let labels = ["Trending", "Mean-Reverting", "Volatile", "Calm"];
                    regimes[j] = labels.get(best).unwrap_or(&"Unknown").to_string();
                } else if output.task_type == TaskType::RiskAssessment {
                    let row = output.predictions.row(j);
                    let high_risk = row.get(2).copied().unwrap_or(0.33);
                    risk_levels[j] = high_risk;
                }
            }

            if total_votes > 0 {
                signals[j] = (bullish_votes as f64 - bearish_votes as f64) / total_votes as f64;
                confidences[j] = (bullish_votes.max(bearish_votes) as f64) / total_votes as f64;
            }
        }

        FusedPrediction {
            signal: signals,
            confidence: confidences,
            regime: regimes,
            risk_level: risk_levels,
            task_contributions: task_outputs
                .iter()
                .map(|o| (o.task_name.clone(), 1.0 / task_outputs.len() as f64))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_dummy_outputs(n_samples: usize) -> Vec<TaskOutput> {
        vec![
            TaskOutput {
                task_type: TaskType::TrendPrediction,
                predictions: Array2::from_shape_fn((n_samples, 3), |(_, j)| {
                    if j == 0 { 0.6 } else if j == 1 { 0.3 } else { 0.1 }
                }),
                task_name: "Trend Prediction".to_string(),
            },
            TaskOutput {
                task_type: TaskType::VolatilityForecast,
                predictions: Array2::from_elem((n_samples, 1), 0.3),
                task_name: "Volatility Forecast".to_string(),
            },
            TaskOutput {
                task_type: TaskType::RegimeDetection,
                predictions: Array2::from_shape_fn((n_samples, 4), |(_, j)| {
                    if j == 0 { 0.5 } else { 0.5 / 3.0 }
                }),
                task_name: "Regime Detection".to_string(),
            },
            TaskOutput {
                task_type: TaskType::RiskAssessment,
                predictions: Array2::from_shape_fn((n_samples, 3), |(_, j)| {
                    if j == 0 { 0.6 } else if j == 1 { 0.3 } else { 0.1 }
                }),
                task_name: "Risk Assessment".to_string(),
            },
        ]
    }

    #[test]
    fn test_weighted_fusion() {
        let fusion = DecisionFusion::new(FusionConfig::default());
        let outputs = make_dummy_outputs(3);
        let result = fusion.fuse(&outputs);

        assert_eq!(result.signal.len(), 3);
        assert_eq!(result.confidence.len(), 3);
        assert_eq!(result.regime.len(), 3);
        assert_eq!(result.risk_level.len(), 3);
    }
}
