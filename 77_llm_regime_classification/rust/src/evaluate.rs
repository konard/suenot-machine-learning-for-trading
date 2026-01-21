//! Evaluation metrics for regime classification.
//!
//! Provides tools to evaluate accuracy and usefulness
//! of regime classification models.

use crate::classifier::{MarketRegime, RegimeResult};
use std::collections::HashMap;

/// Classification metrics for regime evaluation.
#[derive(Debug, Clone)]
pub struct ClassificationMetrics {
    /// Overall accuracy
    pub accuracy: f64,
    /// Precision by regime
    pub precision: HashMap<MarketRegime, f64>,
    /// Recall by regime
    pub recall: HashMap<MarketRegime, f64>,
    /// F1 score by regime
    pub f1_score: HashMap<MarketRegime, f64>,
    /// Regime distribution
    pub regime_distribution: HashMap<MarketRegime, f64>,
}

/// Evaluator for regime classification.
pub struct RegimeEvaluator;

impl RegimeEvaluator {
    /// Create a new evaluator.
    pub fn new() -> Self {
        Self
    }

    /// Evaluate predictions against ground truth.
    pub fn evaluate_vs_ground_truth(
        &self,
        predictions: &[RegimeResult],
        ground_truth: &[MarketRegime],
    ) -> ClassificationMetrics {
        assert_eq!(predictions.len(), ground_truth.len());

        let pred_regimes: Vec<MarketRegime> = predictions.iter().map(|p| p.regime).collect();

        // Calculate accuracy
        let correct = pred_regimes
            .iter()
            .zip(ground_truth.iter())
            .filter(|(p, g)| p == g)
            .count();
        let accuracy = correct as f64 / predictions.len() as f64;

        // Calculate per-regime metrics
        let regimes = MarketRegime::all();
        let mut precision = HashMap::new();
        let mut recall = HashMap::new();
        let mut f1_score = HashMap::new();

        for regime in &regimes {
            let tp = pred_regimes
                .iter()
                .zip(ground_truth.iter())
                .filter(|(p, g)| *p == regime && *g == regime)
                .count();

            let fp = pred_regimes
                .iter()
                .zip(ground_truth.iter())
                .filter(|(p, g)| *p == regime && *g != regime)
                .count();

            let fn_ = pred_regimes
                .iter()
                .zip(ground_truth.iter())
                .filter(|(p, g)| *p != regime && *g == regime)
                .count();

            let prec = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };

            let rec = if tp + fn_ > 0 {
                tp as f64 / (tp + fn_) as f64
            } else {
                0.0
            };

            let f1 = if prec + rec > 0.0 {
                2.0 * prec * rec / (prec + rec)
            } else {
                0.0
            };

            precision.insert(*regime, prec);
            recall.insert(*regime, rec);
            f1_score.insert(*regime, f1);
        }

        // Regime distribution
        let mut regime_distribution = HashMap::new();
        let total = predictions.len() as f64;
        for regime in &regimes {
            let count = pred_regimes.iter().filter(|&r| r == regime).count();
            regime_distribution.insert(*regime, count as f64 / total);
        }

        ClassificationMetrics {
            accuracy,
            precision,
            recall,
            f1_score,
            regime_distribution,
        }
    }

    /// Evaluate regimes based on forward returns.
    pub fn evaluate_forward_returns(
        &self,
        predictions: &[RegimeResult],
        returns: &[f64],
        forward_window: usize,
    ) -> HashMap<MarketRegime, RegimeReturnStats> {
        let mut results: HashMap<MarketRegime, Vec<f64>> = HashMap::new();
        for regime in MarketRegime::all() {
            results.insert(regime, Vec::new());
        }

        for (i, pred) in predictions.iter().enumerate() {
            if i + forward_window < returns.len() {
                let fwd_return: f64 = returns[i..i + forward_window].iter().sum();
                results.get_mut(&pred.regime).unwrap().push(fwd_return);
            }
        }

        let mut stats = HashMap::new();
        for (regime, regime_returns) in results {
            if regime_returns.is_empty() {
                stats.insert(
                    regime,
                    RegimeReturnStats {
                        count: 0,
                        avg_forward_return: 0.0,
                        std_forward_return: 0.0,
                        positive_pct: 0.0,
                    },
                );
            } else {
                let count = regime_returns.len();
                let avg = regime_returns.iter().sum::<f64>() / count as f64;
                let variance = regime_returns.iter().map(|r| (r - avg).powi(2)).sum::<f64>()
                    / count as f64;
                let std = variance.sqrt();
                let positive = regime_returns.iter().filter(|&&r| r > 0.0).count();

                stats.insert(
                    regime,
                    RegimeReturnStats {
                        count,
                        avg_forward_return: avg,
                        std_forward_return: std,
                        positive_pct: positive as f64 / count as f64,
                    },
                );
            }
        }

        stats
    }

    /// Generate evaluation report.
    pub fn generate_report(
        &self,
        predictions: &[RegimeResult],
        returns: &[f64],
    ) -> String {
        let mut report = String::new();

        report.push_str(&"=".repeat(60));
        report.push_str("\nREGIME CLASSIFICATION EVALUATION REPORT\n");
        report.push_str(&"=".repeat(60));
        report.push('\n');

        // Regime distribution
        let mut counts: HashMap<MarketRegime, usize> = HashMap::new();
        for pred in predictions {
            *counts.entry(pred.regime).or_insert(0) += 1;
        }

        report.push_str("\nREGIME DISTRIBUTION:\n");
        report.push_str(&"-".repeat(40));
        report.push('\n');

        let total = predictions.len();
        for regime in MarketRegime::all() {
            let count = counts.get(&regime).unwrap_or(&0);
            let pct = *count as f64 / total as f64 * 100.0;
            report.push_str(&format!(
                "  {:15}: {:5} ({:5.1}%)\n",
                regime.as_str(),
                count,
                pct
            ));
        }

        // Confidence statistics
        let confidences: Vec<f64> = predictions.iter().map(|p| p.confidence).collect();
        let avg_conf = confidences.iter().sum::<f64>() / confidences.len() as f64;
        let min_conf = confidences.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_conf = confidences.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        report.push_str("\nCONFIDENCE STATISTICS:\n");
        report.push_str(&"-".repeat(40));
        report.push_str(&format!("\n  Mean:   {:.3}\n", avg_conf));
        report.push_str(&format!("  Min:    {:.3}\n", min_conf));
        report.push_str(&format!("  Max:    {:.3}\n", max_conf));

        // Forward return analysis
        if returns.len() > 5 {
            let stats = self.evaluate_forward_returns(predictions, returns, 5);

            report.push_str("\nFORWARD RETURN ANALYSIS (5-period):\n");
            report.push_str(&"-".repeat(40));
            report.push('\n');

            for regime in MarketRegime::all() {
                if let Some(s) = stats.get(&regime) {
                    if s.count > 0 {
                        report.push_str(&format!("  {}:\n", regime.as_str()));
                        report.push_str(&format!("    Count:      {}\n", s.count));
                        report.push_str(&format!("    Avg Return: {:.4}\n", s.avg_forward_return));
                        report.push_str(&format!("    Positive:   {:.1}%\n", s.positive_pct * 100.0));
                    }
                }
            }
        }

        report.push('\n');
        report.push_str(&"=".repeat(60));
        report.push('\n');

        report
    }
}

impl Default for RegimeEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for regime returns.
#[derive(Debug, Clone)]
pub struct RegimeReturnStats {
    pub count: usize,
    pub avg_forward_return: f64,
    pub std_forward_return: f64,
    pub positive_pct: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluation() {
        let evaluator = RegimeEvaluator::new();

        let predictions = vec![
            RegimeResult::new(MarketRegime::Bull, 0.8, 0.8, "Bull"),
            RegimeResult::new(MarketRegime::Bull, 0.7, 0.7, "Bull"),
            RegimeResult::new(MarketRegime::Bear, 0.6, 0.6, "Bear"),
        ];

        let ground_truth = vec![
            MarketRegime::Bull,
            MarketRegime::Bull,
            MarketRegime::Sideways,
        ];

        let metrics = evaluator.evaluate_vs_ground_truth(&predictions, &ground_truth);

        assert!(metrics.accuracy > 0.0);
        assert!(metrics.f1_score.contains_key(&MarketRegime::Bull));
    }

    #[test]
    fn test_forward_returns() {
        let evaluator = RegimeEvaluator::new();

        let predictions = vec![
            RegimeResult::new(MarketRegime::Bull, 0.8, 0.8, "Bull"),
            RegimeResult::new(MarketRegime::Bull, 0.8, 0.8, "Bull"),
            RegimeResult::new(MarketRegime::Bear, 0.8, 0.8, "Bear"),
        ];

        let returns = vec![0.01, 0.02, -0.01, 0.01, -0.02, 0.01, 0.01, 0.02];

        let stats = evaluator.evaluate_forward_returns(&predictions, &returns, 3);

        assert!(stats.contains_key(&MarketRegime::Bull));
    }
}
