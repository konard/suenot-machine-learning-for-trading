//! # Classification Metrics
//!
//! Metrics for evaluating binary classification models.

use serde::{Deserialize, Serialize};

/// Classification metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassificationMetrics {
    /// True positives
    pub tp: usize,
    /// True negatives
    pub tn: usize,
    /// False positives
    pub fp: usize,
    /// False negatives
    pub fn_: usize,
}

impl ClassificationMetrics {
    /// Create from predictions and labels
    pub fn from_predictions(predictions: &[i32], labels: &[i32]) -> Self {
        let mut metrics = Self::default();

        for (&pred, &label) in predictions.iter().zip(labels.iter()) {
            match (pred, label) {
                (1, 1) => metrics.tp += 1,
                (0, 0) => metrics.tn += 1,
                (1, 0) => metrics.fp += 1,
                (0, 1) => metrics.fn_ += 1,
                _ => {}
            }
        }

        metrics
    }

    /// Calculate accuracy
    pub fn accuracy(&self) -> f64 {
        let total = self.tp + self.tn + self.fp + self.fn_;
        if total > 0 {
            (self.tp + self.tn) as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Calculate precision
    pub fn precision(&self) -> f64 {
        let denom = self.tp + self.fp;
        if denom > 0 {
            self.tp as f64 / denom as f64
        } else {
            0.0
        }
    }

    /// Calculate recall (sensitivity)
    pub fn recall(&self) -> f64 {
        let denom = self.tp + self.fn_;
        if denom > 0 {
            self.tp as f64 / denom as f64
        } else {
            0.0
        }
    }

    /// Calculate F1 score
    pub fn f1_score(&self) -> f64 {
        let precision = self.precision();
        let recall = self.recall();

        if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        }
    }

    /// Calculate specificity
    pub fn specificity(&self) -> f64 {
        let denom = self.tn + self.fp;
        if denom > 0 {
            self.tn as f64 / denom as f64
        } else {
            0.0
        }
    }

    /// Total samples
    pub fn total(&self) -> usize {
        self.tp + self.tn + self.fp + self.fn_
    }

    /// Print summary
    pub fn summary(&self) -> String {
        format!(
            "Accuracy: {:.2}%, Precision: {:.2}%, Recall: {:.2}%, F1: {:.2}%",
            self.accuracy() * 100.0,
            self.precision() * 100.0,
            self.recall() * 100.0,
            self.f1_score() * 100.0
        )
    }
}

/// Calculate AUC-ROC
pub fn auc_roc(probabilities: &[f64], labels: &[i32]) -> f64 {
    if probabilities.is_empty() || labels.is_empty() {
        return 0.5;
    }

    // Sort by probability descending
    let mut pairs: Vec<_> = probabilities.iter().zip(labels.iter()).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(a.0).unwrap());

    let n_pos = labels.iter().filter(|&&l| l == 1).count() as f64;
    let n_neg = labels.iter().filter(|&&l| l == 0).count() as f64;

    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }

    // Calculate AUC using trapezoidal rule
    let mut auc = 0.0;
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut prev_tp = 0.0;
    let mut prev_fp = 0.0;

    for (_, &label) in pairs {
        if *label == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }

        // Add trapezoid area
        auc += (fp - prev_fp) * (tp + prev_tp) / 2.0;
        prev_tp = tp;
        prev_fp = fp;
    }

    auc / (n_pos * n_neg)
}

/// Calculate Information Coefficient (correlation between predictions and returns)
pub fn information_coefficient(predictions: &[f64], returns: &[f64]) -> f64 {
    if predictions.len() != returns.len() || predictions.is_empty() {
        return 0.0;
    }

    let n = predictions.len() as f64;
    let pred_mean: f64 = predictions.iter().sum::<f64>() / n;
    let ret_mean: f64 = returns.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut pred_var = 0.0;
    let mut ret_var = 0.0;

    for (p, r) in predictions.iter().zip(returns.iter()) {
        let pred_dev = p - pred_mean;
        let ret_dev = r - ret_mean;
        cov += pred_dev * ret_dev;
        pred_var += pred_dev * pred_dev;
        ret_var += ret_dev * ret_dev;
    }

    let denom = (pred_var * ret_var).sqrt();
    if denom > 0.0 {
        cov / denom
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classification_metrics() {
        let predictions = vec![1, 0, 1, 1, 0, 0, 1, 0];
        let labels = vec![1, 0, 0, 1, 0, 1, 1, 0];

        let metrics = ClassificationMetrics::from_predictions(&predictions, &labels);

        assert_eq!(metrics.tp, 3); // Correctly predicted positives
        assert_eq!(metrics.tn, 3); // Correctly predicted negatives
        assert_eq!(metrics.fp, 1); // False alarm
        assert_eq!(metrics.fn_, 1); // Missed positive

        assert!((metrics.accuracy() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_auc_roc() {
        // Perfect predictions
        let probs = vec![0.9, 0.8, 0.3, 0.1];
        let labels = vec![1, 1, 0, 0];

        let auc = auc_roc(&probs, &labels);
        assert!(auc > 0.9);

        // Random predictions
        let probs = vec![0.5, 0.5, 0.5, 0.5];
        let auc = auc_roc(&probs, &labels);
        assert!((auc - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_information_coefficient() {
        // Perfect positive correlation
        let preds = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let rets = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let ic = information_coefficient(&preds, &rets);
        assert!(ic > 0.99);
    }
}
