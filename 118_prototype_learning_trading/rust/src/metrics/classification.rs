//! Classification metrics for model evaluation.

use crate::models::prototype::MarketState;
use std::collections::HashMap;

/// Classification metrics calculator.
#[derive(Debug, Clone)]
pub struct ClassificationMetrics {
    /// Number of classes
    n_classes: usize,
    /// Confusion matrix
    confusion_matrix: Vec<Vec<usize>>,
    /// Total predictions
    total: usize,
}

impl ClassificationMetrics {
    /// Create a new metrics calculator.
    pub fn new(n_classes: usize) -> Self {
        Self {
            n_classes,
            confusion_matrix: vec![vec![0; n_classes]; n_classes],
            total: 0,
        }
    }

    /// Create metrics calculator for market states (5 classes).
    pub fn for_market_states() -> Self {
        Self::new(5)
    }

    /// Update metrics with a single prediction.
    pub fn update(&mut self, true_label: usize, predicted_label: usize) {
        if true_label < self.n_classes && predicted_label < self.n_classes {
            self.confusion_matrix[true_label][predicted_label] += 1;
            self.total += 1;
        }
    }

    /// Update metrics with market states.
    pub fn update_state(&mut self, true_state: MarketState, predicted_state: MarketState) {
        self.update(true_state as usize, predicted_state as usize);
    }

    /// Calculate overall accuracy.
    pub fn accuracy(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }

        let correct: usize = (0..self.n_classes)
            .map(|i| self.confusion_matrix[i][i])
            .sum();

        correct as f64 / self.total as f64
    }

    /// Calculate precision for a specific class.
    pub fn precision(&self, class_idx: usize) -> f64 {
        if class_idx >= self.n_classes {
            return 0.0;
        }

        let predicted_positive: usize = (0..self.n_classes)
            .map(|i| self.confusion_matrix[i][class_idx])
            .sum();

        if predicted_positive == 0 {
            return 0.0;
        }

        self.confusion_matrix[class_idx][class_idx] as f64 / predicted_positive as f64
    }

    /// Calculate recall for a specific class.
    pub fn recall(&self, class_idx: usize) -> f64 {
        if class_idx >= self.n_classes {
            return 0.0;
        }

        let actual_positive: usize = self.confusion_matrix[class_idx].iter().sum();

        if actual_positive == 0 {
            return 0.0;
        }

        self.confusion_matrix[class_idx][class_idx] as f64 / actual_positive as f64
    }

    /// Calculate F1 score for a specific class.
    pub fn f1_score(&self, class_idx: usize) -> f64 {
        let prec = self.precision(class_idx);
        let rec = self.recall(class_idx);

        if prec + rec == 0.0 {
            return 0.0;
        }

        2.0 * prec * rec / (prec + rec)
    }

    /// Calculate macro-averaged F1 score.
    pub fn macro_f1(&self) -> f64 {
        let f1_scores: Vec<f64> = (0..self.n_classes)
            .map(|i| self.f1_score(i))
            .collect();

        f1_scores.iter().sum::<f64>() / self.n_classes as f64
    }

    /// Calculate weighted F1 score.
    pub fn weighted_f1(&self) -> f64 {
        let mut total_f1 = 0.0;
        let mut total_support = 0;

        for class_idx in 0..self.n_classes {
            let support: usize = self.confusion_matrix[class_idx].iter().sum();
            total_f1 += self.f1_score(class_idx) * support as f64;
            total_support += support;
        }

        if total_support == 0 {
            return 0.0;
        }

        total_f1 / total_support as f64
    }

    /// Get per-class metrics.
    pub fn per_class_metrics(&self) -> HashMap<String, ClassMetrics> {
        let mut metrics = HashMap::new();

        for class_idx in 0..self.n_classes {
            let name = MarketState::from_int(class_idx)
                .map(|s| s.name().to_string())
                .unwrap_or_else(|| format!("Class_{}", class_idx));

            let support: usize = self.confusion_matrix[class_idx].iter().sum();

            metrics.insert(
                name,
                ClassMetrics {
                    precision: self.precision(class_idx),
                    recall: self.recall(class_idx),
                    f1_score: self.f1_score(class_idx),
                    support,
                },
            );
        }

        metrics
    }

    /// Get the confusion matrix.
    pub fn confusion_matrix(&self) -> &Vec<Vec<usize>> {
        &self.confusion_matrix
    }

    /// Print a summary of metrics.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("Classification Metrics Summary\n");
        s.push_str("==============================\n\n");

        s.push_str(&format!("Overall Accuracy: {:.4}\n", self.accuracy()));
        s.push_str(&format!("Macro F1 Score: {:.4}\n", self.macro_f1()));
        s.push_str(&format!("Weighted F1 Score: {:.4}\n\n", self.weighted_f1()));

        s.push_str("Per-class metrics:\n");
        s.push_str("-----------------\n");
        s.push_str(&format!(
            "{:<15} {:>10} {:>10} {:>10} {:>10}\n",
            "Class", "Precision", "Recall", "F1", "Support"
        ));

        for class_idx in 0..self.n_classes {
            let name = MarketState::from_int(class_idx)
                .map(|s| s.name())
                .unwrap_or("Unknown");
            let support: usize = self.confusion_matrix[class_idx].iter().sum();

            s.push_str(&format!(
                "{:<15} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
                name,
                self.precision(class_idx),
                self.recall(class_idx),
                self.f1_score(class_idx),
                support
            ));
        }

        s
    }

    /// Reset all metrics.
    pub fn reset(&mut self) {
        self.confusion_matrix = vec![vec![0; self.n_classes]; self.n_classes];
        self.total = 0;
    }
}

/// Per-class metrics.
#[derive(Debug, Clone)]
pub struct ClassMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub support: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy() {
        let mut metrics = ClassificationMetrics::new(3);

        // 8 correct, 2 incorrect
        for _ in 0..3 {
            metrics.update(0, 0);
        }
        for _ in 0..3 {
            metrics.update(1, 1);
        }
        for _ in 0..2 {
            metrics.update(2, 2);
        }
        metrics.update(0, 1); // Error
        metrics.update(1, 2); // Error

        assert!((metrics.accuracy() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_precision_recall() {
        let mut metrics = ClassificationMetrics::new(2);

        // Class 0: 3 TP, 1 FP, 1 FN
        // Class 1: 2 TP, 1 FP, 1 FN
        metrics.update(0, 0);
        metrics.update(0, 0);
        metrics.update(0, 0);
        metrics.update(0, 1); // FN for class 0, FP for class 1
        metrics.update(1, 0); // FP for class 0, FN for class 1
        metrics.update(1, 1);
        metrics.update(1, 1);

        // Class 0 precision: 3 / (3 + 1) = 0.75
        assert!((metrics.precision(0) - 0.75).abs() < 1e-10);

        // Class 0 recall: 3 / (3 + 1) = 0.75
        assert!((metrics.recall(0) - 0.75).abs() < 1e-10);
    }
}
