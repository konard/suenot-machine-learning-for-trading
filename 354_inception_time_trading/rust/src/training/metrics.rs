//! Evaluation metrics for classification
//!
//! This module provides metrics for evaluating model performance.

use tch::Tensor;

/// Calculate accuracy
pub fn accuracy(predictions: &Tensor, targets: &Tensor) -> f64 {
    let pred_classes = predictions.argmax(-1, false);
    let correct = pred_classes.eq_tensor(targets);
    let acc = correct.to_kind(tch::Kind::Float).mean(tch::Kind::Float);
    f64::try_from(acc).unwrap_or(0.0)
}

/// Calculate confusion matrix
pub fn confusion_matrix(predictions: &Tensor, targets: &Tensor, num_classes: i64) -> Vec<Vec<i64>> {
    let pred_classes = predictions.argmax(-1, false);
    let pred_vec: Vec<i64> = Vec::try_from(&pred_classes).unwrap_or_default();
    let target_vec: Vec<i64> = Vec::try_from(targets).unwrap_or_default();

    let mut matrix = vec![vec![0i64; num_classes as usize]; num_classes as usize];

    for (p, t) in pred_vec.iter().zip(target_vec.iter()) {
        if *p >= 0 && *p < num_classes && *t >= 0 && *t < num_classes {
            matrix[*t as usize][*p as usize] += 1;
        }
    }

    matrix
}

/// Calculate F1 score for each class
pub fn f1_score(confusion: &[Vec<i64>]) -> Vec<f64> {
    let num_classes = confusion.len();
    let mut f1_scores = Vec::with_capacity(num_classes);

    for c in 0..num_classes {
        let true_positive = confusion[c][c] as f64;
        let false_positive: f64 = (0..num_classes)
            .filter(|&i| i != c)
            .map(|i| confusion[i][c] as f64)
            .sum();
        let false_negative: f64 = (0..num_classes)
            .filter(|&i| i != c)
            .map(|i| confusion[c][i] as f64)
            .sum();

        let precision = if true_positive + false_positive > 0.0 {
            true_positive / (true_positive + false_positive)
        } else {
            0.0
        };

        let recall = if true_positive + false_negative > 0.0 {
            true_positive / (true_positive + false_negative)
        } else {
            0.0
        };

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        f1_scores.push(f1);
    }

    f1_scores
}

/// Calculate macro F1 score (average across classes)
pub fn macro_f1(confusion: &[Vec<i64>]) -> f64 {
    let f1_scores = f1_score(confusion);
    if f1_scores.is_empty() {
        0.0
    } else {
        f1_scores.iter().sum::<f64>() / f1_scores.len() as f64
    }
}

/// Calculate Cohen's Kappa
pub fn cohens_kappa(confusion: &[Vec<i64>]) -> f64 {
    let num_classes = confusion.len();
    let total: i64 = confusion.iter().flat_map(|row| row.iter()).sum();

    if total == 0 {
        return 0.0;
    }

    let total_f = total as f64;

    // Observed agreement
    let observed: f64 = (0..num_classes).map(|i| confusion[i][i] as f64).sum();
    let p_o = observed / total_f;

    // Expected agreement
    let mut p_e = 0.0;
    for c in 0..num_classes {
        let row_sum: f64 = confusion[c].iter().map(|&x| x as f64).sum();
        let col_sum: f64 = (0..num_classes).map(|i| confusion[i][c] as f64).sum();
        p_e += (row_sum * col_sum) / (total_f * total_f);
    }

    // Kappa
    if (1.0 - p_e).abs() < 1e-8 {
        1.0
    } else {
        (p_o - p_e) / (1.0 - p_e)
    }
}

/// Training metrics container
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub train_accuracy: f64,
    pub val_loss: f64,
    pub val_accuracy: f64,
    pub val_f1_macro: f64,
    pub val_kappa: f64,
    pub learning_rate: f64,
}

impl TrainingMetrics {
    /// Create new metrics for an epoch
    pub fn new(epoch: usize) -> Self {
        Self {
            epoch,
            ..Default::default()
        }
    }

    /// Check if validation improved
    pub fn improved_over(&self, other: &TrainingMetrics) -> bool {
        self.val_f1_macro > other.val_f1_macro
    }

    /// Format as string for logging
    pub fn to_string(&self) -> String {
        format!(
            "Epoch {}: Train Loss: {:.4}, Train Acc: {:.4}, Val Loss: {:.4}, Val Acc: {:.4}, Val F1: {:.4}",
            self.epoch, self.train_loss, self.train_accuracy, self.val_loss, self.val_accuracy, self.val_f1_macro
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confusion_matrix_f1() {
        let confusion = vec![
            vec![10, 2, 1],
            vec![1, 15, 2],
            vec![2, 1, 12],
        ];

        let f1_scores = f1_score(&confusion);
        assert_eq!(f1_scores.len(), 3);

        // All F1 scores should be between 0 and 1
        for &f1 in &f1_scores {
            assert!(f1 >= 0.0 && f1 <= 1.0);
        }
    }

    #[test]
    fn test_cohens_kappa() {
        // Perfect agreement
        let perfect = vec![
            vec![10, 0, 0],
            vec![0, 10, 0],
            vec![0, 0, 10],
        ];
        assert!((cohens_kappa(&perfect) - 1.0).abs() < 1e-8);

        // Some disagreement
        let partial = vec![
            vec![8, 2, 0],
            vec![1, 7, 2],
            vec![1, 1, 8],
        ];
        let kappa = cohens_kappa(&partial);
        assert!(kappa > 0.5 && kappa < 1.0);
    }
}
