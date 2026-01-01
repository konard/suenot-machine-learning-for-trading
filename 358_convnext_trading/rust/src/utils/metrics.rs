//! Performance metrics for model evaluation

use ndarray::{Array1, Array2};

/// Metrics calculator for model evaluation
pub struct Metrics;

impl Metrics {
    /// Calculate accuracy
    pub fn accuracy(predictions: &Array1<usize>, targets: &Array1<usize>) -> f64 {
        if predictions.len() != targets.len() || predictions.is_empty() {
            return 0.0;
        }

        let correct = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(&p, &t)| p == t)
            .count();

        correct as f64 / predictions.len() as f64
    }

    /// Calculate precision for a specific class
    pub fn precision(predictions: &Array1<usize>, targets: &Array1<usize>, class: usize) -> f64 {
        let true_positives = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(&p, &t)| p == class && t == class)
            .count();

        let predicted_positives = predictions.iter().filter(|&&p| p == class).count();

        if predicted_positives == 0 {
            0.0
        } else {
            true_positives as f64 / predicted_positives as f64
        }
    }

    /// Calculate recall for a specific class
    pub fn recall(predictions: &Array1<usize>, targets: &Array1<usize>, class: usize) -> f64 {
        let true_positives = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(&p, &t)| p == class && t == class)
            .count();

        let actual_positives = targets.iter().filter(|&&t| t == class).count();

        if actual_positives == 0 {
            0.0
        } else {
            true_positives as f64 / actual_positives as f64
        }
    }

    /// Calculate F1 score for a specific class
    pub fn f1_score(predictions: &Array1<usize>, targets: &Array1<usize>, class: usize) -> f64 {
        let precision = Self::precision(predictions, targets, class);
        let recall = Self::recall(predictions, targets, class);

        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }

    /// Calculate macro F1 score (average across all classes)
    pub fn macro_f1(predictions: &Array1<usize>, targets: &Array1<usize>, num_classes: usize) -> f64 {
        let f1_sum: f64 = (0..num_classes)
            .map(|c| Self::f1_score(predictions, targets, c))
            .sum();

        f1_sum / num_classes as f64
    }

    /// Calculate confusion matrix
    pub fn confusion_matrix(
        predictions: &Array1<usize>,
        targets: &Array1<usize>,
        num_classes: usize,
    ) -> Array2<usize> {
        let mut matrix = Array2::zeros((num_classes, num_classes));

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            if pred < num_classes && target < num_classes {
                matrix[[target, pred]] += 1;
            }
        }

        matrix
    }

    /// Calculate Mean Squared Error
    pub fn mse(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
        if predictions.len() != targets.len() || predictions.is_empty() {
            return f64::NAN;
        }

        predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / predictions.len() as f64
    }

    /// Calculate Root Mean Squared Error
    pub fn rmse(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
        Self::mse(predictions, targets).sqrt()
    }

    /// Calculate Mean Absolute Error
    pub fn mae(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
        if predictions.len() != targets.len() || predictions.is_empty() {
            return f64::NAN;
        }

        predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>()
            / predictions.len() as f64
    }

    /// Calculate R-squared (coefficient of determination)
    pub fn r_squared(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
        if predictions.len() != targets.len() || predictions.is_empty() {
            return f64::NAN;
        }

        let mean_target = targets.iter().sum::<f64>() / targets.len() as f64;

        let ss_res: f64 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (t - p).powi(2))
            .sum();

        let ss_tot: f64 = targets.iter().map(|t| (t - mean_target).powi(2)).sum();

        if ss_tot == 0.0 {
            0.0
        } else {
            1.0 - ss_res / ss_tot
        }
    }

    /// Calculate direction accuracy (correct sign prediction)
    pub fn direction_accuracy(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
        if predictions.len() != targets.len() || predictions.is_empty() {
            return 0.0;
        }

        let correct = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(&p, &t)| {
                (p > 0.0 && t > 0.0) || (p < 0.0 && t < 0.0) || (p == 0.0 && t == 0.0)
            })
            .count();

        correct as f64 / predictions.len() as f64
    }

    /// Calculate hit rate (profitable predictions)
    pub fn hit_rate(signals: &Array1<i32>, returns: &Array1<f64>) -> f64 {
        if signals.len() != returns.len() || signals.is_empty() {
            return 0.0;
        }

        let profitable = signals
            .iter()
            .zip(returns.iter())
            .filter(|(&s, &r)| {
                if s == 0 {
                    false // Ignore hold signals
                } else {
                    (s as f64 * r) > 0.0
                }
            })
            .count();

        let total_trades = signals.iter().filter(|&&s| s != 0).count();

        if total_trades == 0 {
            0.0
        } else {
            profitable as f64 / total_trades as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_accuracy() {
        let predictions = array![0, 1, 2, 0, 1];
        let targets = array![0, 1, 1, 0, 2];
        let acc = Metrics::accuracy(&predictions, &targets);
        assert!((acc - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_precision_recall() {
        let predictions = array![0, 0, 1, 1, 1];
        let targets = array![0, 1, 1, 1, 0];

        let precision = Metrics::precision(&predictions, &targets, 1);
        assert!((precision - 2.0 / 3.0).abs() < 1e-6);

        let recall = Metrics::recall(&predictions, &targets, 1);
        assert!((recall - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_confusion_matrix() {
        let predictions = array![0, 0, 1, 1, 2, 2];
        let targets = array![0, 1, 1, 2, 2, 0];

        let cm = Metrics::confusion_matrix(&predictions, &targets, 3);

        assert_eq!(cm[[0, 0]], 1); // True class 0, predicted 0
        assert_eq!(cm[[1, 0]], 1); // True class 1, predicted 0
        assert_eq!(cm[[1, 1]], 1); // True class 1, predicted 1
    }

    #[test]
    fn test_mse() {
        let predictions = array![1.0, 2.0, 3.0];
        let targets = array![1.0, 2.0, 4.0];
        let mse = Metrics::mse(&predictions, &targets);
        assert!((mse - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_r_squared() {
        let predictions = array![1.0, 2.0, 3.0];
        let targets = array![1.0, 2.0, 3.0];
        let r2 = Metrics::r_squared(&predictions, &targets);
        assert!((r2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_direction_accuracy() {
        let predictions = array![0.1, -0.2, 0.3, -0.1, 0.0];
        let targets = array![0.2, -0.1, 0.1, 0.1, 0.0];
        let acc = Metrics::direction_accuracy(&predictions, &targets);
        assert!((acc - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_hit_rate() {
        let signals = array![1, -1, 1, -1, 0];
        let returns = array![0.01, 0.02, -0.01, -0.02, 0.01];
        // Trade 1: long, positive return -> hit
        // Trade 2: short, positive return -> miss (short on up move)
        // Trade 3: long, negative return -> miss
        // Trade 4: short, negative return -> hit (short on down move)
        // Trade 5: hold -> ignored
        let hr = Metrics::hit_rate(&signals, &returns);
        assert!((hr - 0.5).abs() < 1e-6);
    }
}
