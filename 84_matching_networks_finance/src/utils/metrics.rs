//! Evaluation metrics for classification
//!
//! Provides accuracy, precision, recall, F1-score, and confusion matrix calculations.

use ndarray::Array1;
use std::collections::HashMap;
use std::fmt;

/// Evaluation metrics
#[derive(Debug, Clone, Default)]
pub struct Metrics {
    /// Overall accuracy
    pub accuracy: f64,
    /// Macro-averaged precision
    pub precision: f64,
    /// Macro-averaged recall
    pub recall: f64,
    /// Macro-averaged F1 score
    pub f1_score: f64,
    /// Per-class metrics
    pub class_metrics: HashMap<usize, ClassMetrics>,
}

/// Per-class metrics
#[derive(Debug, Clone, Default)]
pub struct ClassMetrics {
    /// True positives
    pub true_positives: usize,
    /// False positives
    pub false_positives: usize,
    /// False negatives
    pub false_negatives: usize,
    /// True negatives
    pub true_negatives: usize,
    /// Precision
    pub precision: f64,
    /// Recall
    pub recall: f64,
    /// F1 score
    pub f1_score: f64,
}

impl Metrics {
    /// Calculate metrics from predictions and true labels
    ///
    /// # Arguments
    /// * `predictions` - Predicted class indices
    /// * `labels` - True class indices
    /// * `num_classes` - Total number of classes
    ///
    /// # Example
    /// ```
    /// use ndarray::Array1;
    /// use matching_networks_finance::utils::Metrics;
    ///
    /// let predictions = Array1::from_vec(vec![0, 1, 2, 0, 1]);
    /// let labels = Array1::from_vec(vec![0, 1, 2, 1, 1]);
    ///
    /// let metrics = Metrics::compute(&predictions, &labels, 3);
    /// println!("Accuracy: {:.2}%", metrics.accuracy * 100.0);
    /// ```
    pub fn compute(predictions: &Array1<usize>, labels: &Array1<usize>, num_classes: usize) -> Self {
        let mut class_metrics: HashMap<usize, ClassMetrics> = HashMap::new();

        // Initialize metrics for all classes
        for class_id in 0..num_classes {
            class_metrics.insert(class_id, ClassMetrics::default());
        }

        // Count TP, FP, FN for each class
        for (&pred, &true_label) in predictions.iter().zip(labels.iter()) {
            if pred == true_label {
                // True positive for the predicted class
                if let Some(m) = class_metrics.get_mut(&pred) {
                    m.true_positives += 1;
                }
            } else {
                // False positive for predicted class
                if let Some(m) = class_metrics.get_mut(&pred) {
                    m.false_positives += 1;
                }
                // False negative for true class
                if let Some(m) = class_metrics.get_mut(&true_label) {
                    m.false_negatives += 1;
                }
            }
        }

        // Calculate TN for each class
        let n = predictions.len();
        for (&class_id, metrics) in class_metrics.iter_mut() {
            metrics.true_negatives = n - metrics.true_positives - metrics.false_positives - metrics.false_negatives;
        }

        // Calculate precision, recall, F1 for each class
        for metrics in class_metrics.values_mut() {
            metrics.precision = if metrics.true_positives + metrics.false_positives > 0 {
                metrics.true_positives as f64
                    / (metrics.true_positives + metrics.false_positives) as f64
            } else {
                0.0
            };

            metrics.recall = if metrics.true_positives + metrics.false_negatives > 0 {
                metrics.true_positives as f64
                    / (metrics.true_positives + metrics.false_negatives) as f64
            } else {
                0.0
            };

            metrics.f1_score = if metrics.precision + metrics.recall > 0.0 {
                2.0 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
            } else {
                0.0
            };
        }

        // Calculate macro averages
        let active_classes: Vec<_> = class_metrics
            .values()
            .filter(|m| m.true_positives + m.false_negatives > 0)
            .collect();

        let num_active = active_classes.len() as f64;

        let precision = if num_active > 0.0 {
            active_classes.iter().map(|m| m.precision).sum::<f64>() / num_active
        } else {
            0.0
        };

        let recall = if num_active > 0.0 {
            active_classes.iter().map(|m| m.recall).sum::<f64>() / num_active
        } else {
            0.0
        };

        let f1_score = if num_active > 0.0 {
            active_classes.iter().map(|m| m.f1_score).sum::<f64>() / num_active
        } else {
            0.0
        };

        // Overall accuracy
        let correct: usize = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(&p, &l)| p == l)
            .count();
        let accuracy = correct as f64 / n as f64;

        Metrics {
            accuracy,
            precision,
            recall,
            f1_score,
            class_metrics,
        }
    }

    /// Get metrics for a specific class
    pub fn get_class_metrics(&self, class_id: usize) -> Option<&ClassMetrics> {
        self.class_metrics.get(&class_id)
    }
}

impl fmt::Display for Metrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Overall Metrics:")?;
        writeln!(f, "  Accuracy:  {:.4}", self.accuracy)?;
        writeln!(f, "  Precision: {:.4}", self.precision)?;
        writeln!(f, "  Recall:    {:.4}", self.recall)?;
        writeln!(f, "  F1 Score:  {:.4}", self.f1_score)?;
        writeln!(f)?;
        writeln!(f, "Per-Class Metrics:")?;

        let mut classes: Vec<_> = self.class_metrics.keys().collect();
        classes.sort();

        for &class_id in &classes {
            if let Some(m) = self.class_metrics.get(&class_id) {
                writeln!(
                    f,
                    "  Class {}: P={:.4}, R={:.4}, F1={:.4}",
                    class_id, m.precision, m.recall, m.f1_score
                )?;
            }
        }

        Ok(())
    }
}

/// Confusion matrix
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    /// Matrix data [true_class, predicted_class]
    matrix: Vec<Vec<usize>>,
    /// Number of classes
    num_classes: usize,
    /// Class names (optional)
    class_names: Option<Vec<String>>,
}

impl ConfusionMatrix {
    /// Create a confusion matrix from predictions and labels
    ///
    /// # Arguments
    /// * `predictions` - Predicted class indices
    /// * `labels` - True class indices
    /// * `num_classes` - Total number of classes
    pub fn compute(predictions: &Array1<usize>, labels: &Array1<usize>, num_classes: usize) -> Self {
        let mut matrix = vec![vec![0; num_classes]; num_classes];

        for (&pred, &true_label) in predictions.iter().zip(labels.iter()) {
            if true_label < num_classes && pred < num_classes {
                matrix[true_label][pred] += 1;
            }
        }

        Self {
            matrix,
            num_classes,
            class_names: None,
        }
    }

    /// Set class names for display
    pub fn with_class_names(mut self, names: Vec<String>) -> Self {
        self.class_names = Some(names);
        self
    }

    /// Get the count for a specific cell
    pub fn get(&self, true_class: usize, predicted_class: usize) -> usize {
        self.matrix
            .get(true_class)
            .and_then(|row| row.get(predicted_class))
            .copied()
            .unwrap_or(0)
    }

    /// Get row (all predictions for a true class)
    pub fn get_row(&self, true_class: usize) -> Option<&[usize]> {
        self.matrix.get(true_class).map(|v| v.as_slice())
    }

    /// Get the total number of samples
    pub fn total(&self) -> usize {
        self.matrix.iter().flatten().sum()
    }

    /// Normalize the matrix by rows (convert to percentages)
    pub fn normalize_by_row(&self) -> Vec<Vec<f64>> {
        self.matrix
            .iter()
            .map(|row| {
                let sum: usize = row.iter().sum();
                if sum > 0 {
                    row.iter().map(|&v| v as f64 / sum as f64).collect()
                } else {
                    vec![0.0; self.num_classes]
                }
            })
            .collect()
    }
}

impl fmt::Display for ConfusionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Confusion Matrix:")?;

        // Header
        write!(f, "True\\Pred")?;
        for i in 0..self.num_classes {
            if let Some(names) = &self.class_names {
                write!(f, " {:>8}", &names[i][..names[i].len().min(8)])?;
            } else {
                write!(f, " {:>6}", i)?;
            }
        }
        writeln!(f)?;

        // Rows
        for (i, row) in self.matrix.iter().enumerate() {
            if let Some(names) = &self.class_names {
                write!(f, "{:>9}", &names[i][..names[i].len().min(9)])?;
            } else {
                write!(f, "{:>9}", i)?;
            }
            for &count in row {
                write!(f, " {:>6}", count)?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

/// Complete classification report
#[derive(Debug)]
pub struct ClassificationReport {
    /// Metrics
    pub metrics: Metrics,
    /// Confusion matrix
    pub confusion_matrix: ConfusionMatrix,
}

impl ClassificationReport {
    /// Generate a classification report
    pub fn generate(
        predictions: &Array1<usize>,
        labels: &Array1<usize>,
        num_classes: usize,
    ) -> Self {
        let metrics = Metrics::compute(predictions, labels, num_classes);
        let confusion_matrix = ConfusionMatrix::compute(predictions, labels, num_classes);

        Self {
            metrics,
            confusion_matrix,
        }
    }

    /// Add class names to the report
    pub fn with_class_names(mut self, names: Vec<String>) -> Self {
        self.confusion_matrix = self.confusion_matrix.with_class_names(names);
        self
    }
}

impl fmt::Display for ClassificationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Classification Report")?;
        writeln!(f, "====================")?;
        writeln!(f)?;
        writeln!(f, "{}", self.metrics)?;
        writeln!(f, "{}", self.confusion_matrix)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_perfect() {
        let predictions = Array1::from_vec(vec![0, 1, 2, 0, 1, 2]);
        let labels = Array1::from_vec(vec![0, 1, 2, 0, 1, 2]);

        let metrics = Metrics::compute(&predictions, &labels, 3);

        assert!((metrics.accuracy - 1.0).abs() < 1e-10);
        assert!((metrics.precision - 1.0).abs() < 1e-10);
        assert!((metrics.recall - 1.0).abs() < 1e-10);
        assert!((metrics.f1_score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metrics_mixed() {
        let predictions = Array1::from_vec(vec![0, 1, 2, 0, 1]);
        let labels = Array1::from_vec(vec![0, 1, 2, 1, 1]);

        let metrics = Metrics::compute(&predictions, &labels, 3);

        assert!((metrics.accuracy - 0.8).abs() < 1e-10); // 4/5 correct
    }

    #[test]
    fn test_confusion_matrix() {
        let predictions = Array1::from_vec(vec![0, 0, 1, 1, 2, 2]);
        let labels = Array1::from_vec(vec![0, 1, 0, 1, 1, 2]);

        let cm = ConfusionMatrix::compute(&predictions, &labels, 3);

        // True class 0: predicted 0 once (correct), predicted 1 once
        assert_eq!(cm.get(0, 0), 1);
        assert_eq!(cm.get(0, 1), 0);

        // True class 1: predicted 0 once, 1 once, 2 once
        assert_eq!(cm.get(1, 0), 1);
        assert_eq!(cm.get(1, 1), 1);
        assert_eq!(cm.get(1, 2), 1);

        // True class 2: predicted 2 once (correct)
        assert_eq!(cm.get(2, 2), 1);

        assert_eq!(cm.total(), 6);
    }

    #[test]
    fn test_classification_report() {
        let predictions = Array1::from_vec(vec![0, 1, 2, 0, 1, 2, 0, 1, 2]);
        let labels = Array1::from_vec(vec![0, 1, 2, 0, 0, 2, 1, 1, 2]);

        let report = ClassificationReport::generate(&predictions, &labels, 3);

        // Should not panic and should have valid metrics
        assert!(report.metrics.accuracy > 0.0);
        assert!(report.confusion_matrix.total() == 9);

        // Test display
        let display = format!("{}", report);
        assert!(display.contains("Classification Report"));
    }

    #[test]
    fn test_with_class_names() {
        let predictions = Array1::from_vec(vec![0, 1, 0, 1]);
        let labels = Array1::from_vec(vec![0, 1, 1, 1]);

        let report = ClassificationReport::generate(&predictions, &labels, 2)
            .with_class_names(vec!["Cat".to_string(), "Dog".to_string()]);

        let display = format!("{}", report);
        assert!(display.contains("Cat") || display.contains("Dog"));
    }
}
