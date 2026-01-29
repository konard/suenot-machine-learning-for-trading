//! Performance metrics for evaluating transfer learning models.

/// Confusion matrix for multi-class classification.
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    /// Matrix data: matrix[predicted][actual]
    matrix: Vec<Vec<usize>>,
    /// Number of classes
    num_classes: usize,
}

impl ConfusionMatrix {
    /// Create a new confusion matrix from predictions and true labels.
    pub fn new(predictions: &[usize], labels: &[usize], num_classes: usize) -> Self {
        let mut matrix = vec![vec![0usize; num_classes]; num_classes];
        for (&pred, &label) in predictions.iter().zip(labels.iter()) {
            if pred < num_classes && label < num_classes {
                matrix[pred][label] += 1;
            }
        }
        Self { matrix, num_classes }
    }

    /// Get the accuracy (proportion of correct predictions).
    pub fn accuracy(&self) -> f64 {
        let correct: usize = (0..self.num_classes).map(|i| self.matrix[i][i]).sum();
        let total: usize = self.matrix.iter().flat_map(|row| row.iter()).sum();
        if total == 0 { 0.0 } else { correct as f64 / total as f64 }
    }

    /// Get precision for a specific class.
    pub fn precision(&self, class: usize) -> f64 {
        let true_positive = self.matrix[class][class];
        let predicted_positive: usize = self.matrix[class].iter().sum();
        if predicted_positive == 0 { 0.0 } else { true_positive as f64 / predicted_positive as f64 }
    }

    /// Get recall for a specific class.
    pub fn recall(&self, class: usize) -> f64 {
        let true_positive = self.matrix[class][class];
        let actual_positive: usize = (0..self.num_classes).map(|i| self.matrix[i][class]).sum();
        if actual_positive == 0 { 0.0 } else { true_positive as f64 / actual_positive as f64 }
    }

    /// Get F1 score for a specific class.
    pub fn f1_score(&self, class: usize) -> f64 {
        let p = self.precision(class);
        let r = self.recall(class);
        if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
    }

    /// Get macro-averaged F1 score across all classes.
    pub fn macro_f1(&self) -> f64 {
        let sum: f64 = (0..self.num_classes).map(|c| self.f1_score(c)).sum();
        sum / self.num_classes as f64
    }

    /// Get the raw matrix data.
    pub fn matrix(&self) -> &Vec<Vec<usize>> {
        &self.matrix
    }
}

impl std::fmt::Display for ConfusionMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Confusion Matrix (rows=predicted, cols=actual):")?;
        for (i, row) in self.matrix.iter().enumerate() {
            write!(f, "  Class {}: ", i)?;
            for val in row {
                write!(f, "{:>5}", val)?;
            }
            writeln!(f)?;
        }
        writeln!(f, "  Accuracy: {:.2}%", self.accuracy() * 100.0)?;
        for c in 0..self.num_classes {
            writeln!(
                f,
                "  Class {} - P: {:.2}%, R: {:.2}%, F1: {:.2}%",
                c,
                self.precision(c) * 100.0,
                self.recall(c) * 100.0,
                self.f1_score(c) * 100.0
            )?;
        }
        Ok(())
    }
}

/// Collection of evaluation metrics for transfer learning.
#[derive(Debug, Clone)]
pub struct Metrics {
    /// Source domain accuracy
    pub source_accuracy: f64,
    /// Target domain accuracy
    pub target_accuracy: f64,
    /// Transfer gain (target accuracy - baseline accuracy)
    pub transfer_gain: f64,
    /// Domain similarity score (0.0 to 1.0)
    pub domain_similarity: f64,
    /// A-distance between domains
    pub a_distance: f64,
    /// MMD between domain features
    pub mmd: f64,
}

impl Metrics {
    /// Create metrics from predictions.
    pub fn compute(
        source_predictions: &[usize],
        source_labels: &[usize],
        target_predictions: &[usize],
        target_labels: &[usize],
        baseline_accuracy: f64,
        domain_similarity: f64,
        mmd: f64,
    ) -> Self {
        let source_cm = ConfusionMatrix::new(source_predictions, source_labels, 3);
        let target_cm = ConfusionMatrix::new(target_predictions, target_labels, 3);

        let source_accuracy = source_cm.accuracy();
        let target_accuracy = target_cm.accuracy();
        let transfer_gain = target_accuracy - baseline_accuracy;

        // Approximate A-distance from domain similarity
        let a_distance = 2.0 * (1.0 - domain_similarity);

        Self {
            source_accuracy,
            target_accuracy,
            transfer_gain,
            domain_similarity,
            a_distance,
            mmd,
        }
    }
}

impl std::fmt::Display for Metrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Transfer Learning Metrics ===")?;
        writeln!(f, "Source Accuracy:   {:.2}%", self.source_accuracy * 100.0)?;
        writeln!(f, "Target Accuracy:   {:.2}%", self.target_accuracy * 100.0)?;
        writeln!(f, "Transfer Gain:     {:+.2}%", self.transfer_gain * 100.0)?;
        writeln!(f, "Domain Similarity: {:.4}", self.domain_similarity)?;
        writeln!(f, "A-Distance:        {:.4}", self.a_distance)?;
        writeln!(f, "MMD:               {:.6}", self.mmd)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confusion_matrix_perfect() {
        let predictions = vec![0, 1, 2, 0, 1, 2];
        let labels = vec![0, 1, 2, 0, 1, 2];
        let cm = ConfusionMatrix::new(&predictions, &labels, 3);
        assert_eq!(cm.accuracy(), 1.0);
        assert_eq!(cm.precision(0), 1.0);
        assert_eq!(cm.recall(0), 1.0);
        assert_eq!(cm.f1_score(0), 1.0);
        assert_eq!(cm.macro_f1(), 1.0);
    }

    #[test]
    fn test_confusion_matrix_random() {
        let predictions = vec![0, 0, 0, 0, 0, 0]; // Always predicts class 0
        let labels = vec![0, 1, 2, 0, 1, 2];
        let cm = ConfusionMatrix::new(&predictions, &labels, 3);
        assert!((cm.accuracy() - 1.0 / 3.0).abs() < 1e-10);
        assert!((cm.recall(0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix_display() {
        let predictions = vec![0, 1, 2, 0, 1, 2];
        let labels = vec![0, 1, 2, 0, 1, 2];
        let cm = ConfusionMatrix::new(&predictions, &labels, 3);
        let display = format!("{}", cm);
        assert!(display.contains("Accuracy: 100.00%"));
    }

    #[test]
    fn test_metrics_computation() {
        let source_preds = vec![0, 1, 2, 0, 1, 2];
        let source_labels = vec![0, 1, 2, 0, 1, 2];
        let target_preds = vec![0, 1, 2, 0, 0, 2];
        let target_labels = vec![0, 1, 2, 0, 1, 2];

        let metrics = Metrics::compute(
            &source_preds,
            &source_labels,
            &target_preds,
            &target_labels,
            0.33, // baseline
            0.8,  // domain similarity
            0.05, // MMD
        );

        assert_eq!(metrics.source_accuracy, 1.0);
        assert!(metrics.target_accuracy > 0.5);
        assert!(metrics.transfer_gain > 0.0);
        assert_eq!(metrics.domain_similarity, 0.8);
    }

    #[test]
    fn test_empty_confusion_matrix() {
        let cm = ConfusionMatrix::new(&[], &[], 3);
        assert_eq!(cm.accuracy(), 0.0);
        assert_eq!(cm.precision(0), 0.0);
        assert_eq!(cm.recall(0), 0.0);
    }

    #[test]
    fn test_metrics_display() {
        let metrics = Metrics {
            source_accuracy: 0.85,
            target_accuracy: 0.60,
            transfer_gain: 0.10,
            domain_similarity: 0.75,
            a_distance: 0.50,
            mmd: 0.05,
        };
        let display = format!("{}", metrics);
        assert!(display.contains("Source Accuracy"));
        assert!(display.contains("Transfer Gain"));
    }
}
