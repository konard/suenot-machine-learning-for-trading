//! Classification metrics for evaluating binary classifiers
//!
//! Provides metrics for evaluating price direction prediction models.

use ndarray::Array1;

/// Confusion matrix for binary classification
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    /// True positives
    pub tp: usize,
    /// True negatives
    pub tn: usize,
    /// False positives
    pub fp: usize,
    /// False negatives
    pub fn_: usize,
}

impl ConfusionMatrix {
    /// Calculate confusion matrix from predictions
    pub fn from_predictions(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Self {
        let mut tp = 0;
        let mut tn = 0;
        let mut fp = 0;
        let mut fn_ = 0;

        for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
            let t_bool = t >= 0.5;
            let p_bool = p >= 0.5;

            match (t_bool, p_bool) {
                (true, true) => tp += 1,
                (false, false) => tn += 1,
                (false, true) => fp += 1,
                (true, false) => fn_ += 1,
            }
        }

        Self { tp, tn, fp, fn_ }
    }

    /// Total samples
    pub fn total(&self) -> usize {
        self.tp + self.tn + self.fp + self.fn_
    }

    /// Print formatted confusion matrix
    pub fn display(&self) -> String {
        format!(
            "Confusion Matrix:\n\
             \n\
             Predicted:    0       1\n\
             Actual 0:   {:>5}   {:>5}  (TN/FP)\n\
             Actual 1:   {:>5}   {:>5}  (FN/TP)\n",
            self.tn, self.fp, self.fn_, self.tp
        )
    }
}

/// Collection of classification metrics
#[derive(Debug, Clone)]
pub struct ClassificationMetrics {
    /// Confusion matrix
    pub confusion_matrix: ConfusionMatrix,
    /// Accuracy
    pub accuracy: f64,
    /// Precision
    pub precision: f64,
    /// Recall (sensitivity)
    pub recall: f64,
    /// F1 score
    pub f1: f64,
    /// Specificity
    pub specificity: f64,
    /// Matthews Correlation Coefficient
    pub mcc: f64,
    /// AUC-ROC (if probabilities provided)
    pub auc_roc: Option<f64>,
    /// Log loss (if probabilities provided)
    pub log_loss: Option<f64>,
}

impl ClassificationMetrics {
    /// Calculate all metrics from binary predictions
    pub fn calculate(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Self {
        Self::calculate_with_proba(y_true, y_pred, None)
    }

    /// Calculate metrics with probability predictions for AUC and log loss
    pub fn calculate_with_proba(
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
        y_proba: Option<&Array1<f64>>,
    ) -> Self {
        let cm = ConfusionMatrix::from_predictions(y_true, y_pred);

        let accuracy = Self::accuracy_from_cm(&cm);
        let precision = Self::precision_from_cm(&cm);
        let recall = Self::recall_from_cm(&cm);
        let f1 = Self::f1_from_cm(&cm);
        let specificity = Self::specificity_from_cm(&cm);
        let mcc = Self::mcc_from_cm(&cm);

        let auc_roc = y_proba.map(|p| Self::auc_roc(y_true, p));
        let log_loss = y_proba.map(|p| Self::log_loss(y_true, p));

        Self {
            confusion_matrix: cm,
            accuracy,
            precision,
            recall,
            f1,
            specificity,
            mcc,
            auc_roc,
            log_loss,
        }
    }

    /// Accuracy: (TP + TN) / Total
    fn accuracy_from_cm(cm: &ConfusionMatrix) -> f64 {
        let total = cm.total() as f64;
        if total < 1e-10 {
            return 0.0;
        }
        (cm.tp + cm.tn) as f64 / total
    }

    /// Precision: TP / (TP + FP)
    fn precision_from_cm(cm: &ConfusionMatrix) -> f64 {
        let denom = (cm.tp + cm.fp) as f64;
        if denom < 1e-10 {
            return 0.0;
        }
        cm.tp as f64 / denom
    }

    /// Recall: TP / (TP + FN)
    fn recall_from_cm(cm: &ConfusionMatrix) -> f64 {
        let denom = (cm.tp + cm.fn_) as f64;
        if denom < 1e-10 {
            return 0.0;
        }
        cm.tp as f64 / denom
    }

    /// F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    fn f1_from_cm(cm: &ConfusionMatrix) -> f64 {
        let precision = Self::precision_from_cm(cm);
        let recall = Self::recall_from_cm(cm);
        let denom = precision + recall;
        if denom < 1e-10 {
            return 0.0;
        }
        2.0 * precision * recall / denom
    }

    /// Specificity: TN / (TN + FP)
    fn specificity_from_cm(cm: &ConfusionMatrix) -> f64 {
        let denom = (cm.tn + cm.fp) as f64;
        if denom < 1e-10 {
            return 0.0;
        }
        cm.tn as f64 / denom
    }

    /// Matthews Correlation Coefficient
    fn mcc_from_cm(cm: &ConfusionMatrix) -> f64 {
        let tp = cm.tp as f64;
        let tn = cm.tn as f64;
        let fp = cm.fp as f64;
        let fn_ = cm.fn_ as f64;

        let denom = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
        if denom < 1e-10 {
            return 0.0;
        }
        (tp * tn - fp * fn_) / denom
    }

    /// AUC-ROC (Area Under ROC Curve)
    fn auc_roc(y_true: &Array1<f64>, y_proba: &Array1<f64>) -> f64 {
        // Using Mann-Whitney U statistic approach
        let n = y_true.len();

        // Pair up predictions with labels
        let mut pairs: Vec<(f64, bool)> = y_proba
            .iter()
            .zip(y_true.iter())
            .map(|(&p, &t)| (p, t >= 0.5))
            .collect();

        // Sort by prediction descending
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Count positive and negative samples
        let n_pos = pairs.iter().filter(|(_, t)| *t).count() as f64;
        let n_neg = pairs.iter().filter(|(_, t)| !*t).count() as f64;

        if n_pos < 1e-10 || n_neg < 1e-10 {
            return 0.5;
        }

        // Calculate AUC using trapezoid rule
        let mut tpr_prev = 0.0;
        let mut fpr_prev = 0.0;
        let mut auc = 0.0;
        let mut tp = 0.0;
        let mut fp = 0.0;

        let mut i = 0;
        while i < n {
            // Find all points with same score
            let score = pairs[i].0;
            let mut j = i;
            while j < n && (pairs[j].0 - score).abs() < 1e-10 {
                if pairs[j].1 {
                    tp += 1.0;
                } else {
                    fp += 1.0;
                }
                j += 1;
            }

            let tpr = tp / n_pos;
            let fpr = fp / n_neg;

            // Trapezoid area
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0;

            tpr_prev = tpr;
            fpr_prev = fpr;
            i = j;
        }

        auc
    }

    /// Log Loss (Binary Cross-Entropy)
    fn log_loss(y_true: &Array1<f64>, y_proba: &Array1<f64>) -> f64 {
        let eps = 1e-15;
        let n = y_true.len() as f64;

        -y_true
            .iter()
            .zip(y_proba.iter())
            .map(|(&t, &p)| {
                let p_clipped = p.clamp(eps, 1.0 - eps);
                t * p_clipped.ln() + (1.0 - t) * (1.0 - p_clipped).ln()
            })
            .sum::<f64>()
            / n
    }

    /// Print a summary report
    pub fn report(&self) -> String {
        let mut s = String::new();
        s.push_str("Classification Metrics Report\n");
        s.push_str("=============================\n\n");
        s.push_str(&self.confusion_matrix.display());
        s.push_str("\nMetrics:\n");
        s.push_str(&format!("  Accuracy:    {:.4}\n", self.accuracy));
        s.push_str(&format!("  Precision:   {:.4}\n", self.precision));
        s.push_str(&format!("  Recall:      {:.4}\n", self.recall));
        s.push_str(&format!("  F1 Score:    {:.4}\n", self.f1));
        s.push_str(&format!("  Specificity: {:.4}\n", self.specificity));
        s.push_str(&format!("  MCC:         {:.4}\n", self.mcc));

        if let Some(auc) = self.auc_roc {
            s.push_str(&format!("  AUC-ROC:     {:.4}\n", auc));
        }
        if let Some(ll) = self.log_loss {
            s.push_str(&format!("  Log Loss:    {:.4}\n", ll));
        }

        s
    }
}

/// Calculate precision at different recall thresholds
pub fn precision_recall_curve(
    y_true: &Array1<f64>,
    y_proba: &Array1<f64>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut pairs: Vec<(f64, bool)> = y_proba
        .iter()
        .zip(y_true.iter())
        .map(|(&p, &t)| (p, t >= 0.5))
        .collect();

    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let n_pos = pairs.iter().filter(|(_, t)| *t).count() as f64;

    let mut precisions = Vec::new();
    let mut recalls = Vec::new();
    let mut thresholds = Vec::new();

    let mut tp = 0.0;
    let mut fp = 0.0;

    for (prob, is_pos) in pairs {
        if is_pos {
            tp += 1.0;
        } else {
            fp += 1.0;
        }

        let precision = tp / (tp + fp);
        let recall = tp / n_pos;

        precisions.push(precision);
        recalls.push(recall);
        thresholds.push(prob);
    }

    (precisions, recalls, thresholds)
}

/// Calculate ROC curve points
pub fn roc_curve(
    y_true: &Array1<f64>,
    y_proba: &Array1<f64>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut pairs: Vec<(f64, bool)> = y_proba
        .iter()
        .zip(y_true.iter())
        .map(|(&p, &t)| (p, t >= 0.5))
        .collect();

    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let n_pos = pairs.iter().filter(|(_, t)| *t).count() as f64;
    let n_neg = pairs.iter().filter(|(_, t)| !*t).count() as f64;

    let mut tprs = vec![0.0];
    let mut fprs = vec![0.0];
    let mut thresholds = vec![f64::INFINITY];

    let mut tp = 0.0;
    let mut fp = 0.0;

    for (prob, is_pos) in pairs {
        if is_pos {
            tp += 1.0;
        } else {
            fp += 1.0;
        }

        tprs.push(tp / n_pos);
        fprs.push(fp / n_neg);
        thresholds.push(prob);
    }

    (fprs, tprs, thresholds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confusion_matrix() {
        let y_true = Array1::from_vec(vec![1.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
        let y_pred = Array1::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0]);

        let cm = ConfusionMatrix::from_predictions(&y_true, &y_pred);

        assert_eq!(cm.tp, 2); // Correctly predicted 1
        assert_eq!(cm.tn, 2); // Correctly predicted 0
        assert_eq!(cm.fp, 1); // Predicted 1, was 0
        assert_eq!(cm.fn_, 1); // Predicted 0, was 1
    }

    #[test]
    fn test_accuracy() {
        let y_true = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);
        let y_pred = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);

        let metrics = ClassificationMetrics::calculate(&y_true, &y_pred);
        assert!((metrics.accuracy - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_auc_perfect() {
        let y_true = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);
        let y_proba = Array1::from_vec(vec![0.1, 0.2, 0.8, 0.9]);

        let metrics = ClassificationMetrics::calculate_with_proba(
            &y_true,
            &y_proba.mapv(|p| if p >= 0.5 { 1.0 } else { 0.0 }),
            Some(&y_proba),
        );

        assert!((metrics.auc_roc.unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_f1_score() {
        let y_true = Array1::from_vec(vec![1.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
        let y_pred = Array1::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0]);

        let metrics = ClassificationMetrics::calculate(&y_true, &y_pred);

        // Precision = 2/3, Recall = 2/3, F1 = 2/3
        assert!((metrics.precision - 2.0 / 3.0).abs() < 1e-10);
        assert!((metrics.recall - 2.0 / 3.0).abs() < 1e-10);
        assert!((metrics.f1 - 2.0 / 3.0).abs() < 1e-10);
    }
}
