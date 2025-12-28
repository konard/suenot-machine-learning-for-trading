//! Evaluation metrics for ML models
//!
//! Includes metrics for:
//! - Classification: accuracy, precision, recall, F1, confusion matrix
//! - Regression: MSE, RMSE, MAE, R²

use ndarray::Array1;
use std::collections::HashMap;

/// Metrics calculator
pub struct Metrics;

impl Metrics {
    // ==================== Classification Metrics ====================

    /// Calculate accuracy: (correct predictions) / (total predictions)
    pub fn accuracy(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        assert_eq!(y_true.len(), y_pred.len(), "Arrays must have same length");

        if y_true.is_empty() {
            return 0.0;
        }

        let correct = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(t, p)| (*t - *p).abs() < 1e-10)
            .count();

        correct as f64 / y_true.len() as f64
    }

    /// Calculate precision for binary classification
    /// precision = TP / (TP + FP)
    pub fn precision(y_true: &Array1<f64>, y_pred: &Array1<f64>, positive_class: f64) -> f64 {
        let (tp, fp, _, _) = Self::confusion_matrix_values(y_true, y_pred, positive_class);

        if tp + fp == 0 {
            0.0
        } else {
            tp as f64 / (tp + fp) as f64
        }
    }

    /// Calculate recall for binary classification
    /// recall = TP / (TP + FN)
    pub fn recall(y_true: &Array1<f64>, y_pred: &Array1<f64>, positive_class: f64) -> f64 {
        let (tp, _, fn_, _) = Self::confusion_matrix_values(y_true, y_pred, positive_class);

        if tp + fn_ == 0 {
            0.0
        } else {
            tp as f64 / (tp + fn_) as f64
        }
    }

    /// Calculate F1 score for binary classification
    /// F1 = 2 * (precision * recall) / (precision + recall)
    pub fn f1_score(y_true: &Array1<f64>, y_pred: &Array1<f64>, positive_class: f64) -> f64 {
        let precision = Self::precision(y_true, y_pred, positive_class);
        let recall = Self::recall(y_true, y_pred, positive_class);

        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }

    /// Calculate confusion matrix values (TP, FP, FN, TN)
    fn confusion_matrix_values(
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
        positive_class: f64,
    ) -> (usize, usize, usize, usize) {
        let mut tp = 0;
        let mut fp = 0;
        let mut fn_ = 0;
        let mut tn = 0;

        for (t, p) in y_true.iter().zip(y_pred.iter()) {
            let is_true_positive = (*t - positive_class).abs() < 1e-10;
            let is_pred_positive = (*p - positive_class).abs() < 1e-10;

            match (is_true_positive, is_pred_positive) {
                (true, true) => tp += 1,
                (false, true) => fp += 1,
                (true, false) => fn_ += 1,
                (false, false) => tn += 1,
            }
        }

        (tp, fp, fn_, tn)
    }

    /// Generate a full confusion matrix
    pub fn confusion_matrix(
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
    ) -> HashMap<(i64, i64), usize> {
        let mut matrix: HashMap<(i64, i64), usize> = HashMap::new();

        for (t, p) in y_true.iter().zip(y_pred.iter()) {
            let key = (*t as i64, *p as i64);
            *matrix.entry(key).or_insert(0) += 1;
        }

        matrix
    }

    /// Calculate classification report
    pub fn classification_report(
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
    ) -> Vec<(f64, f64, f64, f64, usize)> {
        // (class, precision, recall, f1, support)
        let mut classes: Vec<f64> = y_true.iter().cloned().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        classes.dedup();

        classes
            .iter()
            .map(|&class| {
                let precision = Self::precision(y_true, y_pred, class);
                let recall = Self::recall(y_true, y_pred, class);
                let f1 = Self::f1_score(y_true, y_pred, class);
                let support = y_true.iter().filter(|&&t| (t - class).abs() < 1e-10).count();
                (class, precision, recall, f1, support)
            })
            .collect()
    }

    // ==================== Regression Metrics ====================

    /// Mean Squared Error
    pub fn mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        assert_eq!(y_true.len(), y_pred.len(), "Arrays must have same length");

        if y_true.is_empty() {
            return 0.0;
        }

        y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f64>()
            / y_true.len() as f64
    }

    /// Root Mean Squared Error
    pub fn rmse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        Self::mse(y_true, y_pred).sqrt()
    }

    /// Mean Absolute Error
    pub fn mae(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        assert_eq!(y_true.len(), y_pred.len(), "Arrays must have same length");

        if y_true.is_empty() {
            return 0.0;
        }

        y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p).abs())
            .sum::<f64>()
            / y_true.len() as f64
    }

    /// R² (coefficient of determination)
    pub fn r2_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        assert_eq!(y_true.len(), y_pred.len(), "Arrays must have same length");

        if y_true.is_empty() {
            return 0.0;
        }

        let mean = y_true.mean().unwrap_or(0.0);

        let ss_res: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum();

        let ss_tot: f64 = y_true.iter().map(|t| (t - mean).powi(2)).sum();

        if ss_tot == 0.0 {
            0.0
        } else {
            1.0 - ss_res / ss_tot
        }
    }

    /// Mean Absolute Percentage Error
    pub fn mape(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        assert_eq!(y_true.len(), y_pred.len(), "Arrays must have same length");

        let valid: Vec<(f64, f64)> = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(t, _)| t.abs() > 1e-10)
            .map(|(&t, &p)| (t, p))
            .collect();

        if valid.is_empty() {
            return 0.0;
        }

        valid
            .iter()
            .map(|(t, p)| ((t - p) / t).abs())
            .sum::<f64>()
            / valid.len() as f64
            * 100.0
    }

    // ==================== Trading-Specific Metrics ====================

    /// Directional accuracy: how often the sign of prediction matches truth
    pub fn directional_accuracy(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        assert_eq!(y_true.len(), y_pred.len(), "Arrays must have same length");

        if y_true.is_empty() {
            return 0.0;
        }

        let correct = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(t, p)| t.signum() == p.signum())
            .count();

        correct as f64 / y_true.len() as f64
    }

    /// Hit ratio: percentage of profitable trades
    pub fn hit_ratio(returns: &Array1<f64>) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let profitable = returns.iter().filter(|&&r| r > 0.0).count();
        profitable as f64 / returns.len() as f64
    }

    /// Profit factor: gross profits / gross losses
    pub fn profit_factor(returns: &Array1<f64>) -> f64 {
        let gross_profit: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
        let gross_loss: f64 = returns.iter().filter(|&&r| r < 0.0).map(|r| r.abs()).sum();

        if gross_loss == 0.0 {
            f64::INFINITY
        } else {
            gross_profit / gross_loss
        }
    }

    /// Sharpe ratio (annualized)
    pub fn sharpe_ratio(returns: &Array1<f64>, risk_free_rate: f64, periods_per_year: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean_return = returns.mean().unwrap_or(0.0);
        let std_return = returns.std(0.0);

        if std_return == 0.0 {
            0.0
        } else {
            (mean_return - risk_free_rate / periods_per_year) / std_return * periods_per_year.sqrt()
        }
    }

    /// Maximum drawdown
    pub fn max_drawdown(cumulative_returns: &Array1<f64>) -> f64 {
        if cumulative_returns.is_empty() {
            return 0.0;
        }

        let mut max_val: f64 = f64::NEG_INFINITY;
        let mut max_dd: f64 = 0.0;

        for &val in cumulative_returns.iter() {
            max_val = max_val.max(val);
            let dd = (max_val - val) / max_val;
            max_dd = max_dd.max(dd);
        }

        max_dd
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_accuracy() {
        let y_true = array![0.0, 1.0, 1.0, 0.0, 1.0];
        let y_pred = array![0.0, 1.0, 0.0, 0.0, 1.0];

        let acc = Metrics::accuracy(&y_true, &y_pred);
        assert!((acc - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_precision_recall() {
        let y_true = array![1.0, 1.0, 1.0, 0.0, 0.0];
        let y_pred = array![1.0, 1.0, 0.0, 1.0, 0.0];

        // TP=2, FP=1, FN=1, TN=1
        let precision = Metrics::precision(&y_true, &y_pred, 1.0);
        let recall = Metrics::recall(&y_true, &y_pred, 1.0);

        assert!((precision - 2.0 / 3.0).abs() < 1e-10);
        assert!((recall - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mse() {
        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = array![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!(Metrics::mse(&y_true, &y_pred) < 1e-10);

        let y_pred2 = array![2.0, 3.0, 4.0, 5.0, 6.0]; // off by 1
        assert!((Metrics::mse(&y_true, &y_pred2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_r2_score() {
        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let r2 = Metrics::r2_score(&y_true, &y_pred);
        assert!((r2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sharpe_ratio() {
        let returns = array![0.01, 0.02, -0.01, 0.03, 0.01];
        let sharpe = Metrics::sharpe_ratio(&returns, 0.0, 252.0);
        assert!(sharpe > 0.0); // Should be positive for mostly positive returns
    }
}
