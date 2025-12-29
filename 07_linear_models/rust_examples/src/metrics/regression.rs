//! Regression metrics for evaluating model performance
//!
//! Provides common metrics for evaluating regression models on
//! financial prediction tasks.

use ndarray::Array1;

/// Collection of regression metrics
#[derive(Debug, Clone)]
pub struct RegressionMetrics {
    /// Mean Squared Error
    pub mse: f64,
    /// Root Mean Squared Error
    pub rmse: f64,
    /// Mean Absolute Error
    pub mae: f64,
    /// R-squared (coefficient of determination)
    pub r2: f64,
    /// Adjusted R-squared
    pub adj_r2: Option<f64>,
    /// Mean Absolute Percentage Error
    pub mape: Option<f64>,
    /// Information Coefficient (correlation)
    pub ic: f64,
    /// Number of samples
    pub n_samples: usize,
}

impl RegressionMetrics {
    /// Calculate all regression metrics
    pub fn calculate(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Self {
        Self::calculate_with_features(y_true, y_pred, None)
    }

    /// Calculate metrics with adjusted R² (requires number of features)
    pub fn calculate_with_features(
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
        n_features: Option<usize>,
    ) -> Self {
        let n = y_true.len();

        let mse = Self::mean_squared_error(y_true, y_pred);
        let rmse = mse.sqrt();
        let mae = Self::mean_absolute_error(y_true, y_pred);
        let r2 = Self::r_squared(y_true, y_pred);
        let ic = Self::information_coefficient(y_true, y_pred);
        let mape = Self::mean_absolute_percentage_error(y_true, y_pred);

        let adj_r2 = n_features.map(|p| {
            let n_f = n as f64;
            let p_f = p as f64;
            1.0 - (1.0 - r2) * (n_f - 1.0) / (n_f - p_f - 1.0)
        });

        Self {
            mse,
            rmse,
            mae,
            r2,
            adj_r2,
            mape,
            ic,
            n_samples: n,
        }
    }

    /// Mean Squared Error: (1/n) * Σ(y_true - y_pred)²
    pub fn mean_squared_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let n = y_true.len() as f64;
        y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&t, &p)| (t - p).powi(2))
            .sum::<f64>()
            / n
    }

    /// Root Mean Squared Error
    pub fn root_mean_squared_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        Self::mean_squared_error(y_true, y_pred).sqrt()
    }

    /// Mean Absolute Error: (1/n) * Σ|y_true - y_pred|
    pub fn mean_absolute_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let n = y_true.len() as f64;
        y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&t, &p)| (t - p).abs())
            .sum::<f64>()
            / n
    }

    /// R-squared (coefficient of determination)
    /// R² = 1 - SS_res / SS_tot
    pub fn r_squared(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let y_mean = y_true.mean().unwrap_or(0.0);

        let ss_res: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&t, &p)| (t - p).powi(2))
            .sum();

        let ss_tot: f64 = y_true.iter().map(|&t| (t - y_mean).powi(2)).sum();

        if ss_tot < 1e-10 {
            return 0.0;
        }

        1.0 - ss_res / ss_tot
    }

    /// Mean Absolute Percentage Error
    /// MAPE = (100/n) * Σ|y_true - y_pred| / |y_true|
    pub fn mean_absolute_percentage_error(
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
    ) -> Option<f64> {
        let valid_pairs: Vec<(f64, f64)> = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(&t, _)| t.abs() > 1e-10)
            .map(|(&t, &p)| (t, p))
            .collect();

        if valid_pairs.is_empty() {
            return None;
        }

        let mape = valid_pairs
            .iter()
            .map(|(t, p)| ((t - p) / t).abs())
            .sum::<f64>()
            / valid_pairs.len() as f64
            * 100.0;

        Some(mape)
    }

    /// Information Coefficient (Pearson correlation between predictions and actuals)
    /// This is a key metric in quantitative finance
    pub fn information_coefficient(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let n = y_true.len() as f64;

        let mean_true = y_true.mean().unwrap_or(0.0);
        let mean_pred = y_pred.mean().unwrap_or(0.0);

        let std_true = y_true.std(0.0);
        let std_pred = y_pred.std(0.0);

        if std_true < 1e-10 || std_pred < 1e-10 {
            return 0.0;
        }

        let cov: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&t, &p)| (t - mean_true) * (p - mean_pred))
            .sum::<f64>()
            / n;

        cov / (std_true * std_pred)
    }

    /// Rank Information Coefficient (Spearman correlation)
    pub fn rank_ic(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let ranks_true = Self::rank(y_true);
        let ranks_pred = Self::rank(y_pred);

        Self::information_coefficient(&ranks_true, &ranks_pred)
    }

    /// Calculate ranks (1-based, average for ties)
    fn rank(arr: &Array1<f64>) -> Array1<f64> {
        let n = arr.len();
        let mut indexed: Vec<(usize, f64)> = arr.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut ranks = vec![0.0; n];
        let mut i = 0;
        while i < n {
            let mut j = i;
            // Find all elements with same value
            while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-10 {
                j += 1;
            }
            // Average rank for ties
            let avg_rank = (i + j) as f64 / 2.0 + 0.5;
            for k in i..j {
                ranks[indexed[k].0] = avg_rank;
            }
            i = j;
        }

        Array1::from_vec(ranks)
    }

    /// Hit rate: percentage of predictions with correct direction
    pub fn hit_rate(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        let correct: usize = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(&t, &p)| (t > 0.0 && p > 0.0) || (t < 0.0 && p < 0.0) || (t == 0.0 && p == 0.0))
            .count();

        correct as f64 / y_true.len() as f64
    }

    /// Profit factor (sum of positive predictions / abs sum of negative)
    /// when predictions are used as position sizes
    pub fn profit_factor(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Option<f64> {
        let pnl: Vec<f64> = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&t, &p)| t * p.signum())
            .collect();

        let gains: f64 = pnl.iter().filter(|&&x| x > 0.0).sum();
        let losses: f64 = pnl.iter().filter(|&&x| x < 0.0).sum::<f64>().abs();

        if losses < 1e-10 {
            return None;
        }

        Some(gains / losses)
    }

    /// Print a summary report
    pub fn report(&self) -> String {
        let mut s = String::new();
        s.push_str("Regression Metrics Report\n");
        s.push_str("=========================\n\n");
        s.push_str(&format!("Samples:     {}\n\n", self.n_samples));
        s.push_str("Error Metrics:\n");
        s.push_str(&format!("  MSE:       {:.6}\n", self.mse));
        s.push_str(&format!("  RMSE:      {:.6}\n", self.rmse));
        s.push_str(&format!("  MAE:       {:.6}\n", self.mae));
        if let Some(mape) = self.mape {
            s.push_str(&format!("  MAPE:      {:.2}%\n", mape));
        }
        s.push_str("\nGoodness of Fit:\n");
        s.push_str(&format!("  R²:        {:.6}\n", self.r2));
        if let Some(adj_r2) = self.adj_r2 {
            s.push_str(&format!("  Adj R²:    {:.6}\n", adj_r2));
        }
        s.push_str("\nPredictive Power:\n");
        s.push_str(&format!("  IC:        {:.6}\n", self.ic));
        s
    }
}

/// Time series cross-validation
pub struct TimeSeriesCV {
    /// Number of splits
    n_splits: usize,
    /// Minimum training size
    min_train_size: usize,
    /// Test size for each split
    test_size: usize,
    /// Gap between train and test
    gap: usize,
}

impl TimeSeriesCV {
    /// Create a new time series cross-validator
    pub fn new(n_splits: usize, min_train_size: usize, test_size: usize, gap: usize) -> Self {
        Self {
            n_splits,
            min_train_size,
            test_size,
            gap,
        }
    }

    /// Generate train/test indices for each split
    pub fn split(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut splits = Vec::new();

        let total_test = self.n_splits * self.test_size;
        let total_gap = self.n_splits * self.gap;

        if n_samples < self.min_train_size + total_test + total_gap {
            return splits;
        }

        for i in 0..self.n_splits {
            let test_end = n_samples - i * self.test_size;
            let test_start = test_end - self.test_size;
            let train_end = test_start - self.gap;
            let train_start = 0;

            if train_end <= self.min_train_size {
                break;
            }

            let train_indices: Vec<usize> = (train_start..train_end).collect();
            let test_indices: Vec<usize> = (test_start..test_end).collect();

            splits.push((train_indices, test_indices));
        }

        splits.reverse();
        splits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse() {
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let mse = RegressionMetrics::mean_squared_error(&y_true, &y_pred);
        assert!((mse - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_r_squared_perfect() {
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let r2 = RegressionMetrics::r_squared(&y_true, &y_pred);
        assert!((r2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_information_coefficient() {
        let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y_pred = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let ic = RegressionMetrics::information_coefficient(&y_true, &y_pred);
        assert!((ic - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hit_rate() {
        let y_true = Array1::from_vec(vec![0.1, -0.1, 0.2, -0.2, 0.1]);
        let y_pred = Array1::from_vec(vec![0.05, -0.05, 0.1, 0.1, -0.05]); // 3 correct out of 5

        let hr = RegressionMetrics::hit_rate(&y_true, &y_pred);
        assert!((hr - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_timeseries_cv() {
        let cv = TimeSeriesCV::new(3, 10, 5, 1);
        let splits = cv.split(30);

        assert_eq!(splits.len(), 3);

        // Check that train comes before test
        for (train, test) in &splits {
            assert!(train.last().unwrap() < test.first().unwrap());
        }
    }
}
