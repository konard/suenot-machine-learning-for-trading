//! Mutual Information calculation for feature selection
//!
//! Mutual information measures the dependency between two variables.
//! It's useful for selecting features that are most predictive of the target.

use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

/// Mutual information calculator
pub struct MutualInformation;

impl MutualInformation {
    /// Calculate mutual information between two continuous variables
    /// using binning (histogram) method
    ///
    /// # Arguments
    /// * `x` - First variable
    /// * `y` - Second variable
    /// * `n_bins` - Number of bins for discretization
    ///
    /// # Returns
    /// Mutual information in bits
    pub fn mutual_info_continuous(x: &[f64], y: &[f64], n_bins: usize) -> f64 {
        assert_eq!(x.len(), y.len(), "Arrays must have same length");

        if x.is_empty() {
            return 0.0;
        }

        // Discretize both variables
        let x_discrete = Self::discretize(x, n_bins);
        let y_discrete = Self::discretize(y, n_bins);

        Self::mutual_info_discrete(&x_discrete, &y_discrete)
    }

    /// Calculate mutual information between two discrete variables
    pub fn mutual_info_discrete(x: &[usize], y: &[usize]) -> f64 {
        let n = x.len() as f64;
        if n == 0.0 {
            return 0.0;
        }

        // Count joint occurrences
        let mut joint_counts: HashMap<(usize, usize), usize> = HashMap::new();
        let mut x_counts: HashMap<usize, usize> = HashMap::new();
        let mut y_counts: HashMap<usize, usize> = HashMap::new();

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            *joint_counts.entry((xi, yi)).or_insert(0) += 1;
            *x_counts.entry(xi).or_insert(0) += 1;
            *y_counts.entry(yi).or_insert(0) += 1;
        }

        // Calculate mutual information
        let mut mi = 0.0;

        for (&(xi, yi), &count) in &joint_counts {
            let p_xy = count as f64 / n;
            let p_x = x_counts[&xi] as f64 / n;
            let p_y = y_counts[&yi] as f64 / n;

            if p_xy > 0.0 && p_x > 0.0 && p_y > 0.0 {
                mi += p_xy * (p_xy / (p_x * p_y)).log2();
            }
        }

        mi.max(0.0)
    }

    /// Discretize continuous values into bins
    fn discretize(values: &[f64], n_bins: usize) -> Vec<usize> {
        if values.is_empty() {
            return vec![];
        }

        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max_val - min_val).abs() < 1e-10 {
            return vec![0; values.len()];
        }

        let bin_width = (max_val - min_val) / n_bins as f64;

        values
            .iter()
            .map(|&v| {
                let bin = ((v - min_val) / bin_width) as usize;
                bin.min(n_bins - 1)
            })
            .collect()
    }

    /// Calculate mutual information for all features against target
    ///
    /// # Arguments
    /// * `x` - Feature matrix (n_samples x n_features)
    /// * `y` - Target variable
    /// * `n_bins` - Number of bins for discretization
    ///
    /// # Returns
    /// Vector of mutual information values for each feature
    pub fn feature_mutual_info(x: &Array2<f64>, y: &Array1<f64>, n_bins: usize) -> Vec<f64> {
        let y_vec: Vec<f64> = y.to_vec();

        (0..x.ncols())
            .map(|i| {
                let feature: Vec<f64> = x.column(i).to_vec();
                Self::mutual_info_continuous(&feature, &y_vec, n_bins)
            })
            .collect()
    }

    /// Calculate entropy of a discrete distribution
    pub fn entropy_discrete(x: &[usize]) -> f64 {
        let n = x.len() as f64;
        if n == 0.0 {
            return 0.0;
        }

        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &val in x {
            *counts.entry(val).or_insert(0) += 1;
        }

        let mut entropy = 0.0;
        for &count in counts.values() {
            let p = count as f64 / n;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Calculate entropy of continuous variable using binning
    pub fn entropy_continuous(x: &[f64], n_bins: usize) -> f64 {
        let discrete = Self::discretize(x, n_bins);
        Self::entropy_discrete(&discrete)
    }

    /// Calculate normalized mutual information (0 to 1)
    pub fn normalized_mutual_info(x: &[f64], y: &[f64], n_bins: usize) -> f64 {
        let mi = Self::mutual_info_continuous(x, y, n_bins);
        let h_x = Self::entropy_continuous(x, n_bins);
        let h_y = Self::entropy_continuous(y, n_bins);

        let denominator = (h_x * h_y).sqrt();
        if denominator == 0.0 {
            0.0
        } else {
            mi / denominator
        }
    }

    /// Select top k features by mutual information
    pub fn select_top_features(
        x: &Array2<f64>,
        y: &Array1<f64>,
        k: usize,
        n_bins: usize,
    ) -> Vec<usize> {
        let mi_scores = Self::feature_mutual_info(x, y, n_bins);

        let mut indexed_scores: Vec<(usize, f64)> =
            mi_scores.into_iter().enumerate().collect();

        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        indexed_scores.into_iter().take(k).map(|(i, _)| i).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_entropy() {
        // Uniform distribution should have max entropy
        let uniform = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let entropy = MutualInformation::entropy_discrete(&uniform);
        assert!((entropy - 2.0).abs() < 0.01); // log2(4) = 2

        // Single value should have 0 entropy
        let constant = vec![1, 1, 1, 1];
        let entropy = MutualInformation::entropy_discrete(&constant);
        assert!(entropy.abs() < 0.01);
    }

    #[test]
    fn test_mutual_info_perfect_correlation() {
        // Identical variables should have MI equal to entropy
        let x = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let y = x.clone();

        let mi = MutualInformation::mutual_info_discrete(&x, &y);
        let entropy = MutualInformation::entropy_discrete(&x);

        assert!((mi - entropy).abs() < 0.01);
    }

    #[test]
    fn test_mutual_info_independent() {
        // Independent variables should have near-zero MI
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let y: Vec<f64> = (0..100).map(|i| (i as f64 * 0.37).cos()).collect();

        let mi = MutualInformation::mutual_info_continuous(&x, &y, 10);
        // MI should be relatively low for independent variables
        assert!(mi < 1.0);
    }

    #[test]
    fn test_feature_selection() {
        let x = array![
            [1.0, 0.1, 2.0],
            [2.0, 0.2, 4.0],
            [3.0, 0.3, 6.0],
            [4.0, 0.4, 8.0],
            [5.0, 0.5, 10.0]
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfectly correlated with feature 2

        let top_features = MutualInformation::select_top_features(&x, &y, 2, 5);
        assert_eq!(top_features.len(), 2);
        // Feature 2 should be first (highest MI with y)
        assert_eq!(top_features[0], 2);
    }
}
