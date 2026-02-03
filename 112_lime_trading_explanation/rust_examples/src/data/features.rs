//! Feature engineering utilities for trading data.

use ndarray::{Array1, Array2};

/// Feature engineering utilities
pub struct FeatureEngineering;

impl FeatureEngineering {
    /// Normalize features using z-score normalization
    ///
    /// # Arguments
    /// * `features` - Feature matrix (n_samples x n_features)
    ///
    /// # Returns
    /// Tuple of (normalized features, means, stds)
    pub fn normalize(features: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let n_features = features.ncols();
        let mut means = Array1::zeros(n_features);
        let mut stds = Array1::zeros(n_features);

        for j in 0..n_features {
            let col = features.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let variance = col.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(1.0);
            let std = variance.sqrt().max(1e-10);

            means[j] = mean;
            stds[j] = std;
        }

        let mut normalized = features.clone();
        for j in 0..n_features {
            for i in 0..features.nrows() {
                normalized[[i, j]] = (features[[i, j]] - means[j]) / stds[j];
            }
        }

        (normalized, means, stds)
    }

    /// Apply normalization using pre-computed parameters
    pub fn apply_normalization(
        features: &Array2<f64>,
        means: &Array1<f64>,
        stds: &Array1<f64>,
    ) -> Array2<f64> {
        let mut normalized = features.clone();
        let n_features = features.ncols();

        for j in 0..n_features {
            for i in 0..features.nrows() {
                normalized[[i, j]] = (features[[i, j]] - means[j]) / stds[j];
            }
        }

        normalized
    }

    /// Split data into train and test sets
    ///
    /// # Arguments
    /// * `features` - Feature matrix
    /// * `targets` - Target vector
    /// * `train_ratio` - Fraction of data to use for training
    ///
    /// # Returns
    /// Tuple of (X_train, X_test, y_train, y_test)
    pub fn train_test_split(
        features: &Array2<f64>,
        targets: &Array1<f64>,
        train_ratio: f64,
    ) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
        let n_samples = features.nrows();
        let train_size = (n_samples as f64 * train_ratio) as usize;

        let x_train = features.slice(ndarray::s![..train_size, ..]).to_owned();
        let x_test = features.slice(ndarray::s![train_size.., ..]).to_owned();
        let y_train = targets.slice(ndarray::s![..train_size]).to_owned();
        let y_test = targets.slice(ndarray::s![train_size..]).to_owned();

        (x_train, x_test, y_train, y_test)
    }

    /// Compute feature statistics
    pub fn feature_statistics(features: &Array2<f64>, feature_names: &[String]) -> Vec<FeatureStats> {
        let mut stats = Vec::new();

        for (j, name) in feature_names.iter().enumerate() {
            let col = features.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let min = col.fold(f64::INFINITY, |a, &b| a.min(b));
            let max = col.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let variance = col.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
            let std = variance.sqrt();

            stats.push(FeatureStats {
                name: name.clone(),
                mean,
                std,
                min,
                max,
            });
        }

        stats
    }
}

/// Statistics for a single feature
#[derive(Debug, Clone)]
pub struct FeatureStats {
    pub name: String,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

impl std::fmt::Display for FeatureStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: mean={:.4}, std={:.4}, min={:.4}, max={:.4}",
            self.name, self.mean, self.std, self.min, self.max
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_normalize() {
        let features = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let (normalized, means, stds) = FeatureEngineering::normalize(&features);

        // Check means are approximately 0
        for j in 0..2 {
            let col_mean: f64 = normalized.column(j).mean().unwrap();
            assert!(col_mean.abs() < 1e-10);
        }

        // Check stds are approximately 1
        for j in 0..2 {
            let col = normalized.column(j);
            let mean = col.mean().unwrap();
            let variance: f64 = col.mapv(|x| (x - mean).powi(2)).mean().unwrap();
            assert!((variance.sqrt() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_train_test_split() {
        let features = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let targets = array![0.0, 1.0, 0.0, 1.0, 0.0];

        let (x_train, x_test, y_train, y_test) =
            FeatureEngineering::train_test_split(&features, &targets, 0.6);

        assert_eq!(x_train.nrows(), 3);
        assert_eq!(x_test.nrows(), 2);
        assert_eq!(y_train.len(), 3);
        assert_eq!(y_test.len(), 2);
    }
}
