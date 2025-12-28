//! Bias-Variance tradeoff analysis
//!
//! Demonstrates the fundamental tradeoff in ML:
//! - High bias (underfitting): Model is too simple
//! - High variance (overfitting): Model is too complex
//!
//! Uses polynomial regression to illustrate the concept.

use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Polynomial regression for bias-variance demonstration
pub struct PolynomialRegression {
    degree: usize,
    coefficients: Option<Array1<f64>>,
}

impl PolynomialRegression {
    /// Create a new polynomial regression model
    pub fn new(degree: usize) -> Self {
        Self {
            degree,
            coefficients: None,
        }
    }

    /// Generate polynomial features from input
    fn polynomial_features(&self, x: &Array1<f64>) -> Array2<f64> {
        let n = x.len();
        let mut features = Array2::zeros((n, self.degree + 1));

        for i in 0..n {
            for d in 0..=self.degree {
                features[[i, d]] = x[i].powi(d as i32);
            }
        }

        features
    }

    /// Fit using ordinary least squares
    pub fn fit(&mut self, x: &Array1<f64>, y: &Array1<f64>) {
        let features = self.polynomial_features(x);

        // Normal equation: (X^T X)^-1 X^T y
        let xt = features.t();
        let xtx = xt.dot(&features);
        let xty = xt.dot(y);

        // Simple matrix inversion using Gaussian elimination
        if let Some(xtx_inv) = Self::inverse(&xtx) {
            self.coefficients = Some(xtx_inv.dot(&xty));
        }
    }

    /// Predict using fitted model
    pub fn predict(&self, x: &Array1<f64>) -> Array1<f64> {
        let features = self.polynomial_features(x);
        let coeffs = self.coefficients.as_ref().expect("Model not fitted");

        features.dot(coeffs)
    }

    /// Simple matrix inversion (for small matrices)
    fn inverse(matrix: &Array2<f64>) -> Option<Array2<f64>> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return None;
        }

        // Create augmented matrix [A | I]
        let mut aug = Array2::zeros((n, 2 * n));
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = matrix[[i, j]];
            }
            aug[[i, n + i]] = 1.0;
        }

        // Gaussian elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            for row in (col + 1)..n {
                if aug[[row, col]].abs() > aug[[max_row, col]].abs() {
                    max_row = row;
                }
            }

            // Swap rows
            for j in 0..(2 * n) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }

            // Check for singular matrix
            if aug[[col, col]].abs() < 1e-10 {
                return None;
            }

            // Scale pivot row
            let pivot = aug[[col, col]];
            for j in 0..(2 * n) {
                aug[[col, j]] /= pivot;
            }

            // Eliminate column
            for row in 0..n {
                if row != col {
                    let factor = aug[[row, col]];
                    for j in 0..(2 * n) {
                        aug[[row, j]] -= factor * aug[[col, j]];
                    }
                }
            }
        }

        // Extract inverse
        let mut inv = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inv[[i, j]] = aug[[i, n + j]];
            }
        }

        Some(inv)
    }

    /// Get the degree of the polynomial
    pub fn get_degree(&self) -> usize {
        self.degree
    }
}

/// Bias-Variance analyzer
pub struct BiasVarianceAnalyzer;

impl BiasVarianceAnalyzer {
    /// Generate synthetic data from a true function with noise
    ///
    /// # Arguments
    /// * `true_fn` - The true underlying function
    /// * `x_range` - (min, max) range for x values
    /// * `n_samples` - Number of samples
    /// * `noise_std` - Standard deviation of Gaussian noise
    pub fn generate_data<F>(
        true_fn: F,
        x_range: (f64, f64),
        n_samples: usize,
        noise_std: f64,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>)
    where
        F: Fn(f64) -> f64,
    {
        let mut rng = rand::thread_rng();
        let noise_dist = Normal::new(0.0, noise_std).unwrap();

        let mut x = Vec::with_capacity(n_samples);
        let mut y_true = Vec::with_capacity(n_samples);
        let mut y_noisy = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let xi = rng.gen_range(x_range.0..x_range.1);
            let yi_true = true_fn(xi);
            let noise = noise_dist.sample(&mut rng);

            x.push(xi);
            y_true.push(yi_true);
            y_noisy.push(yi_true + noise);
        }

        (
            Array1::from_vec(x),
            Array1::from_vec(y_true),
            Array1::from_vec(y_noisy),
        )
    }

    /// Analyze bias-variance decomposition
    ///
    /// Runs multiple experiments with different training samples
    /// to estimate bias and variance components of error.
    ///
    /// # Arguments
    /// * `true_fn` - True underlying function
    /// * `degrees` - Polynomial degrees to test
    /// * `n_experiments` - Number of random experiments
    /// * `n_train` - Training set size per experiment
    /// * `n_test` - Test set size
    /// * `noise_std` - Noise standard deviation
    ///
    /// # Returns
    /// Vector of (degree, bias², variance, total_error) tuples
    pub fn analyze_bias_variance<F>(
        true_fn: F,
        degrees: &[usize],
        n_experiments: usize,
        n_train: usize,
        n_test: usize,
        noise_std: f64,
    ) -> Vec<(usize, f64, f64, f64)>
    where
        F: Fn(f64) -> f64 + Copy,
    {
        // Generate fixed test points
        let (x_test, y_test_true, _) = Self::generate_data(true_fn, (-1.0, 1.0), n_test, 0.0);

        let mut results = Vec::new();

        for &degree in degrees {
            let mut all_predictions: Vec<Array1<f64>> = Vec::new();

            // Run multiple experiments
            for _ in 0..n_experiments {
                let (x_train, _, y_train) =
                    Self::generate_data(true_fn, (-1.0, 1.0), n_train, noise_std);

                let mut model = PolynomialRegression::new(degree);
                model.fit(&x_train, &y_train);

                let predictions = model.predict(&x_test);
                all_predictions.push(predictions);
            }

            // Calculate mean prediction at each test point
            let mut mean_pred = Array1::zeros(n_test);
            for pred in &all_predictions {
                mean_pred = &mean_pred + pred;
            }
            mean_pred = &mean_pred / n_experiments as f64;

            // Bias² = E[(mean_prediction - true_value)²]
            let bias_sq: f64 = mean_pred
                .iter()
                .zip(y_test_true.iter())
                .map(|(&p, &t): (&f64, &f64)| (p - t).powi(2))
                .sum::<f64>()
                / n_test as f64;

            // Variance = E[(prediction - mean_prediction)²]
            let mut variance = 0.0;
            for pred in &all_predictions {
                for (i, &p) in pred.iter().enumerate() {
                    variance += (p - mean_pred[i]).powi(2);
                }
            }
            variance /= (n_experiments * n_test) as f64;

            // Irreducible error
            let irreducible_error = noise_std.powi(2);

            // Total expected error
            let total_error = bias_sq + variance + irreducible_error;

            results.push((degree, bias_sq, variance, total_error));
        }

        results
    }

    /// Calculate learning curve
    /// Shows how train/test error changes with training set size
    ///
    /// # Arguments
    /// * `train_sizes` - Vector of training set sizes to test
    /// * `x` - Feature data
    /// * `y` - Target data
    /// * `degree` - Polynomial degree
    ///
    /// # Returns
    /// Vector of (train_size, train_error, test_error) tuples
    pub fn learning_curve(
        train_sizes: &[usize],
        x: &Array1<f64>,
        y: &Array1<f64>,
        degree: usize,
    ) -> Vec<(usize, f64, f64)> {
        let n = x.len();
        let test_size = n / 5; // Use 20% for testing
        let x_test = x.slice(ndarray::s![(n - test_size)..]).to_owned();
        let y_test = y.slice(ndarray::s![(n - test_size)..]).to_owned();

        let mut results = Vec::new();

        for &size in train_sizes {
            if size > n - test_size {
                continue;
            }

            let x_train = x.slice(ndarray::s![..size]).to_owned();
            let y_train = y.slice(ndarray::s![..size]).to_owned();

            let mut model = PolynomialRegression::new(degree);
            model.fit(&x_train, &y_train);

            let train_pred = model.predict(&x_train);
            let test_pred = model.predict(&x_test);

            let train_mse = train_pred
                .iter()
                .zip(y_train.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / size as f64;

            let test_mse = test_pred
                .iter()
                .zip(y_test.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / test_size as f64;

            results.push((size, train_mse, test_mse));
        }

        results
    }

    /// Calculate validation curve
    /// Shows how train/test error changes with model complexity
    ///
    /// # Returns
    /// Vector of (degree, train_error, test_error) tuples
    pub fn validation_curve(
        degrees: &[usize],
        x: &Array1<f64>,
        y: &Array1<f64>,
        test_ratio: f64,
    ) -> Vec<(usize, f64, f64)> {
        let n = x.len();
        let test_size = (n as f64 * test_ratio) as usize;
        let train_size = n - test_size;

        let x_train = x.slice(ndarray::s![..train_size]).to_owned();
        let y_train = y.slice(ndarray::s![..train_size]).to_owned();
        let x_test = x.slice(ndarray::s![train_size..]).to_owned();
        let y_test = y.slice(ndarray::s![train_size..]).to_owned();

        let mut results = Vec::new();

        for &degree in degrees {
            let mut model = PolynomialRegression::new(degree);
            model.fit(&x_train, &y_train);

            let train_pred = model.predict(&x_train);
            let test_pred = model.predict(&x_test);

            let train_mse = train_pred
                .iter()
                .zip(y_train.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / train_size as f64;

            let test_mse = test_pred
                .iter()
                .zip(y_test.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / test_size as f64;

            results.push((degree, train_mse, test_mse));
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_regression() {
        // Linear data: y = 2x + 1
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let y = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0, 9.0]);

        let mut model = PolynomialRegression::new(1);
        model.fit(&x, &y);

        let pred = model.predict(&x);

        // Predictions should be close to true values
        for (p, t) in pred.iter().zip(y.iter()) {
            assert!((p - t).abs() < 0.01);
        }
    }

    #[test]
    fn test_generate_data() {
        let true_fn = |x: f64| x.sin();
        let (x, y_true, y_noisy) = BiasVarianceAnalyzer::generate_data(true_fn, (-1.0, 1.0), 100, 0.1);

        assert_eq!(x.len(), 100);
        assert_eq!(y_true.len(), 100);
        assert_eq!(y_noisy.len(), 100);

        // y_true should match true function
        for (xi, yi) in x.iter().zip(y_true.iter()) {
            assert!((yi - xi.sin()).abs() < 1e-10);
        }
    }

    #[test]
    fn test_bias_variance_analysis() {
        let true_fn = |x: f64| x.powi(2); // Quadratic function
        let results = BiasVarianceAnalyzer::analyze_bias_variance(
            true_fn,
            &[1, 2, 3],
            10,
            50,
            20,
            0.1,
        );

        assert_eq!(results.len(), 3);

        // Degree 2 should have lowest bias for quadratic function
        let (_, bias_1, _, _) = results[0]; // degree 1
        let (_, bias_2, _, _) = results[1]; // degree 2

        assert!(bias_2 < bias_1);
    }
}
