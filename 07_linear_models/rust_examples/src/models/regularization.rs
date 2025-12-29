//! Regularized Linear Regression: Ridge and Lasso
//!
//! This module provides regularized regression models that help prevent
//! overfitting by penalizing large coefficients.

use ndarray::{s, Array1, Array2, Axis};
use thiserror::Error;

/// Errors for regularized regression models
#[derive(Error, Debug)]
pub enum RegularizationError {
    #[error("Model has not been fitted yet")]
    NotFitted,

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Invalid alpha value: {0}")]
    InvalidAlpha(f64),

    #[error("Computation error: {0}")]
    ComputationError(String),
}

/// Ridge Regression (L2 regularization)
///
/// Minimizes: ||y - Xβ||² + α||β||²
///
/// Ridge regression shrinks coefficients but doesn't set them to zero.
#[derive(Debug, Clone)]
pub struct RidgeRegression {
    /// Regularization strength (alpha)
    alpha: f64,
    /// Fitted coefficients
    pub coefficients: Option<Array1<f64>>,
    /// Intercept term
    pub intercept: Option<f64>,
    /// Whether to fit intercept
    fit_intercept: bool,
    /// Whether to normalize features
    normalize: bool,
}

impl Default for RidgeRegression {
    fn default() -> Self {
        Self::new(1.0, true, false)
    }
}

impl RidgeRegression {
    /// Create a new Ridge Regression model
    ///
    /// # Arguments
    /// * `alpha` - Regularization strength (higher = more regularization)
    /// * `fit_intercept` - Whether to calculate the intercept
    /// * `normalize` - Whether to normalize features before fitting
    pub fn new(alpha: f64, fit_intercept: bool, normalize: bool) -> Self {
        Self {
            alpha,
            coefficients: None,
            intercept: None,
            fit_intercept,
            normalize,
        }
    }

    /// Fit the Ridge regression model
    ///
    /// Uses the closed-form solution: β = (X'X + αI)^(-1) X'y
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), RegularizationError> {
        if self.alpha < 0.0 {
            return Err(RegularizationError::InvalidAlpha(self.alpha));
        }

        let (x_processed, y_processed, x_mean, y_mean, x_std) = if self.normalize {
            self.preprocess(x, y)
        } else if self.fit_intercept {
            let x_mean = x.mean_axis(Axis(0)).unwrap();
            let y_mean = y.mean().unwrap();
            let x_centered = x - &x_mean;
            let y_centered = y - y_mean;
            (
                x_centered,
                y_centered,
                x_mean,
                y_mean,
                Array1::ones(x.ncols()),
            )
        } else {
            (
                x.clone(),
                y.clone(),
                Array1::zeros(x.ncols()),
                0.0,
                Array1::ones(x.ncols()),
            )
        };

        let n_features = x_processed.ncols();

        // Compute X'X + αI
        let xtx = x_processed.t().dot(&x_processed);
        let mut xtx_reg = xtx.clone();

        // Add regularization to diagonal
        for i in 0..n_features {
            xtx_reg[[i, i]] += self.alpha;
        }

        // Compute X'y
        let xty = x_processed.t().dot(&y_processed);

        // Solve the system
        let coefficients = self.solve(&xtx_reg, &xty)?;

        // Rescale coefficients if normalized
        let final_coef = if self.normalize {
            &coefficients / &x_std
        } else {
            coefficients
        };

        // Calculate intercept
        let intercept = if self.fit_intercept {
            y_mean - x_mean.dot(&final_coef)
        } else {
            0.0
        };

        self.coefficients = Some(final_coef);
        self.intercept = Some(intercept);

        Ok(())
    }

    /// Preprocess data (center and scale)
    fn preprocess(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> (Array2<f64>, Array1<f64>, Array1<f64>, f64, Array1<f64>) {
        let x_mean = x.mean_axis(Axis(0)).unwrap();
        let y_mean = y.mean().unwrap();

        let x_std = x.std_axis(Axis(0), 0.0);
        let x_std = x_std.mapv(|s| if s < 1e-10 { 1.0 } else { s });

        let x_centered = x - &x_mean;
        let mut x_scaled = Array2::zeros(x.raw_dim());
        for (j, mut col) in x_scaled.columns_mut().into_iter().enumerate() {
            for (i, val) in col.iter_mut().enumerate() {
                *val = x_centered[[i, j]] / x_std[j];
            }
        }

        let y_centered = y - y_mean;

        (x_scaled, y_centered, x_mean, y_mean, x_std)
    }

    /// Solve the regularized system using Cholesky decomposition
    fn solve(
        &self,
        a: &Array2<f64>,
        b: &Array1<f64>,
    ) -> Result<Array1<f64>, RegularizationError> {
        let n = a.nrows();

        // Cholesky decomposition
        let mut l = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }

                if i == j {
                    let diag = a[[i, i]] - sum;
                    if diag <= 0.0 {
                        return Err(RegularizationError::ComputationError(
                            "Matrix not positive definite".to_string(),
                        ));
                    }
                    l[[i, j]] = diag.sqrt();
                } else {
                    l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
                }
            }
        }

        // Forward substitution: L * z = b
        let mut z = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l[[i, j]] * z[j];
            }
            z[i] = (b[i] - sum) / l[[i, i]];
        }

        // Backward substitution: L' * x = z
        let mut x = Array1::<f64>::zeros(n);
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += l[[j, i]] * x[j];
            }
            x[i] = (z[i] - sum) / l[[i, i]];
        }

        Ok(x)
    }

    /// Make predictions
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, RegularizationError> {
        let coef = self
            .coefficients
            .as_ref()
            .ok_or(RegularizationError::NotFitted)?;
        let intercept = self.intercept.ok_or(RegularizationError::NotFitted)?;

        Ok(x.dot(coef) + intercept)
    }

    /// Get the regularization path for different alpha values
    pub fn regularization_path(
        x: &Array2<f64>,
        y: &Array1<f64>,
        alphas: &[f64],
    ) -> Result<Vec<(f64, Array1<f64>)>, RegularizationError> {
        let mut path = Vec::new();

        for &alpha in alphas {
            let mut model = RidgeRegression::new(alpha, true, false);
            model.fit(x, y)?;

            if let Some(coef) = model.coefficients {
                path.push((alpha, coef));
            }
        }

        Ok(path)
    }
}

/// Lasso Regression (L1 regularization)
///
/// Minimizes: (1/2n)||y - Xβ||² + α||β||₁
///
/// Lasso can shrink some coefficients exactly to zero (feature selection).
#[derive(Debug, Clone)]
pub struct LassoRegression {
    /// Regularization strength
    alpha: f64,
    /// Fitted coefficients
    pub coefficients: Option<Array1<f64>>,
    /// Intercept term
    pub intercept: Option<f64>,
    /// Whether to fit intercept
    fit_intercept: bool,
    /// Maximum iterations
    max_iter: usize,
    /// Tolerance for convergence
    tolerance: f64,
}

impl Default for LassoRegression {
    fn default() -> Self {
        Self::new(1.0, true, 1000, 1e-4)
    }
}

impl LassoRegression {
    /// Create a new Lasso Regression model
    pub fn new(alpha: f64, fit_intercept: bool, max_iter: usize, tolerance: f64) -> Self {
        Self {
            alpha,
            coefficients: None,
            intercept: None,
            fit_intercept,
            max_iter,
            tolerance,
        }
    }

    /// Fit using coordinate descent
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), RegularizationError> {
        if self.alpha < 0.0 {
            return Err(RegularizationError::InvalidAlpha(self.alpha));
        }

        let n_samples = x.nrows() as f64;
        let n_features = x.ncols();

        // Center data if fitting intercept
        let (x_centered, y_centered, x_mean, y_mean) = if self.fit_intercept {
            let x_mean = x.mean_axis(Axis(0)).unwrap();
            let y_mean = y.mean().unwrap();
            let x_c = x - &x_mean;
            let y_c = y - y_mean;
            (x_c, y_c, x_mean, y_mean)
        } else {
            (x.clone(), y.clone(), Array1::zeros(n_features), 0.0)
        };

        // Precompute X'X diagonal for efficiency
        let x_squared_sum: Vec<f64> = x_centered
            .columns()
            .into_iter()
            .map(|col| col.iter().map(|&v| v * v).sum())
            .collect();

        // Initialize coefficients
        let mut coef = Array1::<f64>::zeros(n_features);

        // Coordinate descent
        for _iter in 0..self.max_iter {
            let coef_old = coef.clone();

            for j in 0..n_features {
                // Compute residual without feature j
                let mut residual = y_centered.clone();
                for k in 0..n_features {
                    if k != j {
                        residual = &residual - &(&x_centered.column(k) * coef[k]);
                    }
                }

                // Compute correlation
                let rho: f64 = x_centered.column(j).dot(&residual);

                // Soft thresholding
                let threshold = self.alpha * n_samples;
                coef[j] = if x_squared_sum[j] > 1e-10 {
                    Self::soft_threshold(rho, threshold) / x_squared_sum[j]
                } else {
                    0.0
                };
            }

            // Check convergence
            let diff: f64 = coef
                .iter()
                .zip(coef_old.iter())
                .map(|(&a, &b)| (a - b).abs())
                .sum();

            if diff < self.tolerance {
                break;
            }
        }

        // Calculate intercept
        let intercept = if self.fit_intercept {
            y_mean - x_mean.dot(&coef)
        } else {
            0.0
        };

        self.coefficients = Some(coef);
        self.intercept = Some(intercept);

        Ok(())
    }

    /// Soft thresholding operator
    fn soft_threshold(x: f64, lambda: f64) -> f64 {
        if x > lambda {
            x - lambda
        } else if x < -lambda {
            x + lambda
        } else {
            0.0
        }
    }

    /// Make predictions
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, RegularizationError> {
        let coef = self
            .coefficients
            .as_ref()
            .ok_or(RegularizationError::NotFitted)?;
        let intercept = self.intercept.ok_or(RegularizationError::NotFitted)?;

        Ok(x.dot(coef) + intercept)
    }

    /// Get number of non-zero coefficients
    pub fn n_nonzero(&self) -> usize {
        self.coefficients
            .as_ref()
            .map(|c| c.iter().filter(|&&v| v.abs() > 1e-10).count())
            .unwrap_or(0)
    }

    /// Get indices of selected features
    pub fn selected_features(&self) -> Vec<usize> {
        self.coefficients
            .as_ref()
            .map(|c| {
                c.iter()
                    .enumerate()
                    .filter(|(_, &v)| v.abs() > 1e-10)
                    .map(|(i, _)| i)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get the regularization path
    pub fn regularization_path(
        x: &Array2<f64>,
        y: &Array1<f64>,
        alphas: &[f64],
    ) -> Result<Vec<(f64, Array1<f64>, usize)>, RegularizationError> {
        let mut path = Vec::new();

        for &alpha in alphas {
            let mut model = LassoRegression::new(alpha, true, 1000, 1e-4);
            model.fit(x, y)?;

            if let Some(coef) = model.coefficients.clone() {
                let n_nonzero = model.n_nonzero();
                path.push((alpha, coef, n_nonzero));
            }
        }

        Ok(path)
    }
}

/// Elastic Net Regression (L1 + L2 regularization)
///
/// Minimizes: (1/2n)||y - Xβ||² + α * l1_ratio * ||β||₁ + α * (1 - l1_ratio) / 2 * ||β||²
#[derive(Debug, Clone)]
pub struct ElasticNet {
    /// Overall regularization strength
    alpha: f64,
    /// Balance between L1 and L2 (0 = Ridge, 1 = Lasso)
    l1_ratio: f64,
    /// Fitted coefficients
    pub coefficients: Option<Array1<f64>>,
    /// Intercept
    pub intercept: Option<f64>,
    /// Whether to fit intercept
    fit_intercept: bool,
    /// Maximum iterations
    max_iter: usize,
    /// Convergence tolerance
    tolerance: f64,
}

impl Default for ElasticNet {
    fn default() -> Self {
        Self::new(1.0, 0.5, true, 1000, 1e-4)
    }
}

impl ElasticNet {
    /// Create a new Elastic Net model
    pub fn new(
        alpha: f64,
        l1_ratio: f64,
        fit_intercept: bool,
        max_iter: usize,
        tolerance: f64,
    ) -> Self {
        Self {
            alpha,
            l1_ratio: l1_ratio.clamp(0.0, 1.0),
            coefficients: None,
            intercept: None,
            fit_intercept,
            max_iter,
            tolerance,
        }
    }

    /// Fit using coordinate descent
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), RegularizationError> {
        let n_samples = x.nrows() as f64;
        let n_features = x.ncols();

        // Center data
        let (x_centered, y_centered, x_mean, y_mean) = if self.fit_intercept {
            let x_mean = x.mean_axis(Axis(0)).unwrap();
            let y_mean = y.mean().unwrap();
            (x - &x_mean, y - y_mean, x_mean, y_mean)
        } else {
            (x.clone(), y.clone(), Array1::zeros(n_features), 0.0)
        };

        let l1_penalty = self.alpha * self.l1_ratio * n_samples;
        let l2_penalty = self.alpha * (1.0 - self.l1_ratio) * n_samples;

        // Precompute
        let x_squared_sum: Vec<f64> = x_centered
            .columns()
            .into_iter()
            .map(|col| col.iter().map(|&v| v * v).sum::<f64>() + l2_penalty)
            .collect();

        let mut coef = Array1::<f64>::zeros(n_features);

        for _iter in 0..self.max_iter {
            let coef_old = coef.clone();

            for j in 0..n_features {
                let mut residual = y_centered.clone();
                for k in 0..n_features {
                    if k != j {
                        residual = &residual - &(&x_centered.column(k) * coef[k]);
                    }
                }

                let rho: f64 = x_centered.column(j).dot(&residual);

                coef[j] = if x_squared_sum[j] > 1e-10 {
                    LassoRegression::soft_threshold(rho, l1_penalty) / x_squared_sum[j]
                } else {
                    0.0
                };
            }

            let diff: f64 = coef
                .iter()
                .zip(coef_old.iter())
                .map(|(&a, &b)| (a - b).abs())
                .sum();

            if diff < self.tolerance {
                break;
            }
        }

        let intercept = if self.fit_intercept {
            y_mean - x_mean.dot(&coef)
        } else {
            0.0
        };

        self.coefficients = Some(coef);
        self.intercept = Some(intercept);

        Ok(())
    }

    /// Make predictions
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, RegularizationError> {
        let coef = self
            .coefficients
            .as_ref()
            .ok_or(RegularizationError::NotFitted)?;
        let intercept = self.intercept.ok_or(RegularizationError::NotFitted)?;

        Ok(x.dot(coef) + intercept)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ridge_regression() {
        let x = Array2::from_shape_vec((5, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0])
            .unwrap();
        let y = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]);

        let mut model = RidgeRegression::new(0.1, true, false);
        model.fit(&x, &y).unwrap();

        let predictions = model.predict(&x).unwrap();

        // Predictions should be close to actual values
        for (&pred, &actual) in predictions.iter().zip(y.iter()) {
            assert!((pred - actual).abs() < 1.0);
        }
    }

    #[test]
    fn test_lasso_sparsity() {
        // Create data where one feature is irrelevant
        let x = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 0.1, 0.5, 2.0, 0.2, 0.6, 3.0, 0.3, 0.7, 4.0, 0.4, 0.8, 5.0, 0.5, 0.9,
                6.0, 0.6, 1.0, 7.0, 0.7, 1.1, 8.0, 0.8, 1.2, 9.0, 0.9, 1.3, 10.0, 1.0, 1.4,
            ],
        )
        .unwrap();

        // y only depends on first feature
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);

        let mut model = LassoRegression::new(0.5, true, 1000, 1e-6);
        model.fit(&x, &y).unwrap();

        // With enough regularization, some coefficients should be zero
        let coef = model.coefficients.as_ref().unwrap();
        println!("Lasso coefficients: {:?}", coef);

        // First coefficient should be largest (it's the true predictor)
        assert!(coef[0].abs() > coef[1].abs());
    }

    #[test]
    fn test_elastic_net() {
        let x = Array2::from_shape_vec((5, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0])
            .unwrap();
        let y = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]);

        let mut model = ElasticNet::new(0.1, 0.5, true, 1000, 1e-6);
        model.fit(&x, &y).unwrap();

        assert!(model.coefficients.is_some());
        assert!(model.intercept.is_some());
    }
}
