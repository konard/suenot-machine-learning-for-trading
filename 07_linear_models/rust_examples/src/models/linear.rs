//! Linear Regression implementations
//!
//! This module provides Ordinary Least Squares (OLS) and Gradient Descent
//! implementations of linear regression for predicting cryptocurrency returns.

use ndarray::{s, Array1, Array2, Axis};
use thiserror::Error;

/// Errors that can occur during linear regression
#[derive(Error, Debug)]
pub enum LinearRegressionError {
    #[error("Matrix is singular and cannot be inverted")]
    SingularMatrix,

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Model has not been fitted yet")]
    NotFitted,

    #[error("Computation error: {0}")]
    ComputationError(String),
}

/// Linear Regression model using Ordinary Least Squares
#[derive(Debug, Clone)]
pub struct LinearRegression {
    /// Coefficients (weights) for each feature
    pub coefficients: Option<Array1<f64>>,
    /// Intercept (bias) term
    pub intercept: Option<f64>,
    /// Whether to fit an intercept
    fit_intercept: bool,
    /// R-squared score
    pub r_squared: Option<f64>,
    /// Feature names
    pub feature_names: Option<Vec<String>>,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new(true)
    }
}

impl LinearRegression {
    /// Create a new LinearRegression model
    ///
    /// # Arguments
    /// * `fit_intercept` - Whether to calculate the intercept
    pub fn new(fit_intercept: bool) -> Self {
        Self {
            coefficients: None,
            intercept: None,
            fit_intercept,
            r_squared: None,
            feature_names: None,
        }
    }

    /// Set feature names for interpretation
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Fit the model using Ordinary Least Squares
    ///
    /// Solves the normal equations: β = (X'X)^(-1) X'y
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), LinearRegressionError> {
        if x.nrows() != y.len() {
            return Err(LinearRegressionError::DimensionMismatch {
                expected: x.nrows(),
                got: y.len(),
            });
        }

        let (x_design, n_features) = if self.fit_intercept {
            // Add column of ones for intercept
            let ones = Array2::ones((x.nrows(), 1));
            let x_with_intercept = ndarray::concatenate(Axis(1), &[ones.view(), x.view()])
                .map_err(|e| LinearRegressionError::ComputationError(e.to_string()))?;
            (x_with_intercept, x.ncols() + 1)
        } else {
            (x.clone(), x.ncols())
        };

        // Normal equations: β = (X'X)^(-1) X'y
        let xt = x_design.t();
        let xtx = xt.dot(&x_design);
        let xty = xt.dot(y);

        // Solve using pseudo-inverse (more stable)
        let beta = self.solve_normal_equations(&xtx, &xty)?;

        if self.fit_intercept {
            self.intercept = Some(beta[0]);
            self.coefficients = Some(beta.slice(s![1..]).to_owned());
        } else {
            self.intercept = Some(0.0);
            self.coefficients = Some(beta);
        }

        // Calculate R-squared
        let predictions = self.predict(x)?;
        let y_mean = y.mean().unwrap();
        let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let ss_res: f64 = y
            .iter()
            .zip(predictions.iter())
            .map(|(&yi, &pi)| (yi - pi).powi(2))
            .sum();

        self.r_squared = Some(1.0 - ss_res / ss_tot);

        Ok(())
    }

    /// Solve normal equations using Cholesky decomposition or fallback to pseudoinverse
    fn solve_normal_equations(
        &self,
        xtx: &Array2<f64>,
        xty: &Array1<f64>,
    ) -> Result<Array1<f64>, LinearRegressionError> {
        let n = xtx.nrows();

        // Add small regularization for numerical stability
        let mut xtx_reg = xtx.clone();
        for i in 0..n {
            xtx_reg[[i, i]] += 1e-10;
        }

        // Try Cholesky decomposition
        match self.cholesky_solve(&xtx_reg, xty) {
            Ok(beta) => Ok(beta),
            Err(_) => {
                // Fallback to pseudoinverse using SVD-like approach
                self.pseudoinverse_solve(&xtx_reg, xty)
            }
        }
    }

    /// Solve using Cholesky decomposition
    fn cholesky_solve(
        &self,
        a: &Array2<f64>,
        b: &Array1<f64>,
    ) -> Result<Array1<f64>, LinearRegressionError> {
        let n = a.nrows();
        let mut l = Array2::<f64>::zeros((n, n));

        // Cholesky decomposition: A = L * L^T
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }

                if i == j {
                    let diag = a[[i, i]] - sum;
                    if diag <= 0.0 {
                        return Err(LinearRegressionError::SingularMatrix);
                    }
                    l[[i, j]] = diag.sqrt();
                } else {
                    l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
                }
            }
        }

        // Solve L * z = b (forward substitution)
        let mut z = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l[[i, j]] * z[j];
            }
            z[i] = (b[i] - sum) / l[[i, i]];
        }

        // Solve L^T * x = z (backward substitution)
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

    /// Solve using pseudoinverse (Moore-Penrose)
    fn pseudoinverse_solve(
        &self,
        a: &Array2<f64>,
        b: &Array1<f64>,
    ) -> Result<Array1<f64>, LinearRegressionError> {
        // Use iterative refinement with gradient descent
        let n = a.ncols();
        let mut x = Array1::<f64>::zeros(n);
        let learning_rate = 0.01;
        let max_iter = 1000;
        let tol = 1e-10;

        for _ in 0..max_iter {
            let residual = a.dot(&x) - b;
            let gradient = a.t().dot(&residual);

            let norm: f64 = gradient.iter().map(|&g| g * g).sum::<f64>().sqrt();
            if norm < tol {
                break;
            }

            x = &x - &(&gradient * learning_rate);
        }

        Ok(x)
    }

    /// Make predictions
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, LinearRegressionError> {
        let coefficients = self
            .coefficients
            .as_ref()
            .ok_or(LinearRegressionError::NotFitted)?;
        let intercept = self.intercept.ok_or(LinearRegressionError::NotFitted)?;

        if x.ncols() != coefficients.len() {
            return Err(LinearRegressionError::DimensionMismatch {
                expected: coefficients.len(),
                got: x.ncols(),
            });
        }

        let predictions = x.dot(coefficients) + intercept;
        Ok(predictions)
    }

    /// Get model summary
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("Linear Regression Summary\n");
        s.push_str("=========================\n\n");

        if let Some(ref coef) = self.coefficients {
            s.push_str(&format!("Intercept: {:.6}\n\n", self.intercept.unwrap_or(0.0)));
            s.push_str("Coefficients:\n");

            if let Some(ref names) = self.feature_names {
                for (i, (name, &c)) in names.iter().zip(coef.iter()).enumerate() {
                    s.push_str(&format!("  {:3}. {:20}: {:>12.6}\n", i + 1, name, c));
                }
            } else {
                for (i, &c) in coef.iter().enumerate() {
                    s.push_str(&format!("  {:3}. Feature {:2}: {:>12.6}\n", i + 1, i, c));
                }
            }

            s.push_str(&format!("\nR-squared: {:.6}\n", self.r_squared.unwrap_or(0.0)));
        } else {
            s.push_str("Model not fitted yet.\n");
        }

        s
    }
}

/// Linear Regression using Gradient Descent
#[derive(Debug, Clone)]
pub struct LinearRegressionGD {
    /// Coefficients (weights)
    pub coefficients: Option<Array1<f64>>,
    /// Intercept term
    pub intercept: Option<f64>,
    /// Learning rate
    learning_rate: f64,
    /// Maximum iterations
    max_iter: usize,
    /// Tolerance for convergence
    tolerance: f64,
    /// Whether to fit intercept
    fit_intercept: bool,
    /// Cost history during training
    pub cost_history: Vec<f64>,
}

impl Default for LinearRegressionGD {
    fn default() -> Self {
        Self::new(0.01, 1000, 1e-6, true)
    }
}

impl LinearRegressionGD {
    /// Create a new Gradient Descent Linear Regression model
    pub fn new(learning_rate: f64, max_iter: usize, tolerance: f64, fit_intercept: bool) -> Self {
        Self {
            coefficients: None,
            intercept: None,
            learning_rate,
            max_iter,
            tolerance,
            fit_intercept,
            cost_history: Vec::new(),
        }
    }

    /// Fit the model using batch gradient descent
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), LinearRegressionError> {
        let n_samples = x.nrows() as f64;
        let n_features = x.ncols();

        // Initialize weights
        let mut weights = Array1::<f64>::zeros(n_features);
        let mut bias = 0.0;

        self.cost_history.clear();

        for iter in 0..self.max_iter {
            // Forward pass
            let predictions = x.dot(&weights) + bias;

            // Compute gradients
            let errors = &predictions - y;
            let dw = x.t().dot(&errors) / n_samples;
            let db = errors.sum() / n_samples;

            // Update weights
            weights = &weights - &(&dw * self.learning_rate);
            if self.fit_intercept {
                bias -= self.learning_rate * db;
            }

            // Compute cost (MSE)
            let cost: f64 = errors.iter().map(|e| e.powi(2)).sum::<f64>() / (2.0 * n_samples);
            self.cost_history.push(cost);

            // Check convergence
            if iter > 0 {
                let cost_diff = (self.cost_history[iter - 1] - cost).abs();
                if cost_diff < self.tolerance {
                    log::debug!("Converged at iteration {}", iter);
                    break;
                }
            }
        }

        self.coefficients = Some(weights);
        self.intercept = Some(bias);

        Ok(())
    }

    /// Make predictions
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, LinearRegressionError> {
        let weights = self
            .coefficients
            .as_ref()
            .ok_or(LinearRegressionError::NotFitted)?;
        let bias = self.intercept.ok_or(LinearRegressionError::NotFitted)?;

        Ok(x.dot(weights) + bias)
    }
}

/// Stochastic Gradient Descent Linear Regression
#[derive(Debug, Clone)]
pub struct LinearRegressionSGD {
    /// Coefficients
    pub coefficients: Option<Array1<f64>>,
    /// Intercept
    pub intercept: Option<f64>,
    /// Learning rate
    learning_rate: f64,
    /// Number of epochs
    epochs: usize,
    /// Batch size
    batch_size: usize,
    /// Whether to fit intercept
    fit_intercept: bool,
}

impl Default for LinearRegressionSGD {
    fn default() -> Self {
        Self::new(0.01, 100, 32, true)
    }
}

impl LinearRegressionSGD {
    /// Create a new SGD Linear Regression model
    pub fn new(learning_rate: f64, epochs: usize, batch_size: usize, fit_intercept: bool) -> Self {
        Self {
            coefficients: None,
            intercept: None,
            learning_rate,
            epochs,
            batch_size,
            fit_intercept,
        }
    }

    /// Fit using mini-batch stochastic gradient descent
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), LinearRegressionError> {
        use rand::seq::SliceRandom;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        let mut weights = Array1::<f64>::zeros(n_features);
        let mut bias = 0.0;

        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();

        for _epoch in 0..self.epochs {
            indices.shuffle(&mut rng);

            for batch_start in (0..n_samples).step_by(self.batch_size) {
                let batch_end = (batch_start + self.batch_size).min(n_samples);
                let batch_indices = &indices[batch_start..batch_end];
                let batch_size = batch_indices.len() as f64;

                // Get batch data
                let mut x_batch = Array2::<f64>::zeros((batch_indices.len(), n_features));
                let mut y_batch = Array1::<f64>::zeros(batch_indices.len());

                for (i, &idx) in batch_indices.iter().enumerate() {
                    x_batch.row_mut(i).assign(&x.row(idx));
                    y_batch[i] = y[idx];
                }

                // Compute gradients
                let predictions = x_batch.dot(&weights) + bias;
                let errors = &predictions - &y_batch;

                let dw = x_batch.t().dot(&errors) / batch_size;
                let db = errors.sum() / batch_size;

                // Update weights
                weights = &weights - &(&dw * self.learning_rate);
                if self.fit_intercept {
                    bias -= self.learning_rate * db;
                }
            }
        }

        self.coefficients = Some(weights);
        self.intercept = Some(bias);

        Ok(())
    }

    /// Make predictions
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, LinearRegressionError> {
        let weights = self
            .coefficients
            .as_ref()
            .ok_or(LinearRegressionError::NotFitted)?;
        let bias = self.intercept.ok_or(LinearRegressionError::NotFitted)?;

        Ok(x.dot(weights) + bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_regression_simple() {
        // y = 2 + 3*x
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![5.0, 8.0, 11.0, 14.0, 17.0]);

        let mut model = LinearRegression::new(true);
        model.fit(&x, &y).unwrap();

        let intercept = model.intercept.unwrap();
        let coef = model.coefficients.as_ref().unwrap()[0];

        assert!((intercept - 2.0).abs() < 1e-6);
        assert!((coef - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_regression_multiple() {
        // y = 1 + 2*x1 + 3*x2
        let x = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![6.0, 11.0, 16.0, 21.0]); // 1 + 2*x1 + 3*x2

        let mut model = LinearRegression::new(true);
        model.fit(&x, &y).unwrap();

        let predictions = model.predict(&x).unwrap();

        for (&pred, &actual) in predictions.iter().zip(y.iter()) {
            assert!((pred - actual).abs() < 1e-4);
        }
    }

    #[test]
    fn test_gradient_descent() {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![5.0, 8.0, 11.0, 14.0, 17.0]);

        let mut model = LinearRegressionGD::new(0.1, 10000, 1e-8, true);
        model.fit(&x, &y).unwrap();

        let predictions = model.predict(&x).unwrap();

        for (&pred, &actual) in predictions.iter().zip(y.iter()) {
            assert!((pred - actual).abs() < 0.1);
        }
    }
}
