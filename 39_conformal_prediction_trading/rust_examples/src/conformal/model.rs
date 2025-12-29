//! Simple prediction models for use with conformal prediction
//!
//! These are basic models that can be wrapped with conformal prediction.
//! For production use, consider using more sophisticated models.

use ndarray::{Array1, Array2};

/// Trait for prediction models
pub trait Model {
    /// Fit the model to training data
    fn fit(&mut self, x: &Array2<f64>, y: &[f64]);

    /// Make predictions
    fn predict(&self, x: &Array2<f64>) -> Vec<f64>;

    /// Predict a single sample
    fn predict_one(&self, x: &[f64]) -> f64;
}

/// Simple linear regression model using ordinary least squares
#[derive(Debug, Clone)]
pub struct LinearModel {
    /// Coefficients (including intercept as first element)
    pub coefficients: Option<Array1<f64>>,
    /// Whether to include intercept
    pub fit_intercept: bool,
}

impl Default for LinearModel {
    fn default() -> Self {
        Self::new(true)
    }
}

impl LinearModel {
    /// Create a new linear model
    pub fn new(fit_intercept: bool) -> Self {
        Self {
            coefficients: None,
            fit_intercept,
        }
    }

    /// Get the number of features
    pub fn n_features(&self) -> Option<usize> {
        self.coefficients.as_ref().map(|c| {
            if self.fit_intercept {
                c.len() - 1
            } else {
                c.len()
            }
        })
    }
}

impl Model for LinearModel {
    fn fit(&mut self, x: &Array2<f64>, y: &[f64]) {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 || n_features == 0 {
            return;
        }

        // Add intercept column if needed
        let x_design = if self.fit_intercept {
            let mut design = Array2::ones((n_samples, n_features + 1));
            for i in 0..n_samples {
                for j in 0..n_features {
                    design[[i, j + 1]] = x[[i, j]];
                }
            }
            design
        } else {
            x.clone()
        };

        // Solve using normal equations: (X'X)^(-1) X'y
        // Using simplified pseudo-inverse approximation
        let n_cols = x_design.ncols();

        // Compute X'X
        let mut xtx = Array2::<f64>::zeros((n_cols, n_cols));
        for i in 0..n_cols {
            for j in 0..n_cols {
                let mut sum = 0.0;
                for k in 0..n_samples {
                    sum += x_design[[k, i]] * x_design[[k, j]];
                }
                xtx[[i, j]] = sum;
            }
        }

        // Add regularization for numerical stability
        for i in 0..n_cols {
            xtx[[i, i]] += 1e-6;
        }

        // Compute X'y
        let mut xty = Array1::<f64>::zeros(n_cols);
        for i in 0..n_cols {
            let mut sum = 0.0;
            for k in 0..n_samples {
                sum += x_design[[k, i]] * y[k];
            }
            xty[i] = sum;
        }

        // Solve using Gauss-Jordan elimination
        let coeffs = Self::solve_linear_system(&xtx, &xty);
        self.coefficients = Some(coeffs);
    }

    fn predict(&self, x: &Array2<f64>) -> Vec<f64> {
        let coeffs = match &self.coefficients {
            Some(c) => c,
            None => return vec![0.0; x.nrows()],
        };

        let n_samples = x.nrows();
        let mut predictions = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let row: Vec<f64> = x.row(i).iter().copied().collect();
            predictions.push(self.predict_one(&row));
        }

        predictions
    }

    fn predict_one(&self, x: &[f64]) -> f64 {
        let coeffs = match &self.coefficients {
            Some(c) => c,
            None => return 0.0,
        };

        let mut pred = if self.fit_intercept { coeffs[0] } else { 0.0 };
        let start_idx = if self.fit_intercept { 1 } else { 0 };

        for (i, &xi) in x.iter().enumerate() {
            if start_idx + i < coeffs.len() {
                pred += coeffs[start_idx + i] * xi;
            }
        }

        pred
    }
}

impl LinearModel {
    /// Solve linear system Ax = b using Gauss-Jordan elimination
    fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
        let n = a.nrows();
        if n == 0 || n != a.ncols() || n != b.len() {
            return Array1::zeros(0);
        }

        // Create augmented matrix
        let mut aug = Array2::<f64>::zeros((n, n + 1));
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, n]] = b[i];
        }

        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = aug[[col, col]].abs();
            for row in (col + 1)..n {
                if aug[[row, col]].abs() > max_val {
                    max_val = aug[[row, col]].abs();
                    max_row = row;
                }
            }

            // Swap rows
            if max_row != col {
                for j in 0..=n {
                    let temp = aug[[col, j]];
                    aug[[col, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = temp;
                }
            }

            // Check for singular matrix
            if aug[[col, col]].abs() < 1e-10 {
                continue;
            }

            // Eliminate below
            for row in (col + 1)..n {
                let factor = aug[[row, col]] / aug[[col, col]];
                for j in col..=n {
                    aug[[row, j]] -= factor * aug[[col, j]];
                }
            }
        }

        // Back substitution
        let mut x = Array1::<f64>::zeros(n);
        for i in (0..n).rev() {
            if aug[[i, i]].abs() < 1e-10 {
                continue;
            }
            let mut sum = aug[[i, n]];
            for j in (i + 1)..n {
                sum -= aug[[i, j]] * x[j];
            }
            x[i] = sum / aug[[i, i]];
        }

        x
    }
}

/// Moving average model for simple predictions
#[derive(Debug, Clone)]
pub struct MovingAverageModel {
    window: usize,
    last_values: Vec<f64>,
}

impl MovingAverageModel {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            last_values: Vec::new(),
        }
    }
}

impl Model for MovingAverageModel {
    fn fit(&mut self, _x: &Array2<f64>, y: &[f64]) {
        self.last_values = y.to_vec();
    }

    fn predict(&self, x: &Array2<f64>) -> Vec<f64> {
        vec![self.predict_one(&[]); x.nrows()]
    }

    fn predict_one(&self, _x: &[f64]) -> f64 {
        if self.last_values.is_empty() {
            return 0.0;
        }

        let start = self.last_values.len().saturating_sub(self.window);
        let window_values = &self.last_values[start..];
        window_values.iter().sum::<f64>() / window_values.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_linear_model_simple() {
        // y = 2*x + 1
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = vec![3.0, 5.0, 7.0, 9.0, 11.0];

        let mut model = LinearModel::new(true);
        model.fit(&x, &y);

        let predictions = model.predict(&x);

        // Check predictions are close to actual
        for (pred, &actual) in predictions.iter().zip(y.iter()) {
            assert!((pred - actual).abs() < 0.5);
        }
    }

    #[test]
    fn test_linear_model_predict_one() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = vec![2.0, 4.0, 6.0];

        let mut model = LinearModel::new(true);
        model.fit(&x, &y);

        let pred = model.predict_one(&[4.0]);
        assert!((pred - 8.0).abs() < 0.5);
    }
}
