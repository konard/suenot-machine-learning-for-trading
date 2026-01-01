//! Training methods for ESN output weights

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Training method for output weights
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TrainingMethod {
    /// Ridge regression (Tikhonov regularization)
    RidgeRegression { regularization: f64 },
    /// LASSO regression (L1 regularization)
    Lasso { regularization: f64, max_iter: usize },
    /// Elastic Net (L1 + L2)
    ElasticNet { l1_ratio: f64, regularization: f64, max_iter: usize },
    /// Recursive Least Squares (online)
    RLS { forgetting_factor: f64 },
}

impl Default for TrainingMethod {
    fn default() -> Self {
        Self::RidgeRegression { regularization: 1e-6 }
    }
}

/// Output weight trainer
pub struct OutputTrainer {
    method: TrainingMethod,
}

impl OutputTrainer {
    /// Create new trainer with specified method
    pub fn new(method: TrainingMethod) -> Self {
        Self { method }
    }

    /// Train output weights
    pub fn train(
        &self,
        states: &[Array1<f64>],
        targets: &[Array1<f64>],
    ) -> Array2<f64> {
        match self.method {
            TrainingMethod::RidgeRegression { regularization } => {
                self.ridge_regression(states, targets, regularization)
            }
            TrainingMethod::Lasso { regularization, max_iter } => {
                self.lasso(states, targets, regularization, max_iter)
            }
            TrainingMethod::ElasticNet { l1_ratio, regularization, max_iter } => {
                self.elastic_net(states, targets, l1_ratio, regularization, max_iter)
            }
            TrainingMethod::RLS { .. } => {
                // RLS is online, use ridge for batch
                self.ridge_regression(states, targets, 1e-6)
            }
        }
    }

    /// Ridge regression (standard ESN training)
    fn ridge_regression(
        &self,
        states: &[Array1<f64>],
        targets: &[Array1<f64>],
        regularization: f64,
    ) -> Array2<f64> {
        let n_samples = states.len();
        let state_dim = states[0].len();
        let output_dim = targets[0].len();

        // Build state matrix X
        let mut x = Array2::zeros((n_samples, state_dim));
        for (i, state) in states.iter().enumerate() {
            x.row_mut(i).assign(state);
        }

        // Build target matrix Y
        let mut y = Array2::zeros((n_samples, output_dim));
        for (i, target) in targets.iter().enumerate() {
            y.row_mut(i).assign(target);
        }

        // Ridge regression: W = (X^T X + λI)^(-1) X^T Y
        let xt = x.t();
        let xtx = xt.dot(&x);
        let lambda_i = Array2::eye(state_dim) * regularization;
        let xtx_reg = &xtx + &lambda_i;
        let xty = xt.dot(&y);

        // Solve using Cholesky or pseudo-inverse
        let xtx_inv = pseudo_inverse(&xtx_reg);
        let w = xtx_inv.dot(&xty);

        w.t().to_owned()
    }

    /// LASSO regression using coordinate descent
    fn lasso(
        &self,
        states: &[Array1<f64>],
        targets: &[Array1<f64>],
        regularization: f64,
        max_iter: usize,
    ) -> Array2<f64> {
        let n_samples = states.len();
        let state_dim = states[0].len();
        let output_dim = targets[0].len();

        // Build matrices
        let mut x = Array2::zeros((n_samples, state_dim));
        for (i, state) in states.iter().enumerate() {
            x.row_mut(i).assign(state);
        }

        let mut y = Array2::zeros((n_samples, output_dim));
        for (i, target) in targets.iter().enumerate() {
            y.row_mut(i).assign(target);
        }

        // Initialize weights
        let mut w = Array2::zeros((output_dim, state_dim));

        // Coordinate descent
        for _ in 0..max_iter {
            for j in 0..state_dim {
                for k in 0..output_dim {
                    // Compute residual without j-th feature
                    let mut residual = y.column(k).to_owned();
                    for jj in 0..state_dim {
                        if jj != j {
                            residual = &residual - &(&x.column(jj) * w[[k, jj]]);
                        }
                    }

                    // Compute optimal weight
                    let xj = x.column(j);
                    let rho: f64 = xj.dot(&residual);
                    let xj_sq: f64 = xj.dot(&xj);

                    // Soft thresholding
                    w[[k, j]] = soft_threshold(rho, regularization) / xj_sq;
                }
            }
        }

        w
    }

    /// Elastic Net regression
    fn elastic_net(
        &self,
        states: &[Array1<f64>],
        targets: &[Array1<f64>],
        l1_ratio: f64,
        regularization: f64,
        max_iter: usize,
    ) -> Array2<f64> {
        let n_samples = states.len();
        let state_dim = states[0].len();
        let output_dim = targets[0].len();

        // Build matrices
        let mut x = Array2::zeros((n_samples, state_dim));
        for (i, state) in states.iter().enumerate() {
            x.row_mut(i).assign(state);
        }

        let mut y = Array2::zeros((n_samples, output_dim));
        for (i, target) in targets.iter().enumerate() {
            y.row_mut(i).assign(target);
        }

        let l1_reg = regularization * l1_ratio;
        let l2_reg = regularization * (1.0 - l1_ratio);

        // Initialize weights
        let mut w = Array2::zeros((output_dim, state_dim));

        // Coordinate descent
        for _ in 0..max_iter {
            for j in 0..state_dim {
                for k in 0..output_dim {
                    // Compute residual
                    let mut residual = y.column(k).to_owned();
                    for jj in 0..state_dim {
                        if jj != j {
                            residual = &residual - &(&x.column(jj) * w[[k, jj]]);
                        }
                    }

                    let xj = x.column(j);
                    let rho: f64 = xj.dot(&residual);
                    let xj_sq: f64 = xj.dot(&xj);

                    // Elastic net update
                    w[[k, j]] = soft_threshold(rho, l1_reg) / (xj_sq + l2_reg);
                }
            }
        }

        w
    }
}

/// Recursive Least Squares for online learning
pub struct RLSTrainer {
    /// Forgetting factor (0 < λ <= 1)
    forgetting_factor: f64,
    /// Covariance matrix inverse
    p: Array2<f64>,
    /// Current weights
    w: Array2<f64>,
    /// Initialized flag
    initialized: bool,
}

impl RLSTrainer {
    /// Create new RLS trainer
    pub fn new(state_dim: usize, output_dim: usize, forgetting_factor: f64) -> Self {
        // Initialize P with large diagonal (uncertainty)
        let p = Array2::eye(state_dim) * 1000.0;
        let w = Array2::zeros((output_dim, state_dim));

        Self {
            forgetting_factor,
            p,
            w,
            initialized: true,
        }
    }

    /// Update weights with new observation
    pub fn update(&mut self, state: &Array1<f64>, target: &Array1<f64>) {
        // Compute gain vector: k = P * x / (λ + x^T * P * x)
        let px = self.p.dot(state);
        let xpx: f64 = state.dot(&px);
        let denominator = self.forgetting_factor + xpx;

        let k = &px / denominator;

        // Prediction error
        let prediction = self.w.dot(state);
        let error = target - &prediction;

        // Update weights: W = W + error * k^T
        for i in 0..self.w.nrows() {
            for j in 0..self.w.ncols() {
                self.w[[i, j]] += error[i] * k[j];
            }
        }

        // Update covariance: P = (P - k * x^T * P) / λ
        let outer_kxp = outer_product(&k, &px);
        self.p = (&self.p - &outer_kxp) / self.forgetting_factor;
    }

    /// Get current weights
    pub fn weights(&self) -> &Array2<f64> {
        &self.w
    }

    /// Predict with current weights
    pub fn predict(&self, state: &Array1<f64>) -> Array1<f64> {
        self.w.dot(state)
    }
}

/// Soft thresholding operator for LASSO
fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

/// Compute outer product of two vectors
fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let mut result = Array2::zeros((a.len(), b.len()));
    for i in 0..a.len() {
        for j in 0..b.len() {
            result[[i, j]] = a[i] * b[j];
        }
    }
    result
}

/// Pseudo-inverse using SVD-like approach
fn pseudo_inverse(matrix: &Array2<f64>) -> Array2<f64> {
    let n = matrix.nrows();
    let mut augmented = Array2::zeros((n, 2 * n));

    // [A | I]
    for i in 0..n {
        for j in 0..n {
            augmented[[i, j]] = matrix[[i, j]];
        }
        augmented[[i, n + i]] = 1.0;
    }

    // Gauss-Jordan elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if augmented[[k, i]].abs() > augmented[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows
        for j in 0..(2 * n) {
            let temp = augmented[[i, j]];
            augmented[[i, j]] = augmented[[max_row, j]];
            augmented[[max_row, j]] = temp;
        }

        // Scale pivot row
        let pivot = augmented[[i, i]];
        if pivot.abs() > 1e-10 {
            for j in 0..(2 * n) {
                augmented[[i, j]] /= pivot;
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = augmented[[k, i]];
                    for j in 0..(2 * n) {
                        augmented[[k, j]] -= factor * augmented[[i, j]];
                    }
                }
            }
        }
    }

    // Extract inverse
    let mut result = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] = augmented[[i, n + j]];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ridge_regression() {
        let trainer = OutputTrainer::new(TrainingMethod::RidgeRegression {
            regularization: 1e-6,
        });

        // Simple linear problem: y = 2*x + 1
        let states: Vec<Array1<f64>> = (0..100)
            .map(|i| Array1::from_vec(vec![1.0, i as f64 / 100.0]))
            .collect();

        let targets: Vec<Array1<f64>> = states
            .iter()
            .map(|s| Array1::from_vec(vec![1.0 + 2.0 * s[1]]))
            .collect();

        let w = trainer.train(&states, &targets);

        // Weights should approximate [1, 2]
        assert!((w[[0, 0]] - 1.0).abs() < 0.1);
        assert!((w[[0, 1]] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_rls_online() {
        let mut rls = RLSTrainer::new(2, 1, 0.99);

        // Online learning: y = 2*x + 1
        for i in 0..1000 {
            let x = i as f64 / 1000.0;
            let state = Array1::from_vec(vec![1.0, x]);
            let target = Array1::from_vec(vec![1.0 + 2.0 * x]);

            rls.update(&state, &target);
        }

        let w = rls.weights();

        // Should converge to [1, 2]
        assert!((w[[0, 0]] - 1.0).abs() < 0.2);
        assert!((w[[0, 1]] - 2.0).abs() < 0.2);
    }

    #[test]
    fn test_soft_threshold() {
        assert_eq!(soft_threshold(5.0, 2.0), 3.0);
        assert_eq!(soft_threshold(-5.0, 2.0), -3.0);
        assert_eq!(soft_threshold(1.0, 2.0), 0.0);
    }
}
