//! GLOW Layer implementations
//!
//! Provides the building blocks for GLOW:
//! - ActNorm: Activation normalization
//! - InvertibleConv1x1: Invertible 1x1 convolution
//! - AffineCoupling: Affine coupling layer
//! - FlowStep: Complete flow step

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal};
use serde::{Deserialize, Serialize};

/// Activation Normalization layer
///
/// Performs data-dependent normalization:
/// y = (x - bias) * exp(log_scale)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActNorm {
    /// Number of features
    pub num_features: usize,
    /// Log scale parameter
    pub log_scale: Array1<f64>,
    /// Bias parameter
    pub bias: Array1<f64>,
    /// Whether the layer has been initialized
    pub initialized: bool,
}

impl ActNorm {
    /// Create a new ActNorm layer
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            log_scale: Array1::zeros(num_features),
            bias: Array1::zeros(num_features),
            initialized: false,
        }
    }

    /// Initialize with data statistics
    pub fn initialize(&mut self, data: &Array2<f64>) {
        let mean = data.mean_axis(Axis(0)).expect("Failed to compute mean");
        let std = Self::compute_std(data, &mean);

        self.bias = -mean;
        self.log_scale = std.mapv(|s| -(s + 1e-8).ln());
        self.initialized = true;
    }

    fn compute_std(data: &Array2<f64>, mean: &Array1<f64>) -> Array1<f64> {
        let centered = data - mean;
        let squared = &centered * &centered;
        let variance = squared.mean_axis(Axis(0)).expect("Failed to compute variance");
        variance.mapv(f64::sqrt)
    }

    /// Forward pass: x -> y
    ///
    /// Returns (output, log_det)
    pub fn forward(&mut self, x: &Array2<f64>) -> (Array2<f64>, f64) {
        if !self.initialized {
            self.initialize(x);
        }

        let scale = self.log_scale.mapv(f64::exp);
        let y = (x + &self.bias) * &scale;

        // Log determinant: sum of log_scale for each sample
        let log_det = self.log_scale.sum() * x.nrows() as f64;

        (y, log_det)
    }

    /// Inverse pass: y -> x
    pub fn inverse(&self, y: &Array2<f64>) -> (Array2<f64>, f64) {
        let scale_inv = self.log_scale.mapv(|s| (-s).exp());
        let x = y * &scale_inv - &self.bias;

        let log_det = -self.log_scale.sum() * y.nrows() as f64;

        (x, log_det)
    }

    /// Forward for single sample
    pub fn forward_sample(&mut self, x: &Array1<f64>) -> (Array1<f64>, f64) {
        let x_2d = x.clone().insert_axis(Axis(0));
        let (y_2d, log_det) = self.forward(&x_2d);
        (y_2d.index_axis(Axis(0), 0).to_owned(), log_det)
    }

    /// Inverse for single sample
    pub fn inverse_sample(&self, y: &Array1<f64>) -> (Array1<f64>, f64) {
        let y_2d = y.clone().insert_axis(Axis(0));
        let (x_2d, log_det) = self.inverse(&y_2d);
        (x_2d.index_axis(Axis(0), 0).to_owned(), log_det)
    }
}

/// Invertible 1x1 Convolution
///
/// Uses LU decomposition for efficient computation:
/// W = P @ L @ U, where P is permutation, L is lower triangular, U is upper triangular
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvertibleConv1x1 {
    /// Number of features
    pub num_features: usize,
    /// Weight matrix W
    pub weight: Array2<f64>,
    /// Cached inverse of W
    cached_inverse: Option<Array2<f64>>,
    /// Cached log determinant
    cached_log_det: Option<f64>,
}

impl InvertibleConv1x1 {
    /// Create a new invertible 1x1 convolution
    pub fn new(num_features: usize) -> Self {
        // Initialize with random orthogonal matrix
        let weight = Self::random_orthogonal(num_features);

        Self {
            num_features,
            weight,
            cached_inverse: None,
            cached_log_det: None,
        }
    }

    /// Generate a random orthogonal matrix using QR decomposition approximation
    fn random_orthogonal(n: usize) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Start with random matrix
        let mut q = Array2::from_shape_fn((n, n), |_| normal.sample(&mut rng));

        // Simple Gram-Schmidt orthogonalization
        for i in 0..n {
            // Normalize current column
            let norm: f64 = q.column(i).mapv(|x| x * x).sum().sqrt();
            if norm > 1e-10 {
                q.column_mut(i).mapv_inplace(|x| x / norm);
            }

            // Subtract projections from remaining columns
            for j in (i + 1)..n {
                let dot: f64 = q.column(i).iter().zip(q.column(j).iter())
                    .map(|(a, b)| a * b).sum();
                let col_i = q.column(i).to_owned();
                q.column_mut(j).zip_mut_with(&col_i, |a, b| *a -= dot * b);
            }
        }

        q
    }

    /// Compute log determinant of weight matrix
    fn compute_log_det(&self) -> f64 {
        // Simple approximation: compute using eigenvalues
        // For a more accurate implementation, use proper LU decomposition
        let det = self.approx_determinant();
        det.abs().ln()
    }

    /// Approximate determinant using simple expansion
    fn approx_determinant(&self) -> f64 {
        let n = self.num_features;
        if n == 1 {
            return self.weight[[0, 0]];
        }
        if n == 2 {
            return self.weight[[0, 0]] * self.weight[[1, 1]]
                - self.weight[[0, 1]] * self.weight[[1, 0]];
        }

        // For larger matrices, use row reduction approximation
        let mut matrix = self.weight.clone();
        let mut det = 1.0;

        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            let mut max_val = matrix[[i, i]].abs();
            for j in (i + 1)..n {
                if matrix[[j, i]].abs() > max_val {
                    max_val = matrix[[j, i]].abs();
                    max_row = j;
                }
            }

            if max_val < 1e-10 {
                return 0.0; // Singular matrix
            }

            // Swap rows if needed
            if max_row != i {
                for k in 0..n {
                    let temp = matrix[[i, k]];
                    matrix[[i, k]] = matrix[[max_row, k]];
                    matrix[[max_row, k]] = temp;
                }
                det = -det;
            }

            det *= matrix[[i, i]];

            // Eliminate below
            for j in (i + 1)..n {
                let factor = matrix[[j, i]] / matrix[[i, i]];
                for k in i..n {
                    matrix[[j, k]] -= factor * matrix[[i, k]];
                }
            }
        }

        det
    }

    /// Compute inverse of weight matrix
    fn compute_inverse(&self) -> Array2<f64> {
        let n = self.num_features;

        // Augmented matrix [W | I]
        let mut aug = Array2::zeros((n, 2 * n));
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = self.weight[[i, j]];
            }
            aug[[i, n + i]] = 1.0;
        }

        // Gauss-Jordan elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            let mut max_val = aug[[i, i]].abs();
            for j in (i + 1)..n {
                if aug[[j, i]].abs() > max_val {
                    max_val = aug[[j, i]].abs();
                    max_row = j;
                }
            }

            // Swap rows
            if max_row != i {
                for k in 0..(2 * n) {
                    let temp = aug[[i, k]];
                    aug[[i, k]] = aug[[max_row, k]];
                    aug[[max_row, k]] = temp;
                }
            }

            // Scale pivot row
            let pivot = aug[[i, i]];
            if pivot.abs() > 1e-10 {
                for k in 0..(2 * n) {
                    aug[[i, k]] /= pivot;
                }
            }

            // Eliminate column
            for j in 0..n {
                if j != i {
                    let factor = aug[[j, i]];
                    for k in 0..(2 * n) {
                        aug[[j, k]] -= factor * aug[[i, k]];
                    }
                }
            }
        }

        // Extract inverse
        Array2::from_shape_fn((n, n), |(i, j)| aug[[i, n + j]])
    }

    /// Forward pass: x -> y = x @ W
    pub fn forward(&mut self, x: &Array2<f64>) -> (Array2<f64>, f64) {
        let y = x.dot(&self.weight);

        // Compute log determinant (cached)
        let log_det = if let Some(ld) = self.cached_log_det {
            ld * x.nrows() as f64
        } else {
            let ld = self.compute_log_det();
            self.cached_log_det = Some(ld);
            ld * x.nrows() as f64
        };

        (y, log_det)
    }

    /// Inverse pass: y -> x = y @ W^(-1)
    pub fn inverse(&mut self, y: &Array2<f64>) -> (Array2<f64>, f64) {
        // Compute inverse (cached)
        let w_inv = if let Some(ref inv) = self.cached_inverse {
            inv.clone()
        } else {
            let inv = self.compute_inverse();
            self.cached_inverse = Some(inv.clone());
            inv
        };

        let x = y.dot(&w_inv);

        let log_det = if let Some(ld) = self.cached_log_det {
            -ld * y.nrows() as f64
        } else {
            let ld = self.compute_log_det();
            self.cached_log_det = Some(ld);
            -ld * y.nrows() as f64
        };

        (x, log_det)
    }

    /// Invalidate cache (call after updating weights)
    pub fn invalidate_cache(&mut self) {
        self.cached_inverse = None;
        self.cached_log_det = None;
    }
}

/// Affine Coupling Layer
///
/// Splits input into two halves, transforms one based on the other:
/// y_a = x_a
/// y_b = x_b * exp(s(x_a)) + t(x_a)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffineCoupling {
    /// Number of features
    pub num_features: usize,
    /// Split dimension
    pub split_dim: usize,
    /// Network weights for computing scale and translation
    /// Layer 1: split_dim -> hidden
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    /// Layer 2: hidden -> hidden
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
    /// Layer 3: hidden -> (num_features - split_dim) * 2
    pub w3: Array2<f64>,
    pub b3: Array1<f64>,
    /// Hidden dimension
    pub hidden_dim: usize,
}

impl AffineCoupling {
    /// Create a new affine coupling layer
    pub fn new(num_features: usize, hidden_dim: usize) -> Self {
        let split_dim = num_features / 2;
        let output_dim = (num_features - split_dim) * 2;

        let mut rng = rand::thread_rng();

        // Xavier initialization
        let init_scale_1 = (2.0 / (split_dim + hidden_dim) as f64).sqrt();
        let init_scale_2 = (2.0 / (hidden_dim * 2) as f64).sqrt();
        let init_scale_3 = 0.01; // Small initialization for identity-like behavior

        let w1 = Array2::from_shape_fn((split_dim, hidden_dim), |_| {
            rng.gen::<f64>() * init_scale_1 * 2.0 - init_scale_1
        });
        let b1 = Array1::zeros(hidden_dim);

        let w2 = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| {
            rng.gen::<f64>() * init_scale_2 * 2.0 - init_scale_2
        });
        let b2 = Array1::zeros(hidden_dim);

        let w3 = Array2::from_shape_fn((hidden_dim, output_dim), |_| {
            rng.gen::<f64>() * init_scale_3 * 2.0 - init_scale_3
        });
        let b3 = Array1::zeros(output_dim);

        Self {
            num_features,
            split_dim,
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            hidden_dim,
        }
    }

    /// ReLU activation
    fn relu(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| v.max(0.0))
    }

    /// Compute scale and translation from first half
    fn compute_st(&self, x_a: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        // Forward through network
        let h1 = Self::relu(&(x_a.dot(&self.w1) + &self.b1));
        let h2 = Self::relu(&(h1.dot(&self.w2) + &self.b2));
        let out = h2.dot(&self.w3) + &self.b3;

        // Split output into scale and translation
        let out_dim = self.num_features - self.split_dim;
        let log_s = out.slice(ndarray::s![.., ..out_dim]).to_owned();
        let t = out.slice(ndarray::s![.., out_dim..]).to_owned();

        // Constrain scale
        let log_s = log_s.mapv(|v| v.tanh() * 2.0);

        (log_s, t)
    }

    /// Forward pass
    pub fn forward(&self, x: &Array2<f64>) -> (Array2<f64>, f64) {
        // Split
        let x_a = x.slice(ndarray::s![.., ..self.split_dim]).to_owned();
        let x_b = x.slice(ndarray::s![.., self.split_dim..]).to_owned();

        // Compute scale and translation
        let (log_s, t) = self.compute_st(&x_a);

        // Transform
        let y_b = &x_b * &log_s.mapv(f64::exp) + &t;

        // Concatenate
        let mut y = Array2::zeros((x.nrows(), self.num_features));
        y.slice_mut(ndarray::s![.., ..self.split_dim]).assign(&x_a);
        y.slice_mut(ndarray::s![.., self.split_dim..]).assign(&y_b);

        // Log determinant
        let log_det = log_s.sum();

        (y, log_det)
    }

    /// Inverse pass
    pub fn inverse(&self, y: &Array2<f64>) -> (Array2<f64>, f64) {
        // Split
        let y_a = y.slice(ndarray::s![.., ..self.split_dim]).to_owned();
        let y_b = y.slice(ndarray::s![.., self.split_dim..]).to_owned();

        // Compute scale and translation (using y_a = x_a)
        let (log_s, t) = self.compute_st(&y_a);

        // Inverse transform
        let x_b = (&y_b - &t) * &log_s.mapv(|v| (-v).exp());

        // Concatenate
        let mut x = Array2::zeros((y.nrows(), self.num_features));
        x.slice_mut(ndarray::s![.., ..self.split_dim]).assign(&y_a);
        x.slice_mut(ndarray::s![.., self.split_dim..]).assign(&x_b);

        // Log determinant
        let log_det = -log_s.sum();

        (x, log_det)
    }
}

/// Complete Flow Step: ActNorm -> 1x1 Conv -> Affine Coupling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowStep {
    pub actnorm: ActNorm,
    pub conv1x1: InvertibleConv1x1,
    pub coupling: AffineCoupling,
}

impl FlowStep {
    /// Create a new flow step
    pub fn new(num_features: usize, hidden_dim: usize) -> Self {
        Self {
            actnorm: ActNorm::new(num_features),
            conv1x1: InvertibleConv1x1::new(num_features),
            coupling: AffineCoupling::new(num_features, hidden_dim),
        }
    }

    /// Forward pass through all three components
    pub fn forward(&mut self, x: &Array2<f64>) -> (Array2<f64>, f64) {
        let mut total_log_det = 0.0;

        let (h, log_det) = self.actnorm.forward(x);
        total_log_det += log_det;

        let (h, log_det) = self.conv1x1.forward(&h);
        total_log_det += log_det;

        let (y, log_det) = self.coupling.forward(&h);
        total_log_det += log_det;

        (y, total_log_det)
    }

    /// Inverse pass through all three components (in reverse order)
    pub fn inverse(&mut self, y: &Array2<f64>) -> (Array2<f64>, f64) {
        let mut total_log_det = 0.0;

        let (h, log_det) = self.coupling.inverse(y);
        total_log_det += log_det;

        let (h, log_det) = self.conv1x1.inverse(&h);
        total_log_det += log_det;

        let (x, log_det) = self.actnorm.inverse(&h);
        total_log_det += log_det;

        (x, total_log_det)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_data(n_samples: usize, n_features: usize) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        Array2::from_shape_fn((n_samples, n_features), |_| rng.gen::<f64>())
    }

    #[test]
    fn test_actnorm_invertibility() {
        let mut actnorm = ActNorm::new(8);
        let x = create_test_data(10, 8);

        let (y, log_det_fwd) = actnorm.forward(&x);
        let (x_recovered, log_det_inv) = actnorm.inverse(&y);

        // Check invertibility
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert_relative_eq!(x[[i, j]], x_recovered[[i, j]], epsilon = 1e-10);
            }
        }

        // Check log determinant
        assert_relative_eq!(log_det_fwd, -log_det_inv, epsilon = 1e-10);
    }

    #[test]
    fn test_conv1x1_invertibility() {
        let mut conv = InvertibleConv1x1::new(8);
        let x = create_test_data(10, 8);

        let (y, _) = conv.forward(&x);
        let (x_recovered, _) = conv.inverse(&y);

        // Check invertibility
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert_relative_eq!(x[[i, j]], x_recovered[[i, j]], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_coupling_invertibility() {
        let coupling = AffineCoupling::new(8, 32);
        let x = create_test_data(10, 8);

        let (y, log_det_fwd) = coupling.forward(&x);
        let (x_recovered, log_det_inv) = coupling.inverse(&y);

        // Check invertibility
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert_relative_eq!(x[[i, j]], x_recovered[[i, j]], epsilon = 1e-10);
            }
        }

        // Check log determinant
        assert_relative_eq!(log_det_fwd, -log_det_inv, epsilon = 1e-10);
    }

    #[test]
    fn test_flow_step_invertibility() {
        let mut step = FlowStep::new(8, 32);
        let x = create_test_data(10, 8);

        let (y, _) = step.forward(&x);
        let (x_recovered, _) = step.inverse(&y);

        // Check invertibility (with some tolerance due to accumulated errors)
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert_relative_eq!(x[[i, j]], x_recovered[[i, j]], epsilon = 1e-5);
            }
        }
    }
}
