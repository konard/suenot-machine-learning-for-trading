//! Flow layer implementations

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Activation Normalization layer
/// Learns scale and bias for data-dependent normalization
#[derive(Debug, Clone)]
pub struct ActNorm {
    pub dim: usize,
    pub scale: Array1<f64>,
    pub bias: Array1<f64>,
    pub initialized: bool,
}

impl ActNorm {
    /// Create new ActNorm layer
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            scale: Array1::ones(dim),
            bias: Array1::zeros(dim),
            initialized: false,
        }
    }

    /// Initialize from data (first batch)
    pub fn initialize(&mut self, x: &Array2<f64>) {
        let mean = x.mean_axis(Axis(0)).unwrap();
        let std = x.std_axis(Axis(0), 0.0);

        self.bias = -mean;
        self.scale = std.mapv(|v| 1.0 / (v + 1e-6));
        self.initialized = true;
    }

    /// Forward pass: apply normalization
    pub fn forward(&mut self, x: &Array2<f64>) -> (Array2<f64>, f64) {
        if !self.initialized {
            self.initialize(x);
        }

        let batch_size = x.nrows();
        let y = (x + &self.bias) * &self.scale;

        // Log determinant
        let log_det: f64 = self.scale.iter().map(|s| s.abs().ln()).sum::<f64>()
            * batch_size as f64;

        (y, log_det)
    }

    /// Inverse pass: remove normalization
    pub fn inverse(&self, y: &Array2<f64>) -> Array2<f64> {
        y / &self.scale - &self.bias
    }
}

/// Simple neural network for coupling layers
#[derive(Debug, Clone)]
pub struct SimpleNN {
    pub weights1: Array2<f64>,
    pub bias1: Array1<f64>,
    pub weights2: Array2<f64>,
    pub bias2: Array1<f64>,
    pub weights3: Array2<f64>,
    pub bias3: Array1<f64>,
}

impl SimpleNN {
    /// Create new network with random initialization
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.01).unwrap();

        let weights1 = Array2::from_shape_fn((input_dim, hidden_dim), |_| normal.sample(&mut rng));
        let bias1 = Array1::zeros(hidden_dim);

        let weights2 = Array2::from_shape_fn((hidden_dim, hidden_dim), |_| normal.sample(&mut rng));
        let bias2 = Array1::zeros(hidden_dim);

        let weights3 = Array2::from_shape_fn((hidden_dim, output_dim), |_| normal.sample(&mut rng));
        let bias3 = Array1::zeros(output_dim);

        Self {
            weights1,
            bias1,
            weights2,
            bias2,
            weights3,
            bias3,
        }
    }

    /// Forward pass with ReLU activation
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        // First layer
        let h1 = x.dot(&self.weights1) + &self.bias1;
        let h1 = h1.mapv(|v| v.max(0.0)); // ReLU

        // Second layer
        let h2 = h1.dot(&self.weights2) + &self.bias2;
        let h2 = h2.mapv(|v| v.max(0.0)); // ReLU

        // Output layer
        h2.dot(&self.weights3) + &self.bias3
    }
}

/// Affine Coupling Layer
/// Transforms second half using scale and translation from first half
#[derive(Debug, Clone)]
pub struct AffineCoupling {
    pub dim: usize,
    pub split_dim: usize,
    pub scale_net: SimpleNN,
    pub translate_net: SimpleNN,
}

impl AffineCoupling {
    /// Create new affine coupling layer
    pub fn new(dim: usize, hidden_dim: usize) -> Self {
        let split_dim = dim / 2;
        let output_dim = dim - split_dim;

        Self {
            dim,
            split_dim,
            scale_net: SimpleNN::new(split_dim, hidden_dim, output_dim),
            translate_net: SimpleNN::new(split_dim, hidden_dim, output_dim),
        }
    }

    /// Forward pass: affine transformation
    pub fn forward(&self, x: &Array2<f64>) -> (Array2<f64>, f64) {
        let x1 = x.slice(ndarray::s![.., ..self.split_dim]).to_owned();
        let x2 = x.slice(ndarray::s![.., self.split_dim..]).to_owned();

        // Compute scale and translation
        let s = self.scale_net.forward(&x1).mapv(|v| v.tanh()); // Bounded scale
        let t = self.translate_net.forward(&x1);

        // Transform second half
        let y2 = &x2 * &s.mapv(|v| v.exp()) + &t;

        // Concatenate
        let mut y = Array2::zeros((x.nrows(), self.dim));
        y.slice_mut(ndarray::s![.., ..self.split_dim]).assign(&x1);
        y.slice_mut(ndarray::s![.., self.split_dim..]).assign(&y2);

        // Log determinant
        let log_det: f64 = s.sum();

        (y, log_det)
    }

    /// Inverse pass: reverse affine transformation
    pub fn inverse(&self, y: &Array2<f64>) -> Array2<f64> {
        let y1 = y.slice(ndarray::s![.., ..self.split_dim]).to_owned();
        let y2 = y.slice(ndarray::s![.., self.split_dim..]).to_owned();

        // Compute scale and translation (same as forward)
        let s = self.scale_net.forward(&y1).mapv(|v| v.tanh());
        let t = self.translate_net.forward(&y1);

        // Inverse transform
        let x2 = (&y2 - &t) * &s.mapv(|v| (-v).exp());

        // Concatenate
        let mut x = Array2::zeros((y.nrows(), self.dim));
        x.slice_mut(ndarray::s![.., ..self.split_dim]).assign(&y1);
        x.slice_mut(ndarray::s![.., self.split_dim..]).assign(&x2);

        x
    }
}

/// Fixed random permutation layer
#[derive(Debug, Clone)]
pub struct Permutation {
    pub dim: usize,
    pub perm: Vec<usize>,
    pub inv_perm: Vec<usize>,
}

impl Permutation {
    /// Create new random permutation
    pub fn new(dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut perm: Vec<usize> = (0..dim).collect();

        // Fisher-Yates shuffle
        for i in (1..dim).rev() {
            let j = rng.gen_range(0..=i);
            perm.swap(i, j);
        }

        // Compute inverse permutation
        let mut inv_perm = vec![0; dim];
        for (i, &p) in perm.iter().enumerate() {
            inv_perm[p] = i;
        }

        Self { dim, perm, inv_perm }
    }

    /// Forward: apply permutation
    pub fn forward(&self, x: &Array2<f64>) -> (Array2<f64>, f64) {
        let mut y = Array2::zeros(x.raw_dim());
        for (i, &p) in self.perm.iter().enumerate() {
            y.column_mut(i).assign(&x.column(p));
        }
        (y, 0.0) // Permutation has log-det = 0
    }

    /// Inverse: apply inverse permutation
    pub fn inverse(&self, y: &Array2<f64>) -> Array2<f64> {
        let mut x = Array2::zeros(y.raw_dim());
        for (i, &p) in self.inv_perm.iter().enumerate() {
            x.column_mut(i).assign(&y.column(p));
        }
        x
    }
}

/// Flow layer trait
pub trait FlowLayer {
    fn forward(&mut self, x: &Array2<f64>) -> (Array2<f64>, f64);
    fn inverse(&self, y: &Array2<f64>) -> Array2<f64>;
}

impl FlowLayer for ActNorm {
    fn forward(&mut self, x: &Array2<f64>) -> (Array2<f64>, f64) {
        self.forward(x)
    }

    fn inverse(&self, y: &Array2<f64>) -> Array2<f64> {
        self.inverse(y)
    }
}

impl FlowLayer for AffineCoupling {
    fn forward(&mut self, x: &Array2<f64>) -> (Array2<f64>, f64) {
        self.forward(x)
    }

    fn inverse(&self, y: &Array2<f64>) -> Array2<f64> {
        self.inverse(y)
    }
}

impl FlowLayer for Permutation {
    fn forward(&mut self, x: &Array2<f64>) -> (Array2<f64>, f64) {
        self.forward(x)
    }

    fn inverse(&self, y: &Array2<f64>) -> Array2<f64> {
        self.inverse(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_permutation_invertible() {
        let perm = Permutation::new(8);
        let x = Array2::from_shape_fn((5, 8), |(i, j)| (i * 8 + j) as f64);

        let (y, _) = perm.forward(&x);
        let x_recon = perm.inverse(&y);

        for i in 0..5 {
            for j in 0..8 {
                assert_abs_diff_eq!(x[[i, j]], x_recon[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_actnorm_initializes() {
        let mut actnorm = ActNorm::new(4);
        let x = Array2::from_shape_fn((10, 4), |(i, j)| (i + j) as f64);

        assert!(!actnorm.initialized);
        let _ = actnorm.forward(&x);
        assert!(actnorm.initialized);
    }
}
