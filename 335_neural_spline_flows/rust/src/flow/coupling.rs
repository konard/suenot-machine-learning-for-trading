//! Coupling Layer implementation
//!
//! This module implements the coupling layer with neural spline transformations.

use super::spline::{RationalQuadraticSpline, SplineParams};
use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Neural network layer weights
#[derive(Debug, Clone)]
pub struct LinearLayer {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
}

impl LinearLayer {
    /// Create a new linear layer with Xavier initialization
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let weights = Array2::from_shape_fn((output_dim, input_dim), |_| normal.sample(&mut rng));
        let bias = Array1::zeros(output_dim);

        Self { weights, bias }
    }

    /// Forward pass
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        self.weights.dot(x) + &self.bias
    }
}

/// Simple MLP conditioner network
#[derive(Debug, Clone)]
pub struct Conditioner {
    layers: Vec<LinearLayer>,
    hidden_dim: usize,
}

impl Conditioner {
    /// Create a new conditioner network
    pub fn new(input_dim: usize, output_dim: usize, hidden_dim: usize, num_hidden: usize) -> Self {
        let mut layers = Vec::with_capacity(num_hidden + 1);

        // Input layer
        layers.push(LinearLayer::new(input_dim, hidden_dim));

        // Hidden layers
        for _ in 0..num_hidden.saturating_sub(1) {
            layers.push(LinearLayer::new(hidden_dim, hidden_dim));
        }

        // Output layer
        layers.push(LinearLayer::new(hidden_dim, output_dim));

        Self { layers, hidden_dim }
    }

    /// Forward pass with GELU activation
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut h = x.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h);

            // Apply GELU activation for all but last layer
            if i < self.layers.len() - 1 {
                h = h.mapv(gelu);
            }
        }

        h
    }

    /// Update weights with gradient
    pub fn update_weights(&mut self, learning_rate: f64, gradients: &[LinearLayer]) {
        for (layer, grad) in self.layers.iter_mut().zip(gradients.iter()) {
            layer.weights = &layer.weights - learning_rate * &grad.weights;
            layer.bias = &layer.bias - learning_rate * &grad.bias;
        }
    }
}

/// GELU activation function
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh())
}

/// Coupling Layer with neural spline transformation
#[derive(Debug, Clone)]
pub struct CouplingLayer {
    /// Dimension of input
    dim: usize,
    /// Split dimension (how many dimensions are unchanged)
    split_dim: usize,
    /// Conditioner network
    conditioner: Conditioner,
    /// Spline transformation
    spline: RationalQuadraticSpline,
    /// Number of spline bins
    num_bins: usize,
}

impl CouplingLayer {
    /// Create a new coupling layer
    pub fn new(dim: usize, hidden_dim: usize, num_bins: usize, num_hidden: usize) -> Self {
        let split_dim = dim / 2;
        let transform_dim = dim - split_dim;
        let output_dim = transform_dim * (3 * num_bins + 1);

        let conditioner = Conditioner::new(split_dim, output_dim, hidden_dim, num_hidden);
        let spline = RationalQuadraticSpline::new(num_bins, 3.0, 1e-3);

        Self {
            dim,
            split_dim,
            conditioner,
            spline,
            num_bins,
        }
    }

    /// Forward transformation
    ///
    /// Returns (y, log_det)
    pub fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, f64) {
        assert_eq!(x.len(), self.dim);

        let x1 = x.slice(ndarray::s![..self.split_dim]).to_owned();
        let x2 = x.slice(ndarray::s![self.split_dim..]).to_owned();

        // Get spline parameters from conditioner
        let raw_params = self.conditioner.forward(&x1);

        // Transform x2 using splines
        let transform_dim = self.dim - self.split_dim;
        let params_per_dim = 3 * self.num_bins + 1;

        let mut y2 = Array1::zeros(transform_dim);
        let mut total_log_det = 0.0;

        for i in 0..transform_dim {
            let start = i * params_per_dim;
            let end = start + params_per_dim;
            let raw = raw_params.slice(ndarray::s![start..end]).to_owned();
            let params = self.spline.params_from_raw(&raw);

            let (yi, log_det) = self.spline.forward(x2[i], &params);
            y2[i] = yi;
            total_log_det += log_det;
        }

        // Concatenate y1 = x1 and y2
        let mut y = Array1::zeros(self.dim);
        y.slice_mut(ndarray::s![..self.split_dim]).assign(&x1);
        y.slice_mut(ndarray::s![self.split_dim..]).assign(&y2);

        (y, total_log_det)
    }

    /// Inverse transformation
    ///
    /// Returns (x, log_det)
    pub fn inverse(&self, y: &Array1<f64>) -> (Array1<f64>, f64) {
        assert_eq!(y.len(), self.dim);

        let y1 = y.slice(ndarray::s![..self.split_dim]).to_owned();
        let y2 = y.slice(ndarray::s![self.split_dim..]).to_owned();

        // Get spline parameters from conditioner (using y1 = x1)
        let raw_params = self.conditioner.forward(&y1);

        // Inverse transform y2 using splines
        let transform_dim = self.dim - self.split_dim;
        let params_per_dim = 3 * self.num_bins + 1;

        let mut x2 = Array1::zeros(transform_dim);
        let mut total_log_det = 0.0;

        for i in 0..transform_dim {
            let start = i * params_per_dim;
            let end = start + params_per_dim;
            let raw = raw_params.slice(ndarray::s![start..end]).to_owned();
            let params = self.spline.params_from_raw(&raw);

            let (xi, log_det) = self.spline.inverse(y2[i], &params);
            x2[i] = xi;
            total_log_det += log_det;
        }

        // Concatenate x1 = y1 and x2
        let mut x = Array1::zeros(self.dim);
        x.slice_mut(ndarray::s![..self.split_dim]).assign(&y1);
        x.slice_mut(ndarray::s![self.split_dim..]).assign(&x2);

        (x, total_log_det)
    }

    /// Get mutable reference to conditioner for training
    pub fn conditioner_mut(&mut self) -> &mut Conditioner {
        &mut self.conditioner
    }
}

/// Permutation layer (simple dimension shuffling)
#[derive(Debug, Clone)]
pub struct Permutation {
    /// Forward permutation indices
    perm: Vec<usize>,
    /// Inverse permutation indices
    inv_perm: Vec<usize>,
}

impl Permutation {
    /// Create a new random permutation
    pub fn random(dim: usize) -> Self {
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

        Self { perm, inv_perm }
    }

    /// Create identity permutation
    pub fn identity(dim: usize) -> Self {
        let perm: Vec<usize> = (0..dim).collect();
        let inv_perm = perm.clone();
        Self { perm, inv_perm }
    }

    /// Apply forward permutation
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut y = Array1::zeros(x.len());
        for (i, &p) in self.perm.iter().enumerate() {
            y[i] = x[p];
        }
        y
    }

    /// Apply inverse permutation
    pub fn inverse(&self, y: &Array1<f64>) -> Array1<f64> {
        let mut x = Array1::zeros(y.len());
        for (i, &p) in self.inv_perm.iter().enumerate() {
            x[i] = y[p];
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_linear_layer() {
        let layer = LinearLayer::new(4, 8);
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let y = layer.forward(&x);
        assert_eq!(y.len(), 8);
    }

    #[test]
    fn test_conditioner() {
        let conditioner = Conditioner::new(4, 16, 32, 2);
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let y = conditioner.forward(&x);
        assert_eq!(y.len(), 16);
    }

    #[test]
    fn test_coupling_layer() {
        let layer = CouplingLayer::new(8, 32, 8, 2);
        let x = Array1::from_vec(vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]);
        let (y, _log_det) = layer.forward(&x);
        assert_eq!(y.len(), 8);

        // Check invertibility
        let (x_recovered, _) = layer.inverse(&y);
        for i in 0..8 {
            assert_abs_diff_eq!(x[i], x_recovered[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_permutation() {
        let perm = Permutation::random(8);
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let y = perm.forward(&x);
        let x_recovered = perm.inverse(&y);

        for i in 0..8 {
            assert_abs_diff_eq!(x[i], x_recovered[i], epsilon = 1e-10);
        }
    }
}
