//! Neural network layers for normalizing flows.

use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::StandardNormal;

/// Simple feedforward neural network for coupling layers
#[derive(Debug, Clone)]
pub struct CouplingNetwork {
    /// Weights for each layer
    weights: Vec<Array2<f64>>,
    /// Biases for each layer
    biases: Vec<Array1<f64>>,
    /// Scale output weights
    scale_weights: Array2<f64>,
    /// Scale output bias
    scale_bias: Array1<f64>,
    /// Translation output weights
    translation_weights: Array2<f64>,
    /// Translation output bias
    translation_bias: Array1<f64>,
}

impl CouplingNetwork {
    /// Create a new coupling network
    pub fn new(input_dim: usize, output_dim: usize, hidden_dim: usize, n_hidden: usize) -> Self {
        let mut rng = rand::thread_rng();

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        // Input layer
        let scale = (2.0 / input_dim as f64).sqrt();
        weights.push(random_matrix(&mut rng, hidden_dim, input_dim, scale));
        biases.push(Array1::zeros(hidden_dim));

        // Hidden layers
        let scale = (2.0 / hidden_dim as f64).sqrt();
        for _ in 0..n_hidden - 1 {
            weights.push(random_matrix(&mut rng, hidden_dim, hidden_dim, scale));
            biases.push(Array1::zeros(hidden_dim));
        }

        // Output layers (initialized to zero for identity transform)
        let scale_weights = Array2::zeros((output_dim, hidden_dim));
        let scale_bias = Array1::zeros(output_dim);
        let translation_weights = Array2::zeros((output_dim, hidden_dim));
        let translation_bias = Array1::zeros(output_dim);

        Self {
            weights,
            biases,
            scale_weights,
            scale_bias,
            translation_weights,
            translation_bias,
        }
    }

    /// Forward pass through the network
    pub fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let mut h = x.clone();

        // Hidden layers with ReLU
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            h = w.dot(&h) + b;
            h.mapv_inplace(|v| v.max(0.0)); // ReLU
        }

        // Output: scale (clamped) and translation
        let scale = self.scale_weights.dot(&h) + &self.scale_bias;
        let scale = scale.mapv(|v| v.tanh() * 2.0); // Clamp to [-2, 2]

        let translation = self.translation_weights.dot(&h) + &self.translation_bias;

        (scale, translation)
    }

    /// Get mutable reference to parameters for training
    pub fn parameters_mut(
        &mut self,
    ) -> (
        &mut Vec<Array2<f64>>,
        &mut Vec<Array1<f64>>,
        &mut Array2<f64>,
        &mut Array1<f64>,
        &mut Array2<f64>,
        &mut Array1<f64>,
    ) {
        (
            &mut self.weights,
            &mut self.biases,
            &mut self.scale_weights,
            &mut self.scale_bias,
            &mut self.translation_weights,
            &mut self.translation_bias,
        )
    }
}

/// Affine coupling layer
#[derive(Debug, Clone)]
pub struct AffineCouplingLayer {
    /// Mask indicating which dimensions are unchanged
    mask: Array1<f64>,
    /// Coupling network
    network: CouplingNetwork,
}

impl AffineCouplingLayer {
    /// Create a new affine coupling layer
    pub fn new(dim: usize, mask: Array1<f64>, hidden_dim: usize, n_hidden: usize) -> Self {
        let n_masked = mask.iter().filter(|&&m| m > 0.5).count();
        let n_unmasked = dim - n_masked;

        let network = CouplingNetwork::new(n_masked, n_unmasked, hidden_dim, n_hidden);

        Self { mask, network }
    }

    /// Forward pass: data -> latent
    pub fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, f64) {
        let x_masked = self.apply_mask(x);
        let (scale, translation) = self.network.forward(&x_masked);

        let mut y = x.clone();
        let mut log_det = 0.0;

        let mut idx = 0;
        for i in 0..x.len() {
            if self.mask[i] < 0.5 {
                // Unmasked: apply affine transform
                y[i] = x[i] * scale[idx].exp() + translation[idx];
                log_det += scale[idx];
                idx += 1;
            }
        }

        (y, log_det)
    }

    /// Inverse pass: latent -> data
    pub fn inverse(&self, y: &Array1<f64>) -> (Array1<f64>, f64) {
        let y_masked = self.apply_mask(y);
        let (scale, translation) = self.network.forward(&y_masked);

        let mut x = y.clone();
        let mut log_det = 0.0;

        let mut idx = 0;
        for i in 0..y.len() {
            if self.mask[i] < 0.5 {
                // Unmasked: inverse affine transform
                x[i] = (y[i] - translation[idx]) * (-scale[idx]).exp();
                log_det -= scale[idx];
                idx += 1;
            }
        }

        (x, log_det)
    }

    /// Apply mask to get masked elements
    fn apply_mask(&self, x: &Array1<f64>) -> Array1<f64> {
        x.iter()
            .zip(self.mask.iter())
            .filter(|(_, &m)| m > 0.5)
            .map(|(&v, _)| v)
            .collect::<Vec<_>>()
            .into()
    }

    /// Get mutable reference to network for training
    pub fn network_mut(&mut self) -> &mut CouplingNetwork {
        &mut self.network
    }
}

/// Create a random matrix with given scale
fn random_matrix<R: Rng>(rng: &mut R, rows: usize, cols: usize, scale: f64) -> Array2<f64> {
    let data: Vec<f64> = (0..rows * cols)
        .map(|_| rng.sample::<f64, _>(StandardNormal) * scale)
        .collect();
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

/// Permutation layer
#[derive(Debug, Clone)]
pub struct Permutation {
    /// Permutation indices
    perm: Vec<usize>,
    /// Inverse permutation indices
    inv_perm: Vec<usize>,
}

impl Permutation {
    /// Create a shuffle permutation
    pub fn shuffle(dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut perm: Vec<usize> = (0..dim).collect();

        // Fisher-Yates shuffle
        for i in (1..dim).rev() {
            let j = rng.gen_range(0..=i);
            perm.swap(i, j);
        }

        let mut inv_perm = vec![0; dim];
        for (i, &p) in perm.iter().enumerate() {
            inv_perm[p] = i;
        }

        Self { perm, inv_perm }
    }

    /// Create a reverse permutation
    pub fn reverse(dim: usize) -> Self {
        let perm: Vec<usize> = (0..dim).rev().collect();
        let inv_perm = perm.clone();
        Self { perm, inv_perm }
    }

    /// Apply permutation
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
