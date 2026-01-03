//! # Velocity Field Network
//!
//! Neural network that defines the ODE dynamics: dz/dt = f(z, t; θ)

use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

/// Velocity field network for defining ODE dynamics
///
/// Architecture:
/// - Sinusoidal time embedding
/// - Input projection
/// - Residual MLP blocks with time conditioning
/// - Output projection
#[derive(Debug, Clone)]
pub struct VelocityField {
    /// Input dimension
    pub dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of residual layers
    pub num_layers: usize,
    /// Time embedding dimension
    pub time_embed_dim: usize,

    // Weights
    time_embed_weights: Array2<f64>,
    time_embed_bias: Array1<f64>,
    input_proj_weights: Array2<f64>,
    input_proj_bias: Array1<f64>,
    res_blocks: Vec<ResidualBlock>,
    output_proj_weights: Array2<f64>,
    output_proj_bias: Array1<f64>,
}

/// Residual block with time conditioning
#[derive(Debug, Clone)]
struct ResidualBlock {
    linear1_weights: Array2<f64>,
    linear1_bias: Array1<f64>,
    linear2_weights: Array2<f64>,
    linear2_bias: Array1<f64>,
}

impl VelocityField {
    /// Create a new velocity field network
    pub fn new(dim: usize, hidden_dim: usize, num_layers: usize) -> Self {
        let time_embed_dim = 16;

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.1).unwrap();

        // Initialize weights
        let time_embed_weights = random_matrix(&mut rng, &normal, time_embed_dim, hidden_dim);
        let time_embed_bias = random_vector(&mut rng, &normal, hidden_dim);

        let input_proj_weights = random_matrix(&mut rng, &normal, dim, hidden_dim);
        let input_proj_bias = random_vector(&mut rng, &normal, hidden_dim);

        let mut res_blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            res_blocks.push(ResidualBlock {
                linear1_weights: random_matrix(&mut rng, &normal, hidden_dim * 2, hidden_dim * 4),
                linear1_bias: random_vector(&mut rng, &normal, hidden_dim * 4),
                linear2_weights: random_matrix(&mut rng, &normal, hidden_dim * 4, hidden_dim),
                linear2_bias: random_vector(&mut rng, &normal, hidden_dim),
            });
        }

        // Output projection (initialized to small values for stability)
        let small_normal = Normal::new(0.0, 0.01).unwrap();
        let output_proj_weights = random_matrix(&mut rng, &small_normal, hidden_dim, dim);
        let output_proj_bias = Array1::zeros(dim);

        Self {
            dim,
            hidden_dim,
            num_layers,
            time_embed_dim,
            time_embed_weights,
            time_embed_bias,
            input_proj_weights,
            input_proj_bias,
            res_blocks,
            output_proj_weights,
            output_proj_bias,
        }
    }

    /// Compute velocity at state z and time t
    ///
    /// Returns dz/dt
    pub fn forward(&self, z: &Array1<f64>, t: f64) -> Array1<f64> {
        // Time embedding (sinusoidal)
        let t_emb = self.sinusoidal_embedding(t);
        let t_proj = matmul_add(&t_emb, &self.time_embed_weights, &self.time_embed_bias);
        let t_proj = gelu(&t_proj);

        // Input projection
        let mut h = matmul_add(z, &self.input_proj_weights, &self.input_proj_bias);

        // Residual blocks
        for block in &self.res_blocks {
            h = self.residual_block_forward(&h, &t_proj, block);
        }

        // Output projection
        let output = matmul_add(&h, &self.output_proj_weights, &self.output_proj_bias);

        output
    }

    /// Compute velocity for a batch of states
    pub fn forward_batch(&self, z: &Array2<f64>, t: f64) -> Array2<f64> {
        let batch_size = z.nrows();
        let mut output = Array2::zeros((batch_size, self.dim));

        for i in 0..batch_size {
            let z_i = z.row(i).to_owned();
            let dz = self.forward(&z_i, t);
            for j in 0..self.dim {
                output[[i, j]] = dz[j];
            }
        }

        output
    }

    /// Compute Jacobian-vector product for trace estimation
    ///
    /// Returns (dz/dt, vjp) where vjp = v^T * (∂f/∂z)
    pub fn forward_with_vjp(&self, z: &Array1<f64>, t: f64, v: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let eps = 1e-5;
        let dz = self.forward(z, t);

        // Compute VJP using finite differences
        let mut vjp = Array1::zeros(self.dim);
        for i in 0..self.dim {
            let mut z_plus = z.clone();
            z_plus[i] += eps;
            let dz_plus = self.forward(&z_plus, t);

            for j in 0..self.dim {
                vjp[i] += v[j] * (dz_plus[j] - dz[j]) / eps;
            }
        }

        (dz, vjp)
    }

    /// Sinusoidal time embedding
    fn sinusoidal_embedding(&self, t: f64) -> Array1<f64> {
        let half = self.time_embed_dim / 2;
        let mut embedding = Array1::zeros(self.time_embed_dim);

        let max_period = 10000.0_f64;
        for i in 0..half {
            let freq = (-max_period.ln() * i as f64 / half as f64).exp();
            embedding[i] = (t * freq).cos();
            embedding[i + half] = (t * freq).sin();
        }

        embedding
    }

    /// Forward pass through a residual block
    fn residual_block_forward(
        &self,
        x: &Array1<f64>,
        t_emb: &Array1<f64>,
        block: &ResidualBlock,
    ) -> Array1<f64> {
        // Concatenate x and t_emb
        let mut concat = Array1::zeros(x.len() + t_emb.len());
        for (i, &val) in x.iter().enumerate() {
            concat[i] = val;
        }
        for (i, &val) in t_emb.iter().enumerate() {
            concat[x.len() + i] = val;
        }

        // First linear + GELU
        let h = matmul_add(&concat, &block.linear1_weights, &block.linear1_bias);
        let h = gelu(&h);

        // Second linear
        let out = matmul_add(&h, &block.linear2_weights, &block.linear2_bias);

        // Residual connection
        x + &out
    }

    /// Get all trainable parameters as a flat vector
    pub fn get_params(&self) -> Vec<f64> {
        let mut params = Vec::new();

        params.extend(self.time_embed_weights.iter().cloned());
        params.extend(self.time_embed_bias.iter().cloned());
        params.extend(self.input_proj_weights.iter().cloned());
        params.extend(self.input_proj_bias.iter().cloned());

        for block in &self.res_blocks {
            params.extend(block.linear1_weights.iter().cloned());
            params.extend(block.linear1_bias.iter().cloned());
            params.extend(block.linear2_weights.iter().cloned());
            params.extend(block.linear2_bias.iter().cloned());
        }

        params.extend(self.output_proj_weights.iter().cloned());
        params.extend(self.output_proj_bias.iter().cloned());

        params
    }

    /// Set parameters from a flat vector
    pub fn set_params(&mut self, params: &[f64]) {
        let mut idx = 0;

        // Time embedding
        for val in self.time_embed_weights.iter_mut() {
            *val = params[idx];
            idx += 1;
        }
        for val in self.time_embed_bias.iter_mut() {
            *val = params[idx];
            idx += 1;
        }

        // Input projection
        for val in self.input_proj_weights.iter_mut() {
            *val = params[idx];
            idx += 1;
        }
        for val in self.input_proj_bias.iter_mut() {
            *val = params[idx];
            idx += 1;
        }

        // Residual blocks
        for block in &mut self.res_blocks {
            for val in block.linear1_weights.iter_mut() {
                *val = params[idx];
                idx += 1;
            }
            for val in block.linear1_bias.iter_mut() {
                *val = params[idx];
                idx += 1;
            }
            for val in block.linear2_weights.iter_mut() {
                *val = params[idx];
                idx += 1;
            }
            for val in block.linear2_bias.iter_mut() {
                *val = params[idx];
                idx += 1;
            }
        }

        // Output projection
        for val in self.output_proj_weights.iter_mut() {
            *val = params[idx];
            idx += 1;
        }
        for val in self.output_proj_bias.iter_mut() {
            *val = params[idx];
            idx += 1;
        }
    }
}

/// Matrix-vector multiplication with bias: y = x @ W + b
fn matmul_add(x: &Array1<f64>, w: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let mut result = b.clone();
    let (in_dim, out_dim) = (w.nrows(), w.ncols());

    for j in 0..out_dim {
        for i in 0..in_dim {
            result[j] += x[i] * w[[i, j]];
        }
    }

    result
}

/// GELU activation function
fn gelu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| {
        0.5 * v * (1.0 + (v * 0.7978845608 * (1.0 + 0.044715 * v * v)).tanh())
    })
}

/// Generate random matrix
fn random_matrix<R: Rng>(rng: &mut R, dist: &Normal<f64>, rows: usize, cols: usize) -> Array2<f64> {
    let mut m = Array2::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            m[[i, j]] = dist.sample(rng);
        }
    }
    m
}

/// Generate random vector
fn random_vector<R: Rng>(rng: &mut R, dist: &Normal<f64>, size: usize) -> Array1<f64> {
    let mut v = Array1::zeros(size);
    for i in 0..size {
        v[i] = dist.sample(rng);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_velocity_field_forward() {
        let vf = VelocityField::new(5, 32, 2);
        let z = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let t = 0.5;

        let dz = vf.forward(&z, t);

        assert_eq!(dz.len(), 5);
        assert!(dz.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_velocity_field_with_vjp() {
        let vf = VelocityField::new(5, 32, 2);
        let z = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let v = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        let t = 0.5;

        let (dz, vjp) = vf.forward_with_vjp(&z, t, &v);

        assert_eq!(dz.len(), 5);
        assert_eq!(vjp.len(), 5);
        assert!(dz.iter().all(|&x| x.is_finite()));
        assert!(vjp.iter().all(|&x| x.is_finite()));
    }
}
