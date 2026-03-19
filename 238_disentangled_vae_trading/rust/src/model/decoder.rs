//! VAE Decoder
//!
//! Maps latent vectors back to reconstructed input space.

use ndarray::{Array1, Array2};
use rand::Rng;
use crate::model::config::DisentangledVAEConfig;

/// VAE Decoder network
///
/// Takes a latent vector and reconstructs the original input features.
#[derive(Debug, Clone)]
pub struct Decoder {
    /// Weight matrices for hidden layers
    weights: Vec<Array2<f64>>,
    /// Bias vectors for hidden layers
    biases: Vec<Array1<f64>>,
    /// Weight matrix for output
    w_out: Array2<f64>,
    /// Bias for output
    b_out: Array1<f64>,
    /// Latent dimension
    latent_dim: usize,
    /// Output dimension (seq_len * input_dim)
    output_dim: usize,
}

impl Decoder {
    /// Create a new decoder from config
    pub fn new(config: &DisentangledVAEConfig) -> Self {
        let output_dim = config.seq_len * config.input_dim;
        let mut rng = rand::thread_rng();

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        // Build hidden layers (mirror of encoder)
        let mut prev_dim = config.latent_dim;
        for _ in 0..config.num_hidden_layers {
            let scale = (2.0 / prev_dim as f64).sqrt();
            let w = Array2::from_shape_fn((prev_dim, config.hidden_dim), |_| {
                rng.gen_range(-scale..scale)
            });
            let b = Array1::zeros(config.hidden_dim);
            weights.push(w);
            biases.push(b);
            prev_dim = config.hidden_dim;
        }

        // Output projection
        let scale = (2.0 / prev_dim as f64).sqrt();
        let w_out = Array2::from_shape_fn((prev_dim, output_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let b_out = Array1::zeros(output_dim);

        Self {
            weights,
            biases,
            w_out,
            b_out,
            latent_dim: config.latent_dim,
            output_dim,
        }
    }

    /// Forward pass: latent -> reconstruction
    ///
    /// # Arguments
    /// * `z` - Latent vector of shape [latent_dim]
    ///
    /// # Returns
    /// Reconstructed output of shape [seq_len * input_dim]
    pub fn forward(&self, z: &Array1<f64>) -> Array1<f64> {
        assert_eq!(
            z.len(),
            self.latent_dim,
            "Latent dim mismatch: expected {}, got {}",
            self.latent_dim,
            z.len()
        );

        let mut h = z.clone();

        // Hidden layers with ReLU activation
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            h = h.dot(w) + b;
            h.mapv_inplace(|x| x.max(0.0));
        }

        // Output projection (no activation for reconstruction)
        h.dot(&self.w_out) + &self.b_out
    }

    /// Forward pass for a batch
    ///
    /// # Arguments
    /// * `zs` - Batch of latent vectors [batch_size, latent_dim]
    ///
    /// # Returns
    /// Batch of reconstructions [batch_size, seq_len * input_dim]
    pub fn forward_batch(&self, zs: &Array2<f64>) -> Array2<f64> {
        let batch_size = zs.nrows();
        let mut outputs = Array2::zeros((batch_size, self.output_dim));

        for i in 0..batch_size {
            let z = zs.row(i).to_owned();
            let out = self.forward(&z);
            outputs.row_mut(i).assign(&out);
        }

        outputs
    }

    /// Get the output dimension
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_forward() {
        let config = DisentangledVAEConfig {
            input_dim: 8,
            seq_len: 10,
            latent_dim: 4,
            hidden_dim: 32,
            num_hidden_layers: 2,
            ..Default::default()
        };

        let decoder = Decoder::new(&config);
        let z = Array1::zeros(4);

        let output = decoder.forward(&z);

        assert_eq!(output.len(), 80); // 10 * 8
    }

    #[test]
    fn test_decoder_batch() {
        let config = DisentangledVAEConfig {
            input_dim: 8,
            seq_len: 10,
            latent_dim: 4,
            hidden_dim: 32,
            num_hidden_layers: 2,
            ..Default::default()
        };

        let decoder = Decoder::new(&config);
        let zs = Array2::zeros((5, 4));

        let outputs = decoder.forward_batch(&zs);

        assert_eq!(outputs.dim(), (5, 80));
    }

    #[test]
    fn test_decoder_output_dim() {
        let config = DisentangledVAEConfig {
            input_dim: 4,
            seq_len: 20,
            latent_dim: 8,
            ..Default::default()
        };

        let decoder = Decoder::new(&config);
        assert_eq!(decoder.output_dim(), 80); // 20 * 4
    }
}
