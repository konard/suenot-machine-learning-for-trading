//! Encoder network for VAE
//!
//! Maps input sequences to latent distribution parameters.

use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, Normal};

use super::config::VAEConfig;

/// Encoder network
///
/// Takes market data sequences and outputs mu and log_var for the latent space.
#[derive(Debug, Clone)]
pub struct Encoder {
    config: VAEConfig,
    // Weight matrices (simplified representation)
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
}

impl Encoder {
    /// Create a new encoder from config
    pub fn new(config: VAEConfig) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        // Initialize weights (Xavier initialization)
        let mut rng = rand::thread_rng();
        let mut prev_dim = config.input_dim * config.sequence_length;

        for &hidden_dim in &config.hidden_dims {
            let scale = (2.0 / (prev_dim + hidden_dim) as f64).sqrt();
            let normal = Normal::new(0.0, scale).unwrap();

            let weight = Array2::from_shape_fn((prev_dim, hidden_dim), |_| {
                normal.sample(&mut rng)
            });
            let bias = Array1::zeros(hidden_dim);

            weights.push(weight);
            biases.push(bias);
            prev_dim = hidden_dim;
        }

        // Output layers for mu and log_var
        let final_dim = *config.hidden_dims.last().unwrap_or(&config.input_dim);

        // Mu layer
        let mu_weight = Array2::from_shape_fn((final_dim, config.latent_dim), |_| {
            rng.gen_range(-0.1..0.1)
        });
        let mu_bias = Array1::zeros(config.latent_dim);
        weights.push(mu_weight);
        biases.push(mu_bias);

        // Log_var layer
        let logvar_weight = Array2::from_shape_fn((final_dim, config.latent_dim), |_| {
            rng.gen_range(-0.1..0.1)
        });
        let logvar_bias = Array1::zeros(config.latent_dim);
        weights.push(logvar_weight);
        biases.push(logvar_bias);

        Self {
            config,
            weights,
            biases,
        }
    }

    /// Forward pass through encoder
    ///
    /// # Arguments
    /// * `x` - Input tensor flattened from (seq_len, input_dim)
    ///
    /// # Returns
    /// Tuple of (mu, log_var) each of shape (latent_dim,)
    pub fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let mut h = x.clone();

        // Hidden layers with LeakyReLU
        let n_hidden = self.config.hidden_dims.len();
        for i in 0..n_hidden {
            h = h.dot(&self.weights[i]) + &self.biases[i];
            // LeakyReLU activation
            h.mapv_inplace(|v| if v > 0.0 { v } else { 0.2 * v });
        }

        // Output mu and log_var
        let mu = h.dot(&self.weights[n_hidden]) + &self.biases[n_hidden];
        let log_var = h.dot(&self.weights[n_hidden + 1]) + &self.biases[n_hidden + 1];

        (mu, log_var)
    }

    /// Encode a batch of sequences
    pub fn encode_batch(&self, batch: &[Array1<f64>]) -> Vec<(Array1<f64>, Array1<f64>)> {
        batch.iter().map(|x| self.forward(x)).collect()
    }

    /// Get the latent dimension
    pub fn latent_dim(&self) -> usize {
        self.config.latent_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_forward() {
        let config = VAEConfig::new(13, 8, vec![64, 32])
            .with_sequence_length(60);

        let encoder = Encoder::new(config.clone());

        // Create random input
        let input = Array1::from_vec(vec![0.1; 13 * 60]);
        let (mu, log_var) = encoder.forward(&input);

        assert_eq!(mu.len(), 8);
        assert_eq!(log_var.len(), 8);
    }
}
