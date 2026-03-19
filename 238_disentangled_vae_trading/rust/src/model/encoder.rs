//! VAE Encoder
//!
//! Maps input feature sequences to latent distributions (mu, logvar).

use ndarray::{Array1, Array2};
use rand::Rng;
use crate::model::config::DisentangledVAEConfig;

/// VAE Encoder network
///
/// Takes flattened input features and produces mean (mu) and
/// log-variance (logvar) vectors for the latent distribution.
#[derive(Debug, Clone)]
pub struct Encoder {
    /// Weight matrices for hidden layers
    weights: Vec<Array2<f64>>,
    /// Bias vectors for hidden layers
    biases: Vec<Array1<f64>>,
    /// Weight matrix for mu output
    w_mu: Array2<f64>,
    /// Bias for mu output
    b_mu: Array1<f64>,
    /// Weight matrix for logvar output
    w_logvar: Array2<f64>,
    /// Bias for logvar output
    b_logvar: Array1<f64>,
    /// Input dimension (seq_len * input_dim)
    flat_input_dim: usize,
    /// Latent dimension
    latent_dim: usize,
}

impl Encoder {
    /// Create a new encoder from config
    pub fn new(config: &DisentangledVAEConfig) -> Self {
        let flat_input_dim = config.seq_len * config.input_dim;
        let mut rng = rand::thread_rng();

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        // Build hidden layers
        let mut prev_dim = flat_input_dim;
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

        // Mu and logvar projection layers
        let scale = (2.0 / prev_dim as f64).sqrt();
        let w_mu = Array2::from_shape_fn((prev_dim, config.latent_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let b_mu = Array1::zeros(config.latent_dim);

        let w_logvar = Array2::from_shape_fn((prev_dim, config.latent_dim), |_| {
            rng.gen_range(-scale..scale)
        });
        let b_logvar = Array1::zeros(config.latent_dim);

        Self {
            weights,
            biases,
            w_mu,
            b_mu,
            w_logvar,
            b_logvar,
            flat_input_dim,
            latent_dim: config.latent_dim,
        }
    }

    /// Forward pass: input -> (mu, logvar)
    ///
    /// # Arguments
    /// * `input` - Flattened input vector of shape [seq_len * input_dim]
    ///
    /// # Returns
    /// Tuple of (mu, logvar) each of shape [latent_dim]
    pub fn forward(&self, input: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        assert_eq!(
            input.len(),
            self.flat_input_dim,
            "Input dim mismatch: expected {}, got {}",
            self.flat_input_dim,
            input.len()
        );

        let mut h = input.clone();

        // Hidden layers with ReLU activation
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            h = h.dot(w) + b;
            // ReLU
            h.mapv_inplace(|x| x.max(0.0));
        }

        // Project to mu and logvar
        let mu = h.dot(&self.w_mu) + &self.b_mu;
        let logvar = h.dot(&self.w_logvar) + &self.b_logvar;

        // Clamp logvar for numerical stability
        let logvar = logvar.mapv(|x| x.clamp(-10.0, 10.0));

        (mu, logvar)
    }

    /// Forward pass for a batch
    ///
    /// # Arguments
    /// * `inputs` - Batch of flattened inputs [batch_size, seq_len * input_dim]
    ///
    /// # Returns
    /// Tuple of (mus, logvars) each of shape [batch_size, latent_dim]
    pub fn forward_batch(&self, inputs: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let batch_size = inputs.nrows();
        let mut mus = Array2::zeros((batch_size, self.latent_dim));
        let mut logvars = Array2::zeros((batch_size, self.latent_dim));

        for i in 0..batch_size {
            let input = inputs.row(i).to_owned();
            let (mu, logvar) = self.forward(&input);
            mus.row_mut(i).assign(&mu);
            logvars.row_mut(i).assign(&logvar);
        }

        (mus, logvars)
    }

    /// Get the flat input dimension
    pub fn flat_input_dim(&self) -> usize {
        self.flat_input_dim
    }

    /// Get the latent dimension
    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_forward() {
        let config = DisentangledVAEConfig {
            input_dim: 8,
            seq_len: 10,
            latent_dim: 4,
            hidden_dim: 32,
            num_hidden_layers: 2,
            ..Default::default()
        };

        let encoder = Encoder::new(&config);
        let input = Array1::zeros(80); // 10 * 8

        let (mu, logvar) = encoder.forward(&input);

        assert_eq!(mu.len(), 4);
        assert_eq!(logvar.len(), 4);
    }

    #[test]
    fn test_encoder_batch() {
        let config = DisentangledVAEConfig {
            input_dim: 8,
            seq_len: 10,
            latent_dim: 4,
            hidden_dim: 32,
            num_hidden_layers: 2,
            ..Default::default()
        };

        let encoder = Encoder::new(&config);
        let inputs = Array2::zeros((5, 80));

        let (mus, logvars) = encoder.forward_batch(&inputs);

        assert_eq!(mus.dim(), (5, 4));
        assert_eq!(logvars.dim(), (5, 4));
    }

    #[test]
    fn test_logvar_clamped() {
        let config = DisentangledVAEConfig {
            input_dim: 4,
            seq_len: 5,
            latent_dim: 2,
            hidden_dim: 16,
            num_hidden_layers: 1,
            ..Default::default()
        };

        let encoder = Encoder::new(&config);
        let input = Array1::from_elem(20, 1000.0); // extreme values

        let (_, logvar) = encoder.forward(&input);

        for &v in logvar.iter() {
            assert!(v >= -10.0 && v <= 10.0);
        }
    }
}
