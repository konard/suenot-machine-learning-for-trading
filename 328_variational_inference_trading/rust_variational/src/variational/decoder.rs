//! Decoder network for VAE
//!
//! Maps latent codes back to output space.

use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, Normal};

use super::config::VAEConfig;

/// Decoder network
///
/// Takes latent vectors and predicts:
/// - Reconstructed input features
/// - Return distribution (mu, sigma)
/// - Regime probabilities
#[derive(Debug, Clone)]
pub struct Decoder {
    config: VAEConfig,
    // Weight matrices
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
    // Output head weights
    return_mu_weight: Array2<f64>,
    return_mu_bias: Array1<f64>,
    return_logvar_weight: Array2<f64>,
    return_logvar_bias: Array1<f64>,
    regime_weight: Array2<f64>,
    regime_bias: Array1<f64>,
}

impl Decoder {
    /// Create a new decoder from config
    pub fn new(config: VAEConfig) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        // Initialize weights
        let mut rng = rand::thread_rng();

        // Reversed hidden dims for decoder
        let hidden_dims: Vec<usize> = config.hidden_dims.iter().rev().cloned().collect();
        let mut prev_dim = config.latent_dim;

        for &hidden_dim in &hidden_dims {
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

        // Output projection to reconstructed features
        let output_dim = config.input_dim * config.sequence_length;
        let recon_weight = Array2::from_shape_fn((prev_dim, output_dim), |_| {
            rng.gen_range(-0.1..0.1)
        });
        let recon_bias = Array1::zeros(output_dim);
        weights.push(recon_weight);
        biases.push(recon_bias);

        // Return prediction head
        let return_mu_weight = Array2::from_shape_fn((prev_dim, 1), |_| {
            rng.gen_range(-0.1..0.1)
        });
        let return_mu_bias = Array1::zeros(1);
        let return_logvar_weight = Array2::from_shape_fn((prev_dim, 1), |_| {
            rng.gen_range(-0.1..0.1)
        });
        let return_logvar_bias = Array1::zeros(1);

        // Regime classification head
        let regime_weight = Array2::from_shape_fn((prev_dim, config.n_regimes), |_| {
            rng.gen_range(-0.1..0.1)
        });
        let regime_bias = Array1::zeros(config.n_regimes);

        Self {
            config,
            weights,
            biases,
            return_mu_weight,
            return_mu_bias,
            return_logvar_weight,
            return_logvar_bias,
            regime_weight,
            regime_bias,
        }
    }

    /// Forward pass through decoder
    ///
    /// # Arguments
    /// * `z` - Latent vector of shape (latent_dim,)
    ///
    /// # Returns
    /// Tuple of:
    /// - x_recon: Reconstructed features
    /// - return_mu: Predicted return mean
    /// - return_logvar: Predicted return log variance
    /// - regime_logits: Regime classification logits
    pub fn forward(&self, z: &Array1<f64>) -> DecoderOutput {
        let mut h = z.clone();

        // Hidden layers with LeakyReLU
        let n_hidden = self.config.hidden_dims.len();
        for i in 0..n_hidden {
            h = h.dot(&self.weights[i]) + &self.biases[i];
            // LeakyReLU activation
            h.mapv_inplace(|v| if v > 0.0 { v } else { 0.2 * v });
        }

        // Reconstruction
        let x_recon = h.dot(&self.weights[n_hidden]) + &self.biases[n_hidden];

        // Return prediction
        let return_mu = h.dot(&self.return_mu_weight) + &self.return_mu_bias;
        let return_logvar = h.dot(&self.return_logvar_weight) + &self.return_logvar_bias;

        // Regime logits
        let regime_logits = h.dot(&self.regime_weight) + &self.regime_bias;

        DecoderOutput {
            x_recon,
            return_mu: return_mu[0],
            return_logvar: return_logvar[0],
            regime_logits,
        }
    }

    /// Decode a batch of latent vectors
    pub fn decode_batch(&self, batch: &[Array1<f64>]) -> Vec<DecoderOutput> {
        batch.iter().map(|z| self.forward(z)).collect()
    }
}

/// Output from decoder forward pass
#[derive(Debug, Clone)]
pub struct DecoderOutput {
    /// Reconstructed features
    pub x_recon: Array1<f64>,
    /// Predicted return mean
    pub return_mu: f64,
    /// Predicted return log variance
    pub return_logvar: f64,
    /// Regime classification logits
    pub regime_logits: Array1<f64>,
}

impl DecoderOutput {
    /// Get predicted return standard deviation
    pub fn return_std(&self) -> f64 {
        (0.5 * self.return_logvar).exp()
    }

    /// Get regime probabilities (softmax of logits)
    pub fn regime_probs(&self) -> Array1<f64> {
        let max_logit = self.regime_logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_logits: Vec<f64> = self.regime_logits.iter()
            .map(|&x| (x - max_logit).exp())
            .collect();
        let sum: f64 = exp_logits.iter().sum();
        Array1::from_vec(exp_logits.iter().map(|x| x / sum).collect())
    }

    /// Get most likely regime
    pub fn predicted_regime(&self) -> usize {
        self.regime_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_forward() {
        let config = VAEConfig::new(13, 8, vec![64, 32])
            .with_sequence_length(60);

        let decoder = Decoder::new(config);

        // Create random latent vector
        let z = Array1::from_vec(vec![0.1; 8]);
        let output = decoder.forward(&z);

        assert_eq!(output.x_recon.len(), 13 * 60);
        assert!(output.return_std() > 0.0);
        assert_eq!(output.regime_logits.len(), 3);
    }

    #[test]
    fn test_regime_probs() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 0.5]);
        let output = DecoderOutput {
            x_recon: Array1::zeros(10),
            return_mu: 0.0,
            return_logvar: 0.0,
            regime_logits: logits,
        };

        let probs = output.regime_probs();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert_eq!(output.predicted_regime(), 1); // Highest logit is index 1
    }
}
