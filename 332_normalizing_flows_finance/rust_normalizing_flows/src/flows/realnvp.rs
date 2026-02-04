//! RealNVP (Real-valued Non-Volume Preserving) flow implementation.

use super::config::FlowConfig;
use super::layers::{AffineCouplingLayer, Permutation};
use super::traits::{NormalizingFlow, TrainingHistory};
use anyhow::Result;
use ndarray::Array1;
use rand::Rng;
use rand_distr::StandardNormal;
use std::f64::consts::PI;

/// RealNVP normalizing flow
#[derive(Debug)]
pub struct RealNVP {
    /// Model configuration
    config: FlowConfig,
    /// Affine coupling layers
    coupling_layers: Vec<AffineCouplingLayer>,
    /// Permutation layers
    permutations: Vec<Permutation>,
}

impl RealNVP {
    /// Create a new RealNVP flow with the given configuration
    pub fn new(config: FlowConfig) -> Self {
        let dim = config.dim;
        let n_layers = config.n_layers;
        let hidden_dim = config.hidden_dim;
        let n_hidden = config.n_hidden_layers;

        let mut coupling_layers = Vec::new();
        let mut permutations = Vec::new();

        for i in 0..n_layers {
            // Create alternating masks
            let mask = if i % 2 == 0 {
                create_mask(dim, true) // First half masked
            } else {
                create_mask(dim, false) // Second half masked
            };

            coupling_layers.push(AffineCouplingLayer::new(dim, mask, hidden_dim, n_hidden));

            // Add permutation between layers (except after last)
            if i < n_layers - 1 {
                permutations.push(Permutation::shuffle(dim));
            }
        }

        Self {
            config,
            coupling_layers,
            permutations,
        }
    }

    /// Get a reference to the configuration
    pub fn config(&self) -> &FlowConfig {
        &self.config
    }

    /// Compute the negative log-likelihood loss
    pub fn nll_loss(&self, batch: &[f64]) -> f64 {
        let n = batch.len();
        let mut total_nll = 0.0;

        for &x in batch {
            let input = Array1::from_vec(vec![x]);
            let log_prob = self.log_prob(&input);
            total_nll -= log_prob;
        }

        total_nll / n as f64
    }

    /// Simple gradient descent training step
    fn training_step(&mut self, batch: &[f64], learning_rate: f64) -> f64 {
        let eps = 1e-4; // Finite difference epsilon
        let loss_before = self.nll_loss(batch);

        // Update each coupling layer's network parameters
        for layer in &mut self.coupling_layers {
            let network = layer.network_mut();
            let (weights, biases, sw, sb, tw, tb) = network.parameters_mut();

            // Update hidden layer weights
            for w in weights.iter_mut() {
                for i in 0..w.nrows() {
                    for j in 0..w.ncols() {
                        let original = w[[i, j]];

                        // Compute gradient via finite differences
                        w[[i, j]] = original + eps;
                        let loss_plus = self.compute_loss_for_grad(batch, layer);
                        w[[i, j]] = original - eps;
                        let loss_minus = self.compute_loss_for_grad(batch, layer);
                        w[[i, j]] = original;

                        let grad = (loss_plus - loss_minus) / (2.0 * eps);
                        w[[i, j]] -= learning_rate * grad.clamp(-1.0, 1.0);
                    }
                }
            }

            // Update hidden layer biases
            for b in biases.iter_mut() {
                for i in 0..b.len() {
                    let original = b[i];

                    b[i] = original + eps;
                    let loss_plus = self.compute_loss_for_grad(batch, layer);
                    b[i] = original - eps;
                    let loss_minus = self.compute_loss_for_grad(batch, layer);
                    b[i] = original;

                    let grad = (loss_plus - loss_minus) / (2.0 * eps);
                    b[i] -= learning_rate * grad.clamp(-1.0, 1.0);
                }
            }

            // Update scale weights (simplified - just a few)
            for i in 0..sw.nrows().min(5) {
                for j in 0..sw.ncols().min(10) {
                    let original = sw[[i, j]];
                    sw[[i, j]] = original + eps;
                    let loss_plus = self.compute_loss_for_grad(batch, layer);
                    sw[[i, j]] = original - eps;
                    let loss_minus = self.compute_loss_for_grad(batch, layer);
                    sw[[i, j]] = original;

                    let grad = (loss_plus - loss_minus) / (2.0 * eps);
                    sw[[i, j]] -= learning_rate * grad.clamp(-1.0, 1.0);
                }
            }

            // Update translation weights
            for i in 0..tw.nrows().min(5) {
                for j in 0..tw.ncols().min(10) {
                    let original = tw[[i, j]];
                    tw[[i, j]] = original + eps;
                    let loss_plus = self.compute_loss_for_grad(batch, layer);
                    tw[[i, j]] = original - eps;
                    let loss_minus = self.compute_loss_for_grad(batch, layer);
                    tw[[i, j]] = original;

                    let grad = (loss_plus - loss_minus) / (2.0 * eps);
                    tw[[i, j]] -= learning_rate * grad.clamp(-1.0, 1.0);
                }
            }
        }

        loss_before
    }

    /// Helper to compute loss for gradient computation
    fn compute_loss_for_grad(&self, batch: &[f64], _layer: &AffineCouplingLayer) -> f64 {
        self.nll_loss(batch)
    }
}

impl NormalizingFlow for RealNVP {
    fn forward(&self, z: &Array1<f64>) -> Array1<f64> {
        let mut x = z.clone();

        // Apply layers in reverse order (from latent to data)
        for i in (0..self.coupling_layers.len()).rev() {
            // Inverse permutation (if not first layer)
            if i < self.permutations.len() {
                x = self.permutations[i].inverse(&x);
            }

            // Inverse coupling
            let (x_new, _) = self.coupling_layers[i].inverse(&x);
            x = x_new;
        }

        x
    }

    fn inverse(&self, x: &Array1<f64>) -> (Array1<f64>, f64) {
        let mut z = x.clone();
        let mut total_log_det = 0.0;

        // Apply layers in forward order (from data to latent)
        for i in 0..self.coupling_layers.len() {
            // Coupling layer
            let (z_new, log_det) = self.coupling_layers[i].forward(&z);
            z = z_new;
            total_log_det += log_det;

            // Permutation (if not last layer)
            if i < self.permutations.len() {
                z = self.permutations[i].forward(&z);
            }
        }

        (z, total_log_det)
    }

    fn log_prob(&self, x: &Array1<f64>) -> f64 {
        let (z, log_det) = self.inverse(x);

        // Log prob under standard normal base distribution
        let log_pz: f64 = z.iter().map(|&zi| -0.5 * (zi * zi + (2.0 * PI).ln())).sum();

        log_pz + log_det
    }

    fn sample(&self, n_samples: usize) -> Vec<Array1<f64>> {
        let mut rng = rand::thread_rng();
        let dim = self.config.dim;

        (0..n_samples)
            .map(|_| {
                // Sample from standard normal
                let z: Array1<f64> = (0..dim)
                    .map(|_| rng.sample::<f64, _>(StandardNormal))
                    .collect::<Vec<_>>()
                    .into();

                // Transform through flow
                self.forward(&z)
            })
            .collect()
    }

    fn train(&mut self, data: &[f64], epochs: usize) -> Result<TrainingHistory> {
        let mut history = TrainingHistory::default();

        let n = data.len();
        let batch_size = 64.min(n);
        let learning_rate = self.config.learning_rate;

        // Split into train/val
        let val_size = (n as f64 * 0.15) as usize;
        let train_data = &data[..n - val_size];
        let val_data = &data[n - val_size..];

        log::info!("Training RealNVP: {} epochs, {} train samples", epochs, train_data.len());

        let mut best_val_loss = f64::INFINITY;
        let mut patience_counter = 0;
        let patience = 20;

        for epoch in 0..epochs {
            // Shuffle training data
            let mut indices: Vec<usize> = (0..train_data.len()).collect();
            let mut rng = rand::thread_rng();
            for i in (1..indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }

            // Train on batches
            let mut epoch_loss = 0.0;
            let n_batches = (train_data.len() + batch_size - 1) / batch_size;

            for batch_idx in 0..n_batches {
                let start = batch_idx * batch_size;
                let end = (start + batch_size).min(train_data.len());

                let batch: Vec<f64> = indices[start..end]
                    .iter()
                    .map(|&i| train_data[i])
                    .collect();

                let loss = self.training_step(&batch, learning_rate);
                epoch_loss += loss;
            }

            epoch_loss /= n_batches as f64;

            // Validation loss
            let val_loss = self.nll_loss(val_data);

            history.train_loss.push(epoch_loss);
            history.val_loss.push(val_loss);
            history.learning_rate.push(learning_rate);

            // Early stopping
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= patience {
                    log::info!("Early stopping at epoch {}", epoch + 1);
                    break;
                }
            }

            if (epoch + 1) % 10 == 0 {
                log::info!(
                    "Epoch {}/{}: train_loss={:.4}, val_loss={:.4}",
                    epoch + 1,
                    epochs,
                    epoch_loss,
                    val_loss
                );
            }
        }

        Ok(history)
    }

    fn dim(&self) -> usize {
        self.config.dim
    }
}

/// Create a mask for coupling layers
fn create_mask(dim: usize, first_half: bool) -> Array1<f64> {
    let mut mask = Array1::zeros(dim);
    let half = dim / 2;

    if first_half {
        for i in 0..half {
            mask[i] = 1.0;
        }
    } else {
        for i in half..dim {
            mask[i] = 1.0;
        }
    }

    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realnvp_creation() {
        let config = FlowConfig::new(2);
        let flow = RealNVP::new(config);
        assert_eq!(flow.dim(), 2);
    }

    #[test]
    fn test_forward_inverse() {
        let config = FlowConfig::new(2).with_n_layers(4);
        let flow = RealNVP::new(config);

        let z = Array1::from_vec(vec![0.5, -0.3]);
        let x = flow.forward(&z);
        let (z_recovered, _) = flow.inverse(&x);

        // Check reconstruction
        for i in 0..2 {
            assert!((z[i] - z_recovered[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_sampling() {
        let config = FlowConfig::new(1);
        let flow = RealNVP::new(config);

        let samples = flow.sample(100);
        assert_eq!(samples.len(), 100);
    }
}
