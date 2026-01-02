//! Restricted Boltzmann Machine (RBM) implementation
//!
//! RBM is a generative stochastic neural network with visible and hidden units.
//! Energy function: E(v,h) = -v·W·h - a·v - b·h

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::Normal;

/// Restricted Boltzmann Machine
///
/// A two-layer network with visible and hidden units connected by weights W.
/// The energy is defined as: E(v,h) = -Σ_i a_i v_i - Σ_j b_j h_j - Σ_ij v_i W_ij h_j
#[derive(Debug, Clone)]
pub struct RBM {
    /// Weight matrix (n_visible x n_hidden)
    pub weights: Array2<f64>,
    /// Visible bias (n_visible)
    pub visible_bias: Array1<f64>,
    /// Hidden bias (n_hidden)
    pub hidden_bias: Array1<f64>,
    /// Number of visible units
    pub n_visible: usize,
    /// Number of hidden units
    pub n_hidden: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Momentum for gradient updates
    pub momentum: f64,
    /// Weight decay regularization
    pub weight_decay: f64,
}

impl RBM {
    /// Create a new RBM
    ///
    /// # Arguments
    /// * `n_visible` - Number of visible (input) units
    /// * `n_hidden` - Number of hidden units
    pub fn new(n_visible: usize, n_hidden: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.01).unwrap();

        let weights = Array2::from_shape_fn((n_visible, n_hidden), |_| rng.sample(normal));
        let visible_bias = Array1::zeros(n_visible);
        let hidden_bias = Array1::zeros(n_hidden);

        Self {
            weights,
            visible_bias,
            hidden_bias,
            n_visible,
            n_hidden,
            learning_rate: 0.01,
            momentum: 0.9,
            weight_decay: 0.0001,
        }
    }

    /// Sigmoid activation function
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Compute hidden unit probabilities given visible units
    ///
    /// P(h_j = 1 | v) = sigmoid(b_j + Σ_i v_i W_ij)
    pub fn hidden_probabilities(&self, visible: &Array1<f64>) -> Array1<f64> {
        let activation = visible.dot(&self.weights) + &self.hidden_bias;
        activation.mapv(Self::sigmoid)
    }

    /// Sample hidden units given visible units
    pub fn sample_hidden(&self, visible: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let probs = self.hidden_probabilities(visible);
        let mut rng = rand::thread_rng();
        let samples = probs.mapv(|p| if rng.gen::<f64>() < p { 1.0 } else { 0.0 });
        (probs, samples)
    }

    /// Compute visible unit probabilities given hidden units
    ///
    /// P(v_i = 1 | h) = sigmoid(a_i + Σ_j h_j W_ij)
    pub fn visible_probabilities(&self, hidden: &Array1<f64>) -> Array1<f64> {
        let activation = self.weights.dot(hidden) + &self.visible_bias;
        activation.mapv(Self::sigmoid)
    }

    /// Sample visible units given hidden units
    pub fn sample_visible(&self, hidden: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let probs = self.visible_probabilities(hidden);
        let mut rng = rand::thread_rng();
        let samples = probs.mapv(|p| if rng.gen::<f64>() < p { 1.0 } else { 0.0 });
        (probs, samples)
    }

    /// Compute free energy F(v) = -log Σ_h exp(-E(v,h))
    ///
    /// This marginalizes over hidden units analytically.
    pub fn free_energy(&self, visible: &Array1<f64>) -> f64 {
        // Visible term: -Σ_i a_i v_i
        let visible_term = -visible.dot(&self.visible_bias);

        // Hidden term: -Σ_j log(1 + exp(b_j + Σ_i v_i W_ij))
        let wx_b = visible.dot(&self.weights) + &self.hidden_bias;
        let hidden_term: f64 = -wx_b.iter().map(|&x| (1.0 + x.exp()).ln()).sum();

        visible_term + hidden_term
    }

    /// Compute free energy for a batch
    pub fn free_energy_batch(&self, data: &Array2<f64>) -> Array1<f64> {
        let n = data.nrows();
        let mut energies = Array1::zeros(n);

        for i in 0..n {
            energies[i] = self.free_energy(&data.row(i).to_owned());
        }

        energies
    }

    /// Train using Contrastive Divergence (CD-k)
    ///
    /// # Arguments
    /// * `data` - Training data (n_samples x n_visible)
    /// * `epochs` - Number of training epochs
    /// * `k` - Number of Gibbs sampling steps
    pub fn train_cd(&mut self, data: &Array2<f64>, epochs: usize, k: usize) {
        let n_samples = data.nrows();

        // Momentum terms
        let mut weight_velocity = Array2::zeros((self.n_visible, self.n_hidden));
        let mut visible_bias_velocity = Array1::zeros(self.n_visible);
        let mut hidden_bias_velocity = Array1::zeros(self.n_hidden);

        log::info!(
            "Training RBM with CD-{} for {} epochs on {} samples",
            k,
            epochs,
            n_samples
        );

        for epoch in 0..epochs {
            let mut total_error = 0.0;

            for i in 0..n_samples {
                let v0 = data.row(i).to_owned();

                // Positive phase
                let (ph0, h0) = self.sample_hidden(&v0);

                // Negative phase (k-step Gibbs sampling)
                let mut hk = h0;
                let mut vk = v0.clone();

                for _ in 0..k {
                    let (pv, v_sample) = self.sample_visible(&hk);
                    vk = pv; // Use probabilities for smoother gradients
                    let (ph, h_sample) = self.sample_hidden(&vk);
                    hk = h_sample;
                }

                let phk = self.hidden_probabilities(&vk);

                // Compute gradients
                let positive_grad = outer_product(&v0, &ph0);
                let negative_grad = outer_product(&vk, &phk);

                let weight_grad = &positive_grad - &negative_grad;
                let visible_bias_grad = &v0 - &vk;
                let hidden_bias_grad = &ph0 - &phk;

                // Update with momentum
                weight_velocity = self.momentum * &weight_velocity
                    + self.learning_rate * (&weight_grad - self.weight_decay * &self.weights);
                visible_bias_velocity =
                    self.momentum * &visible_bias_velocity + self.learning_rate * &visible_bias_grad;
                hidden_bias_velocity =
                    self.momentum * &hidden_bias_velocity + self.learning_rate * &hidden_bias_grad;

                self.weights = &self.weights + &weight_velocity;
                self.visible_bias = &self.visible_bias + &visible_bias_velocity;
                self.hidden_bias = &self.hidden_bias + &hidden_bias_velocity;

                // Reconstruction error
                let error: f64 = (&v0 - &vk).mapv(|x| x * x).sum();
                total_error += error;
            }

            if epoch % 10 == 0 || epoch == epochs - 1 {
                log::info!(
                    "Epoch {}/{}: reconstruction_error = {:.6}",
                    epoch + 1,
                    epochs,
                    total_error / n_samples as f64
                );
            }
        }
    }

    /// Reconstruct visible units from input
    pub fn reconstruct(&self, visible: &Array1<f64>) -> Array1<f64> {
        let (_, hidden) = self.sample_hidden(visible);
        let (reconstructed, _) = self.sample_visible(&hidden);
        reconstructed
    }

    /// Get reconstruction error for a sample
    pub fn reconstruction_error(&self, visible: &Array1<f64>) -> f64 {
        let reconstructed = self.reconstruct(visible);
        (&visible.clone() - &reconstructed).mapv(|x| x * x).sum()
    }

    /// Compute anomaly scores based on free energy
    ///
    /// Higher free energy = more anomalous
    pub fn anomaly_scores(&self, data: &Array2<f64>) -> Array1<f64> {
        let energies = self.free_energy_batch(data);

        // Normalize to [0, 1]
        let min = energies.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        if range < 1e-10 {
            Array1::zeros(energies.len())
        } else {
            energies.mapv(|e| (e - min) / range)
        }
    }

    /// Detect anomalies using free energy threshold
    pub fn detect_anomalies(&self, data: &Array2<f64>, threshold: f64) -> Vec<bool> {
        let scores = self.anomaly_scores(data);
        scores.iter().map(|&s| s > threshold).collect()
    }

    /// Generate samples from the model
    pub fn generate(&self, n_samples: usize, n_gibbs_steps: usize) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let mut samples = Array2::zeros((n_samples, self.n_visible));

        for i in 0..n_samples {
            // Initialize randomly
            let mut v: Array1<f64> =
                Array1::from_shape_fn(self.n_visible, |_| if rng.gen::<f64>() > 0.5 { 1.0 } else { 0.0 });

            // Run Gibbs sampling
            for _ in 0..n_gibbs_steps {
                let (_, h) = self.sample_hidden(&v);
                let (pv, _) = self.sample_visible(&h);
                v = pv;
            }

            samples.row_mut(i).assign(&v);
        }

        samples
    }
}

/// Compute outer product of two vectors
fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let m = b.len();
    let mut result = Array2::zeros((n, m));

    for i in 0..n {
        for j in 0..m {
            result[[i, j]] = a[i] * b[j];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_rbm_creation() {
        let rbm = RBM::new(10, 5);
        assert_eq!(rbm.n_visible, 10);
        assert_eq!(rbm.n_hidden, 5);
        assert_eq!(rbm.weights.shape(), &[10, 5]);
    }

    #[test]
    fn test_hidden_probabilities() {
        let rbm = RBM::new(5, 3);
        let visible = array![0.5, 0.5, 0.5, 0.5, 0.5];
        let probs = rbm.hidden_probabilities(&visible);

        assert_eq!(probs.len(), 3);
        assert!(probs.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_free_energy() {
        let rbm = RBM::new(5, 3);
        let visible = array![1.0, 0.0, 1.0, 0.0, 1.0];
        let energy = rbm.free_energy(&visible);
        assert!(energy.is_finite());
    }

    #[test]
    fn test_reconstruct() {
        let rbm = RBM::new(5, 3);
        let visible = array![0.8, 0.2, 0.9, 0.1, 0.7];
        let reconstructed = rbm.reconstruct(&visible);

        assert_eq!(reconstructed.len(), 5);
        assert!(reconstructed.iter().all(|&v| v >= 0.0 && v <= 1.0));
    }
}
