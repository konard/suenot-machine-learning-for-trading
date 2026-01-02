//! Neural network energy function implementation
//!
//! Simple feedforward network that computes energy E(x) for input x.
//! Low energy = high probability (typical), High energy = low probability (anomalous)

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::Normal;

/// Activation function types
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// SiLU/Swish: x * sigmoid(x)
    SiLU,
    /// No activation (linear)
    Linear,
}

impl Activation {
    /// Apply activation function
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::SiLU => x * (1.0 / (1.0 + (-x).exp())),
            Activation::Linear => x,
        }
    }

    /// Apply activation derivative
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Activation::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            Activation::Tanh => 1.0 - x.tanh().powi(2),
            Activation::SiLU => {
                let s = 1.0 / (1.0 + (-x).exp());
                s + x * s * (1.0 - s)
            }
            Activation::Linear => 1.0,
        }
    }
}

/// Single layer in the energy network
#[derive(Debug, Clone)]
pub struct Layer {
    /// Weight matrix (input_dim x output_dim)
    pub weights: Array2<f64>,
    /// Bias vector (output_dim)
    pub bias: Array1<f64>,
    /// Activation function
    pub activation: Activation,
}

impl Layer {
    /// Create a new layer with random initialization
    pub fn new(input_dim: usize, output_dim: usize, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, (2.0 / input_dim as f64).sqrt()).unwrap();

        let weights = Array2::from_shape_fn((input_dim, output_dim), |_| rng.sample(normal));
        let bias = Array1::zeros(output_dim);

        Self {
            weights,
            bias,
            activation,
        }
    }

    /// Forward pass through the layer
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let z = input.dot(&self.weights) + &self.bias;
        z.mapv(|x| self.activation.apply(x))
    }

    /// Forward pass for batch
    pub fn forward_batch(&self, input: &Array2<f64>) -> Array2<f64> {
        let z = input.dot(&self.weights) + &self.bias;
        z.mapv(|x| self.activation.apply(x))
    }
}

/// Neural network energy function
///
/// Computes energy E(x) for input x. The probability is:
/// p(x) = exp(-E(x)) / Z
///
/// where Z is the partition function (normalizing constant).
#[derive(Debug, Clone)]
pub struct EnergyModel {
    /// Network layers
    pub layers: Vec<Layer>,
    /// Input dimension
    pub input_dim: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Regularization coefficient
    pub reg_coef: f64,
}

impl EnergyModel {
    /// Create a new energy model
    ///
    /// # Arguments
    /// * `input_dim` - Number of input features
    pub fn new(input_dim: usize) -> Self {
        Self::with_architecture(input_dim, &[64, 32, 16])
    }

    /// Create an energy model with custom architecture
    ///
    /// # Arguments
    /// * `input_dim` - Number of input features
    /// * `hidden_dims` - Sizes of hidden layers
    pub fn with_architecture(input_dim: usize, hidden_dims: &[usize]) -> Self {
        let mut layers = Vec::new();
        let mut prev_dim = input_dim;

        for &hidden_dim in hidden_dims {
            layers.push(Layer::new(prev_dim, hidden_dim, Activation::SiLU));
            prev_dim = hidden_dim;
        }

        // Output layer produces scalar energy
        layers.push(Layer::new(prev_dim, 1, Activation::Linear));

        Self {
            layers,
            input_dim,
            learning_rate: 0.001,
            reg_coef: 0.01,
        }
    }

    /// Compute energy for a single input
    pub fn energy(&self, input: &Array1<f64>) -> f64 {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x[0]
    }

    /// Compute energy for a batch of inputs
    pub fn energy_batch(&self, input: &Array2<f64>) -> Array1<f64> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward_batch(&x);
        }
        x.column(0).to_owned()
    }

    /// Compute approximate log probability (up to normalizing constant)
    pub fn log_prob(&self, input: &Array1<f64>) -> f64 {
        -self.energy(input)
    }

    /// Train the model using contrastive divergence
    ///
    /// # Arguments
    /// * `data` - Training data (n_samples x n_features)
    /// * `epochs` - Number of training epochs
    pub fn train(&mut self, data: &Array2<f64>, epochs: usize) {
        self.train_with_config(data, epochs, 10, 0.01)
    }

    /// Train with custom configuration
    ///
    /// # Arguments
    /// * `data` - Training data
    /// * `epochs` - Number of training epochs
    /// * `n_steps` - Number of Langevin dynamics steps for negative sampling
    /// * `step_size` - Step size for Langevin dynamics
    pub fn train_with_config(
        &mut self,
        data: &Array2<f64>,
        epochs: usize,
        n_steps: usize,
        step_size: f64,
    ) {
        let n_samples = data.nrows();
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        log::info!("Training EBM for {} epochs on {} samples", epochs, n_samples);

        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for i in 0..n_samples {
                let x_real = data.row(i).to_owned();

                // Energy of real sample
                let energy_real = self.energy(&x_real);

                // Generate negative sample via Langevin dynamics
                let mut x_neg: Array1<f64> =
                    Array1::from_shape_fn(self.input_dim, |_| rng.sample(normal));

                for _ in 0..n_steps {
                    let grad = self.energy_gradient(&x_neg);
                    let noise: Array1<f64> =
                        Array1::from_shape_fn(self.input_dim, |_| rng.sample(normal) * 0.01);
                    x_neg = &x_neg - step_size * &grad + noise;
                }

                // Energy of negative sample
                let energy_neg = self.energy(&x_neg);

                // Contrastive divergence loss
                let loss = energy_real - energy_neg;
                total_loss += loss;

                // Update weights (simplified gradient descent)
                self.update_weights(&x_real, &x_neg);
            }

            if epoch % 10 == 0 || epoch == epochs - 1 {
                log::info!(
                    "Epoch {}/{}: avg_loss = {:.6}",
                    epoch + 1,
                    epochs,
                    total_loss / n_samples as f64
                );
            }
        }
    }

    /// Compute energy gradient with respect to input (numerical approximation)
    fn energy_gradient(&self, input: &Array1<f64>) -> Array1<f64> {
        let eps = 1e-5;
        let mut grad = Array1::zeros(self.input_dim);

        for i in 0..self.input_dim {
            let mut x_plus = input.clone();
            let mut x_minus = input.clone();
            x_plus[i] += eps;
            x_minus[i] -= eps;

            grad[i] = (self.energy(&x_plus) - self.energy(&x_minus)) / (2.0 * eps);
        }

        grad
    }

    /// Update weights using contrastive divergence
    fn update_weights(&mut self, x_real: &Array1<f64>, x_neg: &Array1<f64>) {
        // Simplified weight update using finite differences
        let eps = 1e-4;

        for layer in &mut self.layers {
            // Update weights
            for i in 0..layer.weights.nrows() {
                for j in 0..layer.weights.ncols() {
                    let original = layer.weights[[i, j]];

                    layer.weights[[i, j]] = original + eps;
                    let e_plus = self.energy(x_real) - self.energy(x_neg);

                    layer.weights[[i, j]] = original - eps;
                    let e_minus = self.energy(x_real) - self.energy(x_neg);

                    layer.weights[[i, j]] = original;

                    let grad = (e_plus - e_minus) / (2.0 * eps);
                    layer.weights[[i, j]] -= self.learning_rate * (grad + self.reg_coef * original);
                }
            }

            // Update biases
            for j in 0..layer.bias.len() {
                let original = layer.bias[j];

                layer.bias[j] = original + eps;
                let e_plus = self.energy(x_real) - self.energy(x_neg);

                layer.bias[j] = original - eps;
                let e_minus = self.energy(x_real) - self.energy(x_neg);

                layer.bias[j] = original;

                let grad = (e_plus - e_minus) / (2.0 * eps);
                layer.bias[j] -= self.learning_rate * grad;
            }
        }
    }

    /// Compute anomaly scores for data
    ///
    /// Higher score = more anomalous
    pub fn anomaly_scores(&self, data: &Array2<f64>) -> Array1<f64> {
        let energies = self.energy_batch(data);

        // Normalize to [0, 1] range
        let min = energies.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        if range < 1e-10 {
            Array1::zeros(energies.len())
        } else {
            energies.mapv(|e| (e - min) / range)
        }
    }

    /// Detect anomalies using threshold
    ///
    /// Returns vector of booleans (true = anomaly)
    pub fn detect_anomalies(&self, data: &Array2<f64>, threshold: f64) -> Vec<bool> {
        let scores = self.anomaly_scores(data);
        scores.iter().map(|&s| s > threshold).collect()
    }

    /// Get energy statistics for a dataset
    pub fn energy_stats(&self, data: &Array2<f64>) -> EnergyStats {
        let energies = self.energy_batch(data);
        EnergyStats::from_energies(&energies)
    }
}

/// Statistics about energy distribution
#[derive(Debug, Clone)]
pub struct EnergyStats {
    /// Mean energy
    pub mean: f64,
    /// Standard deviation of energy
    pub std: f64,
    /// Minimum energy
    pub min: f64,
    /// Maximum energy
    pub max: f64,
    /// Median energy
    pub median: f64,
}

impl EnergyStats {
    /// Compute statistics from energy values
    pub fn from_energies(energies: &Array1<f64>) -> Self {
        let n = energies.len() as f64;
        let mean = energies.sum() / n;
        let variance = energies.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();
        let min = energies.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut sorted: Vec<f64> = energies.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        Self {
            mean,
            std,
            min,
            max,
            median,
        }
    }

    /// Check if a given energy is anomalous (beyond n standard deviations)
    pub fn is_anomalous(&self, energy: f64, n_std: f64) -> bool {
        (energy - self.mean).abs() > n_std * self.std
    }

    /// Get normalized energy score
    pub fn normalize(&self, energy: f64) -> f64 {
        (energy - self.mean) / self.std
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_energy_model_creation() {
        let model = EnergyModel::new(10);
        assert_eq!(model.input_dim, 10);
        assert_eq!(model.layers.len(), 4); // 3 hidden + 1 output
    }

    #[test]
    fn test_energy_computation() {
        let model = EnergyModel::new(5);
        let input = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let energy = model.energy(&input);
        assert!(energy.is_finite());
    }

    #[test]
    fn test_batch_energy() {
        let model = EnergyModel::new(3);
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let energies = model.energy_batch(&data);
        assert_eq!(energies.len(), 3);
    }

    #[test]
    fn test_activation_functions() {
        assert_eq!(Activation::ReLU.apply(-1.0), 0.0);
        assert_eq!(Activation::ReLU.apply(1.0), 1.0);

        let sigmoid_0 = Activation::Sigmoid.apply(0.0);
        assert!((sigmoid_0 - 0.5).abs() < 1e-10);
    }
}
