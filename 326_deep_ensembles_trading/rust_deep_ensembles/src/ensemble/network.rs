//! Neural network implementation for Gaussian MLP
//!
//! A simple feedforward neural network that outputs Gaussian parameters (mean, std).

use ndarray::{Array1, Array2};
use rand::Rng;
use rand::SeedableRng;
use rand_distr::Normal;

/// Gaussian MLP that outputs mean and standard deviation
#[derive(Debug, Clone)]
pub struct GaussianMLP {
    /// Weight matrices for hidden layers
    pub weights: Vec<Array2<f64>>,
    /// Bias vectors for hidden layers
    pub biases: Vec<Array1<f64>>,
    /// Weight matrix for mean output
    pub w_mean: Array2<f64>,
    /// Bias for mean output
    pub b_mean: Array1<f64>,
    /// Weight matrix for log_std output
    pub w_log_std: Array2<f64>,
    /// Bias for log_std output
    pub b_log_std: Array1<f64>,
    /// Cached activations for backpropagation
    activations: Vec<Array2<f64>>,
    /// Minimum standard deviation
    min_std: f64,
}

impl GaussianMLP {
    /// Create a new Gaussian MLP
    ///
    /// # Arguments
    /// * `input_dim` - Number of input features
    /// * `hidden_dims` - Dimensions of hidden layers
    /// * `seed` - Random seed for initialization
    pub fn new(input_dim: usize, hidden_dims: &[usize], seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        // Build hidden layers
        let mut prev_dim = input_dim;
        for &hidden_dim in hidden_dims {
            let std = (2.0 / (prev_dim + hidden_dim) as f64).sqrt();
            let normal = Normal::new(0.0, std).unwrap();

            let w = Array2::from_shape_fn((prev_dim, hidden_dim), |_| rng.sample(normal));
            let b = Array1::zeros(hidden_dim);

            weights.push(w);
            biases.push(b);
            prev_dim = hidden_dim;
        }

        // Output layers for mean and log_std
        let out_std = (2.0 / (prev_dim + 1) as f64).sqrt();
        let out_normal = Normal::new(0.0, out_std).unwrap();

        let w_mean = Array2::from_shape_fn((prev_dim, 1), |_| rng.sample(out_normal));
        let b_mean = Array1::zeros(1);

        let w_log_std = Array2::from_shape_fn((prev_dim, 1), |_| rng.sample(out_normal));
        let b_log_std = Array1::zeros(1);

        Self {
            weights,
            biases,
            w_mean,
            b_mean,
            w_log_std,
            b_log_std,
            activations: Vec::new(),
            min_std: 1e-6,
        }
    }

    /// ReLU activation function
    fn relu(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| v.max(0.0))
    }

    /// Derivative of ReLU
    fn relu_derivative(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }

    /// Softplus activation for positive outputs
    fn softplus(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| (1.0 + v.clamp(-20.0, 20.0).exp()).ln())
    }

    /// Forward pass through the network
    ///
    /// # Arguments
    /// * `x` - Input features [batch_size, input_dim]
    ///
    /// # Returns
    /// Tuple of (mean, std) arrays
    pub fn forward(&mut self, x: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
        self.activations.clear();
        self.activations.push(x.clone());

        // Hidden layers
        let mut h = x.clone();
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            let z = h.dot(w) + b;
            h = Self::relu(&z);
            self.activations.push(h.clone());
        }

        // Output layers
        let mean_out = h.dot(&self.w_mean) + &self.b_mean;
        let log_std_out = h.dot(&self.w_log_std) + &self.b_log_std;

        // Ensure std is positive
        let std_out = Self::softplus(&log_std_out).mapv(|v| v + self.min_std);

        // Flatten outputs
        let mean = mean_out.column(0).to_owned();
        let std = std_out.column(0).to_owned();

        (mean, std)
    }

    /// Compute NLL loss
    ///
    /// NLL = 0.5 * log(2*pi) + log(std) + 0.5 * ((target - mean) / std)^2
    pub fn nll_loss(&self, mean: &Array1<f64>, std: &Array1<f64>, target: &Array1<f64>) -> f64 {
        let n = mean.len() as f64;
        let half_log_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();

        let mut total_loss = 0.0;
        for i in 0..mean.len() {
            let z = (target[i] - mean[i]) / std[i];
            total_loss += half_log_2pi + std[i].ln() + 0.5 * z * z;
        }

        total_loss / n
    }

    /// Backward pass with gradient descent update
    ///
    /// # Arguments
    /// * `target` - True target values
    /// * `mean` - Predicted mean
    /// * `std` - Predicted standard deviation
    /// * `learning_rate` - Learning rate
    pub fn backward(
        &mut self,
        target: &Array1<f64>,
        mean: &Array1<f64>,
        std: &Array1<f64>,
        learning_rate: f64,
    ) {
        let batch_size = target.len() as f64;

        // Gradient of NLL w.r.t. mean: (mean - target) / std^2
        let d_mean: Array1<f64> = (mean - target) / std.mapv(|s| s * s) / batch_size;

        // Gradient of NLL w.r.t. std: 1/std - (target - mean)^2 / std^3
        let residual = target - mean;
        let d_std: Array1<f64> = (std.mapv(|s| 1.0 / s)
            - &residual * &residual / std.mapv(|s| s * s * s))
            / batch_size;

        // Gradient through softplus
        let log_std = std.mapv(|s| (s - self.min_std).ln());
        let d_log_std: Array1<f64> = &d_std * std / log_std.mapv(|l| 1.0 + (-l).exp());

        // Reshape for matrix operations
        let d_mean_clone = d_mean.clone();
        let d_log_std_clone = d_log_std.clone();
        let d_mean_2d = d_mean.insert_axis(ndarray::Axis(1));
        let d_log_std_2d = d_log_std.insert_axis(ndarray::Axis(1));

        // Get last hidden activation
        let h = &self.activations[self.activations.len() - 1];

        // Update output layer weights
        self.w_mean = &self.w_mean - learning_rate * h.t().dot(&d_mean_2d);
        self.b_mean = &self.b_mean - learning_rate * d_mean_clone;

        self.w_log_std = &self.w_log_std - learning_rate * h.t().dot(&d_log_std_2d);
        self.b_log_std = &self.b_log_std - learning_rate * d_log_std_clone;

        // Backpropagate through hidden layers
        let mut d_h = d_mean_2d.dot(&self.w_mean.t()) + d_log_std_2d.dot(&self.w_log_std.t());

        for i in (0..self.weights.len()).rev() {
            // Gradient through ReLU
            d_h = d_h * Self::relu_derivative(&self.activations[i + 1]);

            // Update weights and biases
            let prev_activation = &self.activations[i];
            self.weights[i] = &self.weights[i] - learning_rate * prev_activation.t().dot(&d_h);
            self.biases[i] = &self.biases[i] - learning_rate * d_h.sum_axis(ndarray::Axis(0));

            // Propagate gradient
            if i > 0 {
                d_h = d_h.dot(&self.weights[i].t());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_forward() {
        let mut mlp = GaussianMLP::new(4, &[16, 8], 42);

        let x = Array2::from_shape_fn((10, 4), |_| rand::random::<f64>());
        let (mean, std) = mlp.forward(&x);

        assert_eq!(mean.len(), 10);
        assert_eq!(std.len(), 10);

        // Check std is positive
        for &s in std.iter() {
            assert!(s > 0.0);
        }
    }

    #[test]
    fn test_nll_loss() {
        let mlp = GaussianMLP::new(4, &[8], 42);

        let mean = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let std = Array1::from_vec(vec![0.1, 0.1, 0.1]);
        let target = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let loss = mlp.nll_loss(&mean, &std, &target);

        // Loss should be relatively low when predictions match targets
        assert!(loss < 5.0);
    }
}
