//! Bayesian Neural Network Layers.
//!
//! Implements weight uncertainty using variational inference.
//! Based on "Weight Uncertainty in Neural Networks" (Blundell et al., 2015).

use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal};
use std::f64::consts::PI;

/// Bayesian Linear Layer with weight uncertainty.
///
/// Each weight and bias is represented as a distribution:
/// w ~ N(mu, sigma^2) where sigma = softplus(rho)
#[derive(Debug, Clone)]
pub struct BayesianLinear {
    /// Input dimension
    pub in_features: usize,
    /// Output dimension
    pub out_features: usize,

    // Weight parameters (variational posterior)
    /// Weight mean
    pub weight_mu: Array2<f64>,
    /// Weight rho (sigma = softplus(rho))
    pub weight_rho: Array2<f64>,

    // Bias parameters
    /// Bias mean
    pub bias_mu: Array1<f64>,
    /// Bias rho
    pub bias_rho: Array1<f64>,

    // Prior parameters
    prior_sigma_1: f64,
    prior_sigma_2: f64,
    prior_pi: f64,

    // Cached samples for KL computation
    weight_sample: Option<Array2<f64>>,
    bias_sample: Option<Array1<f64>>,
}

impl BayesianLinear {
    /// Create a new Bayesian linear layer.
    pub fn new(
        in_features: usize,
        out_features: usize,
        prior_sigma_1: f64,
        prior_sigma_2: f64,
        prior_pi: f64,
    ) -> Self {
        let mut rng = rand::thread_rng();

        // Initialize weight_mu with Kaiming initialization
        let std = (2.0 / in_features as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let weight_mu = Array2::from_shape_fn((out_features, in_features), |_| {
            normal.sample(&mut rng)
        });

        // Initialize rho to give small initial variance
        // softplus(-5) ≈ 0.007
        let weight_rho = Array2::from_elem((out_features, in_features), -5.0);

        // Initialize bias
        let bias_mu = Array1::zeros(out_features);
        let bias_rho = Array1::from_elem(out_features, -5.0);

        Self {
            in_features,
            out_features,
            weight_mu,
            weight_rho,
            bias_mu,
            bias_rho,
            prior_sigma_1,
            prior_sigma_2,
            prior_pi,
            weight_sample: None,
            bias_sample: None,
        }
    }

    /// Softplus function: log(1 + exp(x))
    fn softplus(x: f64) -> f64 {
        if x > 20.0 {
            x
        } else {
            (1.0 + x.exp()).ln()
        }
    }

    /// Sample weights using reparameterization trick.
    fn sample_weights(&mut self) -> (Array2<f64>, Array1<f64>) {
        let mut rng = rand::thread_rng();

        // Sample epsilon from N(0, 1)
        let weight_epsilon: Array2<f64> = Array2::from_shape_fn(
            (self.out_features, self.in_features),
            |_| rng.sample(StandardNormal),
        );

        let bias_epsilon: Array1<f64> = Array1::from_shape_fn(
            self.out_features,
            |_| rng.sample(StandardNormal),
        );

        // Compute sigma = softplus(rho)
        let weight_sigma = self.weight_rho.mapv(Self::softplus);
        let bias_sigma = self.bias_rho.mapv(Self::softplus);

        // Sample: w = mu + sigma * epsilon
        let weight = &self.weight_mu + &(&weight_sigma * &weight_epsilon);
        let bias = &self.bias_mu + &(&bias_sigma * &bias_epsilon);

        self.weight_sample = Some(weight.clone());
        self.bias_sample = Some(bias.clone());

        (weight, bias)
    }

    /// Forward pass with optional weight sampling.
    ///
    /// # Arguments
    /// * `x` - Input array of shape (batch_size, in_features)
    /// * `sample` - If true, sample weights; if false, use mean weights
    ///
    /// # Returns
    /// Output array of shape (batch_size, out_features)
    pub fn forward(&mut self, x: &Array2<f64>, sample: bool) -> Array2<f64> {
        let (weight, bias) = if sample {
            self.sample_weights()
        } else {
            self.weight_sample = Some(self.weight_mu.clone());
            self.bias_sample = Some(self.bias_mu.clone());
            (self.weight_mu.clone(), self.bias_mu.clone())
        };

        // Compute x @ weight.T + bias
        let output = x.dot(&weight.t());

        // Add bias to each row
        let mut result = output;
        for mut row in result.rows_mut() {
            row += &bias;
        }

        result
    }

    /// Compute KL divergence from variational posterior to prior.
    pub fn kl_divergence(&self) -> f64 {
        let weight_sample = self.weight_sample.as_ref().expect("Must call forward first");
        let bias_sample = self.bias_sample.as_ref().expect("Must call forward first");

        let weight_sigma = self.weight_rho.mapv(Self::softplus);
        let bias_sigma = self.bias_rho.mapv(Self::softplus);

        // KL for weights
        let kl_weight: f64 = weight_sample
            .iter()
            .zip(self.weight_mu.iter())
            .zip(weight_sigma.iter())
            .map(|((w, mu), sigma)| {
                let log_posterior = Self::log_gaussian(*w, *mu, *sigma);
                let log_prior = self.log_scale_mixture_prior(*w);
                log_posterior - log_prior
            })
            .sum();

        // KL for bias
        let kl_bias: f64 = bias_sample
            .iter()
            .zip(self.bias_mu.iter())
            .zip(bias_sigma.iter())
            .map(|((b, mu), sigma)| {
                let log_posterior = Self::log_gaussian(*b, *mu, *sigma);
                let log_prior = self.log_scale_mixture_prior(*b);
                log_posterior - log_prior
            })
            .sum();

        kl_weight + kl_bias
    }

    /// Log probability under Gaussian distribution.
    fn log_gaussian(x: f64, mu: f64, sigma: f64) -> f64 {
        -0.5 * (2.0 * PI).ln() - sigma.ln() - 0.5 * ((x - mu) / sigma).powi(2)
    }

    /// Log probability under scale mixture of Gaussians prior.
    fn log_scale_mixture_prior(&self, x: f64) -> f64 {
        let log_p1 = Self::log_gaussian(x, 0.0, self.prior_sigma_1);
        let log_p2 = Self::log_gaussian(x, 0.0, self.prior_sigma_2);

        let log_pi = self.prior_pi.ln();
        let log_one_minus_pi = (1.0 - self.prior_pi).ln();

        // log-sum-exp for numerical stability
        let max_log = (log_pi + log_p1).max(log_one_minus_pi + log_p2);
        let sum = ((log_pi + log_p1 - max_log).exp()
            + (log_one_minus_pi + log_p2 - max_log).exp())
        .ln();

        max_log + sum
    }

    /// Get weight statistics for monitoring.
    pub fn get_statistics(&self) -> LayerStatistics {
        let weight_sigma = self.weight_rho.mapv(Self::softplus);
        let bias_sigma = self.bias_rho.mapv(Self::softplus);

        LayerStatistics {
            weight_mu_mean: self.weight_mu.mean().unwrap_or(0.0),
            weight_mu_std: self.weight_mu.std(0.0),
            weight_sigma_mean: weight_sigma.mean().unwrap_or(0.0),
            weight_sigma_std: weight_sigma.std(0.0),
            bias_mu_mean: self.bias_mu.mean().unwrap_or(0.0),
            bias_sigma_mean: bias_sigma.mean().unwrap_or(0.0),
        }
    }

    /// Update parameters using gradients (simplified SGD).
    pub fn update_parameters(
        &mut self,
        weight_mu_grad: &Array2<f64>,
        weight_rho_grad: &Array2<f64>,
        bias_mu_grad: &Array1<f64>,
        bias_rho_grad: &Array1<f64>,
        learning_rate: f64,
    ) {
        self.weight_mu = &self.weight_mu - &(weight_mu_grad * learning_rate);
        self.weight_rho = &self.weight_rho - &(weight_rho_grad * learning_rate);
        self.bias_mu = &self.bias_mu - &(bias_mu_grad * learning_rate);
        self.bias_rho = &self.bias_rho - &(bias_rho_grad * learning_rate);
    }
}

/// Statistics about a Bayesian layer.
#[derive(Debug, Clone)]
pub struct LayerStatistics {
    pub weight_mu_mean: f64,
    pub weight_mu_std: f64,
    pub weight_sigma_mean: f64,
    pub weight_sigma_std: f64,
    pub bias_mu_mean: f64,
    pub bias_sigma_mean: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayesian_linear_forward() {
        let mut layer = BayesianLinear::new(10, 5, 1.0, 0.0025, 0.5);
        let input = Array2::ones((3, 10));

        let output = layer.forward(&input, true);
        assert_eq!(output.shape(), &[3, 5]);
    }

    #[test]
    fn test_kl_divergence() {
        let mut layer = BayesianLinear::new(10, 5, 1.0, 0.0025, 0.5);
        let input = Array2::ones((3, 10));

        let _ = layer.forward(&input, true);
        let kl = layer.kl_divergence();

        assert!(kl.is_finite());
    }
}
