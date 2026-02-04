//! Bayesian Neural Network Model.

use anyhow::Result;
use ndarray::{Array1, Array2};

use super::config::BNNConfig;
use super::layers::BayesianLinear;
use super::uncertainty::UncertaintyEstimate;

/// Bayesian Neural Network for trading.
#[derive(Debug)]
pub struct BayesianNN {
    /// Model configuration
    pub config: BNNConfig,
    /// Hidden layers
    layers: Vec<BayesianLinear>,
    /// Classifier head
    classifier: BayesianLinear,
    /// Return prediction head (if heteroscedastic)
    return_head: Option<BayesianLinear>,
}

impl BayesianNN {
    /// Create a new Bayesian Neural Network.
    pub fn new(config: BNNConfig) -> Self {
        let mut layers = Vec::new();
        let mut prev_dim = config.input_dim;

        // Create hidden layers
        for &hidden_dim in &config.hidden_dims {
            layers.push(BayesianLinear::new(
                prev_dim,
                hidden_dim,
                config.prior_sigma_1,
                config.prior_sigma_2,
                config.prior_pi,
            ));
            prev_dim = hidden_dim;
        }

        // Classifier head
        let classifier = BayesianLinear::new(
            prev_dim,
            config.num_classes,
            config.prior_sigma_1,
            config.prior_sigma_2,
            config.prior_pi,
        );

        // Return head for heteroscedastic output
        let return_head = if config.heteroscedastic {
            Some(BayesianLinear::new(
                prev_dim,
                2, // mean and log_variance
                config.prior_sigma_1,
                config.prior_sigma_2,
                config.prior_pi,
            ))
        } else {
            None
        };

        Self {
            config,
            layers,
            classifier,
            return_head,
        }
    }

    /// Forward pass through the network.
    ///
    /// # Arguments
    /// * `x` - Input features of shape (batch_size, input_dim)
    /// * `sample` - If true, sample weights; if false, use mean weights
    ///
    /// # Returns
    /// Classification logits of shape (batch_size, num_classes)
    pub fn forward(&mut self, x: &Array2<f64>, sample: bool) -> Array2<f64> {
        let mut h = x.clone();

        // Pass through hidden layers with LeakyReLU activation
        for layer in &mut self.layers {
            h = layer.forward(&h, sample);
            h = leaky_relu(&h, 0.1);
        }

        // Classifier output
        self.classifier.forward(&h, sample)
    }

    /// Forward pass with return prediction.
    ///
    /// # Returns
    /// (logits, return_mean, return_log_var) if heteroscedastic
    /// (logits, None, None) otherwise
    pub fn forward_with_returns(
        &mut self,
        x: &Array2<f64>,
        sample: bool,
    ) -> (Array2<f64>, Option<Array1<f64>>, Option<Array1<f64>>) {
        let mut h = x.clone();

        // Hidden layers
        for layer in &mut self.layers {
            h = layer.forward(&h, sample);
            h = leaky_relu(&h, 0.1);
        }

        // Classifier
        let logits = self.classifier.forward(&h, sample);

        // Return head
        if let Some(ref mut return_head) = self.return_head {
            let return_out = return_head.forward(&h, sample);
            let return_mean = return_out.column(0).to_owned();
            let return_log_var = return_out.column(1).to_owned();
            (logits, Some(return_mean), Some(return_log_var))
        } else {
            (logits, None, None)
        }
    }

    /// Compute total KL divergence for all layers.
    pub fn kl_divergence(&self) -> f64 {
        let mut kl = 0.0;

        for layer in &self.layers {
            kl += layer.kl_divergence();
        }

        kl += self.classifier.kl_divergence();

        if let Some(ref return_head) = self.return_head {
            kl += return_head.kl_divergence();
        }

        kl
    }

    /// Predict with uncertainty estimation using Monte Carlo sampling.
    ///
    /// # Arguments
    /// * `x` - Input features
    ///
    /// # Returns
    /// Uncertainty estimate with mean, std, and decomposed uncertainties
    pub fn predict_with_uncertainty(&mut self, x: &Array2<f64>) -> UncertaintyEstimate {
        let n_samples = self.config.n_samples_inference;
        let mut all_probs = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let logits = self.forward(x, true);
            let probs = softmax(&logits);

            // For single sample, take first row
            if probs.nrows() == 1 {
                all_probs.push(probs.row(0).to_vec());
            } else {
                // Average across batch
                let mean_probs: Vec<f64> = (0..probs.ncols())
                    .map(|j| probs.column(j).mean().unwrap_or(0.0))
                    .collect();
                all_probs.push(mean_probs);
            }
        }

        UncertaintyEstimate::from_samples(&all_probs)
    }

    /// Predict returns with uncertainty decomposition.
    pub fn predict_returns_with_uncertainty(
        &mut self,
        x: &Array2<f64>,
    ) -> Option<UncertaintyEstimate> {
        if self.return_head.is_none() {
            return None;
        }

        let n_samples = self.config.n_samples_inference;
        let mut mean_samples = Vec::with_capacity(n_samples);
        let mut var_samples = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let (_, return_mean, return_log_var) = self.forward_with_returns(x, true);

            if let (Some(mean), Some(log_var)) = (return_mean, return_log_var) {
                mean_samples.push(mean.to_vec());
                var_samples.push(log_var.mapv(f64::exp).to_vec());
            }
        }

        Some(UncertaintyEstimate::from_samples_heteroscedastic(
            &mean_samples,
            &var_samples,
        ))
    }

    /// Get total number of parameters.
    pub fn num_parameters(&self) -> usize {
        self.config.total_parameters()
    }

    /// Get layer statistics for monitoring.
    pub fn get_layer_statistics(&self) -> Vec<super::layers::LayerStatistics> {
        let mut stats = Vec::new();

        for layer in &self.layers {
            stats.push(layer.get_statistics());
        }

        stats.push(self.classifier.get_statistics());

        if let Some(ref return_head) = self.return_head {
            stats.push(return_head.get_statistics());
        }

        stats
    }
}

/// LeakyReLU activation function.
fn leaky_relu(x: &Array2<f64>, negative_slope: f64) -> Array2<f64> {
    x.mapv(|v| if v > 0.0 { v } else { negative_slope * v })
}

/// Softmax function (row-wise).
fn softmax(x: &Array2<f64>) -> Array2<f64> {
    let mut result = x.clone();

    for mut row in result.rows_mut() {
        let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = row.iter().map(|&v| (v - max).exp()).sum();

        for v in row.iter_mut() {
            *v = (*v - max).exp() / exp_sum;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bnn_forward() {
        let config = BNNConfig::with_input_dim(10);
        let mut model = BayesianNN::new(config);

        let x = Array2::ones((5, 10));
        let output = model.forward(&x, true);

        assert_eq!(output.shape(), &[5, 3]);
    }

    #[test]
    fn test_predict_with_uncertainty() {
        let config = BNNConfig::with_input_dim(10).n_samples_inference(10);
        let mut model = BayesianNN::new(config);

        let x = Array2::ones((1, 10));
        let estimate = model.predict_with_uncertainty(&x);

        assert_eq!(estimate.mean.len(), 3);
        assert!(estimate.mean_confidence() > 0.0);
    }
}
