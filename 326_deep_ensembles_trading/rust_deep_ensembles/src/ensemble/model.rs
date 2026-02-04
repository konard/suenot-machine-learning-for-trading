//! Deep Ensemble model
//!
//! Combines multiple neural networks for uncertainty-aware predictions.

use super::config::EnsembleConfig;
use super::network::GaussianMLP;
use super::prediction::EnsemblePrediction;
use ndarray::{Array1, Array2};

/// Deep Ensemble model
#[derive(Debug)]
pub struct DeepEnsemble {
    /// Ensemble configuration
    pub config: EnsembleConfig,
    /// Individual neural network models
    pub models: Vec<GaussianMLP>,
    /// Whether the ensemble has been trained
    pub is_trained: bool,
}

impl DeepEnsemble {
    /// Create a new Deep Ensemble
    pub fn new(config: EnsembleConfig) -> Self {
        config.validate().expect("Invalid configuration");

        // Create models with different random seeds
        let models: Vec<GaussianMLP> = (0..config.num_models)
            .map(|i| {
                GaussianMLP::new(
                    config.input_dim,
                    &config.hidden_dims,
                    42 + i as u64, // Different seed for each model
                )
            })
            .collect();

        Self {
            config,
            models,
            is_trained: false,
        }
    }

    /// Train the ensemble
    ///
    /// # Arguments
    /// * `x_train` - Training features [n_samples, n_features]
    /// * `y_train` - Training targets [n_samples]
    /// * `x_val` - Optional validation features
    /// * `y_val` - Optional validation targets
    /// * `verbose` - Whether to print training progress
    ///
    /// # Returns
    /// Training history as a vector of (train_loss, val_loss) tuples per model
    pub fn train(
        &mut self,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        x_val: Option<&Array2<f64>>,
        y_val: Option<&Array1<f64>>,
        verbose: bool,
    ) -> Vec<(f64, f64)> {
        let n_samples = x_train.nrows();
        let n_batches = n_samples / self.config.batch_size;

        let mut history = Vec::new();

        for (model_idx, model) in self.models.iter_mut().enumerate() {
            if verbose {
                println!("Training model {}/{}", model_idx + 1, self.config.num_models);
            }

            let mut best_val_loss = f64::INFINITY;
            let mut patience_counter = 0;
            let mut final_train_loss = 0.0;
            let mut final_val_loss = 0.0;

            for epoch in 0..self.config.max_epochs {
                // Shuffle indices
                let mut indices: Vec<usize> = (0..n_samples).collect();
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                indices.shuffle(&mut rng);

                let mut epoch_loss = 0.0;

                for batch_idx in 0..n_batches {
                    let start = batch_idx * self.config.batch_size;
                    let end = start + self.config.batch_size;

                    // Get batch
                    let batch_indices: Vec<usize> = indices[start..end].to_vec();
                    let x_batch = x_train.select(ndarray::Axis(0), &batch_indices);
                    let y_batch = y_train.select(ndarray::Axis(0), &batch_indices);

                    // Forward pass
                    let (mean, std) = model.forward(&x_batch);

                    // Compute loss
                    let loss = model.nll_loss(&mean, &std, &y_batch);
                    epoch_loss += loss;

                    // Backward pass
                    model.backward(&y_batch, &mean, &std, self.config.learning_rate);
                }

                epoch_loss /= n_batches as f64;
                final_train_loss = epoch_loss;

                // Validation
                let val_loss = if let (Some(x_v), Some(y_v)) = (x_val, y_val) {
                    let (mean_val, std_val) = model.forward(x_v);
                    model.nll_loss(&mean_val, &std_val, y_v)
                } else {
                    0.0
                };
                final_val_loss = val_loss;

                // Early stopping
                if x_val.is_some() {
                    if val_loss < best_val_loss {
                        best_val_loss = val_loss;
                        patience_counter = 0;
                    } else {
                        patience_counter += 1;
                    }

                    if patience_counter >= self.config.early_stopping_patience {
                        if verbose {
                            println!("  Early stopping at epoch {}", epoch + 1);
                        }
                        break;
                    }
                }

                if verbose && (epoch + 1) % 10 == 0 {
                    println!(
                        "  Epoch {}/{}, Train Loss: {:.4}, Val Loss: {:.4}",
                        epoch + 1,
                        self.config.max_epochs,
                        epoch_loss,
                        val_loss
                    );
                }
            }

            history.push((final_train_loss, final_val_loss));
        }

        self.is_trained = true;
        history
    }

    /// Make predictions with uncertainty estimation
    ///
    /// # Arguments
    /// * `x` - Input features [n_samples, n_features]
    ///
    /// # Returns
    /// EnsemblePrediction with mean, std, and uncertainty decomposition
    pub fn predict(&mut self, x: &Array2<f64>) -> EnsemblePrediction {
        let mut individual_means = Vec::new();
        let mut individual_stds = Vec::new();

        for model in &mut self.models {
            let (mean, std) = model.forward(x);
            individual_means.push(mean);
            individual_stds.push(std);
        }

        EnsemblePrediction::new(individual_means, individual_stds)
    }

    /// Get the number of models in the ensemble
    pub fn num_models(&self) -> usize {
        self.models.len()
    }

    /// Check if the ensemble is trained
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }
}

/// Compute calibration error
pub fn compute_calibration_error(
    predictions: &EnsemblePrediction,
    targets: &Array1<f64>,
    num_bins: usize,
) -> f64 {
    let n = predictions.num_samples();
    let confidence = predictions.confidence();

    // Compute accuracy (whether target is within 1 std)
    let mut accuracies = Array1::zeros(n);
    for i in 0..n {
        let error = (predictions.mean[i] - targets[i]).abs();
        accuracies[i] = if error < predictions.total_std[i] {
            1.0
        } else {
            0.0
        };
    }

    let mut ece = 0.0;

    for bin_idx in 0..num_bins {
        let bin_lower = bin_idx as f64 / num_bins as f64;
        let bin_upper = (bin_idx + 1) as f64 / num_bins as f64;

        let mut in_bin_count = 0;
        let mut conf_sum = 0.0;
        let mut acc_sum = 0.0;

        for i in 0..n {
            if confidence[i] > bin_lower && confidence[i] <= bin_upper {
                in_bin_count += 1;
                conf_sum += confidence[i];
                acc_sum += accuracies[i];
            }
        }

        if in_bin_count > 0 {
            let avg_conf = conf_sum / in_bin_count as f64;
            let avg_acc = acc_sum / in_bin_count as f64;
            ece += (avg_acc - avg_conf).abs() * (in_bin_count as f64 / n as f64);
        }
    }

    ece
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_synthetic_data(n: usize, d: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        use rand::SeedableRng;
        use rand_distr::Distribution;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();

        let x = Array2::from_shape_fn((n, d), |_| normal.sample(&mut rng));

        // Generate targets: linear combination + noise
        let weights = Array1::from_shape_fn(d, |_| normal.sample(&mut rng));
        let y = x.dot(&weights) + Array1::from_shape_fn(n, |_| normal.sample(&mut rng) * 0.5);

        (x, y)
    }

    #[test]
    fn test_ensemble_creation() {
        let config = EnsembleConfig::minimal();
        let ensemble = DeepEnsemble::new(config.clone());

        assert_eq!(ensemble.num_models(), config.num_models);
        assert!(!ensemble.is_trained());
    }

    #[test]
    fn test_ensemble_training() {
        let config = EnsembleConfig {
            num_models: 2,
            input_dim: 4,
            hidden_dims: vec![8],
            max_epochs: 10,
            batch_size: 16,
            ..EnsembleConfig::default()
        };

        let mut ensemble = DeepEnsemble::new(config);

        let (x_train, y_train) = generate_synthetic_data(100, 4, 42);
        let (x_val, y_val) = generate_synthetic_data(20, 4, 43);

        let history = ensemble.train(
            &x_train,
            &y_train,
            Some(&x_val),
            Some(&y_val),
            false,
        );

        assert!(ensemble.is_trained());
        assert_eq!(history.len(), 2); // 2 models
    }

    #[test]
    fn test_ensemble_prediction() {
        let config = EnsembleConfig {
            num_models: 3,
            input_dim: 4,
            hidden_dims: vec![8],
            max_epochs: 5,
            batch_size: 16,
            ..EnsembleConfig::default()
        };

        let mut ensemble = DeepEnsemble::new(config);

        let (x_train, y_train) = generate_synthetic_data(50, 4, 42);
        ensemble.train(&x_train, &y_train, None, None, false);

        let x_test = Array2::from_shape_fn((10, 4), |_| rand::random::<f64>());
        let predictions = ensemble.predict(&x_test);

        assert_eq!(predictions.num_samples(), 10);
        assert_eq!(predictions.num_models(), 3);

        // Check uncertainties are positive
        for i in 0..predictions.num_samples() {
            assert!(predictions.total_std[i] > 0.0);
            assert!(predictions.epistemic_std[i] >= 0.0);
            assert!(predictions.aleatoric_std[i] >= 0.0);
        }
    }
}
