//! # Model Training
//!
//! Training utilities for Neural ODE portfolio models.

use ndarray::Array1;
use rand::Rng;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use super::portfolio::NeuralODEPortfolio;
use crate::data::Features;

/// Loss function types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LossFunction {
    /// Mean Squared Error
    MSE,
    /// Mean Squared Error + Sharpe penalty
    MSEWithSharpe { risk_aversion: f64 },
    /// Full portfolio loss (return - risk + costs)
    Portfolio {
        risk_aversion: f64,
        cost_weight: f64,
    },
}

impl LossFunction {
    /// Compute loss given predicted weights, target weights, and optional returns
    pub fn compute(
        &self,
        predicted: &[f64],
        target: &[f64],
        returns: Option<&[f64]>,
    ) -> f64 {
        match self {
            LossFunction::MSE => {
                let mse: f64 = predicted
                    .iter()
                    .zip(target.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>()
                    / predicted.len() as f64;
                mse
            }
            LossFunction::MSEWithSharpe { risk_aversion } => {
                let mse: f64 = predicted
                    .iter()
                    .zip(target.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>()
                    / predicted.len() as f64;

                // Add Sharpe-like penalty if returns available
                if let Some(rets) = returns {
                    let port_return: f64 = predicted.iter()
                        .zip(rets.iter())
                        .map(|(w, r)| w * r)
                        .sum();
                    let variance: f64 = predicted.iter()
                        .zip(rets.iter())
                        .map(|(w, r)| w * (r - port_return).powi(2))
                        .sum();
                    mse + risk_aversion * variance
                } else {
                    mse
                }
            }
            LossFunction::Portfolio { risk_aversion, cost_weight } => {
                if let Some(rets) = returns {
                    // Expected return
                    let port_return: f64 = predicted.iter()
                        .zip(rets.iter())
                        .map(|(w, r)| w * r)
                        .sum();

                    // Risk (variance proxy)
                    let variance: f64 = predicted.iter()
                        .zip(rets.iter())
                        .map(|(w, r)| w * (r - port_return).powi(2))
                        .sum();

                    // Transaction costs (deviation from target)
                    let turnover: f64 = predicted.iter()
                        .zip(target.iter())
                        .map(|(p, t)| (p - t).abs())
                        .sum();

                    // Negative return (we minimize) + risk + costs
                    -port_return + risk_aversion * variance + cost_weight * turnover
                } else {
                    // Fall back to MSE
                    predicted
                        .iter()
                        .zip(target.iter())
                        .map(|(p, t)| (p - t).powi(2))
                        .sum::<f64>()
                        / predicted.len() as f64
                }
            }
        }
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Number of epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Loss function
    pub loss_fn: LossFunction,
    /// Gradient clip norm
    pub grad_clip: f64,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Patience for early stopping
    pub patience: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            epochs: 100,
            batch_size: 32,
            loss_fn: LossFunction::Portfolio {
                risk_aversion: 1.0,
                cost_weight: 0.01,
            },
            grad_clip: 1.0,
            weight_decay: 1e-5,
            patience: 10,
        }
    }
}

/// Training sample
#[derive(Debug, Clone)]
pub struct TrainingSample {
    /// Initial weights
    pub initial_weights: Vec<f64>,
    /// Market features
    pub features: Features,
    /// Target weights (optimal allocation)
    pub target_weights: Vec<f64>,
    /// Realized returns (optional)
    pub returns: Option<Vec<f64>>,
}

/// Trainer for Neural ODE Portfolio
pub struct Trainer {
    config: TrainingConfig,
    best_loss: f64,
    patience_counter: usize,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            best_loss: f64::INFINITY,
            patience_counter: 0,
        }
    }

    /// Train the model using evolution strategy (gradient-free)
    ///
    /// This is a simple approach suitable for small models.
    /// For production, consider using proper automatic differentiation.
    pub fn train_evolution(
        &mut self,
        model: &mut NeuralODEPortfolio,
        samples: &[TrainingSample],
    ) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut loss_history = Vec::with_capacity(self.config.epochs);

        let n_params = model.num_params();
        let sigma = 0.1; // Noise scale
        let n_perturbations = 20;

        info!("Starting evolution training with {} parameters", n_params);

        for epoch in 0..self.config.epochs {
            // Evaluate current model
            let current_loss = self.evaluate(model, samples);

            // Store best parameters
            if current_loss < self.best_loss {
                self.best_loss = current_loss;
                self.patience_counter = 0;
            } else {
                self.patience_counter += 1;
                if self.patience_counter >= self.config.patience {
                    info!("Early stopping at epoch {}", epoch);
                    break;
                }
            }

            loss_history.push(current_loss);

            if epoch % 10 == 0 {
                info!("Epoch {}: loss = {:.6}", epoch, current_loss);
            }

            // Evolution step: try random perturbations
            let mut best_perturbation_loss = current_loss;
            let mut best_dynamics_params = model.dynamics_mut().dynamics_net_mut().get_params();

            for _ in 0..n_perturbations {
                // Perturb dynamics network
                let original_params = model.dynamics_mut().dynamics_net_mut().get_params();

                model.dynamics_mut().dynamics_net_mut().perturb(sigma);

                let perturbed_loss = self.evaluate(model, samples);

                if perturbed_loss < best_perturbation_loss {
                    best_perturbation_loss = perturbed_loss;
                    best_dynamics_params = model.dynamics_mut().dynamics_net_mut().get_params();
                }

                // Restore original
                model.dynamics_mut().dynamics_net_mut().set_params(&original_params);
            }

            // Apply best perturbation
            if best_perturbation_loss < current_loss {
                model.dynamics_mut().dynamics_net_mut().set_params(&best_dynamics_params);
                debug!("Improved loss: {:.6} -> {:.6}", current_loss, best_perturbation_loss);
            }
        }

        info!("Training completed. Best loss: {:.6}", self.best_loss);
        loss_history
    }

    /// Evaluate model on samples
    pub fn evaluate(
        &self,
        model: &NeuralODEPortfolio,
        samples: &[TrainingSample],
    ) -> f64 {
        let mut total_loss = 0.0;

        for sample in samples {
            let predicted = model.get_target_weights(
                &sample.initial_weights,
                &sample.features,
                1.0,  // 1.0 time horizon
            );

            let loss = self.config.loss_fn.compute(
                &predicted,
                &sample.target_weights,
                sample.returns.as_deref(),
            );

            total_loss += loss;
        }

        total_loss / samples.len() as f64
    }

    /// Simple gradient estimation using finite differences
    pub fn estimate_gradient(
        &self,
        model: &mut NeuralODEPortfolio,
        samples: &[TrainingSample],
        epsilon: f64,
    ) -> Vec<f64> {
        let base_loss = self.evaluate(model, samples);
        let params = model.dynamics_mut().dynamics_net_mut().get_params();
        let mut gradient = vec![0.0; params.len()];

        for i in 0..params.len() {
            // Perturb parameter i
            let mut perturbed_params = params.clone();
            perturbed_params[i] += epsilon;
            model.dynamics_mut().dynamics_net_mut().set_params(&perturbed_params);

            let perturbed_loss = self.evaluate(model, samples);
            gradient[i] = (perturbed_loss - base_loss) / epsilon;

            // Restore
            model.dynamics_mut().dynamics_net_mut().set_params(&params);
        }

        gradient
    }

    /// Train using gradient descent with finite differences
    pub fn train_gradient(
        &mut self,
        model: &mut NeuralODEPortfolio,
        samples: &[TrainingSample],
    ) -> Vec<f64> {
        let mut loss_history = Vec::with_capacity(self.config.epochs);

        info!("Starting gradient descent training");

        for epoch in 0..self.config.epochs {
            // Evaluate current loss
            let current_loss = self.evaluate(model, samples);
            loss_history.push(current_loss);

            if current_loss < self.best_loss {
                self.best_loss = current_loss;
                self.patience_counter = 0;
            } else {
                self.patience_counter += 1;
                if self.patience_counter >= self.config.patience {
                    info!("Early stopping at epoch {}", epoch);
                    break;
                }
            }

            if epoch % 10 == 0 {
                info!("Epoch {}: loss = {:.6}", epoch, current_loss);
            }

            // Estimate gradient
            let gradient = self.estimate_gradient(model, samples, 1e-5);

            // Clip gradient
            let grad_norm: f64 = gradient.iter().map(|g| g.powi(2)).sum::<f64>().sqrt();
            let clip_factor = if grad_norm > self.config.grad_clip {
                self.config.grad_clip / grad_norm
            } else {
                1.0
            };

            // Update parameters
            let mut params = model.dynamics_mut().dynamics_net_mut().get_params();
            for (p, g) in params.iter_mut().zip(gradient.iter()) {
                *p -= self.config.learning_rate * g * clip_factor;
                *p *= 1.0 - self.config.weight_decay; // L2 regularization
            }
            model.dynamics_mut().dynamics_net_mut().set_params(&params);
        }

        info!("Training completed. Best loss: {:.6}", self.best_loss);
        loss_history
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_dummy_sample() -> TrainingSample {
        TrainingSample {
            initial_weights: vec![0.5, 0.5],
            features: Features {
                n_assets: 2,
                n_features: 3,
                data: vec![vec![0.1, 0.2, 0.3], vec![0.2, 0.3, 0.4]],
                names: vec!["a".into(), "b".into(), "c".into()],
            },
            target_weights: vec![0.6, 0.4],
            returns: Some(vec![0.02, 0.01]),
        }
    }

    #[test]
    fn test_mse_loss() {
        let loss_fn = LossFunction::MSE;
        let pred = vec![0.5, 0.5];
        let target = vec![0.6, 0.4];

        let loss = loss_fn.compute(&pred, &target, None);
        assert!(loss > 0.0);
        assert!((loss - 0.01).abs() < 1e-6); // (0.1^2 + 0.1^2) / 2
    }

    #[test]
    fn test_portfolio_loss() {
        let loss_fn = LossFunction::Portfolio {
            risk_aversion: 1.0,
            cost_weight: 0.01,
        };

        let pred = vec![0.5, 0.5];
        let target = vec![0.6, 0.4];
        let returns = vec![0.02, 0.01];

        let loss = loss_fn.compute(&pred, &target, Some(&returns));
        // Should be negative (we want positive returns)
        // But with risk and cost penalties, could be positive
        assert!(!loss.is_nan());
    }

    #[test]
    fn test_trainer_evaluate() {
        let config = TrainingConfig::default();
        let trainer = Trainer::new(config);

        let model = NeuralODEPortfolio::new(2, 3, 8);
        let samples = vec![create_dummy_sample()];

        let loss = trainer.evaluate(&model, &samples);
        assert!(!loss.is_nan());
        assert!(loss.is_finite());
    }

    #[test]
    fn test_training_sample() {
        let sample = create_dummy_sample();
        assert_eq!(sample.initial_weights.len(), 2);
        assert_eq!(sample.target_weights.len(), 2);
        assert_eq!(sample.features.n_assets, 2);
    }
}
