//! InceptionTime Ensemble
//!
//! This module implements an ensemble of InceptionTime networks
//! for more robust predictions.

use anyhow::Result;
use tch::{nn, Device, Tensor};

use super::network::{InceptionTimeNetwork, NetworkConfig};

/// Ensemble of InceptionTime networks
#[derive(Debug)]
pub struct InceptionTimeEnsemble {
    /// Individual models in the ensemble
    models: Vec<InceptionTimeNetwork>,
    /// Variable stores for each model
    var_stores: Vec<nn::VarStore>,
    /// Number of classes
    num_classes: i64,
    /// Device
    device: Device,
}

impl InceptionTimeEnsemble {
    /// Create a new ensemble of InceptionTime models
    ///
    /// # Arguments
    /// * `ensemble_size` - Number of models in the ensemble
    /// * `num_classes` - Number of output classes
    pub fn new(ensemble_size: usize, num_classes: i64) -> Result<Self> {
        Self::with_config(ensemble_size, NetworkConfig {
            num_classes,
            ..Default::default()
        })
    }

    /// Create a new ensemble with custom configuration
    pub fn with_config(ensemble_size: usize, config: NetworkConfig) -> Result<Self> {
        let device = Device::cuda_if_available();
        let mut models = Vec::with_capacity(ensemble_size);
        let mut var_stores = Vec::with_capacity(ensemble_size);

        for i in 0..ensemble_size {
            let vs = nn::VarStore::new(device);
            let model = InceptionTimeNetwork::new(&vs.root() / format!("model_{}", i), config.clone())?;
            models.push(model);
            var_stores.push(vs);
        }

        Ok(Self {
            models,
            var_stores,
            num_classes: config.num_classes,
            device,
        })
    }

    /// Get ensemble size
    pub fn size(&self) -> usize {
        self.models.len()
    }

    /// Forward pass through all models
    ///
    /// # Arguments
    /// * `x` - Input tensor
    /// * `train` - Whether in training mode
    ///
    /// # Returns
    /// Averaged logits from all models
    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let outputs: Vec<Tensor> = self
            .models
            .iter()
            .map(|model| model.forward(x, train))
            .collect();

        // Average the outputs
        let stacked = Tensor::stack(&outputs, 0);
        stacked.mean_dim(Some([0].as_slice()), false, tch::Kind::Float)
    }

    /// Get prediction probabilities (averaged across ensemble)
    pub fn predict_proba(&self, x: &Tensor) -> Tensor {
        let probs: Vec<Tensor> = self
            .models
            .iter()
            .map(|model| model.predict_proba(x))
            .collect();

        // Average probabilities
        let stacked = Tensor::stack(&probs, 0);
        stacked.mean_dim(Some([0].as_slice()), false, tch::Kind::Float)
    }

    /// Get predicted class
    pub fn predict(&self, x: &Tensor) -> Tensor {
        let probs = self.predict_proba(x);
        probs.argmax(-1, false)
    }

    /// Get prediction with confidence score
    pub fn predict_with_confidence(&self, x: &Tensor) -> (Tensor, Tensor) {
        let probs = self.predict_proba(x);
        let (confidence, predictions) = probs.max_dim(-1, false);
        (predictions, confidence)
    }

    /// Get individual model predictions (for uncertainty estimation)
    pub fn predict_all(&self, x: &Tensor) -> Vec<Tensor> {
        self.models
            .iter()
            .map(|model| model.predict_proba(x))
            .collect()
    }

    /// Calculate prediction uncertainty (standard deviation across ensemble)
    pub fn prediction_uncertainty(&self, x: &Tensor) -> Tensor {
        let predictions = self.predict_all(x);
        let stacked = Tensor::stack(&predictions, 0);

        // Calculate std across ensemble dimension
        stacked.std_dim(Some([0].as_slice()), true, tch::Kind::Float)
    }

    /// Get variable store for a specific model (for training)
    pub fn var_store(&self, model_idx: usize) -> Option<&nn::VarStore> {
        self.var_stores.get(model_idx)
    }

    /// Get mutable variable store for a specific model (for training)
    pub fn var_store_mut(&mut self, model_idx: usize) -> Option<&mut nn::VarStore> {
        self.var_stores.get_mut(model_idx)
    }

    /// Get a specific model
    pub fn model(&self, idx: usize) -> Option<&InceptionTimeNetwork> {
        self.models.get(idx)
    }

    /// Get device
    pub fn device(&self) -> Device {
        self.device
    }

    /// Save all models
    pub fn save(&self, base_path: &str) -> Result<()> {
        for (i, vs) in self.var_stores.iter().enumerate() {
            let path = format!("{}_{}.pt", base_path, i);
            vs.save(&path)?;
        }
        Ok(())
    }

    /// Load all models
    pub fn load(&mut self, base_path: &str) -> Result<()> {
        for (i, vs) in self.var_stores.iter_mut().enumerate() {
            let path = format!("{}_{}.pt", base_path, i);
            vs.load(&path)?;
        }
        Ok(())
    }
}

/// Voting strategies for ensemble predictions
#[derive(Debug, Clone, Copy)]
pub enum VotingStrategy {
    /// Average probabilities and take argmax
    SoftVoting,
    /// Each model votes for a class, majority wins
    HardVoting,
    /// Weighted average based on model confidence
    WeightedVoting,
}

/// Apply voting strategy to ensemble predictions
pub fn apply_voting(
    predictions: &[Tensor],
    strategy: VotingStrategy,
    weights: Option<&[f64]>,
) -> Tensor {
    match strategy {
        VotingStrategy::SoftVoting => {
            let stacked = Tensor::stack(predictions, 0);
            let avg = stacked.mean_dim(Some([0].as_slice()), false, tch::Kind::Float);
            avg.argmax(-1, false)
        }
        VotingStrategy::HardVoting => {
            let votes: Vec<Tensor> = predictions
                .iter()
                .map(|p| p.argmax(-1, false))
                .collect();
            let stacked = Tensor::stack(&votes, 0);
            stacked.mode(0, false).0
        }
        VotingStrategy::WeightedVoting => {
            let weights = weights.unwrap_or(&[1.0; 5]);
            let weight_sum: f64 = weights.iter().sum();

            let mut weighted_sum = predictions[0].shallow_clone() * (weights[0] / weight_sum);
            for (pred, &weight) in predictions.iter().zip(weights.iter()).skip(1) {
                weighted_sum = weighted_sum + pred * (weight / weight_sum);
            }
            weighted_sum.argmax(-1, false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voting_strategy_default() {
        // Just test that the enum exists and has the expected variants
        let _soft = VotingStrategy::SoftVoting;
        let _hard = VotingStrategy::HardVoting;
        let _weighted = VotingStrategy::WeightedVoting;
    }
}
