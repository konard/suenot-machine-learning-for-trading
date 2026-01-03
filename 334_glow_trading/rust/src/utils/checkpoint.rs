//! Model checkpoint management

use crate::model::GLOWModel;
use crate::data::Normalizer;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Model checkpoint containing model and training state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// GLOW model
    pub model: GLOWModel,
    /// Feature normalizer
    pub normalizer: Option<Normalizer>,
    /// Current epoch
    pub epoch: usize,
    /// Best validation loss
    pub best_val_loss: f64,
    /// Training history
    pub history: TrainingHistory,
}

/// Training history
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingHistory {
    /// Training losses per epoch
    pub train_losses: Vec<f64>,
    /// Validation losses per epoch
    pub val_losses: Vec<f64>,
    /// Learning rates per epoch
    pub learning_rates: Vec<f64>,
}

impl Checkpoint {
    /// Create a new checkpoint
    pub fn new(model: GLOWModel) -> Self {
        Self {
            model,
            normalizer: None,
            epoch: 0,
            best_val_loss: f64::INFINITY,
            history: TrainingHistory::default(),
        }
    }

    /// Update checkpoint with training progress
    pub fn update(&mut self, epoch: usize, train_loss: f64, val_loss: f64, lr: f64) {
        self.epoch = epoch;
        self.history.train_losses.push(train_loss);
        self.history.val_losses.push(val_loss);
        self.history.learning_rates.push(lr);

        if val_loss < self.best_val_loss {
            self.best_val_loss = val_loss;
        }
    }

    /// Save checkpoint to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let bytes = bincode::serialize(self)?;
        fs::write(path, bytes)?;
        Ok(())
    }

    /// Load checkpoint from file
    pub fn load<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let bytes = fs::read(path)?;
        let checkpoint = bincode::deserialize(&bytes)?;
        Ok(checkpoint)
    }

    /// Get the model
    pub fn model(&self) -> &GLOWModel {
        &self.model
    }

    /// Get mutable model reference
    pub fn model_mut(&mut self) -> &mut GLOWModel {
        &mut self.model
    }

    /// Set normalizer
    pub fn set_normalizer(&mut self, normalizer: Normalizer) {
        self.normalizer = Some(normalizer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::GLOWConfig;
    use tempfile::tempdir;

    #[test]
    fn test_checkpoint_save_load() {
        let config = GLOWConfig::with_features(8);
        let model = GLOWModel::new(config);
        let checkpoint = Checkpoint::new(model);

        let dir = tempdir().unwrap();
        let path = dir.path().join("checkpoint.bin");

        checkpoint.save(&path).unwrap();
        let loaded = Checkpoint::load(&path).unwrap();

        assert_eq!(checkpoint.epoch, loaded.epoch);
        assert_eq!(
            checkpoint.model.config.num_features,
            loaded.model.config.num_features
        );
    }

    #[test]
    fn test_checkpoint_update() {
        let config = GLOWConfig::with_features(8);
        let model = GLOWModel::new(config);
        let mut checkpoint = Checkpoint::new(model);

        checkpoint.update(1, 1.5, 1.4, 0.001);
        checkpoint.update(2, 1.3, 1.2, 0.001);

        assert_eq!(checkpoint.epoch, 2);
        assert_eq!(checkpoint.history.train_losses.len(), 2);
        assert_eq!(checkpoint.best_val_loss, 1.2);
    }
}
