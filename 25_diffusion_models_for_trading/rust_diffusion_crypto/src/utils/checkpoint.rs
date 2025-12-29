//! Model checkpointing.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Checkpoint metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Model path
    pub model_path: String,
    /// Training epoch
    pub epoch: usize,
    /// Best validation loss
    pub best_loss: f64,
    /// Configuration used
    pub config: super::Config,
    /// Training losses history
    pub losses: Vec<f64>,
}

impl Checkpoint {
    /// Create a new checkpoint.
    pub fn new(
        model_path: String,
        epoch: usize,
        best_loss: f64,
        config: super::Config,
        losses: Vec<f64>,
    ) -> Self {
        Self {
            model_path,
            epoch,
            best_loss,
            config,
            losses,
        }
    }

    /// Save checkpoint metadata.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Load checkpoint metadata.
    pub fn load<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let checkpoint: Checkpoint = serde_json::from_str(&content)?;
        Ok(checkpoint)
    }
}
