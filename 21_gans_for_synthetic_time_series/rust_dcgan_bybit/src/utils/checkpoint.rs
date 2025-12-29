//! Checkpoint save/load utilities
//!
//! Provides functions for saving and loading model checkpoints
//! along with training state.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::model::DCGAN;
use crate::training::TrainingMetrics;

/// Checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    /// Current epoch
    pub epoch: usize,
    /// Generator loss at checkpoint
    pub gen_loss: f64,
    /// Discriminator loss at checkpoint
    pub disc_loss: f64,
    /// Timestamp of checkpoint
    pub timestamp: String,
    /// Model configuration (as JSON)
    pub config: String,
}

/// Save a complete checkpoint (model + metadata)
///
/// # Arguments
///
/// * `model` - DCGAN model to save
/// * `metrics` - Training metrics
/// * `epoch` - Current epoch number
/// * `dir` - Directory to save checkpoint
///
/// # Returns
///
/// Path to saved checkpoint
pub fn save_checkpoint(
    model: &DCGAN,
    metrics: &TrainingMetrics,
    epoch: usize,
    dir: &str,
) -> anyhow::Result<String> {
    std::fs::create_dir_all(dir)?;

    let checkpoint_name = format!("checkpoint_epoch_{:04}", epoch);
    let checkpoint_dir = format!("{}/{}", dir, checkpoint_name);
    std::fs::create_dir_all(&checkpoint_dir)?;

    // Save model weights
    let gen_path = format!("{}/generator.pt", checkpoint_dir);
    let disc_path = format!("{}/discriminator.pt", checkpoint_dir);
    model.save(&gen_path, &disc_path)?;

    // Save metadata
    let meta = CheckpointMeta {
        epoch,
        gen_loss: metrics.latest_gen_loss().unwrap_or(0.0),
        disc_loss: metrics.latest_disc_loss().unwrap_or(0.0),
        timestamp: chrono::Utc::now().to_rfc3339(),
        config: serde_json::json!({
            "latent_dim": model.latent_dim(),
            "sequence_length": model.sequence_length(),
            "num_features": model.num_features(),
        })
        .to_string(),
    };

    let meta_path = format!("{}/meta.json", checkpoint_dir);
    let meta_json = serde_json::to_string_pretty(&meta)?;
    std::fs::write(&meta_path, meta_json)?;

    // Save metrics
    let metrics_path = format!("{}/metrics.csv", checkpoint_dir);
    metrics.save_csv(&metrics_path)?;

    tracing::info!("Saved checkpoint to {}", checkpoint_dir);
    Ok(checkpoint_dir)
}

/// Load checkpoint metadata
pub fn load_checkpoint_meta(checkpoint_dir: &str) -> anyhow::Result<CheckpointMeta> {
    let meta_path = format!("{}/meta.json", checkpoint_dir);
    let content = std::fs::read_to_string(&meta_path)?;
    let meta: CheckpointMeta = serde_json::from_str(&content)?;
    Ok(meta)
}

/// Load a complete checkpoint
///
/// # Arguments
///
/// * `model` - DCGAN model to load weights into
/// * `checkpoint_dir` - Directory containing checkpoint
///
/// # Returns
///
/// Tuple of (epoch, metrics)
pub fn load_checkpoint(
    model: &mut DCGAN,
    checkpoint_dir: &str,
) -> anyhow::Result<(usize, TrainingMetrics)> {
    // Load model weights
    let gen_path = format!("{}/generator.pt", checkpoint_dir);
    let disc_path = format!("{}/discriminator.pt", checkpoint_dir);
    model.load(&gen_path, &disc_path)?;

    // Load metadata
    let meta = load_checkpoint_meta(checkpoint_dir)?;

    // Load metrics
    let metrics_path = format!("{}/metrics.csv", checkpoint_dir);
    let metrics = if Path::new(&metrics_path).exists() {
        TrainingMetrics::load_csv(&metrics_path)?
    } else {
        TrainingMetrics::new()
    };

    tracing::info!("Loaded checkpoint from {} (epoch {})", checkpoint_dir, meta.epoch);
    Ok((meta.epoch, metrics))
}

/// Find the latest checkpoint in a directory
pub fn find_latest_checkpoint(dir: &str) -> Option<String> {
    let path = Path::new(dir);
    if !path.exists() {
        return None;
    }

    let mut checkpoints: Vec<_> = std::fs::read_dir(path)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().ok().map(|t| t.is_dir()).unwrap_or(false))
        .filter(|e| {
            e.file_name()
                .to_str()
                .map(|n| n.starts_with("checkpoint_epoch_"))
                .unwrap_or(false)
        })
        .collect();

    checkpoints.sort_by(|a, b| b.file_name().cmp(&a.file_name()));

    checkpoints
        .first()
        .map(|e| e.path().to_string_lossy().to_string())
}

/// List all checkpoints in a directory
pub fn list_checkpoints(dir: &str) -> Vec<(String, CheckpointMeta)> {
    let path = Path::new(dir);
    if !path.exists() {
        return vec![];
    }

    std::fs::read_dir(path)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().ok().map(|t| t.is_dir()).unwrap_or(false))
        .filter(|e| {
            e.file_name()
                .to_str()
                .map(|n| n.starts_with("checkpoint_epoch_"))
                .unwrap_or(false)
        })
        .filter_map(|e| {
            let path = e.path().to_string_lossy().to_string();
            load_checkpoint_meta(&path).ok().map(|meta| (path, meta))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_meta_serialization() {
        let meta = CheckpointMeta {
            epoch: 10,
            gen_loss: 0.5,
            disc_loss: 0.6,
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            config: "{}".to_string(),
        };

        let json = serde_json::to_string(&meta).unwrap();
        let loaded: CheckpointMeta = serde_json::from_str(&json).unwrap();

        assert_eq!(meta.epoch, loaded.epoch);
    }
}
