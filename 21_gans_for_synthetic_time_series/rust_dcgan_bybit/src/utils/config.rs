//! Configuration management
//!
//! Provides unified configuration for the entire DCGAN pipeline.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Data configuration
    pub data: DataConfig,
    /// Model configuration
    pub model: ModelConfig,
    /// Training configuration
    pub training: TrainingConfigFile,
}

/// Data-related configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Trading symbol (e.g., "BTCUSDT")
    pub symbol: String,
    /// Kline interval (e.g., "60" for 1 hour)
    pub interval: String,
    /// Sequence length for training
    pub sequence_length: usize,
    /// Batch size
    pub batch_size: usize,
    /// Features to use: "ohlcv" or "returns"
    pub feature_mode: String,
    /// Path to save/load data
    pub data_path: String,
}

/// Model-related configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Latent dimension size
    pub latent_dim: i64,
    /// Number of features (5 for OHLCV)
    pub num_features: i64,
    /// Base filters for generator
    pub gen_base_filters: i64,
    /// Base filters for discriminator
    pub disc_base_filters: i64,
    /// Dropout rate for discriminator
    pub dropout: f64,
}

/// Training-related configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfigFile {
    /// Number of epochs
    pub epochs: usize,
    /// Generator learning rate
    pub gen_lr: f64,
    /// Discriminator learning rate
    pub disc_lr: f64,
    /// Discriminator steps per generator step
    pub disc_steps: usize,
    /// Checkpoint save frequency
    pub checkpoint_every: usize,
    /// Checkpoint directory
    pub checkpoint_dir: String,
    /// Use label smoothing
    pub label_smoothing: bool,
    /// Device: "cpu" or "cuda"
    pub device: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data: DataConfig {
                symbol: "BTCUSDT".to_string(),
                interval: "60".to_string(),
                sequence_length: 24,
                batch_size: 64,
                feature_mode: "ohlcv".to_string(),
                data_path: "data".to_string(),
            },
            model: ModelConfig {
                latent_dim: 100,
                num_features: 5,
                gen_base_filters: 256,
                disc_base_filters: 64,
                dropout: 0.3,
            },
            training: TrainingConfigFile {
                epochs: 100,
                gen_lr: 2e-4,
                disc_lr: 2e-4,
                disc_steps: 1,
                checkpoint_every: 10,
                checkpoint_dir: "checkpoints".to_string(),
                label_smoothing: false,
                device: "cpu".to_string(),
            },
        }
    }
}

impl Config {
    /// Create a new default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Load configuration from TOML file
    pub fn from_toml(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to TOML file
    pub fn save_toml(&self, path: &str) -> anyhow::Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Load configuration from JSON file
    pub fn from_json(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to JSON file
    pub fn save_json(&self, path: &str) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Get device from configuration
    pub fn get_device(&self) -> tch::Device {
        match self.training.device.to_lowercase().as_str() {
            "cuda" | "gpu" => {
                if tch::Cuda::is_available() {
                    tch::Device::Cuda(0)
                } else {
                    tracing::warn!("CUDA requested but not available, falling back to CPU");
                    tch::Device::Cpu
                }
            }
            _ => tch::Device::Cpu,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.data.sequence_length == 0 {
            anyhow::bail!("Sequence length must be > 0");
        }
        if self.data.batch_size == 0 {
            anyhow::bail!("Batch size must be > 0");
        }
        if self.model.latent_dim <= 0 {
            anyhow::bail!("Latent dimension must be > 0");
        }
        if self.training.epochs == 0 {
            anyhow::bail!("Number of epochs must be > 0");
        }
        Ok(())
    }
}

/// Create default configuration file if it doesn't exist
pub fn ensure_config_exists(path: &str) -> anyhow::Result<Config> {
    if Path::new(path).exists() {
        if path.ends_with(".toml") {
            Config::from_toml(path)
        } else {
            Config::from_json(path)
        }
    } else {
        let config = Config::default();
        if path.ends_with(".toml") {
            config.save_toml(path)?;
        } else {
            config.save_json(path)?;
        }
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.data.symbol, "BTCUSDT");
        assert_eq!(config.model.latent_dim, 100);
    }

    #[test]
    fn test_config_json_roundtrip() {
        let config = Config::default();
        let json = serde_json::to_string(&config).unwrap();
        let loaded: Config = serde_json::from_str(&json).unwrap();

        assert_eq!(config.data.symbol, loaded.data.symbol);
        assert_eq!(config.model.latent_dim, loaded.model.latent_dim);
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        assert!(config.validate().is_ok());

        config.data.sequence_length = 0;
        assert!(config.validate().is_err());
    }
}
