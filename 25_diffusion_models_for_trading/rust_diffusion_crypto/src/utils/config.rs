//! Configuration handling.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Main configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub data: DataConfig,
    pub model: ModelConfig,
    pub training: TrainingConfig,
}

/// Data configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Trading symbol
    pub symbol: String,
    /// Kline interval
    pub interval: String,
    /// Sequence length for historical data
    pub sequence_length: usize,
    /// Forecast horizon
    pub forecast_horizon: usize,
    /// Batch size
    pub batch_size: usize,
}

/// Model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Hidden dimension
    pub hidden_dim: i64,
    /// Time embedding dimension
    pub time_emb_dim: i64,
    /// Number of diffusion steps
    pub num_diffusion_steps: usize,
    /// Noise schedule type (linear, cosine, sigmoid)
    pub noise_schedule: String,
}

/// Training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of epochs
    pub epochs: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Gradient clipping
    pub grad_clip: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            data: DataConfig {
                symbol: "BTCUSDT".to_string(),
                interval: "60".to_string(),
                sequence_length: 100,
                forecast_horizon: 24,
                batch_size: 32,
            },
            model: ModelConfig {
                hidden_dim: 256,
                time_emb_dim: 64,
                num_diffusion_steps: 1000,
                noise_schedule: "cosine".to_string(),
            },
            training: TrainingConfig {
                epochs: 100,
                learning_rate: 0.0001,
                grad_clip: 1.0,
            },
        }
    }
}

impl Config {
    /// Load configuration from file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: Config = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to file.
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }
}
