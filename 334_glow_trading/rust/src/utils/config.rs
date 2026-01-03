//! Configuration management

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Trading symbol
    pub symbol: String,
    /// Data interval
    pub interval: String,
    /// Number of features
    pub num_features: usize,
    /// GLOW model levels
    pub num_levels: usize,
    /// Flow steps per level
    pub num_steps: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Training epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Lookback period
    pub lookback: usize,
    /// Initial capital
    pub initial_capital: f64,
    /// Transaction cost
    pub transaction_cost: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            symbol: "BTCUSDT".to_string(),
            interval: "1h".to_string(),
            num_features: 16,
            num_levels: 3,
            num_steps: 4,
            hidden_dim: 64,
            epochs: 100,
            batch_size: 256,
            learning_rate: 1e-4,
            lookback: 20,
            initial_capital: 10000.0,
            transaction_cost: 0.001,
        }
    }
}

impl Config {
    /// Load configuration from file
    pub fn load<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let config = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Create config for specific symbol
    pub fn for_symbol(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            ..Default::default()
        }
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
        assert_eq!(config.symbol, "BTCUSDT");
        assert_eq!(config.num_features, 16);
    }

    #[test]
    fn test_config_save_load() {
        let config = Config::default();

        let mut temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_path_buf();

        config.save(&path).unwrap();
        let loaded = Config::load(&path).unwrap();

        assert_eq!(config.symbol, loaded.symbol);
        assert_eq!(config.num_features, loaded.num_features);
    }
}
